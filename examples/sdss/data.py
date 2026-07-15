"""Data layer for the SDSS spectral-reconstruction example: spectra streamed in place.

Pipeline
--------
1. **Index (build time, once).** Each SDSS ``spPlate-PLATE-MJD.fits`` holds all ~640 fibers of
   one plate observation as a single ``(fiber, wave) float32`` image on a *common* log-wavelength
   grid (unlike the per-object ``spec-lite`` files, which are trimmed to ragged lengths).
   :func:`build_store` opens each plate's ``PRIMARY`` flux HDU as a *virtual* dataset
   (VirtualiZarr -> ``kerchunk.fits``), concatenates them along the ``fiber`` axis, and commits
   the byte-range references to a local Icechunk repo. No flux is copied or resampled -- the store
   is a few kB of references pointing at the original FITS objects on ``data.sdss.org``.

2. **Stream (train time).** :func:`open_store` reopens the Icechunk repo (resolving the virtual
   chunks straight over HTTPS), and :func:`reconstruct_dataset` builds an
   :class:`~insitubatch.InSituDataset` with ``sample_axis=0`` (one fiber = one sample). Because a
   whole plate is one chunk, each decoded chunk yields ~640 samples -- the O(chunks)
   decode-amortization regime (contrast the Hubble example, one image per chunk). A per-fiber
   robust normalization (:func:`normalize`) runs vectorized on the decode pool; the per-sample
   reconstruction noise lives in the :class:`Corrupt` batch stage, per the transform-cost contract.

The task mirrors astroML's ``compute_sdss_pca`` (spectral reconstruction / eigenspectra): recover
the clean spectrum through a low-dimensional bottleneck. The baseline is **PCA** at the same latent
dimension -- the optimal *linear* reconstruction; a small autoencoder, trained by SGD over the
streamed mini-batches, beats it when the spectra lie on a nonlinear manifold. astroML fits its PCA
after downloading + resampling the archive into one ``spec4000.npz``; here we stream the common-grid
plates in place, no reshard.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import zarr
from zarr.abc.store import Store

if TYPE_CHECKING:
    from virtualizarr.manifests import ManifestArray

from insitubatch import (
    Batch,
    ChunkTransform,
    DecodedChunk,
    InSituDataset,
    ensure_local_dir,
    obstore_store,
    open_geometries,
    split_by_chunk,
)

FLUX_VAR = "flux"
SDSS_HOST = "https://data.sdss.org"
REDUX = "/sas/dr17/sdss/spectro/redux/26"  # legacy (DR8-era) reduction; spPlate lives per-plate
LATENT_DIM = 16  # bottleneck / PCA components -- the reconstruction budget both methods share
NOISE_SIGMA = 0.3  # per-sample input noise (normalized units), added in the batch stage

HERE = Path(__file__).resolve().parent
DEFAULT_URIS = HERE / "dr17_plate_uris.json"
DEFAULT_STORE = HERE / "sdss_store"


# --------------------------------------------------------------------------- offline synthetic
def make_synthetic_store(
    url: str,
    *,
    n_plates: int = 12,
    fibers_per_plate: int = 128,
    n_wave: int = 512,
    intrinsic_dim: int = 8,
    seed: int = 0,
) -> None:
    """Write a synthetic ``flux (fiber, wave)`` zarr mimicking the spPlate-derived store's geometry.

    For offline runs (``--source synthetic``) and drift tests -- no network, no FITS/VZ stack.
    One plate per chunk (``chunks=(fibers_per_plate, n_wave)``), like the real spPlate layout, so
    each chunk carries many fiber samples (the decode-amortization regime).

    Each spectrum is a gentle continuum plus a set of emission lines shifted by a per-fiber
    redshift, with per-fiber line amplitudes -- a low-dimensional but strongly *nonlinear* manifold
    (a moving line is not a linear combination of a few fixed templates). This mirrors real galaxy
    spectra, where varying redshift is exactly what makes a fixed-dimension linear PCA reconstruct
    poorly: at ``LATENT_DIM`` components PCA leaves the shifted lines smeared, while a nonlinear
    autoencoder of the same bottleneck learns the shift -- so the trained model beats the baseline.
    ``intrinsic_dim`` is accepted for API stability but the redshift+amplitude latent sets the true
    dimension.
    """
    del intrinsic_dim  # latent dimension is set by the redshift + per-line amplitudes below
    rng = np.random.default_rng(seed)
    n_fiber = n_plates * fibers_per_plate
    wave = np.linspace(0.0, 1.0, n_wave)

    rest_lines = np.linspace(0.12, 0.88, 5)  # rest-frame line centers
    width = 0.012
    zshift = rng.uniform(-0.12, 0.12, size=(n_fiber, 1))  # dominant nonlinear (redshift) parameter
    slope = rng.uniform(-1.0, 1.0, size=(n_fiber, 1))
    flux = 8.0 + 1.5 * slope * (wave[None, :] - 0.5)  # gentle continuum, e-/s-like pedestal
    for c in rest_lines:
        centers = c + zshift  # (n_fiber, 1): the line sweeps across the grid with redshift
        amp = rng.uniform(0.5, 3.0, size=(n_fiber, 1))
        flux = flux + amp * np.exp(-((wave[None, :] - centers) ** 2) / (2 * width**2))

    flux = flux.astype("f4")
    flux += rng.normal(0.0, 0.05, size=flux.shape).astype("f4")  # small intrinsic scatter
    bad = (rng.integers(0, n_fiber, 3 * n_plates), rng.integers(0, n_wave, 3 * n_plates))
    flux[bad] = np.nan  # a few bad pixels, as real coadds carry

    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    arr = group.create_array(
        FLUX_VAR,
        shape=flux.shape,
        chunks=(fibers_per_plate, n_wave),
        dtype="f4",
        dimension_names=("fiber", "wave"),
    )
    arr[:] = flux


# --------------------------------------------------------------------------- build time
#
# A single plate is ~640 fibers x ~3864 wave (~10 MB) -- it fits in memory, so one plate is a demo,
# not a streaming workload. The two build modes trade off along the FITS byte layout (no reshard in
# either -- both only rewrite the *chunk manifest*, never the pixels):
#
#   one plate   -> _single_plate_fiber_chunks: full wave width, many fibers per chunk
#                  (the O(chunks) decode-amortization regime; toy scale).
#   many plates -> _many_plates_common_grid:   plates cropped to a shared wavelength window and
#                  folded into one flat fiber axis, one fiber per chunk (the streaming regime; the
#                  only no-reshard way to concat ragged-width plates -- scales to the archive).
#
FIBERS_PER_CHUNK = 64  # one plate's 640 fibers -> 10 contiguous chunks


def _chunk_ref(array: ManifestArray) -> tuple[str, int]:
    """The ``(path, byte offset)`` of the single virtual chunk kerchunk emits for a plate HDU."""
    manifest = array.manifest
    return str(manifest._paths[0, 0]), int(manifest._offsets[0, 0])


def _grid_start(plate: xr.Dataset) -> float:
    return float(plate[FLUX_VAR].attrs["COEFF0"])  # log10(wavelength) of bin 0


def _grid_step(plate: xr.Dataset) -> float:
    return float(plate[FLUX_VAR].attrs["COEFF1"])  # dloglam per bin (identical across SDSS plates)


def _virtual_flux(
    template: xr.Dataset,
    shape: tuple[int, int],
    chunk_shape: tuple[int, int],
    entries: dict[str, dict[str, object]],
) -> xr.Dataset:
    """Assemble a virtual ``flux (fiber, wave)`` dataset from explicit byte-range chunk refs.

    Reuses ``template``'s dtype/codec metadata (from the opened plate) and overrides only the shape
    and chunk grid -- the chunks point back into the original FITS bytes, so nothing is copied.
    """
    from virtualizarr.manifests import ChunkManifest, ManifestArray

    metadata = template[FLUX_VAR].data.metadata
    grid = replace(metadata.chunk_grid, chunk_shape=chunk_shape)
    array = ManifestArray(
        metadata=replace(metadata, shape=shape, chunk_grid=grid),
        chunkmanifest=ChunkManifest(entries=entries),
    )
    return xr.Dataset({FLUX_VAR: xr.Variable(("fiber", "wave"), array)})


def _single_plate_fiber_chunks(plate: xr.Dataset, fibers_per_chunk: int) -> xr.Dataset:
    """One plate -> contiguous fiber-block chunks at full wave width (many fibers per chunk).

    kerchunk maps the ``(fiber, wave)`` HDU to ONE virtual chunk: the byte range
    ``[offset, offset + n_fiber*row_bytes)``. The fiber axis is the outer (row) axis, so fiber block
    ``i`` is the contiguous sub-range ``offset + i*fibers_per_chunk*row_bytes`` -- pure byte
    arithmetic. Each chunk still holds many fibers: the decode-amortization regime.
    """
    array = plate[FLUX_VAR].data
    path, base = _chunk_ref(array)
    n_fiber, n_wave = array.shape
    row_bytes = n_wave * array.dtype.itemsize
    if n_fiber % fibers_per_chunk:
        raise ValueError(f"{n_fiber} fibers not divisible by fibers_per_chunk={fibers_per_chunk}")

    block_bytes = fibers_per_chunk * row_bytes
    entries: dict[str, dict[str, object]] = {
        f"{i}.0": {"path": path, "offset": base + i * block_bytes, "length": block_bytes}
        for i in range(n_fiber // fibers_per_chunk)
    }
    return _virtual_flux(plate, (n_fiber, n_wave), (fibers_per_chunk, n_wave), entries)


def _many_plates_common_grid(plates: list[xr.Dataset]) -> xr.Dataset:
    """N plates -> one flat fiber axis on a shared wavelength window, one fiber per chunk.

    Plates cover slightly different wavelength ranges, so they cannot share a rectangular ``wave``
    axis at full width. But every plate uses the same ``dloglam`` (COEFF1) and their start
    wavelengths (COEFF0) differ by whole bins, so cropping each plate to the common overlap window
    lands them on ONE grid *exactly* -- no resampling. After the crop each fiber's window is still a
    contiguous byte sub-range, so a fiber is one virtual chunk and the plates concatenate into a
    flat ``(total_fiber, width)`` sample axis. One fiber per chunk is the streaming regime, but it
    scales to the whole archive (thousands of plates) with no reshard and no download.
    """
    step = _grid_step(plates[0])
    starts = [_grid_start(p) for p in plates]
    window_lo = max(starts)
    window_hi = min(
        start + p[FLUX_VAR].data.shape[1] * step for p, start in zip(plates, starts, strict=True)
    )
    width = round((window_hi - window_lo) / step)

    entries: dict[str, dict[str, object]] = {}
    fiber = 0
    for plate, start in zip(plates, starts, strict=True):
        array = plate[FLUX_VAR].data
        path, base = _chunk_ref(array)
        n_fiber, n_wave = array.shape
        itemsize = array.dtype.itemsize
        row_bytes = n_wave * itemsize
        offset_bins = round((window_lo - start) / step)
        if abs((window_lo - start) / step - offset_bins) > 1e-3:
            raise ValueError("plate wavelength grids are not bin-aligned; cannot crop losslessly")
        window_bytes = width * itemsize
        for f in range(n_fiber):
            byte0 = base + f * row_bytes + offset_bins * itemsize
            entries[f"{fiber}.0"] = {"path": path, "offset": byte0, "length": window_bytes}
            fiber += 1
    return _virtual_flux(plates[0], (fiber, width), (1, width), entries)


def build_store(
    plate_urls: list[str],
    store_path: str | os.PathLike[str] = DEFAULT_STORE,
    *,
    fibers_per_chunk: int = FIBERS_PER_CHUNK,
) -> str:
    """Index one or more SDSS ``spPlate`` FITS (HTTPS) into a local Icechunk repo of virtual refs.

    Idempotent: rebuilds ``store_path`` from scratch. Requires the ``astronomy`` extra
    (``virtualizarr``, ``kerchunk``, ``astropy``, ``icechunk``). A **single** plate becomes
    full-width fiber-block chunks (many fibers per chunk -- decode-amortization); **several** plates
    are cropped to a shared wavelength window and folded into one flat fiber axis (one fiber per
    chunk -- streaming at archive scale). Both modes move no pixels -- see
    :func:`_single_plate_fiber_chunks` and :func:`_many_plates_common_grid`.
    """
    import shutil

    import icechunk
    from obstore.store import HTTPStore
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import FITSParser
    from virtualizarr.registry import ObjectStoreRegistry

    store_path = str(store_path)
    shutil.rmtree(store_path, ignore_errors=True)

    registry = ObjectStoreRegistry({SDSS_HOST: HTTPStore.from_url(SDSS_HOST)})
    plates = [
        open_virtual_dataset(url=u, registry=registry, parser=FITSParser()).rename(
            {"PRIMARY": FLUX_VAR}
        )
        for u in plate_urls
    ]
    virtual = (
        _single_plate_fiber_chunks(plates[0], fibers_per_chunk)
        if len(plates) == 1
        else _many_plates_common_grid(plates)
    )

    ice_prefix = SDSS_HOST + "/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(ice_prefix, icechunk.http_store())
    )
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(store_path),
        config=config,
        authorize_virtual_chunk_access=icechunk.containers_credentials({ice_prefix: None}),
    )
    session = repo.writable_session("main")
    virtual.virtualize.to_icechunk(session.store)
    session.commit(f"index {len(plate_urls)} SDSS spPlate frame(s)")
    return store_path


def open_store(store_path: str | os.PathLike[str] = DEFAULT_STORE) -> Store:
    """Reopen the Icechunk repo built by :func:`build_store` as a read-only zarr ``Store``.

    Resolves virtual chunks straight over HTTPS from ``data.sdss.org``. Needs ``icechunk`` (a Rust
    read path -- not ``kerchunk``): the build-time index libraries are absent from here.
    """
    import icechunk

    ice_prefix = SDSS_HOST + "/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(ice_prefix, icechunk.http_store())
    )
    repo = icechunk.Repository.open(
        icechunk.local_filesystem_storage(str(store_path)),
        config=config,
        authorize_virtual_chunk_access=icechunk.containers_credentials({ice_prefix: None}),
    )
    return repo.readonly_session("main").store


# --------------------------------------------------------------------------- transforms
def normalize(chunk: DecodedChunk) -> DecodedChunk:
    """chunk_transform: NaN-scrub + per-fiber robust standardization, vectorized.

    Coadded spectra span a wide flux range with bad-pixel NaNs. Subtract the per-fiber median and
    divide by a MAD-based scale, then clip -- so a reconstruction MSE is well-conditioned across
    fibers. Per-(variable, chunk), deterministic, pure numpy: the cacheable chunk stage, and it
    releases the GIL on the decode pool.
    """
    if chunk.read.array != FLUX_VAR:
        return chunk
    x = np.nan_to_num(chunk.data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True)
    x = (x - med) / (1.4826 * mad + 1e-6)
    chunk.data = np.clip(x, -5.0, 5.0).astype(np.float32)
    return chunk


@dataclass
class Corrupt:
    """batch_transform: split each spectrum into a ``(noisy, clean)`` pair with fresh input noise.

    The reconstruction target is the clean normalized spectrum; the model sees a noisy copy. The
    noise is per-sample random, so it belongs in the (uncached) batch stage, not the chunk stage.
    ``clean``/``noisy`` are ``(B, W)`` float32.
    """

    sigma: float = NOISE_SIGMA
    seed: int | None = None

    def __call__(self, batch: Batch) -> Batch:
        clean = batch.arrays.pop(FLUX_VAR).astype(np.float32)  # (B, W)
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, self.sigma, size=clean.shape).astype(np.float32)
        batch.arrays["clean"] = clean
        batch.arrays["noisy"] = clean + noise
        return batch


# --------------------------------------------------------------------------- dataset
def reconstruct_dataset(
    store: Store,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    sigma: float = NOISE_SIGMA,
    cache_dir: str | None = None,
    max_inflight: int | None = None,
) -> InSituDataset:
    """One reconstruction dataset over the SDSS store: iterate ``.train`` / ``.val`` / ``.test``.

    Each batch carries ``noisy`` and ``clean`` ``(B, W)`` spectra. Split is by chunk (plate), so no
    fiber leaks across train/val/test. ``max_inflight`` is unset by default (``data.sdss.org`` over
    HTTPS is not an anonymous-S3 throttler like MAST); raise/cap it to tune read-ahead.
    """
    geoms = open_geometries(store, variables=[FLUX_VAR], sample_axis=0)
    chunk_transforms: list[ChunkTransform] = [normalize]
    return InSituDataset(
        store,
        split_by_chunk(geoms[FLUX_VAR], fractions=(0.7, 0.15, 0.15)),
        geometries=geoms,
        batch_size=batch_size,
        shuffle=shuffle,
        cache_dir=cache_dir,
        max_inflight=max_inflight,
        chunk_transforms=chunk_transforms,
        batch_transforms=[Corrupt(sigma=sigma)],
    )


# --------------------------------------------------------------------------- metrics / baseline
def collect(ds: InSituDataset, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a split into ``(noisy, clean)`` matrices ``(N, W)``.

    Used to fit/score the PCA baseline -- which, like astroML's ``spec4000`` workflow (and unlike
    the streamed autoencoder), needs every spectrum resident at once.
    """
    noisy, clean = [], []
    for b in getattr(ds, split):
        noisy.append(b.arrays["noisy"])
        clean.append(b.arrays["clean"])
    return np.concatenate(noisy), np.concatenate(clean)


def fit_pca(clean: np.ndarray, k: int = LATENT_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Fit a ``k``-component PCA on clean training spectra: return ``(mean, components)``."""
    mean = clean.mean(axis=0)
    _, _, vt = np.linalg.svd(clean - mean, full_matrices=False)
    return mean.astype(np.float32), vt[:k].astype(np.float32)


def pca_reconstruct(noisy: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Reconstruct spectra by projecting onto the top-``k`` PCA subspace (the linear baseline)."""
    centered = noisy - mean
    return (mean + (centered @ components.T) @ components).astype(np.float32)


def recon_mse(pred: np.ndarray, clean: np.ndarray) -> float:
    """Mean squared reconstruction error (lower is better)."""
    return float(np.mean((pred.astype(np.float64) - clean.astype(np.float64)) ** 2))


# --------------------------------------------------------------------------- CLI glue
def load_uris(path: str | os.PathLike[str] = DEFAULT_URIS) -> list[str]:
    """Load the cached list of SDSS spPlate URLs."""
    return json.loads(Path(path).read_text())


def build_datasets(args: argparse.Namespace) -> InSituDataset:
    """One reconstruction dataset from CLI args: offline ``synthetic`` spectra (written fresh to a
    temp store) or the real ``sdss`` Icechunk store (built from URIs first if ``--build``)."""
    if args.source == "synthetic":
        import tempfile

        url = f"file://{tempfile.mkdtemp()}/sdss_synth.zarr"
        make_synthetic_store(url, n_plates=args.n_plates, fibers_per_plate=args.fibers)
        store = obstore_store(url)
    else:
        if args.build:
            plate_urls = load_uris(args.uris)[: args.plates]
            print(f"building store from {len(plate_urls)} plate(s): {plate_urls} ...")
            build_store(plate_urls, args.store)
        store = open_store(args.store)
    return reconstruct_dataset(store, batch_size=args.batch_size, sigma=args.sigma)


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDSS spectral reconstruction -- train in place.")
    p.add_argument(
        "--source",
        choices=("synthetic", "sdss"),
        default="synthetic",
        help="offline synthetic spectra | real SDSS spPlate frames (needs --build once)",
    )
    p.add_argument("--uris", default=str(DEFAULT_URIS), help="JSON list of spPlate FITS URLs")
    p.add_argument("--store", default=str(DEFAULT_STORE), help="local Icechunk repo path")
    p.add_argument("--build", action="store_true", help="(re)build the sdss store first")
    p.add_argument(
        "--plates",
        type=int,
        default=1,
        help="real spPlate count: 1 = full-width many-fibers/chunk (decode-amortization); "
        ">1 = common-window 1-fiber/chunk (streaming at archive scale)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    p.add_argument("--sigma", type=float, default=NOISE_SIGMA)
    p.add_argument("--n-plates", type=int, default=12, help="synthetic plate count")
    p.add_argument("--fibers", type=int, default=128, help="synthetic fibers per plate")
    return p.parse_args(argv)
