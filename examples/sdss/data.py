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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from zarr.abc.store import Store

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
FIBERS_PER_CHUNK = 64  # split one plate's fibers into contiguous virtual chunks (640 -> 10 chunks)


def _rechunk_fibers(plate: xr.Dataset, fibers_per_chunk: int) -> xr.Dataset:
    """Split a plate's whole-image virtual chunk into contiguous fiber-block chunks -- no reshard.

    ``kerchunk.fits`` maps the ``(fiber, wave)`` HDU to a single virtual chunk: one byte range
    ``[offset, offset + n_fiber*n_wave*itemsize)``. Because the fiber axis is the *outer* axis of
    the row-major image, fiber block ``i`` occupies a *contiguous* sub-range
    ``offset + i*fibers_per_chunk*n_wave*itemsize`` of length ``fibers_per_chunk*n_wave*itemsize``.
    So we rebuild the manifest as ``n_fiber // fibers_per_chunk`` sub-refs -- pure byte arithmetic,
    no pixels moved -- giving insitubatch multiple sample-axis chunks (each still many fibers, the
    decode-amortization regime) that :func:`~insitubatch.split_by_chunk` can partition.
    """
    import dataclasses

    from virtualizarr.manifests import ChunkManifest, ManifestArray

    ma = plate[FLUX_VAR].data
    manifest = ma.manifest
    path = str(manifest._paths[0, 0])
    base_offset = int(manifest._offsets[0, 0])
    n_fiber, n_wave = ma.shape
    row_bytes = n_wave * ma.dtype.itemsize
    if n_fiber % fibers_per_chunk:
        raise ValueError(f"{n_fiber} fibers not divisible by fibers_per_chunk={fibers_per_chunk}")

    block_bytes = fibers_per_chunk * row_bytes
    entries = {
        f"{i}.0": {"path": path, "offset": base_offset + i * block_bytes, "length": block_bytes}
        for i in range(n_fiber // fibers_per_chunk)
    }
    grid = dataclasses.replace(ma.metadata.chunk_grid, chunk_shape=(fibers_per_chunk, n_wave))
    rechunked = ManifestArray(
        metadata=dataclasses.replace(ma.metadata, chunk_grid=grid),
        chunkmanifest=ChunkManifest(entries=entries),
    )
    return xr.Dataset({FLUX_VAR: xr.Variable(("fiber", "wave"), rechunked)})


def build_store(
    plate_url: str,
    store_path: str | os.PathLike[str] = DEFAULT_STORE,
    *,
    fibers_per_chunk: int = FIBERS_PER_CHUNK,
) -> str:
    """Index one SDSS ``spPlate`` FITS (over HTTPS) into a local Icechunk repo of virtual refs.

    Idempotent: rebuilds ``store_path`` from scratch. Requires the ``astronomy`` extra
    (``virtualizarr``, ``kerchunk``, ``astropy``, ``icechunk``). The plate's ``PRIMARY`` flux HDU
    -- all ~640 fibers on one common log-wavelength grid -- is virtually re-chunked along the fiber
    axis (:func:`_rechunk_fibers`) so each fiber is one sample and each chunk is many fibers. (Only
    one plate: different plates cover slightly different wavelength ranges, so they cannot share a
    rectangular ``wave`` axis without resampling -- exactly the reshard this example avoids.)
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
    plate = open_virtual_dataset(url=plate_url, registry=registry, parser=FITSParser()).rename(
        {"PRIMARY": FLUX_VAR}
    )
    rechunked = _rechunk_fibers(plate, fibers_per_chunk)

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
    rechunked.virtualize.to_icechunk(session.store)
    session.commit(f"index SDSS spPlate {plate_url.rsplit('/', 1)[-1]}")
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
            plate_url = load_uris(args.uris)[0]
            print(f"building store from {plate_url} ...")
            build_store(plate_url, args.store)
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    p.add_argument("--sigma", type=float, default=NOISE_SIGMA)
    p.add_argument("--n-plates", type=int, default=12, help="synthetic plate count")
    p.add_argument("--fibers", type=int, default=128, help="synthetic fibers per plate")
    return p.parse_args(argv)
