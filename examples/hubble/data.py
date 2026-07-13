"""Data layer for the Hubble denoising example: real WFC3/IR frames, streamed in place.

Pipeline
--------
1. **Index (build time, once).** Each Hubble ``_flt.fits`` on MAST's public S3 bucket
   holds one ``SCI`` image ``(1014, 1014) float32``. :func:`build_store` opens each as a
   *virtual* dataset (VirtualiZarr -> ``kerchunk.fits``), concatenates them along a new
   ``frame`` axis, and commits the byte-range references to a local Icechunk repo. No pixels
   are copied -- the store is a few kB of references pointing at the original FITS objects.
   ``kerchunk``/``virtualizarr``/``astropy`` are needed *only here*, never at train time.

2. **Stream (train time).** :func:`open_store` reopens the Icechunk repo (resolving the
   virtual chunks straight from S3), and :func:`denoise_dataset` builds an
   :class:`~insitubatch.InSituDataset`: ``sample_axis=0`` makes each frame one sample (one
   image = one chunk). Two chunk stages -- :func:`clean_normalize` then
   :class:`~examples.transforms.Coarsen` -- run vectorized on the decode pool; the per-sample
   random noise lives in the :class:`AddNoise` batch stage, per the transform-cost contract.

The ML task is deliberately simple (Gaussian-noise removal, a didactic stand-in) -- the point
is that we train on the real archive with no reshard, not that this is a SOTA denoiser.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from zarr.abc.store import Store

from examples.transforms import Coarsen
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

SCI_VAR = "SCI"
BUCKET = "stpubdata"
S3_PREFIX = "s3://stpubdata"
REGION = "us-east-1"
COARSEN = 4  # 1014 -> 253 on each axis: keeps the CPU demo light, still whole-frame reads
NOISE_SIGMA = 0.6  # in normalized (robust-standardized) units

HERE = Path(__file__).resolve().parent
DEFAULT_URIS = HERE / "wfc3ir_m16_uris.json"
DEFAULT_STORE = HERE / "hubble_store"


# --------------------------------------------------------------------------- offline synthetic
def make_synthetic_store(url: str, *, n_frames: int = 12, size: int = 128, seed: int = 0) -> None:
    """Write a synthetic ``SCI (frame, y, x)`` zarr mimicking the FITS-derived store's geometry.

    For offline runs (``--source synthetic``) and drift tests -- no network, no FITS/VZ stack.
    One frame per chunk (``chunks=(1, size, size)``), like the real ``_flt.fits`` layout. Each
    frame holds sharp point sources (stars) on a bright pedestal (a wide e-/s-like dynamic range)
    with a handful of NaN bad pixels, so :func:`clean_normalize`'s NaN-scrub and robust scaling are
    exercised as on real WFC3/IR data. The sources are sharp on purpose: a median filter blurs
    them, so a CNN that preserves point sources beats the baseline (as it does on real stars).
    """
    rng = np.random.default_rng(seed)
    gy, gx = np.mgrid[0:size, 0:size].astype(np.float64)
    frames = np.empty((n_frames, size, size), dtype="f4")
    for i in range(n_frames):
        img = np.full((size, size), rng.uniform(20.0, 80.0))  # bright background pedestal
        for _ in range(20):  # sharp stars: a 3x3 median blurs/erases them, the CNN keeps them
            cy, cx = rng.uniform(0, size, size=2)
            r = rng.uniform(1.0, 3.0)
            amp = rng.uniform(80.0, 600.0)
            img += amp * np.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / (2 * r * r))
        bad = (rng.integers(0, size, 5), rng.integers(0, size, 5))
        img[bad] = np.nan  # a few bad pixels, as WFC3/IR flt frames carry
        frames[i] = img.astype("f4")

    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    arr = group.create_array(
        SCI_VAR,
        shape=frames.shape,
        chunks=(1, size, size),
        dtype="f4",
        dimension_names=("frame", "y", "x"),
    )
    arr[:] = frames


# --------------------------------------------------------------------------- build time
def build_store(
    uris: list[str],
    store_path: str | os.PathLike[str] = DEFAULT_STORE,
    *,
    region: str = REGION,
) -> str:
    """Index ``uris`` (Hubble ``_flt.fits`` on S3) into a local Icechunk repo of virtual refs.

    Idempotent: rebuilds ``store_path`` from scratch. Requires the build-time stack
    (``virtualizarr``, ``kerchunk``, ``astropy``, ``icechunk``, ``s3fs`` for anonymous S3).
    """
    import shutil

    import icechunk
    import xarray as xr
    from obstore.store import S3Store
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import FITSParser
    from virtualizarr.registry import ObjectStoreRegistry

    store_path = str(store_path)
    shutil.rmtree(store_path, ignore_errors=True)

    registry = ObjectStoreRegistry({S3_PREFIX: S3Store(BUCKET, region=region, skip_signature=True)})
    parser = FITSParser(reader_options={"storage_options": {"anon": True}})
    frames = [
        open_virtual_dataset(url=u, registry=registry, parser=parser).expand_dims("frame")
        for u in uris
    ]
    combined = xr.concat(frames, dim="frame")  # (frame, 1014, 1014), one image per chunk

    ice_prefix = S3_PREFIX + "/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(ice_prefix, icechunk.s3_store(region=region, anonymous=True))
    )
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(store_path),
        config=config,
        authorize_virtual_chunk_access=icechunk.containers_credentials(
            {ice_prefix: icechunk.s3_anonymous_credentials()}
        ),
    )
    session = repo.writable_session("main")
    combined.virtualize.to_icechunk(session.store)
    session.commit(f"index {len(uris)} Hubble WFC3/IR frames")
    return store_path


def open_store(
    store_path: str | os.PathLike[str] = DEFAULT_STORE, *, region: str = REGION
) -> Store:
    """Reopen the Icechunk repo built by :func:`build_store` as a read-only zarr ``Store``.

    Resolves virtual chunks straight from the public S3 objects. Needs ``icechunk`` (a Rust
    read path -- not ``kerchunk``): the build-time index libraries are absent from here.
    """
    import icechunk

    ice_prefix = S3_PREFIX + "/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(ice_prefix, icechunk.s3_store(region=region, anonymous=True))
    )
    repo = icechunk.Repository.open(
        icechunk.local_filesystem_storage(str(store_path)),
        config=config,
        authorize_virtual_chunk_access=icechunk.containers_credentials(
            {ice_prefix: icechunk.s3_anonymous_credentials()}
        ),
    )
    return repo.readonly_session("main").store


# --------------------------------------------------------------------------- transforms
def clean_normalize(chunk: DecodedChunk) -> DecodedChunk:
    """chunk_transform: NaN-scrub + per-frame robust standardization, vectorized.

    WFC3/IR ``SCI`` frames are in e-/s with bad-pixel NaNs and a huge dynamic range. Subtract
    the per-frame median and divide by a MAD-based scale, then clip -- so MSE denoising is
    well-conditioned. Per-(variable, chunk), deterministic, pure numpy: the cacheable chunk
    stage, and it releases the GIL on the decode pool.
    """
    if chunk.read.array != SCI_VAR:
        return chunk
    x = np.nan_to_num(chunk.data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(x, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(x - med), axis=(1, 2), keepdims=True)
    x = (x - med) / (1.4826 * mad + 1e-6)
    chunk.data = np.clip(x, -5.0, 5.0).astype(np.float32)
    return chunk


@dataclass
class AddNoise:
    """batch_transform: split each frame into a ``(noisy, clean)`` pair with fresh Gaussian noise.

    Per-sample random, so it belongs in the (uncached) batch stage, not the chunk stage.
    Adds a channel axis: ``clean``/``noisy`` are ``(B, 1, H, W)`` float32.
    """

    sigma: float = NOISE_SIGMA
    seed: int | None = None

    def __call__(self, batch: Batch) -> Batch:
        clean = batch.arrays.pop(SCI_VAR)[:, None, :, :].astype(np.float32)  # (B, 1, H, W)
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, self.sigma, size=clean.shape).astype(np.float32)
        batch.arrays["clean"] = clean
        batch.arrays["noisy"] = clean + noise
        return batch


# --------------------------------------------------------------------------- dataset
def denoise_dataset(
    store: Store,
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    coarsen: int = COARSEN,
    sigma: float = NOISE_SIGMA,
    cache_dir: str | None = None,
    max_inflight: int | None = 4,
) -> InSituDataset:
    """One denoising dataset over the Hubble store: iterate ``.train`` / ``.val`` / ``.test``.

    Each batch carries ``noisy`` and ``clean`` ``(B, 1, H, W)`` frames. ``coarsen`` block-means
    the frame (``1014 -> 1014//coarsen``) to keep the CPU demo light while still reading whole
    frames from S3 (the no-reshard, train-in-place stance).

    ``max_inflight`` is capped low by default: MAST's *anonymous* public bucket throttles
    (HTTP 503 SlowDown) under heavy concurrent read-ahead, and the virtual-reference fetch does
    not retry. A benchmark on AWS would use authenticated/retrying access and raise this.
    """
    geoms = open_geometries(store, variables=[SCI_VAR], sample_axis=0)
    chunk_transforms: list[ChunkTransform] = [clean_normalize, Coarsen(factor=coarsen)]
    return InSituDataset(
        store,
        split_by_chunk(geoms[SCI_VAR], fractions=(0.7, 0.15, 0.15)),
        geometries=geoms,
        batch_size=batch_size,
        shuffle=shuffle,
        cache_dir=cache_dir,
        max_inflight=max_inflight,
        chunk_transforms=chunk_transforms,
        batch_transforms=[AddNoise(sigma=sigma)],
    )


# --------------------------------------------------------------------------- metrics / baseline
def psnr(pred: np.ndarray, clean: np.ndarray, data_range: float = 10.0) -> float:
    """Peak signal-to-noise ratio (dB). ``data_range`` is the clipped span (-5..5 -> 10)."""
    mse = float(np.mean((pred.astype(np.float64) - clean.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(data_range**2 / mse)


def median_baseline(noisy: np.ndarray, size: int = 3) -> np.ndarray:
    """Naive denoiser: a per-frame median filter -- the no-training reference to beat."""
    from scipy.ndimage import median_filter

    out = np.empty_like(noisy)
    for i in range(noisy.shape[0]):  # (B, 1, H, W)
        out[i, 0] = median_filter(noisy[i, 0], size=size)
    return out


# --------------------------------------------------------------------------- CLI glue
def load_uris(path: str | os.PathLike[str] = DEFAULT_URIS) -> list[str]:
    """Load the cached list of Hubble frame S3 URIs (curated from MAST via astroquery)."""
    return json.loads(Path(path).read_text())


def build_datasets(args: argparse.Namespace) -> InSituDataset:
    """One denoising dataset from CLI args: offline ``synthetic`` frames (written fresh to a temp
    store) or the real ``hubble`` Icechunk store (built from URIs first if ``--build``)."""
    if args.source == "synthetic":
        import tempfile

        url = f"file://{tempfile.mkdtemp()}/hubble_synth.zarr"
        make_synthetic_store(url, n_frames=args.n_frames, size=args.size)
        store = obstore_store(url)
    else:
        if args.build:
            print(f"building store from {args.uris} ...")
            build_store(load_uris(args.uris), args.store)
        store = open_store(args.store)
    return denoise_dataset(
        store, batch_size=args.batch_size, coarsen=args.coarsen, sigma=args.sigma
    )


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hubble WFC3/IR denoising -- train in place.")
    p.add_argument(
        "--source",
        choices=("synthetic", "hubble"),
        default="synthetic",
        help="offline synthetic frames | real Hubble WFC3/IR frames (needs --build once)",
    )
    p.add_argument("--uris", default=str(DEFAULT_URIS), help="JSON list of _flt.fits S3 URIs")
    p.add_argument("--store", default=str(DEFAULT_STORE), help="local Icechunk repo path")
    p.add_argument("--build", action="store_true", help="(re)build the hubble store first")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--coarsen", type=int, default=COARSEN)
    p.add_argument("--sigma", type=float, default=NOISE_SIGMA)
    p.add_argument("--n-frames", type=int, default=12, help="synthetic frame count")
    p.add_argument("--size", type=int, default=128, help="synthetic frame size (Y=X)")
    return p.parse_args(argv)
