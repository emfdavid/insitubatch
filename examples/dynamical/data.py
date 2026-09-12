"""Shared data for the downscaling example: the store, the one dataset, the eval.

**The task -- spatial downscaling of surface solar irradiance.** Each sample is one hourly
field. The field is cropped to a European window, block-averaged to a coarse grid, and the
model must put the fine structure back. Input and target are therefore two resolutions of the
*same* array, which is what keeps this to **one variable** -- on ``noaa-gfs-analysis`` a second
variable costs another 13.7 GiB of residency, because a stored chunk is 6.87 GiB and the
residency floor is two blocks of it.

**Bilinear upsampling is the no-model baseline** a useful model must beat -- the downscaling
analog of persistence in the advection example. It is also a data fingerprint: its RMSE depends
only on the field, so a change in it across runs means the loader handed over different data.

On the synthetic store the sub-grid structure is a deterministic function of the smooth field,
so a CNN beats bilinear by construction. On the real GFS archive the claim is "same pipeline,
real data, no reshard" -- not SOTA downscaling.

``downscaling_dataset`` returns one dataset whose label is always ``ghi`` regardless of the
store, so the training file is store-agnostic.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import zarr
from zarr.abc.store import Store

from insitubatch import (
    Batch,
    InSituDataset,
    ensure_local_dir,
    icechunk_store,
    obstore_store,
    open_geometries,
    split_by_chunk,
)

from .._logging import add_log_level, configure_logging

# dynamical.org's NOAA GFS analysis: hourly global 0.25-degree fields in a public Icechunk
# repository, read anonymously. `dynamical_catalog.get_store("noaa-gfs-analysis")` returns the
# same zarr Store if you would rather address it by catalog id than by URL.
GFS_URL = "s3://dynamical-noaa-gfs/noaa-gfs-analysis/v0.1.0.icechunk"
GFS_REGION = "us-west-2"
GFS_VAR = "downward_short_wave_radiation_flux_surface"  # surface GHI, W/m^2
GFS_SAMPLES_PER_CHUNK = 1440

LABEL = "ghi"  # canonical dataset label (store-independent)

# The model domain, as (south, north, west, east) degrees with longitude in [-180, 180). Both
# grids run latitude 90 -> -90 and longitude 0 -> 360, so a European window wraps the prime
# meridian -- `region_index` returns wrapping column indices rather than a slice.
REGION = (35.0, 65.0, -15.0, 25.0)
COARSEN = 4  # 0.25 degree -> 1 degree on GFS


def _gfs_store() -> Store:
    """Anonymous read-only session on dynamical.org's public GFS analysis repository."""
    return icechunk_store(GFS_URL, anonymous=True, region=GFS_REGION)


def region_index(
    nlat: int, nlon: int, region: tuple[float, float, float, float] = REGION
) -> tuple[np.ndarray, np.ndarray]:
    """Row and (wrapping) column indices of a lat/lon box on a global equiangular grid.

    Latitude runs 90 -> -90 over ``nlat`` points and longitude 0 -> 360 over ``nlon``, which is
    how both the GFS archive and the synthetic store are laid out, so one window definition in
    degrees serves every resolution. Columns are returned in west-to-east order and wrapped
    modulo ``nlon``, since a European box straddles the prime meridian. Both are trimmed to a
    multiple of :data:`COARSEN` so the block average divides evenly.
    """
    south, north, west, east = region
    dlat, dlon = 180.0 / (nlat - 1), 360.0 / nlon
    r0 = int(np.ceil((90.0 - north) / dlat))
    n_rows = int(np.floor((90.0 - south) / dlat)) - r0 + 1
    c0 = int(np.ceil((west % 360.0) / dlon))
    n_cols = int(np.floor(((east - west) % 360.0) / dlon)) + 1
    n_rows -= n_rows % COARSEN
    n_cols -= n_cols % COARSEN
    if n_rows < COARSEN or n_cols < COARSEN:
        raise ValueError(
            f"region {region} covers {n_rows}x{n_cols} cells of a {nlat}x{nlon} grid, which is "
            f"smaller than the coarsening factor {COARSEN}. Widen the region or the grid."
        )
    return np.arange(r0, r0 + n_rows), (c0 + np.arange(n_cols)) % nlon


def block_mean(field: np.ndarray, factor: int = COARSEN) -> np.ndarray:
    """Average ``(B, H, W) -> (B, H//factor, W//factor)``. Both axes must divide evenly."""
    b, h, w = field.shape
    return field.reshape(b, h // factor, factor, w // factor, factor).mean(axis=(2, 4))


def bilinear_upsample(coarse: np.ndarray, factor: int = COARSEN) -> np.ndarray:
    """Bilinear upsample ``(B, h, w) -> (B, h*factor, w*factor)``, separable and vectorized.

    Cell centres, edge-clamped: output cell ``j`` samples source position
    ``(j + 0.5) / factor - 0.5``. This is the baseline prediction -- everything a downscaler
    gets for free from the coarse field alone, with no model.
    """

    def axis_weights(n_out: int, n_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.clip((np.arange(n_out) + 0.5) / factor - 0.5, 0, n_in - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, n_in - 1)
        return lo, hi, (pos - lo).astype(np.float32)

    lo_y, hi_y, wy = axis_weights(coarse.shape[1] * factor, coarse.shape[1])
    lo_x, hi_x, wx = axis_weights(coarse.shape[2] * factor, coarse.shape[2])
    rows = coarse[:, lo_y] * (1 - wy)[:, None] + coarse[:, hi_y] * wy[:, None]
    return rows[:, :, lo_x] * (1 - wx) + rows[:, :, hi_x] * wx


def downscaling_dataset(
    store: Store,
    *,
    var: str = GFS_VAR,
    sample_range: tuple[int, int] | None = None,
    batch_size: int = 32,
    block_chunks: int = 16,
    max_inflight: int | None = None,
    shuffle: bool = True,
    cache_dir: str | None = None,
) -> InSituDataset:
    """One dataset holding one variable, labelled ``ghi``, sampled over time.

    ``store`` is a zarr Store -- build it with :func:`~insitubatch.icechunk_store` for a
    dynamical.org repository or :func:`~insitubatch.obstore_store` for a plain zarr URL.
    ``var`` names the array in the store; the dataset's label is always ``ghi`` so the model
    code is store-agnostic. ``sample_range`` restricts the split to a finite time window --
    the way to try a deep-chunked archive without committing to all of it, remembering that a
    range landing mid-chunk still pulls that chunk in whole.

    ``block_chunks`` is the memory knob that matters here: the residency floor is two blocks of
    stored chunks, so on GFS (6.87 GiB a chunk) anything above ``1`` will not fit a workstation.
    """
    opened = open_geometries(store, variables=[var])
    geoms = {LABEL: opened[var]}
    # Splits are chunk-granular, so the usual (0.8, 0.1, 0.1) rounds val and test to zero
    # chunks on an archive this deep -- a quarter each is the smallest honest split here.
    manifest = split_by_chunk(opened[var], fractions=(0.5, 0.25, 0.25), sample_range=sample_range)
    return InSituDataset(
        store,
        manifest,
        geometries=geoms,
        batch_size=batch_size,
        block_chunks=block_chunks,
        max_inflight=max_inflight,
        shuffle=shuffle,
        cache_dir=cache_dir,
    )


def inputs_and_targets(batch: Batch, factor: int = COARSEN) -> tuple[np.ndarray, np.ndarray]:
    """Split a ``Batch`` into ``(x, target)``: the upsampled coarse field and the native one.

    ``ghi`` arrives as ``(B, nlat, nlon)`` -- the whole global field, because the field axes are
    carried whole and the stored shard is the read unit. The model's domain is the European
    window, so the crop happens here rather than in the loader. ``target`` is the native-
    resolution crop; ``x`` is that crop block-averaged by ``factor`` and bilinearly restored to
    the same shape, so both are ``(B, H, W)`` float32 in W/m^2 and ``x`` *is* the baseline
    prediction.
    """
    field = batch.arrays[LABEL]
    rows, cols = region_index(field.shape[1], field.shape[2])
    target = field[:, rows][:, :, cols].astype("f4")
    return bilinear_upsample(block_mean(target, factor), factor), target


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root-mean-square error in W/m^2 over every cell of every sample."""
    return float(np.sqrt(np.mean((pred.astype("f8") - target.astype("f8")) ** 2)))


def evaluate(view: Iterable[Batch], predict: Callable[[Batch], np.ndarray]) -> tuple[float, float]:
    """RMSE on a held-out split view (e.g. ``ds.val``): ``(model_rmse, bilinear_rmse)``.

    ``predict`` maps a ``Batch`` to the model's ``(B, H, W)`` field, so this stays framework-
    neutral. Bilinear upsampling -- the best the coarse field can do with no model -- is the
    baseline. The view is deterministic (eval splits don't shuffle), so no epoch is set.
    """
    preds, targets, bilinears = [], [], []
    for batch in view:
        x, target = inputs_and_targets(batch)
        preds.append(predict(batch))
        targets.append(target)
        bilinears.append(x)
    if not preds:
        raise RuntimeError(
            "no batches to evaluate: the split this view covers is empty. A deep-chunked store "
            "splits by whole chunks, so a narrow --sample-range can round a split to zero of "
            "them -- widen the range. Reporting RMSE over no samples would be worse than "
            "stopping here."
        )
    pred, target, bilinear = (np.concatenate(a) for a in (preds, targets, bilinears))
    return rmse(pred, target), rmse(bilinear, target)


def make_irradiance_store(
    url: str,
    *,
    n_steps: int = 512,
    nlat: int = 91,
    nlon: int = 180,
    sample_chunk: int = 128,
    tile: int = 32,
    seed: int = 0,
    compress: bool = True,
) -> None:
    """Write a synthetic global irradiance archive shaped like the GFS one, only small.

    ``(n_steps, nlat, nlon)`` chunked ``(sample_chunk, tile, tile)``: many samples inside one
    stored chunk, the field tiled across several of them, and a tile grid that does *not*
    divide the field -- the same three properties that make the real archive interesting and
    expensive, at a size that runs anywhere.

    Irradiance is a clear-sky diurnal cycle attenuated by an advecting cloud field. The cloud
    carries structure below the coarsening scale (``sin`` of the smooth field, which oscillates
    rapidly wherever the smooth field has a gradient), and that structure is a *deterministic*
    function of the smooth field -- so block-averaging destroys it while a CNN can infer it
    back, and the example has real headroom over bilinear by construction.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype="f8")[:, None, None]
    la = np.deg2rad(np.linspace(90.0, -90.0, nlat))[None, :, None]
    lo = np.deg2rad(np.linspace(0.0, 360.0, nlon, endpoint=False))[None, None, :]

    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * t / (24 * 365.25))
    hour_angle = 2 * np.pi * ((t % 24) / 24) + lo - np.pi
    clear = 1100.0 * np.clip(
        np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(hour_angle), 0.0, None
    )

    smooth = np.zeros_like(clear)
    for _ in range(4):
        kx, ky = rng.integers(1, 4, size=2)
        smooth += rng.normal() * np.cos(
            kx * lo + ky * la + rng.uniform(0, 2 * np.pi) + rng.uniform(-0.05, 0.05) * t
        )
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9)
    cloud = np.clip(smooth + 0.25 * np.sin(9.0 * np.pi * smooth), 0.0, 1.0)
    ghi = (clear * (1.0 - 0.8 * cloud)).astype("f4")

    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    compressors: Literal["auto"] | None = "auto" if compress else None
    print(
        f"make_irradiance_store: {url}\n"
        f"  {GFS_VAR} ({n_steps}, {nlat}, {nlon}) chunks ({sample_chunk}, {tile}, {tile})  "
        f"~{ghi.nbytes / 1e6:.0f} MB  compress={'auto' if compress else 'none'}",
        flush=True,
    )
    t0 = time.perf_counter()
    arr = group.create_array(
        GFS_VAR,
        shape=ghi.shape,
        chunks=(sample_chunk, tile, tile),
        dtype="f4",
        compressors=compressors,
        dimension_names=("time", "latitude", "longitude"),
    )
    arr[:] = ghi
    print(f"  done: {url} in {time.perf_counter() - t0:.1f}s", flush=True)


def _synthetic_ready(url: str, n_steps: int, nlat: int, nlon: int, sample_chunk: int) -> bool:
    """True if a synthetic store with the requested geometry already exists at ``url``.

    A mismatch in shape or sample chunking, or any open failure (absent / partial), returns
    False so a changed request regenerates rather than training on a stale store.
    """
    try:
        arr = zarr.open_group(store=obstore_store(url), mode="r")[GFS_VAR]
        return (
            isinstance(arr, zarr.Array)
            and arr.shape == (n_steps, nlat, nlon)
            and arr.chunks[0] == sample_chunk
        )
    except Exception:
        return False


def _range(s: str) -> tuple[int, int]:
    start, stop = (int(x) for x in s.split(","))
    return (start, stop)


def build_parser() -> argparse.ArgumentParser:
    """The shared example CLI (source, device, dataset geometry)."""
    p = argparse.ArgumentParser(description="Solar irradiance downscaling on a deep-chunked store")
    p.add_argument(
        "--source",
        choices=("synthetic", "gfs"),
        default="synthetic",
        help="offline synthetic irradiance | dynamical.org's NOAA GFS analysis (streamed "
        "anonymously; needs insitubatch[icechunk] and ~20 GiB of RAM)",
    )
    p.add_argument(
        "--device", default="cpu", help="cpu or cuda -- the train loop moves tensors there"
    )
    p.add_argument(
        "--sample-range",
        type=_range,
        default=None,
        metavar="START,STOP",
        help="finite training window on the time axis (default: 4 chunks on --source gfs)",
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--block-chunks",
        type=int,
        default=None,
        help="shuffle window in chunks; the residency floor is two blocks of stored chunks "
        "(default: 1 on --source gfs, where one chunk is 6.87 GiB)",
    )
    p.add_argument("--n-steps", type=int, default=512, help="synthetic archive length (time)")
    p.add_argument("--nlat", type=int, default=91, help="synthetic grid rows")
    p.add_argument("--nlon", type=int, default=180, help="synthetic grid columns")
    p.add_argument("--sample-chunk", type=int, default=128, help="synthetic samples per chunk")
    p.add_argument(
        "--url",
        default=None,
        help="synthetic store path (file:// temp default; gs://... / s3://... for a cloud store)",
    )
    p.add_argument(
        "--regenerate",
        action="store_true",
        help="force-rewrite the synthetic store even if one with the same geometry exists",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="spill the decoded-chunk cache here (e.g. NVMe) for cross-epoch reuse",
    )
    p.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="throttle read-ahead depth; each slot costs a whole stored chunk on this geometry",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="print the full memory and geometry report before training (fetches nothing)",
    )
    add_log_level(p)
    return p


def cli() -> argparse.Namespace:
    """Parse the shared CLI."""
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    return args


def build_datasets(args: argparse.Namespace) -> InSituDataset:
    """One downscaling dataset from CLI args -- iterate ``ds.train`` / ``ds.val``.

    ``--source`` picks the offline synthetic archive (written fresh) or dynamical.org's real
    GFS analysis. The GFS defaults are the conservative ones: four chunks of the time axis, one
    chunk per block, two reads in flight. It still estimates ~20 GiB of peak, which is what
    ``--print-summary`` is for -- the report opens no store and fetches nothing, so checking
    before you commit costs a second.
    """
    if args.source == "gfs":
        ds = downscaling_dataset(
            _gfs_store(),
            sample_range=args.sample_range or (0, GFS_SAMPLES_PER_CHUNK * 4),
            batch_size=args.batch_size,
            block_chunks=args.block_chunks or 1,
            max_inflight=args.max_inflight or 2,
            cache_dir=args.cache_dir,
        )
    else:
        url = args.url or "file:///tmp/insitu_irradiance.zarr"
        if args.regenerate or not _synthetic_ready(
            url, args.n_steps, args.nlat, args.nlon, args.sample_chunk
        ):
            make_irradiance_store(
                url,
                n_steps=args.n_steps,
                nlat=args.nlat,
                nlon=args.nlon,
                sample_chunk=args.sample_chunk,
            )
        ds = downscaling_dataset(
            obstore_store(url),
            sample_range=args.sample_range,
            batch_size=args.batch_size,
            block_chunks=args.block_chunks or 16,
            max_inflight=args.max_inflight,
            cache_dir=args.cache_dir,
        )
    if args.print_summary:
        ds.print_summary()
    else:
        peak = ds.describe()["memory"]["estimated_peak_bytes"]
        print(f"estimated peak {peak / 2**30:.2f} GiB -- --print-summary for the full report")
    return ds
