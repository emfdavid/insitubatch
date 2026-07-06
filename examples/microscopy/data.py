"""Shared data for the cell-segmentation example: the store, the one dataset, the eval.

**The task -- per-plane foreground segmentation.** Each sample is one Z-plane of a confocal
stack. Two variables are gathered at the *same* anchor: ``raw`` (a 2-channel fluorescence
image) and ``mask`` (its binary foreground label). Nothing is resharded -- the two arrays are
chunked differently along Z (``raw`` one plane deep, ``mask`` many planes deep) and carry a
different channel count, yet the engine batches them row-for-row off a single sample grid.
That is the arbitrary-sample-axis (``sample_axis=2``) + per-variable-chunking unlock.

A **global intensity threshold (Otsu)** is the no-context baseline a useful model must beat --
the segmentation analog of persistence in the advection example. Otsu reads each pixel's
intensity alone, so a smooth autofluorescence *haze* gradient defeats it: a bright-haze
background pixel can outshine a dim cell elsewhere, and no single threshold separates them. A
tiny CNN that *sees the neighborhood* (sharp cells vs low-frequency haze) does better. On the
synthetic store it beats Otsu by construction; on the real IDR store the same code runs on a
real OME-NGFF image (we claim "same pipeline, real data, no reshard" -- not SOTA Dice).

``segmentation_dataset`` returns one dataset whose labels are always ``raw, mask`` regardless
of the store, so the training file is store-agnostic.
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
    ensure_local_dir,
    obstore_store,
    open_geometries,
    split_by_chunk,
)
from insitubatch.source import InSituDataset
from insitubatch.types import Batch

# A public OME-NGFF (zarr v0.1) image in the EMBL-EBI Image Data Repository: a 3D two-channel
# confocal stack with an expert instance-segmentation label. Read anonymously off the IDR S3.
IDR_URL = "s3://idr/zarr/v0.1/6001240.zarr"
IDR_ENDPOINT = "https://uk1s3.embassy.ebi.ac.uk"
IDR_RAW = "0"  # full-resolution image (T=1, C=2, Z=236, Y=275, X=271), Z-chunk 1
IDR_MASK = "labels/masks/0"  # the label (T=1, C=1, Z=236, Y=275, X=271), Z-chunk 30, Y/X tiled

# OME-NGFF is (T, C, Z, Y, X); we sample over Z. The synthetic store uses the same 5-D layout
# (leading T=1) so one code path serves both sources.
SAMPLE_AXIS = 2
LABELS = ("raw", "mask")  # canonical labels (store-independent)


def _idr_store() -> Store:
    """Anonymous read of the public IDR bucket (custom S3 endpoint, path-style, no signing)."""
    return obstore_store(
        IDR_URL,
        endpoint=IDR_ENDPOINT,
        skip_signature=True,
        virtual_hosted_style_request=False,
        region="us-east-1",
    )


def _smooth_field(
    rng: np.random.Generator, yy: np.ndarray, xx: np.ndarray, modes: int
) -> np.ndarray:
    """A non-negative smooth low-frequency field in [0, 1] -- the autofluorescence haze.

    A few random low modes give a gradient that a *global* threshold cannot separate from dim
    cells, but whose low spatial frequency a CNN trivially distinguishes from sharp cells.
    """
    f = np.zeros_like(yy)
    for _ in range(modes):
        kx, ky = rng.integers(1, 3, size=2)
        f += rng.normal() * np.cos(2 * np.pi * (kx * xx + ky * yy) + rng.uniform(0, 2 * np.pi))
    f -= f.min()
    return f / (f.max() + 1e-9)


def make_cells_store(
    url: str,
    *,
    n_planes: int = 64,
    size: int = 96,
    mask_chunk: int = 16,
    inner_chunk: int | None = None,
    cells_per_plane: int = 14,
    seed: int = 0,
    compress: bool = True,
) -> None:
    """Write a synthetic two-channel cell stack (``raw``) + its foreground label (``mask``).

    Mirrors the real IDR geometry: ``raw`` is ``(1, 2, n_planes, size, size)`` chunked **one
    plane deep** on Z; ``mask`` is ``(1, 1, n_planes, size, size)`` chunked ``mask_chunk``
    planes deep and tiled in Y/X. Same Z length, different chunking and channel count -- the
    per-variable-chunking case the engine handles with no reshard.

    Each plane holds a few Gaussian cells (sharp, high spatial frequency). Both channels add a
    different smooth *haze* gradient and read noise, so a global threshold on intensity cannot
    separate a dim cell from bright background -- but a CNN reading the neighborhood can. The
    ``mask`` is the haze-free, noise-free ground truth (cells above half-max). The stack is
    small (fits in RAM); scale ``n_planes``/``size`` for a larger store.
    """
    ic = inner_chunk or (size // 2)
    if not 1 <= ic <= size:
        raise ValueError(f"inner_chunk {ic} must be in 1..size ({size})")
    rng = np.random.default_rng(seed)
    yy, xx = (np.mgrid[0:size, 0:size] / size).astype(np.float64)
    haze0 = _smooth_field(rng, yy, xx, modes=3)
    haze1 = _smooth_field(rng, yy, xx, modes=3)

    raw = np.empty((1, 2, n_planes, size, size), dtype="f4")
    mask = np.empty((1, 1, n_planes, size, size), dtype="i4")
    gy, gx = np.mgrid[0:size, 0:size].astype(np.float64)
    for z in range(n_planes):
        clean = np.zeros((size, size))
        for _ in range(cells_per_plane):
            cy, cx = rng.uniform(0, size, size=2)
            r = rng.uniform(3, 7)
            bright = rng.uniform(0.6, 1.0)
            clean += bright * np.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / (2 * r * r))
        clean = np.clip(clean, 0, 1)
        mask[0, 0, z] = (clean > 0.5).astype("i4")
        # Cells at ~0.6 haze gain; haze background up to ~0.5 -- overlapping ranges, so no
        # global threshold works, but the CNN reads sharp-vs-smooth structure.
        noise0 = rng.normal(0, 0.05, size=(size, size))
        noise1 = rng.normal(0, 0.05, size=(size, size))
        raw[0, 0, z] = (0.6 * clean + 0.5 * haze0 + noise0).astype("f4")
        raw[0, 1, z] = (0.6 * clean + 0.5 * haze1 + noise1).astype("f4")

    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    compressors: Literal["auto"] | None = "auto" if compress else None
    dims = ("t", "c", "z", "y", "x")
    total_mb = (raw.nbytes + mask.nbytes) / 1e6
    print(
        f"make_cells_store: {url}\n"
        f"  raw (1, 2, {n_planes}, {size}, {size}) Z-chunk 1  +  "
        f"mask (1, 1, {n_planes}, {size}, {size}) Z-chunk {mask_chunk}  "
        f"~{total_mb:.0f} MB  compress={'auto' if compress else 'none'}",
        flush=True,
    )
    t0 = time.perf_counter()
    raw_arr = group.create_array(
        IDR_RAW,
        shape=raw.shape,
        chunks=(1, 1, 1, size, size),
        dtype="f4",
        compressors=compressors,
        dimension_names=dims,
    )
    mask_arr = group.create_array(
        IDR_MASK,
        shape=mask.shape,
        chunks=(1, 1, mask_chunk, ic, ic),
        dtype="i4",
        compressors=compressors,
        dimension_names=dims,
    )
    raw_arr[:] = raw
    mask_arr[:] = mask
    print(f"  done: {url} in {time.perf_counter() - t0:.1f}s", flush=True)


def _synthetic_ready(url: str, n_planes: int, size: int, mask_chunk: int) -> bool:
    """True if a synthetic store with the requested geometry already exists at ``url``.

    A mismatch in shape or Z-chunking, or any open failure (absent / partial), returns False so
    a changed request regenerates rather than training on a stale store.
    """
    try:
        group = zarr.open_group(store=obstore_store(url), mode="r")
        raw, mask = group[IDR_RAW], group[IDR_MASK]
        return (
            isinstance(raw, zarr.Array)
            and isinstance(mask, zarr.Array)
            and raw.shape == (1, 2, n_planes, size, size)
            and mask.chunks[2] == mask_chunk
        )
    except Exception:
        return False


def segmentation_dataset(
    store: Store,
    *,
    raw_var: str = IDR_RAW,
    mask_var: str = IDR_MASK,
    sample_axis: int = SAMPLE_AXIS,
    sample_range: tuple[int, int] | None = None,
    batch_size: int = 16,
    shuffle: bool = True,
    cache_dir: str | None = None,
    max_inflight: int | None = None,
) -> InSituDataset:
    """One dataset: ``raw`` (2-channel image) + ``mask`` (foreground label), sampled over Z.

    ``store`` is a zarr Store -- build it with :func:`~insitubatch.obstore_store` /
    :func:`~insitubatch.fsspec_store`. ``raw_var`` / ``mask_var`` name the two arrays in the
    store; the dataset's labels are always ``raw, mask`` so the model code is store-agnostic.
    Both share the Z (sample) axis *length* but may chunk it differently -- the engine maps the
    single sample grid (the ``raw`` chunking, one plane per anchor) onto each variable's own
    chunks. ``sample_range`` restricts the split to a finite Z window. Iterate the returned
    dataset's ``.train`` / ``.val`` views.
    """
    opened = open_geometries(store, variables=[raw_var, mask_var], sample_axis=sample_axis)
    geoms = {"raw": opened[raw_var], "mask": opened[mask_var]}
    # The raw (one plane per chunk) is the reference sample grid: one Z-plane = one sample.
    manifest = split_by_chunk(opened[raw_var], fractions=(0.8, 0.1, 0.1), sample_range=sample_range)
    return InSituDataset(
        store,
        manifest,
        geometries=geoms,
        batch_size=batch_size,
        shuffle=shuffle,
        cache_dir=cache_dir,
        max_inflight=max_inflight,
    )


def inputs_and_targets(batch: Batch) -> tuple[np.ndarray, np.ndarray]:
    """Split a ``Batch`` into ``(x, target)``: the 2-channel image and the binary foreground.

    ``raw`` arrives as ``(B, T=1, C=2, Y, X)`` and ``mask`` as ``(B, T=1, C=1, Y, X)`` -- the
    singleton OME-NGFF ``T`` axis is a field axis carried whole; we squeeze it here. ``x`` is
    ``(B, 2, Y, X)`` float; ``target`` is ``(B, 1, Y, X)`` in {0, 1} (the label is instance
    IDs in the real store, binarized to foreground). The framework loop only touches these.
    """
    x = batch.arrays["raw"][:, 0].astype("f4")  # (B, 2, Y, X)
    target = (batch.arrays["mask"][:, 0] > 0).astype("f4")  # (B, 1, Y, X)
    return x, target


def _otsu(img: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold for one image: the intensity maximizing between-class variance."""
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return hi
    hist, edges = np.histogram(img, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(np.float64)
    wf = np.cumsum(w)  # cumulative weight of the below-threshold (foreground-candidate) class
    wb = w.sum() - wf
    cs = np.cumsum(w * centers)
    mf = cs / np.maximum(wf, 1)
    mb = (cs[-1] - cs) / np.maximum(wb, 1)
    between = wf * wb * (mf - mb) ** 2
    return float(centers[int(np.argmax(between))])


def otsu_foreground(batch: Batch) -> np.ndarray:
    """The no-context baseline: per-plane Otsu threshold on the mean of the two channels.

    Returns a ``(B, 1, Y, X)`` {0, 1} foreground mask -- what you get reading pixel intensity
    alone, blind to the neighborhood (and so to the smooth haze). The CNN must beat this.
    """
    x, _ = inputs_and_targets(batch)
    intensity = x.mean(axis=1)  # (B, Y, X) -- combine both channels
    out = np.stack([plane > _otsu(plane) for plane in intensity])  # (B, Y, X)
    return out[:, None].astype("f4")


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Micro-averaged foreground IoU (Jaccard) over all pixels of all planes."""
    p, t = pred > 0.5, target > 0.5
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(inter / union) if union else 1.0


def evaluate(view: Iterable[Batch], predict: Callable[[Batch], np.ndarray]) -> tuple[float, float]:
    """Foreground IoU on a held-out split view (e.g. ``ds.val``): ``(model_iou, otsu_iou)``.

    ``predict`` maps a ``Batch`` to the model's ``(B, 1, Y, X)`` foreground probability (each
    framework supplies its own, so this stays framework-neutral). Otsu -- the best a global
    intensity threshold can do -- is the baseline a useful model must beat. The view is
    deterministic (eval splits don't shuffle), so no epoch is set.
    """
    preds, targets, otsus = [], [], []
    for batch in view:
        _x, target = inputs_and_targets(batch)
        preds.append(predict(batch))
        targets.append(target)
        otsus.append(otsu_foreground(batch))
    pred, target, otsu = (np.concatenate(a) for a in (preds, targets, otsus))
    return iou(pred, target), iou(otsu, target)


def _range(s: str) -> tuple[int, int]:
    start, stop = (int(x) for x in s.split(","))
    return (start, stop)


def build_parser() -> argparse.ArgumentParser:
    """The shared example CLI (source, device, dataset geometry)."""
    p = argparse.ArgumentParser(description="OME-NGFF cell segmentation over Z")
    p.add_argument(
        "--source",
        choices=("synthetic", "idr"),
        default="synthetic",
        help="offline synthetic cells | the real IDR OME-NGFF image (streamed anonymously)",
    )
    p.add_argument(
        "--device", default="cpu", help="cpu or cuda -- the train loop moves tensors there"
    )
    p.add_argument(
        "--sample-range",
        type=_range,
        default=None,
        metavar="START,STOP",
        help="finite training window on the Z axis (subset the stack)",
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-planes", type=int, default=64, help="synthetic stack depth (Z)")
    p.add_argument("--size", type=int, default=96, help="synthetic plane size (Y=X)")
    p.add_argument("--mask-chunk", type=int, default=16, help="synthetic mask Z-chunk depth")
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
        help="throttle read-ahead depth (lower = lower cold TTFB over a high-latency network)",
    )
    return p


def cli() -> argparse.Namespace:
    """Parse the shared CLI."""
    return build_parser().parse_args()


def build_datasets(args: argparse.Namespace) -> InSituDataset:
    """One segmentation dataset from CLI args -- iterate ``ds.train`` / ``ds.val``. ``--source``
    picks offline synthetic cells (written fresh) or the real IDR OME-NGFF image."""
    if args.source == "idr":
        store = _idr_store()
        return segmentation_dataset(
            store,
            raw_var=IDR_RAW,
            mask_var=IDR_MASK,
            sample_range=args.sample_range,
            batch_size=args.batch_size,
            shuffle=True,
            cache_dir=args.cache_dir,
            max_inflight=args.max_inflight,
        )
    url = args.url or "file:///tmp/insitu_cells.zarr"
    if args.regenerate or not _synthetic_ready(url, args.n_planes, args.size, args.mask_chunk):
        make_cells_store(url, n_planes=args.n_planes, size=args.size, mask_chunk=args.mask_chunk)
    return segmentation_dataset(
        obstore_store(url),
        sample_range=args.sample_range,
        batch_size=args.batch_size,
        shuffle=True,
        cache_dir=args.cache_dir,
        max_inflight=args.max_inflight,
    )
