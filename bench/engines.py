"""Benchmark engines — one dispatch over the loaders being compared.

Engines (see benchmark_plan.md):
  insitu  — InSituDataset (our async loader); cache = none|memory|disk
  naive   — B0: sequential per-chunk zarr reads, no torch, no concurrency (floor)
  memory  — B3: whole split preloaded into RAM, then iterate (compute-bound ceiling)
  workers — B1: map-style Dataset + torch DataLoader(num_workers) (the realistic baseline)

Each engine returns one Result per epoch. A per-batch ``compute_ms`` step
(``time.sleep``, which releases the GIL like a CUDA kernel launch) simulates the
train/infer step so prefetch overlap is observable; GPU uses a real step in M2.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import zarr

from insitubatch import (
    DiskCache,
    MemoryCache,
    SplitName,
    open_geometries,
    split_by_chunk,
    store_from_url,
)
from insitubatch.source import InSituDataset
from insitubatch.types import ArrayGeometry

from .result import Result, peak_rss_mb


@dataclass
class Cfg:
    engine: str
    url: str
    storage: str  # file | s3
    sample_chunk: int  # for the JSONL row (the dataset's chunk size)
    var: str = "t2m"
    batch_size: int = 32
    block_chunks: int = 16
    max_inflight: int = 16
    prefetch_depth: int = 2
    num_workers: int = 4
    cache: str = "none"  # none | memory | disk
    compute_ms: float = 0.0
    epochs: int = 1
    shuffle: bool = True
    seed: int = 0


def _compute(compute_ms: float) -> None:
    if compute_ms > 0:
        time.sleep(compute_ms / 1000.0)  # releases the GIL; proxy for a GPU step


def _bytes_per_sample(geom: ArrayGeometry) -> int:
    return int(np.prod(geom.inner_shape)) * geom.dtype.itemsize


def _drive(batches: Iterator, count: object, compute_ms: float) -> tuple[float, int, float]:
    """Iterate batches, timing total seconds, sample count, time-to-first-batch."""
    t0 = time.perf_counter()
    ttfb = 0.0
    n = 0
    first = True
    for batch in batches:
        if first:
            ttfb = time.perf_counter() - t0
            first = False
        n += count(batch)  # type: ignore[operator]
        _compute(compute_ms)
    return time.perf_counter() - t0, n, ttfb


def _result(
    cfg: Cfg, geom: ArrayGeometry, epoch: int, seconds: float, n: int, ttfb: float
) -> Result:
    bps = _bytes_per_sample(geom)
    return Result(
        engine=cfg.engine,
        cache=cfg.cache,
        storage=cfg.storage,
        sample_chunk=cfg.sample_chunk,
        n_samples=n,
        epoch=epoch,
        batch_size=cfg.batch_size,
        block_chunks=cfg.block_chunks,
        prefetch_depth=cfg.prefetch_depth,
        num_workers=cfg.num_workers,
        compute_ms=cfg.compute_ms,
        seconds=seconds,
        samples_per_s=(n / seconds if seconds else 0.0),
        mb_per_s=(n * bps / 1e6 / seconds if seconds else 0.0),
        ttfb_ms=ttfb * 1e3,
        peak_rss_mb=peak_rss_mb(),
    )


def run(
    cfg: Cfg, *, cache_dir: str | None = None, store_kwargs: dict | None = None
) -> list[Result]:
    store_kwargs = store_kwargs or {}
    geom = open_geometries(cfg.url, **store_kwargs)[cfg.var]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))
    engines = {
        "insitu": _run_insitu,
        "naive": _run_naive,
        "memory": _run_memory,
        "workers": _run_workers,
    }
    return engines[cfg.engine](cfg, geom, manifest, cache_dir, store_kwargs)


def _run_insitu(cfg, geom, manifest, cache_dir, store_kwargs) -> list[Result]:
    cache = None
    if cfg.cache == "memory":
        cache = MemoryCache(64 << 30)
    elif cfg.cache == "disk":
        cache = DiskCache(cache_dir or "/tmp/insitubatch-cache", 64 << 30)
    ds = InSituDataset(
        cfg.url,
        manifest,
        geometries={cfg.var: geom},
        split=SplitName.TRAIN,
        batch_size=cfg.batch_size,
        block_chunks=cfg.block_chunks,
        max_inflight=cfg.max_inflight,
        prefetch_depth=cfg.prefetch_depth,
        cache=cache,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
        to_tensor=False,
        **store_kwargs,
    )
    out = []
    for epoch in range(cfg.epochs):
        ds.set_epoch(epoch)
        sec, n, ttfb = _drive(iter(ds), lambda b: b.arrays[cfg.var].shape[0], cfg.compute_ms)
        out.append(_result(cfg, geom, epoch, sec, n, ttfb))
    return out


def _run_naive(cfg, geom, manifest, cache_dir, store_kwargs) -> list[Result]:
    arr = zarr.open_group(store=store_from_url(cfg.url, **store_kwargs), mode="r")[cfg.var]
    spc = geom.sample_chunk_size

    def batches():
        for c in manifest.chunks[SplitName.TRAIN.value]:
            s0, s1 = c * spc, min(c * spc + spc, geom.n_samples)
            yield np.asarray(arr[s0:s1])

    out = []
    for epoch in range(cfg.epochs):
        sec, n, ttfb = _drive(batches(), lambda a: a.shape[0], cfg.compute_ms)
        out.append(_result(cfg, geom, epoch, sec, n, ttfb))
    return out


def _run_memory(cfg, geom, manifest, cache_dir, store_kwargs) -> list[Result]:
    group = zarr.open_group(store=store_from_url(cfg.url, **store_kwargs), mode="r")
    full = np.asarray(group[cfg.var][:])
    idx0 = manifest.sample_indices(SplitName.TRAIN, geom)
    out = []
    for epoch in range(cfg.epochs):
        idx = idx0.copy()
        if cfg.shuffle:
            np.random.default_rng((cfg.seed, epoch)).shuffle(idx)

        def batches(idx=idx):
            for i in range(0, len(idx), cfg.batch_size):
                yield full[idx[i : i + cfg.batch_size]]

        sec, n, ttfb = _drive(batches(), lambda a: a.shape[0], cfg.compute_ms)
        out.append(_result(cfg, geom, epoch, sec, n, ttfb))
    return out


def _run_workers(cfg, geom, manifest, cache_dir, store_kwargs) -> list[Result]:
    from torch.utils.data import DataLoader, Dataset  # optional baseline dependency

    idx = manifest.sample_indices(SplitName.TRAIN, geom)
    url, var, sk = cfg.url, cfg.var, store_kwargs

    class _MapDataset(Dataset):
        def __init__(self) -> None:
            self._arr = None

        def __len__(self) -> int:
            return len(idx)

        def __getitem__(self, i: int):
            if self._arr is None:  # open once per worker process
                self._arr = zarr.open_group(store=store_from_url(url, **sk), mode="r")[var]
            return np.asarray(self._arr[int(idx[i])])

    out = []
    for epoch in range(cfg.epochs):
        loader = DataLoader(
            _MapDataset(),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=cfg.shuffle,
            persistent_workers=False,
            prefetch_factor=(2 if cfg.num_workers else None),
        )
        sec, n, ttfb = _drive(iter(loader), lambda t: t.shape[0], cfg.compute_ms)
        del loader
        out.append(_result(cfg, geom, epoch, sec, n, ttfb))
    return out
