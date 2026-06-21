"""Benchmark engines — one dispatch over the loaders being compared.

Engines (see benchmark_plan.md):
  insitu  — InSituDataset (our async loader); cache = none|memory|disk
  naive   — B0: sequential per-chunk zarr reads, no torch, no concurrency (floor)
  memory  — B3: whole split preloaded into RAM, then iterate (compute-bound ceiling)
  workers — B1: map-style Dataset + torch DataLoader(num_workers) (the realistic baseline)
  xbatcher— B2: xbatcher.BatchGenerator + torch DataLoader (the domain-standard stack)

Each engine returns one Result per epoch. A per-batch ``compute_ms`` step
(``time.sleep``, which releases the GIL like a CUDA kernel launch) simulates the
train/infer step so prefetch overlap is observable; GPU uses a real step in M2.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import zarr

from insitubatch import (
    SplitName,
    open_geometries,
    split_by_chunk,
    store_from_url,
)
from insitubatch.source import InSituDataset
from insitubatch.types import ArrayGeometry

from .result import Result, peak_rss_mb, rss_breakdown_mb


@dataclass
class Cfg:
    engine: str
    url: str
    storage: str  # file | s3
    sample_chunk: int  # for the JSONL row (the dataset's chunk size)
    var: str = "t2m"
    batch_size: int = 32
    block_chunks: int = 16
    max_inflight: int | None = None  # None -> follows block_chunks (read concurrency)
    prefetch_depth: int = 2
    num_workers: int = 4
    cache: str = "none"  # none | memory | disk
    compute_ms: float = 0.0
    epochs: int = 1
    max_batches: int = 0  # cap batches/epoch (0 = full) -> bounded, predictable run time
    shuffle: bool = True
    seed: int = 0


def _compute(compute_ms: float) -> None:
    if compute_ms > 0:
        time.sleep(compute_ms / 1000.0)  # releases the GIL; proxy for a GPU step


def _bytes_per_sample(geom: ArrayGeometry) -> int:
    return int(np.prod(geom.inner_shape)) * geom.dtype.itemsize


def _drive(
    batches: Iterator, count: object, compute_ms: float, max_batches: int = 0
) -> tuple[float, int, float]:
    """Iterate batches, timing total seconds, sample count, time-to-first-batch.

    ``max_batches`` caps the work per epoch (0 = full) so a config's run time is
    bounded and predictable -- the lever for sub-hour exploratory suite runs.
    """
    t0 = time.perf_counter()
    ttfb = 0.0
    n = 0
    for i, batch in enumerate(batches):
        if i == 0:
            ttfb = time.perf_counter() - t0
        n += count(batch)  # type: ignore[operator]
        _compute(compute_ms)
        if max_batches and i + 1 >= max_batches:
            break
    return time.perf_counter() - t0, n, ttfb


def _result(
    cfg: Cfg, geom: ArrayGeometry, epoch: int, seconds: float, n: int, ttfb: float
) -> Result:
    bps = _bytes_per_sample(geom)
    anon, file = rss_breakdown_mb()
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
        rss_anon_mb=anon,
        rss_file_mb=file,
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
        "xbatcher": _run_xbatcher,
    }
    return engines[cfg.engine](cfg, geom, manifest, cache_dir, store_kwargs)


def _run_insitu(cfg, geom, manifest, cache_dir, store_kwargs) -> list[Result]:
    if cfg.cache != "none":
        # B1 ships read-once (the V2 throughput/memory decoupling). Cross-epoch
        # reuse returns in B2 on the ChunkPool; fail loudly rather than silently
        # mislabel a cache-off run as cached.
        raise NotImplementedError(
            f"insitu cache={cfg.cache!r} is B2 (ChunkPool LRU); B1 is read-once. Use cache=none."
        )
    ds = InSituDataset(
        cfg.url,
        manifest,
        geometries={cfg.var: geom},
        split=SplitName.TRAIN,
        batch_size=cfg.batch_size,
        block_chunks=cfg.block_chunks,
        max_inflight=cfg.max_inflight,
        prefetch_depth=cfg.prefetch_depth,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
        to_tensor=False,
        **store_kwargs,
    )
    out = []
    for epoch in range(cfg.epochs):
        ds.set_epoch(epoch)
        sec, n, ttfb = _drive(
            iter(ds), lambda b: b.arrays[cfg.var].shape[0], cfg.compute_ms, cfg.max_batches
        )
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
        sec, n, ttfb = _drive(batches(), lambda a: a.shape[0], cfg.compute_ms, cfg.max_batches)
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

        sec, n, ttfb = _drive(batches(), lambda a: a.shape[0], cfg.compute_ms, cfg.max_batches)
        out.append(_result(cfg, geom, epoch, sec, n, ttfb))
    return out


class _SampleReader:
    """Picklable map-style dataset for the B1 baseline: one sample per index,
    opening its own zarr handle per worker. Must be top-level (not a closure) so
    it pickles to DataLoader worker processes under the `spawn` start method."""

    def __init__(self, url: str, var: str, idx, store_kwargs: dict) -> None:
        self.url = url
        self.var = var
        self.idx = idx
        self.store_kwargs = store_kwargs
        self._arr = None  # opened lazily, once per worker

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        if self._arr is None:
            grp = zarr.open_group(store=store_from_url(self.url, **self.store_kwargs), mode="r")
            self._arr = grp[self.var]
        return np.asarray(self._arr[int(self.idx[i])])


def _worker_mp_context(num_workers: int) -> str | None:
    """Start method for DataLoader worker processes.

    obstore's Rust tokio runtime is not fork-safe (fork copies only the calling
    thread, leaving the runtime's threads dead with their locks held), so a forked
    worker deadlocks on first read. forkserver forks each worker from a pristine
    server that never touched the runtime, and is cheaper than spawn — so prefer it
    on Linux; fall back to spawn elsewhere (macOS, where forkserver is unreliable).
    None when there are no workers (torch rejects a context without them).
    """
    if not num_workers:
        return None
    if sys.platform != "darwin" and "forkserver" in mp.get_all_start_methods():
        return "forkserver"
    return "spawn"


def _run_workers(cfg, geom, manifest, cache_dir, store_kwargs) -> list[Result]:
    from torch.utils.data import DataLoader  # optional baseline dependency

    idx = manifest.sample_indices(SplitName.TRAIN, geom)
    # Build the loader once and re-iterate per epoch so persistent_workers keeps the
    # pool warm: the worker model's startup is then paid once (epoch-0 TTFB), not every
    # epoch. That's the tuned baseline (benchmark_plan.md: no strawman). shuffle still
    # reshuffles each iteration. persistent_workers requires num_workers>0.
    loader = DataLoader(
        _SampleReader(cfg.url, cfg.var, idx, store_kwargs),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.shuffle,
        persistent_workers=bool(cfg.num_workers),
        prefetch_factor=(2 if cfg.num_workers else None),
        multiprocessing_context=_worker_mp_context(cfg.num_workers),
    )
    out = []
    for epoch in range(cfg.epochs):
        sec, n, ttfb = _drive(iter(loader), lambda t: t.shape[0], cfg.compute_ms, cfg.max_batches)
        out.append(_result(cfg, geom, epoch, sec, n, ttfb))
    del loader
    return out


def _run_xbatcher(cfg, geom, manifest, cache_dir, store_kwargs) -> list[Result]:
    import xarray as xr  # optional bench dependency
    import xbatcher
    from torch.utils.data import DataLoader
    from xbatcher.loaders.torch import MapDataset

    da = xr.open_zarr(store_from_url(cfg.url, **store_kwargs), consolidated=False)[cfg.var]
    da = da.isel({da.dims[0]: manifest.sample_indices(SplitName.TRAIN, geom)})
    # one timestep per sample (full inner dims); the DataLoader collates batch_size.
    input_dims = {da.dims[0]: 1, **{d: int(da.sizes[d]) for d in da.dims[1:]}}

    # Built once + re-iterated so persistent_workers keeps the pool warm (see _run_workers).
    loader = DataLoader(
        MapDataset(xbatcher.BatchGenerator(da, input_dims=input_dims)),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.shuffle,
        persistent_workers=bool(cfg.num_workers),
        prefetch_factor=(2 if cfg.num_workers else None),
        multiprocessing_context=_worker_mp_context(cfg.num_workers),
    )
    out = []
    for epoch in range(cfg.epochs):
        sec, n, ttfb = _drive(iter(loader), lambda t: t.shape[0], cfg.compute_ms, cfg.max_batches)
        out.append(_result(cfg, geom, epoch, sec, n, ttfb))
    del loader
    return out
