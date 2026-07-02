"""An insitubatch data loader paralleling the Earthmover ``dataloader-demo``.

Mirrors https://github.com/earth-mover/dataloader-demo: load an ERA5-style zarr
from the cloud, feed a simulated training loop, and report per-batch wait time.
What differs:

- **Storage is obstore.** Pass any zarr ``--url`` (e.g. an S3 ERA5 store); reads
  go through ``obstore_store`` (so ``--request-payer`` works for Requester-Pays).
- **Parallelism is in the event loop, not workers.** There is no ``num_workers``
  knob — insitubatch fans out IO on its async loop and prefetches; a different
  parallelism model for the same batches.
- **A spatial subregion is extracted with a ``batch_transform``** (random crop per
  sample), to echo the demo's patches.

**Multi-timestep windows:** the demo samples 48x48 patches over *3 timesteps*. That
windowed, multi-offset access is now supported — declare each timestep as an offset view
(``g``, ``g.shift(1)``, …); see ``examples/advection/`` for an input@t / target@t+h
forecast. This script keeps a single-timestep spatial crop to isolate the storage +
parallelism comparison.

    uv run python -m examples.wb2_dataloader --wb2 --max-batches 100   # public WeatherBench2 GCS
    uv run python -m examples.wb2_dataloader --wb2 --num-epochs 2 --cache-resident  # decode-once
    uv run python -m examples.wb2_dataloader \
        --url s3://bucket/era5.zarr --var 2m_temperature --subregion 48,48 \
        --batch-size 32 --train-step-ms 10 --request-payer
    uv run python -m examples.wb2_dataloader            # tiny synthetic data, no network

The companion ``examples/wb2_xbatcher.py`` runs the same task on the xbatcher +
DataLoader worker stack (and shows how to cut its startup cost).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from collections.abc import Callable

import numpy as np
import zarr

from insitubatch import (
    SplitName,
    ensure_local_dir,
    obstore_store,
    open_geometries,
    split_by_chunk,
)
from insitubatch.source import InSituDataset
from insitubatch.types import Batch

# The public WeatherBench2 ERA5 used by the Earthmover demo (zarr v2, consolidated).
WB2_URL = (
    "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr"
)
WB2_VAR = "2m_temperature"


def _synthetic(
    tmp: str, *, n: int = 64, lat: int = 32, lon: int = 64, spc: int = 8
) -> tuple[str, str]:
    url = f"file://{tmp}/era5.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    arr = group.create_array(
        "t2m",
        shape=(n, lat, lon),
        chunks=(spc, lat, lon),
        dtype="f4",
        dimension_names=("time", "lat", "lon"),
    )
    arr[:] = np.random.default_rng(0).standard_normal((n, lat, lon)).astype("f4")
    return url, "t2m"


def _subregion_crop(var: str, subregion: tuple[int, int], seed: int) -> Callable[[Batch], Batch]:
    """A batch_transform: crop a random (h, w) spatial subregion per sample."""
    h, w = subregion
    rng = np.random.default_rng(seed)

    def crop(batch: Batch) -> Batch:
        a = batch.arrays[var]  # (batch, ..., LAT, LON)
        lat, lon = a.shape[-2], a.shape[-1]
        out = np.empty(a.shape[:-2] + (h, w), dtype=a.dtype)
        for b in range(a.shape[0]):
            i = int(rng.integers(0, lat - h + 1))
            j = int(rng.integers(0, lon - w + 1))
            out[b] = a[b, ..., i : i + h, j : j + w]
        batch.arrays[var] = out
        return batch

    return crop


def run_demo(
    *,
    url: str | None = None,
    var: str = "t2m",
    subregion: tuple[int, int] = (16, 16),
    batch_size: int = 16,
    block_chunks: int = 8,
    prefetch_depth: int = 2,
    cache_resident: bool = False,
    cache_dir: str | None = None,
    train_step_ms: float = 0.0,
    num_epochs: int = 1,
    max_batches: int = 0,
    shuffle: bool = True,
    seed: int = 0,
    request_payer: bool = False,
    verbose: bool = True,
) -> dict:
    tmp = None
    if url is None:
        tmp = tempfile.mkdtemp(prefix="wb2-demo-")
        url, var = _synthetic(tmp)

    store_kwargs: dict = {}
    if url.startswith("gs://"):
        store_kwargs["skip_signature"] = True  # public bucket, anonymous read
    if request_payer:
        store_kwargs["request_payer"] = True

    geom = open_geometries(obstore_store(url), variables=[var], **store_kwargs)[var]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))
    # Caching is the pool's policy (V2: "don't evict"). --cache-resident sizes the budget
    # to hold the whole train split, so epoch 2+ is served from the pool (decode-once);
    # --cache-dir spills the slots to NVMe (mmap) instead of heap. Default: read-once.
    cache_budget_bytes = None
    if cache_resident:
        per_chunk = geom.sample_chunk_size * int(np.prod(geom.inner_shape)) * geom.dtype.itemsize
        cache_budget_bytes = (len(manifest.chunks[SplitName.TRAIN.value]) + 2) * per_chunk

    ds = InSituDataset(
        obstore_store(url),
        manifest,
        geometries={var: geom},
        batch_size=batch_size,
        block_chunks=block_chunks,
        prefetch_depth=prefetch_depth,
        cache_dir=cache_dir,
        cache_budget_bytes=cache_budget_bytes,
        shuffle=shuffle,
        seed=seed,
        batch_transforms=[_subregion_crop(var, subregion, seed)],
        **store_kwargs,
    )

    waits: list[float] = []
    total = 0
    sample_shape: tuple[int, ...] = ()
    t_start = time.perf_counter()
    for epoch in range(num_epochs):
        ds.set_epoch(epoch)
        t_prev = time.perf_counter()
        for i, batch in enumerate(ds.train):
            now = time.perf_counter()
            waits.append(now - t_prev)
            a = batch.arrays[var]
            total += a.shape[0]
            sample_shape = tuple(a.shape[1:])
            if verbose:
                print(
                    f"epoch {epoch} batch {i}: {tuple(a.shape)}  wait {1e3 * (now - t_prev):.1f} ms"
                )
            time.sleep(train_step_ms / 1000.0)  # simulated train step
            t_prev = time.perf_counter()
            if max_batches and i + 1 >= max_batches:
                break
    dt = time.perf_counter() - t_start

    summary = {
        "samples": total,
        "sample_shape": sample_shape,
        "ttfb_ms": 1e3 * waits[0] if waits else 0.0,  # cold-start time-to-first-batch
        "seconds": dt,
        "samples_per_s": total / dt if dt else 0.0,
        "mean_wait_ms": 1e3 * float(np.mean(waits)) if waits else 0.0,
        "cache_resident": cache_resident,
        "train_step_ms": train_step_ms,
    }
    if verbose:
        print(f"\nsummary: {summary}")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return summary


def _subregion(s: str) -> tuple[int, int]:
    h, w = (int(x) for x in s.split(","))
    return (h, w)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--url", default=None, help="zarr URL (gs://, s3://, file://); default = synthetic"
    )
    p.add_argument(
        "--wb2", action="store_true", help="use the public WeatherBench2 ERA5 (sets --url/--var)"
    )
    p.add_argument("--var", default="t2m")
    p.add_argument("--subregion", type=_subregion, default=(16, 16), help="crop H,W e.g. 48,48")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-chunks", type=int, default=8)
    p.add_argument("--prefetch-depth", type=int, default=2)
    p.add_argument(
        "--cache-resident",
        action="store_true",
        help="hold the whole train split in the pool -> epoch 2+ decode-once (else read-once)",
    )
    p.add_argument("--cache-dir", default=None, help="spill the resident cache to NVMe (mmap)")
    p.add_argument("--train-step-ms", type=float, default=0.0)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=0, help="cap batches per epoch (0 = all)")
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--request-payer", action="store_true")
    a = p.parse_args()
    url, var = (WB2_URL, WB2_VAR) if a.wb2 else (a.url, a.var)
    run_demo(
        url=url,
        var=var,
        subregion=a.subregion,
        batch_size=a.batch_size,
        block_chunks=a.block_chunks,
        prefetch_depth=a.prefetch_depth,
        cache_resident=a.cache_resident,
        cache_dir=a.cache_dir,
        train_step_ms=a.train_step_ms,
        num_epochs=a.num_epochs,
        max_batches=a.max_batches,
        shuffle=not a.no_shuffle,
        request_payer=a.request_payer,
    )


if __name__ == "__main__":
    main()
