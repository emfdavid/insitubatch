"""Stream WeatherBench2 ERA5 from Arraylake/Icechunk straight into insitubatch.

The point of this example is the handoff: an Icechunk session store is passed
*directly* to ``InSituDataset``. An Icechunk store has no URL that round-trips to
it (it is bound to a repository snapshot/branch), so the engine accepts a prebuilt
zarr-v3 Store -- the hot path is unchanged, since it only ever speaks the Store
interface. One in-process event loop fans out the IO; there is no ``num_workers``.

The data is ``earthmover-public/weatherbench2`` (public). ERA5 is grouped by
resolution (``era5/6h_64x32``, ``6h_240x121``, ``6h_1440x721``); pick one with
``--group``. Each sample is a single timestep (the outer axis) cropped to a random
spatial patch (a ``batch_transform``). Needs ``al auth login`` (or
``ARRAYLAKE_TOKEN``) and ``uv sync --extra arraylake``.

    uv run python -m examples.wb2_arraylake

    # heavier-IO grid, a few epochs, simulated 10ms train step
    uv run python -m examples.wb2_arraylake \
        --group era5/6h_1440x721 --num-epochs 2 --train-step-ms 10

    # Reasonable settings for WAN access: subset to a window, cache to NVMe, full epochs.
    # Caveat: under --cache-resident the pool budget holds the whole subset, which removes
    # the engine's read-ahead backpressure -- so --max-inflight here doubles as the
    # read-ahead throttle. Keep it modest (~16): a high value oversubscribes the network at
    # the start of an uncached epoch (block 0 competes with the whole-subset prefetch) and
    # inflates cold TTFB. A decoupled read-ahead cap would fix this (DESIGN.md roadmap, M-RA).
    uv run python -m examples.wb2_arraylake \
        --group era5/6h_1440x721 --max-inflight 16 --sample-range 0,180 --max-batches 0 \
        --num-epochs 2 --cache-resident --cache-dir /tmp/insitu-cache

    # Measured -- proximity to the store dominates (run near your data):
    #   home network (high-latency WAN, tuned as above): TTFB 8.5 s, 7.9 samples/s
    #   in-region AWS EC2 (untuned defaults):             TTFB 1.3 s,  60 samples/s
    #     home: {'samples': 288, 'ttfb_ms': 8470.3, 'samples_per_s': 7.94, 'mean_wait_ms': 1006.9}
    #     ec2:  {'samples': 320, 'ttfb_ms': 1269.6, 'samples_per_s': 60.24, 'mean_wait_ms': 115.6}
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
from zarr.abc.store import Store

from insitubatch import SplitName, arraylake_store, open_geometries, split_by_chunk
from insitubatch.source import InSituDataset
from insitubatch.types import Batch

# The public WeatherBench2 ERA5, as an Arraylake/Icechunk repo. ERA5 is grouped by
# resolution; 240x121 is a sensible default real workload.
ARRAYLAKE_REPO = "earthmover-public/weatherbench2"
WB2_GROUP = "era5/6h_240x121"
# Surface field, dims (time, lon, lat). Single-variable keeps the sample geometry
# v1-friendly (specific_humidity carries a `level` axis; add it explicitly if wanted).
DEFAULT_VAR = "2m_temperature"


def _crop(patch: tuple[int, int], seed: int) -> Callable[[Batch], Batch]:
    """batch_transform: crop a random (h, w) subregion from the last two dims."""
    h, w = patch
    rng = np.random.default_rng(seed)

    def crop(batch: Batch) -> Batch:
        for name, a in batch.arrays.items():
            lat, lon = a.shape[-2], a.shape[-1]
            i = int(rng.integers(0, lat - h + 1))
            j = int(rng.integers(0, lon - w + 1))
            batch.arrays[name] = a[..., i : i + h, j : j + w]
        return batch

    return crop


def run(
    store: Store,
    *,
    var: str = DEFAULT_VAR,
    group: str = WB2_GROUP,
    patch: tuple[int, int] = (32, 32),
    batch_size: int = 16,
    block_chunks: int = 8,
    prefetch_depth: int = 2,
    max_inflight: int | None = None,
    sample_range: tuple[int, int] | None = None,
    cache_resident: bool = False,
    cache_dir: str | None = None,
    train_step_ms: float = 0.0,
    num_epochs: int = 1,
    max_batches: int = 0,
    shuffle: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Stream patches from ``store`` and report time-to-first-batch + throughput.

    ``max_inflight`` is the latency dial: over a high-latency network (streaming across the
    WAN) you need many concurrent requests to keep the read-ahead fed, or block-boundary
    waits show as a sawtooth (the next block isn't fetched while the current one drains).
    Raise it well above the default 32 when the network round-trip is the bottleneck.

    ``sample_range`` restricts training to a sample-axis (time) window -- essential for a
    multi-decade store: the whole archive is hundreds of GB, and ``cache_resident`` sizes
    the pool to the *whole train split*, so without subsetting it tries to hold (and, on
    the first epoch, eagerly fetch) the entire thing -- minutes of allocation before the
    first batch. Pick a window (e.g. one year of 6-hourly steps) so the cache fits; pair
    with ``cache_dir`` to spill the slots to NVMe (mmap) instead of RAM. With a bounded,
    cached split, epoch 2+ is served decode-once from the pool (no re-fetch).
    """
    qual = f"{group}/{var}" if group else var
    geom = open_geometries(store, variables=[qual])[qual]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1), sample_range=sample_range)

    cache_budget_bytes = None
    if cache_resident:
        per_chunk = geom.sample_chunk_size * int(np.prod(geom.inner_shape)) * geom.dtype.itemsize
        n_train = len(manifest.chunks[SplitName.TRAIN.value])
        cache_budget_bytes = (n_train + 2) * per_chunk
        if verbose:
            gib = cache_budget_bytes / 1024**3
            where = f"NVMe at {cache_dir}" if cache_dir else "RAM"
            print(f"cache-resident: {n_train}-chunk train split (~{gib:.1f} GiB) held in {where}")
            if cache_dir is None and gib > 8:
                print(
                    "  warning: large in-RAM cache -- pass --sample-range to subset, or "
                    "--cache-dir to spill to NVMe"
                )
            if max_batches and num_epochs > 1:
                print(
                    "  note: --max-batches caps each epoch, so epoch 0 caches only those "
                    "(reshuffled) chunks; drop it (full epochs) for the cache to pay off"
                )

    ds = InSituDataset(
        store,
        manifest,
        geometries={qual: geom},
        batch_size=batch_size,
        block_chunks=block_chunks,
        prefetch_depth=prefetch_depth,
        max_inflight=max_inflight,
        cache_budget_bytes=cache_budget_bytes,
        cache_dir=cache_dir,
        shuffle=shuffle,
        seed=seed,
        batch_transforms=[_crop(patch, seed)],
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
            a = batch.arrays[qual]
            total += int(a.shape[0])
            sample_shape = tuple(int(d) for d in a.shape[1:])
            if verbose:
                print(
                    f"  epoch {epoch} batch {i}: {tuple(a.shape)}  "
                    f"wait {1e3 * (now - t_prev):.1f} ms"
                )
            time.sleep(train_step_ms / 1000.0)
            t_prev = time.perf_counter()
            if max_batches and i + 1 >= max_batches:
                break
    dt = time.perf_counter() - t_start

    summary = {
        "samples": total,
        "sample_shape": sample_shape,
        "ttfb_ms": 1e3 * waits[0] if waits else 0.0,  # cold time-to-first-batch
        "seconds": dt,
        "samples_per_s": total / dt if dt else 0.0,
        "mean_wait_ms": 1e3 * float(np.mean(waits)) if waits else 0.0,
    }
    if verbose:
        print(f"\nsummary: {summary}")
    return summary


def _patch(s: str) -> tuple[int, int]:
    h, w = (int(x) for x in s.split(","))
    return (h, w)


def _range(s: str) -> tuple[int, int]:
    start, stop = (int(x) for x in s.split(","))
    return (start, stop)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", default=ARRAYLAKE_REPO, help="Arraylake repo (org/name)")
    p.add_argument("--branch", default="main")
    p.add_argument("--var", default=DEFAULT_VAR)
    p.add_argument("--group", default=WB2_GROUP, help="group path within the repo ('' for root)")
    p.add_argument("--patch", type=_patch, default=(32, 32), help="spatial patch H,W")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-chunks", type=int, default=8, help="shuffle window (outer chunks)")
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=20, help="cap batches per epoch (0 = all)")
    p.add_argument("--train-step-ms", type=float, default=0.0, help="simulated train step")
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="stored tiles in flight (the latency dial); raise it for a high-latency network",
    )
    p.add_argument(
        "--sample-range",
        type=_range,
        default=None,
        metavar="START,STOP",
        help="train on a sample-axis (time) window -- subset this multi-decade store so a "
        "cache fits (e.g. one year of 6-hourly steps is ~1460)",
    )
    p.add_argument(
        "--cache-resident",
        action="store_true",
        help="hold the (subset) train split resident -> epoch 2+ is decode-once (no re-fetch)",
    )
    p.add_argument("--cache-dir", default=None, help="spill cached slots to this NVMe dir (mmap)")
    a = p.parse_args()

    store = arraylake_store(a.repo, branch=a.branch)
    run(
        store,
        var=a.var,
        group=a.group,
        patch=a.patch,
        batch_size=a.batch_size,
        block_chunks=a.block_chunks,
        num_epochs=a.num_epochs,
        max_batches=a.max_batches,
        train_step_ms=a.train_step_ms,
        shuffle=not a.no_shuffle,
        max_inflight=a.max_inflight,
        sample_range=a.sample_range,
        cache_resident=a.cache_resident,
        cache_dir=a.cache_dir,
    )


if __name__ == "__main__":
    main()
