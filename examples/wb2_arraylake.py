"""Stream WeatherBench2 ERA5 from Arraylake/Icechunk straight into insitubatch.

The point of this example is the handoff: an Icechunk session store is passed
*directly* to ``InSituDataset``. An Icechunk store has no URL that round-trips to
it (it is bound to a repository snapshot/branch), so the engine accepts a prebuilt
zarr-v3 Store -- the hot path is unchanged, since it only ever speaks the Store
interface. One in-process event loop fans out the IO; there is no ``num_workers``.

The data is ``earthmover-public/weatherbench2`` (public). ERA5 is grouped by
resolution (``era5/6h_64x32``, ``6h_240x121``, ``6h_1440x721``); pick one with
``--group``. Each sample is a single timestep (the outer axis) cropped to a random
spatial patch (a ``batch_transform``).

    # real data: needs `al auth login` (or ARRAYLAKE_TOKEN) and:
    #   uv sync --extra arraylake
    uv run python -m examples.wb2_arraylake --source arraylake

    # heavier-IO grid, a few epochs, simulated 10ms train step
    uv run python -m examples.wb2_arraylake --source arraylake \
        --group era5/6h_1440x721 --num-epochs 2 --train-step-ms 10

    # offline: tiny synthetic local Icechunk repo, no network or credentials
    uv run python -m examples.wb2_arraylake
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from collections.abc import Callable

import numpy as np
from zarr.abc.store import Store

from insitubatch import SplitName, open_geometries, split_by_chunk
from insitubatch.source import InSituDataset
from insitubatch.types import Batch

# The public WeatherBench2 ERA5, as an Arraylake/Icechunk repo. ERA5 is grouped by
# resolution; 240x121 is a sensible default real workload.
ARRAYLAKE_REPO = "earthmover-public/weatherbench2"
WB2_GROUP = "era5/6h_240x121"
# Surface field, dims (time, lon, lat). Single-variable keeps the sample geometry
# v1-friendly (specific_humidity carries a `level` axis; add it explicitly if wanted).
DEFAULT_VAR = "2m_temperature"


# --------------------------------------------------------------------------- #
# Store access: one Icechunk store, from the cloud or a local synthetic repo.
# --------------------------------------------------------------------------- #
def open_arraylake_store(repo: str, *, branch: str = "main") -> Store:
    """Open an Arraylake repo and return its read-only Icechunk session store.

    Auth comes from a cached ``al auth login`` or ``ARRAYLAKE_TOKEN``; the client
    vends the bucket credentials for the (public) repo. The returned store is a
    zarr-v3 Store -- exactly what ``InSituDataset`` accepts directly.
    """
    from arraylake import Client

    ic_repo = Client().get_repo(repo)
    return ic_repo.readonly_session(branch).store


def open_synthetic_store(
    tmp: str, *, n: int = 256, lat: int = 64, lon: int = 64, spc: int = 8, var: str = DEFAULT_VAR
) -> Store:
    """Write a tiny local Icechunk repo shaped like WB2 (nested under WB2_GROUP).

    No network, no credentials -- exercises the same nested-group, store-passing
    code path as the real repo so the example is runnable offline.
    """
    import icechunk
    import zarr

    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(tmp))
    session = repo.writable_session("main")
    group = zarr.open_group(store=session.store, mode="a", path=WB2_GROUP)
    arr = group.create_array(
        var,
        shape=(n, lat, lon),
        chunks=(spc, lat, lon),
        dtype="f4",
        dimension_names=("time", "longitude", "latitude"),
    )
    arr[:] = np.random.default_rng(0).standard_normal((n, lat, lon)).astype("f4")
    session.commit("synthetic wb2")
    return repo.readonly_session("main").store


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
    train_step_ms: float = 0.0,
    num_epochs: int = 1,
    max_batches: int = 0,
    shuffle: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Stream patches from ``store`` and report time-to-first-batch + throughput."""
    qual = f"{group}/{var}" if group else var
    geom = open_geometries(store, variables=[qual])[qual]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))

    ds = InSituDataset(
        store,
        manifest,
        geometries={qual: geom},
        split=SplitName.TRAIN,
        batch_size=batch_size,
        block_chunks=block_chunks,
        prefetch_depth=prefetch_depth,
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
        for i, batch in enumerate(ds):
            now = time.perf_counter()
            waits.append(now - t_prev)
            a = batch.arrays[qual]
            total += int(a.shape[0])
            sample_shape = tuple(int(d) for d in a.shape[1:])
            if verbose:
                print(f"  epoch {epoch} batch {i}: {tuple(a.shape)}  "
                      f"wait {1e3 * (now - t_prev):.1f} ms")
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


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", choices=["arraylake", "synthetic"], default="synthetic")
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
    a = p.parse_args()

    tmp = None
    if a.source == "synthetic":
        tmp = tempfile.mkdtemp(prefix="wb2-al-")
        store = open_synthetic_store(tmp, var=a.var)
    else:
        store = open_arraylake_store(a.repo, branch=a.branch)

    try:
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
        )
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
