"""Prefetch: the producer assembles batches ahead of the consumer.

The pre-M1.5 loop was demand-driven (a batch was assembled only when pulled), so
the event loop sat idle during the compute step. Prefetch decouples production
from consumption via a bounded queue.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import zarr

from insitubatch import (
    SplitName,
    ensure_local_dir,
    open_geometries,
    split_by_chunk,
    store_from_url,
)
from insitubatch.source import InSituDataset


def _write(tmp_path, *, n=160, spc=8, inner=(2, 2)) -> str:
    url = f"file://{tmp_path}/d.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=store_from_url(url, read_only=False), mode="w")
    arr = group.create_array("t2m", shape=(n, *inner), chunks=(spc, *inner), dtype="f4")
    arr[:] = np.arange(n * int(np.prod(inner)), dtype="f4").reshape(n, *inner)
    return url


def test_producer_runs_ahead_of_consumer(tmp_path) -> None:
    url = _write(tmp_path)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))

    produced: list[int] = []

    def record(batch):  # runs in the producer thread
        produced.append(int(batch.sample_indices[0]))
        return batch

    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        batch_size=4,
        block_chunks=2,
        prefetch_depth=2,
        batch_transforms=[record],
    )
    ds.set_epoch(0)

    it = iter(ds)
    first = next(it)  # pulling one batch starts the producer, which runs ahead
    time.sleep(0.2)  # let the background producer fill the bounded queue

    # With prefetch_depth=2 the producer assembles the consumed batch + fills the
    # queue + parks on the next put => >= depth + 1 produced. Demand-driven would
    # have produced exactly 1.
    assert len(produced) >= 3, f"producer did not run ahead: produced={produced}"

    rest = list(it)  # draining the rest still works
    assert first is not None
    assert len(rest) >= 1


def test_prefetch_preserves_values_and_coverage(tmp_path) -> None:
    url = _write(tmp_path)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))
    src = zarr.open_group(store=store_from_url(url), mode="r")["t2m"][:]

    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        batch_size=4,
        block_chunks=2,
        prefetch_depth=3,
    )
    ds.set_epoch(0)

    seen: list[np.ndarray] = []
    for batch in ds:
        idx = batch.sample_indices
        np.testing.assert_array_equal(batch.arrays["t2m"], src[idx])
        seen.append(idx)

    all_seen = np.concatenate(seen)
    train = manifest.sample_indices(SplitName.TRAIN, geom)
    assert sorted(all_seen.tolist()) == sorted(train.tolist())


def test_partial_iteration_reaps_producer(tmp_path) -> None:
    """Stopping early (a bare ``break``) must reap the producer thread.

    Every other test drains fully (``for b in ds`` / ``list``), so the early-break
    teardown path -- which ``--max-batches``, ``islice``, step budgets, and early
    stopping all hit -- went uncovered. That is exactly where task cancellation lives
    and where the leaked-pin deadlock hid. Assert the prefetch thread is joined, not
    left parked. (Reuse-after-partial is covered, with a timeout guard, by the next
    test.)
    """
    url = _write(tmp_path, n=160, spc=8)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        batch_size=4,
        block_chunks=2,
        prefetch_depth=2,
    )

    ds.set_epoch(0)
    it = iter(ds)
    for i, _ in enumerate(it):  # consume a couple, then stop short
        if i == 1:
            break
    it.close()  # deterministic teardown (a bare break leaves this to GC)

    # The producer thread must be joined by teardown, not left parked on wait_ready.
    # (Reuse-after-partial is covered, with a timeout guard, by the next test.)
    assert not any(t.name == "insitu-prefetch" and t.is_alive() for t in threading.enumerate()), (
        "prefetch thread survived partial-iteration teardown"
    )
    ds.close()


def test_early_break_then_next_epoch_does_not_deadlock(tmp_path) -> None:
    """A capped epoch (early break) must not poison the next epoch.

    Regression: ``try_admit`` pins each chunk the driver admits; the producer only
    unpins a block after consuming it. Breaking mid-epoch leaves the read-ahead (and
    the un-drained current block) **pinned** in the persistent pool, and the
    scheduler's teardown cancels tasks without unpinning. The next epoch then
    inherits those leaked pins -- once they fill the residency budget, admission can
    free no room and the driver deadlocks on ``_capacity`` while the consumer hangs
    in ``wait_ready`` (observed on the bench: scheduler loop idle, prefetch thread
    parked, no in-flight work). Pins must not survive an epoch.
    """
    url = _write(tmp_path, n=160, spc=8)  # 20 chunks; train split ~16
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))

    # budget = 2 * block_chunks chunks; batch_size < one block so a block yields
    # several batches -> breaking after one batch leaves the current block pinned too.
    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        batch_size=4,
        block_chunks=2,
        prefetch_depth=2,
    )

    # epoch 0: pull one batch, let the producer read ahead (pinning), then abort.
    ds.set_epoch(0)
    it = iter(ds)
    next(it)
    time.sleep(0.3)  # let read-ahead pin a block or two before we tear down
    it.close()  # deterministic generator teardown (GeneratorExit -> __iter__ finally)

    # epoch 1: must run to completion, not deadlock on leaked pins.
    done = threading.Event()
    seen: list[np.ndarray] = []

    def drain() -> None:
        for batch in ds:
            seen.append(batch.sample_indices)
        done.set()

    worker = threading.Thread(target=drain, daemon=True)
    worker.start()
    assert done.wait(timeout=30), "epoch 1 deadlocked after an early break in epoch 0"

    all_seen = np.concatenate(seen)
    train = manifest.sample_indices(SplitName.TRAIN, geom)
    assert sorted(all_seen.tolist()) == sorted(train.tolist())
