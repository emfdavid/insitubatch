"""An abandoned pass must give the pool back everything it held.

Breaking out of a loader loop early is ordinary code -- a smoke run over the first N
batches, an evaluation subset, ``itertools.islice`` -- and the pass that gets abandoned
is usually abandoned *because* it was slow, so its driver is still fetching when the
generator is finalized.

The defect these tests pin: the release ran *inside* the scheduler's ``with`` block, so
it happened before the driver was stopped. Closing is not synchronous on a borrowed
event loop, so a cancelled-but-not-yet-unwound driver could still reach ``try_admit``
and re-pin a slot under an owner that had just been released. Nothing releases that pin
and the slot stays FILLING -- its sibling tiles were cancelled -- so it never becomes
READY and therefore never becomes evictable. Every abandoned pass burned part of the
budget for the life of the dataset, until a later epoch starved on a pool it could not
free.

Why the store injects latency: with an instant store the driver is usually done by the
time the consumer breaks, so the race does not run and the test passes on broken code.
The delay makes the driver certainly-in-flight at teardown, which is the condition the
bug needs. It is not a timing *assertion* -- nothing here measures time.

These assert the *invariant* (nothing is left held), not the consequence. The
consequence -- a later epoch raising "residency budget exhausted ... this would hang" --
was reproduced on cloud stores (the `era5_c1` family on GCS and S3, batch 16 over a
16-chunk block) but not on any local fixture: each abandoned pass leaks only a few
slots, and a fixture small enough for CI survives them. A version of that test was
written, confirmed to pass against the broken code, and deleted -- a test that cannot
fail is worse than no test, because it reads like coverage.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from zarr.storage import WrapperStore

from insitubatch import obstore_store, open_geometries, split_by_chunk
from insitubatch.pool import SlotState
from insitubatch.source import InSituDataset

# One sample per chunk, and a batch as wide as the shuffle block: the shape where a
# batch pins a whole block, so a leak of even a few slots is fatal rather than absorbed
# by slack. This is the benchmark suite's `c1` configuration (bench/run.py defaults).
N, SPC, BATCH, BLOCK = 256, 1, 16, 16
DELAY_S = 0.01


class SlowStore(WrapperStore):  # type: ignore[type-arg]
    """Delays chunk reads, so the driver is still fetching when the consumer breaks."""

    async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
        if "/c/" in key:
            await asyncio.sleep(DELAY_S)
        return await self._store.get(key, prototype, byte_range=byte_range)


@pytest.fixture
def slow(write_zarr):
    url, _ = write_zarr(n=N, spc=SPC, inner=(8, 8))

    def _open() -> Any:
        return SlowStore(obstore_store(url))

    geometries = open_geometries(_open())
    manifest = split_by_chunk(geometries["t2m"], fractions=(1.0, 0.0, 0.0))
    return _open, geometries, manifest


def _dataset(slow) -> InSituDataset:
    _open, geometries, manifest = slow
    return InSituDataset(
        _open(),
        manifest,
        geometries=geometries,
        batch_size=BATCH,
        block_chunks=BLOCK,
        shuffle=True,
        seed=0,
    )


def _abandon(ds: InSituDataset, after: int = 1) -> None:
    """Consume `after` batches, then walk away -- the `break` every user writes."""
    for i, _batch in enumerate(ds.train):
        if i + 1 >= after:
            break


def test_an_abandoned_pass_leaves_no_references_behind(slow) -> None:
    """The pool's reference map is empty once the pass is gone.

    Asserted on the map rather than on throughput or memory because a leaked pin is
    invisible to both until the budget happens to run out -- which is a later epoch's
    problem, on a different machine, presenting as a hang.
    """
    ds = _dataset(slow)
    _abandon(ds)

    pool = ds._pool
    assert pool._pinned == {}, f"references survived the pass: {pool._pinned}"
    assert pool.active_owners == 0


def test_an_abandoned_pass_leaves_no_unfinishable_slots(slow) -> None:
    """No slot is left that can never finish.

    A partial whose sibling tiles were cancelled can never reach READY, and only READY
    slots are evictable -- so a partial left behind is budget that never comes back.

    Settled first, deliberately. Closing cancels the driver's coroutines, but decode work
    already handed to the process-wide executor keeps running: a slot can legitimately be
    mid-write at the instant the pass ends, and it resolves a moment later (measured: 0
    survivors in 40 trials). Asserting the instant instead of the outcome is a flake --
    this test had one. What must never survive the settle is an *orphan*: quiescent,
    unreferenced, and not READY, which is exactly what the leak left behind.
    """
    ds = _dataset(slow)
    _abandon(ds)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        pool = ds._pool
        with pool._cv:
            orphans = {
                k: sl.state.name
                for k, sl in pool._slots.items()
                if sl.state is not SlotState.READY and sl.quiescent and pool._refs(k) == 0
            }
            busy = any(not sl.quiescent for sl in pool._slots.values())
        if not busy:
            break
        time.sleep(0.05)
    assert orphans == {}, f"slots left unfinishable: {orphans}"


def test_repeated_abandonment_does_not_accumulate(slow) -> None:
    """Ten abandoned passes leave the pool exactly as empty as one does.

    The failure mode was cumulative -- each pass burned a few more slots -- so a single
    abandonment can look survivable on a generous budget while a training loop that
    breaks every epoch dies on epoch six.
    """
    ds = _dataset(slow)
    for epoch in range(10):
        ds.set_epoch(epoch)
        _abandon(ds)

    assert ds._pool._pinned == {}
    assert ds._pool.active_owners == 0
