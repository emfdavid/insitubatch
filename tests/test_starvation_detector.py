"""A stall is only terminal if the waiters really cannot proceed.

`Scheduler._starvation` turns an admission stall into a raise instead of a hang, and it
proves the stall terminal from three facts, one of which is "a consumer is blocked in
`wait_ready`, so it can never reach its next unpin and free budget". That fact was read
off the registration in `_waiting`, which is a weaker claim than it looks: a waiter is
registered *before* it tests its condition and is removed only after its thread is
rescheduled, so a waiter whose chunk is already READY -- one that is about to wake,
gather and unpin, i.e. exactly the thing that frees budget -- was reported as blocked.

Observed on a real store: at the instant the detector fired, the sole waiter was on a
chunk in state READY, pinned by its own owner. Nothing was wrong with the run; it was
killed with "Nothing can free a slot, so this would hang" while holding a full, healthy,
two-block working set. It reproduced roughly once in twenty passes on GCS and S3 with a
second pass sharing the loop, and never on a quiet process -- the shape of a scheduling
race, not of a budget.

These tests fix the *question* rather than the timing: they construct the registered-but-
satisfiable waiter directly, so nothing here depends on which thread the OS runs next.
"""

from __future__ import annotations

import numpy as np
import pytest

from insitubatch import obstore_store, open_geometries, split_by_chunk
from insitubatch.pool import ChunkPool
from insitubatch.scheduler import Scheduler
from insitubatch.source import InSituDataset
from insitubatch.types import ArrayGeometry

ARRAY, CID, OWNER = "t2m", 3, 1


@pytest.fixture
def pool() -> ChunkPool:
    geom = ArrayGeometry(
        path=ARRAY,
        shape=(32, 4, 4),
        chunks=(4, 4, 4),
        dtype=np.dtype("f4"),
        sample_axis=0,
    )
    return ChunkPool({ARRAY: geom})


def _ready_chunk(pool: ChunkPool) -> tuple[str, int]:
    """Admit one chunk and deliver its tile, so the slot is READY and referenced."""
    assert pool.try_admit(ARRAY, CID, OWNER)
    with pool.tile_write(ARRAY, CID, ()) as write:
        write.deliver(np.zeros(pool._by_path[ARRAY].tile_shape(), dtype="f4"))
    return (ARRAY, CID)


def _register_waiter(pool: ChunkPool, key: tuple[str, int], owner: int) -> None:
    """The window: a thread inside `wait_ready` that has registered and not yet re-run.

    Poked in directly rather than raced for. A thread really parked here is
    indistinguishable to the pool, and driving the OS scheduler into the window on demand
    is exactly the flakiness this test exists to avoid.
    """
    with pool._cv:
        pool._waiting[(key, owner)] = pool._waiting.get((key, owner), 0) + 1


def test_a_waiter_whose_chunk_is_ready_is_not_blocked(pool: ChunkPool) -> None:
    """The defect: this waiter is one scheduling quantum from freeing budget."""
    key = _ready_chunk(pool)
    _register_waiter(pool, key, OWNER)

    assert pool.blocked_waiters() == []


def test_a_waiter_on_an_unadmitted_chunk_is_blocked(pool: ChunkPool) -> None:
    """The control: the detector must still see a genuine stall.

    Without this, "report nothing" would pass the test above and silently convert every
    real deadlock back into the hang the detector exists to prevent.
    """
    _register_waiter(pool, (ARRAY, 7), OWNER)

    assert pool.blocked_waiters() == [(ARRAY, 7)]


def test_a_ready_chunk_referenced_by_another_owner_still_blocks(pool: ChunkPool) -> None:
    """Readiness alone is not the condition -- the slot must be referenced by *this* owner.

    A second iteration's pin satisfies nothing for ours (#35), so a waiter in that state
    is genuinely stuck and must be reported.
    """
    key = _ready_chunk(pool)
    _register_waiter(pool, key, owner=OWNER + 1)

    assert pool.blocked_waiters() == [key]


def test_a_failed_chunk_does_not_block(pool: ChunkPool) -> None:
    """A waiter on a failed slot wakes and raises; it is not waiting on the budget."""
    assert pool.try_admit(ARRAY, CID, OWNER)
    with pool.tile_write(ARRAY, CID, ()) as write:
        write.fail(RuntimeError("bad chunk"))
    _register_waiter(pool, (ARRAY, CID), OWNER)

    assert pool.blocked_waiters() == []


# -- premise 2: "nothing in flight" must mean "no delivery pending" -----------


def test_a_queued_tile_task_counts_as_delivery_pending(write_zarr) -> None:
    """`max_inflight=1` serialises fetches, so the gap between one tile releasing the
    semaphore and the next acquiring it is wide -- and `_inflight_now` counts only tasks
    that have acquired it. A run whose deliveries are merely *queued* is not stalled.

    Observed on a real store at `max_inflight=1`: the detector fired with
    `inflight_now=0`, 28 tile tasks created and unfinished, and its one waiter on a chunk
    in state FILLING that one of those 28 was about to fill.
    """
    url, _ = write_zarr(n=512, spc=1, inner=(8, 8))
    geometries = open_geometries(obstore_store(url))
    manifest = split_by_chunk(geometries["t2m"], fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        obstore_store(url),
        manifest,
        geometries=geometries,
        batch_size=16,
        block_chunks=4,
        max_inflight=1,
        shuffle=True,
        seed=0,
    )
    ds.set_epoch(0)

    got = sum(int(b.arrays["t2m"].shape[0]) for b in ds.train)
    assert got == 512


# -- premise 2, at the level the detector reads it ---------------------------


class _UnfinishedTask:
    """Stands in for a tile task that has been created and has not finished.

    A real one needs a running loop and a store to fetch from; the detector only ever
    asks `done()`, and *that* is the whole question -- whether a delivery is still coming.
    """

    def done(self) -> bool:
        return False


def _scheduler(write_zarr, pool_out: list[ChunkPool]) -> Scheduler:
    url, _ = write_zarr(n=32, spc=4, inner=(4, 4))
    geometries = open_geometries(obstore_store(url))
    pool = ChunkPool(geometries)
    pool_out.append(pool)
    return Scheduler(obstore_store(url), geometries, pool)


def test_a_queued_tile_task_is_not_a_terminal_stall(write_zarr) -> None:
    """A stall with deliveries outstanding is a slow run, not a wedged one."""
    pools: list[ChunkPool] = []
    sched = _scheduler(write_zarr, pools)
    pool = pools[0]
    _register_waiter(pool, ("t2m", 5), OWNER)  # genuinely blocked: never admitted
    sched._tiles.add(_UnfinishedTask())  # type: ignore[arg-type]

    assert sched._starvation("t2m", 5) is None


def test_a_stall_with_nothing_outstanding_is_still_terminal(write_zarr) -> None:
    """The control: with no delivery coming and a genuinely blocked waiter, it must raise.

    Without this, "never report a stall" would satisfy the test above and turn every real
    deadlock back into the silent hang the detector exists to replace.
    """
    pools: list[ChunkPool] = []
    sched = _scheduler(write_zarr, pools)
    _register_waiter(pools[0], ("t2m", 5), OWNER)

    err = sched._starvation("t2m", 5)
    assert err is not None and "residency budget exhausted" in str(err)


# -- premise 3: the blocked consumer that matters is *this* pass's -----------
#
# `_starvation` reads the pool's waiters pool-wide on purpose: an admission stall is about
# the shared byte budget, and another iteration's blocked consumer is holding some of it.
# A fetch-ahead permit is the opposite -- `_ahead` belongs to one Scheduler and is handed
# back only by that pass's own `unpin_block` -- so another pass being blocked says nothing
# about whether ours can proceed.


def test_another_passs_blocked_consumer_is_not_our_permit_stall(write_zarr) -> None:
    """Two iterations over one pool are the normal case, not the pathological one.

    `zip(ds.train, ds.val)` interleaves two passes by construction, so at any instant one
    of them is very likely parked in `wait_ready` while the other's driver wants a permit.
    Reading the waiters pool-wide made that ordinary overlap terminal: the pass that was
    running fine raised "read-ahead permits exhausted" because the *other* one was mid-wait.
    """
    pools: list[ChunkPool] = []
    sched = _scheduler(write_zarr, pools)
    other = pools[0].new_owner()
    assert other != sched.owner
    _register_waiter(pools[0], ("t2m", 5), owner=other)

    assert sched._ahead_starvation("t2m", 5) is None


def test_our_own_blocked_consumer_is_a_permit_stall(write_zarr) -> None:
    """The control: a pass whose own consumer cannot unpin will never see a permit again.

    Without this, "never report a permit stall" would satisfy the test above and restore
    the unbounded wait -- three idle threads and no message -- that the bound replaced.
    """
    pools: list[ChunkPool] = []
    sched = _scheduler(write_zarr, pools)
    _register_waiter(pools[0], ("t2m", 5), owner=sched.owner)

    err = sched._ahead_starvation("t2m", 5)
    assert err is not None and "read-ahead permits exhausted" in str(err)


def test_a_permit_stall_with_a_delivery_coming_is_not_terminal(write_zarr) -> None:
    """The other control, unchanged from the admission detector: a delivery in flight
    will land, be gathered and unpinned, and hand a permit back."""
    pools: list[ChunkPool] = []
    sched = _scheduler(write_zarr, pools)
    _register_waiter(pools[0], ("t2m", 5), owner=sched.owner)
    sched._tiles.add(_UnfinishedTask())  # type: ignore[arg-type]

    assert sched._ahead_starvation("t2m", 5) is None
