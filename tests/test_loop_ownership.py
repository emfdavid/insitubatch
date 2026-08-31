"""Teardown ownership: what a scheduler may cancel, and whose loop it may stop.

These are the regressions behind the #30 shared-loop decision. Every failure they
cover lives in ``close()``, not in steady state -- a back-pressured ``_drive`` on a
shared loop does *not* starve other work, but a naive teardown destroys it:

1. ``asyncio.all_tasks(loop)`` is the **whole loop's** task set, so cancelling it
   kills unrelated ``zarr`` sync work mid-flight (observed as ``CancelledError``
   raised inside an innocent read on another thread).
2. ``loop.stop()`` / ``loop.close()`` on a borrowed loop leaves zarr's
   process-global loop dead for the rest of the process.

Both are silent and intermittent in the wild, so they are pinned here.
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest
import zarr
from zarr.core.sync import _get_loop

from insitubatch import ensure_local_dir, obstore_store, open_geometries
from insitubatch.pool import ChunkPool
from insitubatch.scheduler import Scheduler, SchedulerConfig


@pytest.fixture
def store_url(tmp_path):
    url = f"file://{tmp_path}/own.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    arr = group.create_array("v", shape=(4, 6, 6), chunks=(1, 6, 6), dtype="f4")
    arr[:] = np.random.default_rng(0).standard_normal((4, 6, 6)).astype("f4")
    return url


def _sched(url) -> Scheduler:
    geoms = open_geometries(obstore_store(url))
    return Scheduler(url, geoms, ChunkPool(geoms), SchedulerConfig(max_inflight=2))


def test_shutdown_cancels_only_our_tasks(store_url):
    """A foreign task on our loop survives ``close()``; ours are cancelled."""
    sched = _sched(store_url)
    started = threading.Event()
    foreign_cancelled = threading.Event()
    foreign_finished = threading.Event()

    async def foreign() -> None:
        # Stands in for unrelated zarr-sync work sharing the loop.
        started.set()
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            foreign_cancelled.set()
            raise
        foreign_finished.set()

    fut = asyncio.run_coroutine_threadsafe(foreign(), sched._loop)
    assert started.wait(timeout=5)
    # Cancel + drain OUR tasks only. (close() would also stop the loop, which would
    # orphan the foreign task for reasons unrelated to what this test pins.)
    done = asyncio.run_coroutine_threadsafe(sched._shutdown(), sched._loop)
    done.result(timeout=5)

    assert not foreign_cancelled.is_set(), "shutdown cancelled a task it did not create"
    fut.result(timeout=5)
    assert foreign_finished.is_set()
    sched.close()


def test_shutdown_cancels_what_we_registered(store_url):
    """The flip side of the pair: anything in ``_tasks`` IS cancelled.

    Together with the test above this pins the whole contract -- in ``_tasks`` means
    ours and gets cancelled, absent from it means someone else's and is left alone --
    without racing a real driver to completion.
    """
    sched = _sched(store_url)
    running = threading.Event()

    async def ours() -> None:
        running.set()
        await asyncio.sleep(30)

    async def register() -> asyncio.Task:
        task = asyncio.create_task(ours())
        sched._tasks.add(task)
        task.add_done_callback(sched._tasks.discard)
        return task

    task = asyncio.run_coroutine_threadsafe(register(), sched._loop).result(timeout=5)
    assert running.wait(timeout=5)
    asyncio.run_coroutine_threadsafe(sched._shutdown(), sched._loop).result(timeout=5)

    assert task.cancelled(), "shutdown left one of our own tasks running"
    sched.close()


def test_parked_driver_is_registered_and_cancelled(store_url):
    """A real driver, parked on a full budget, is tracked and torn down by close().

    Budget holds one chunk while four are requested and nothing consumes, so ``_admit``
    parks -- the state ``close()`` actually has to unwind mid-epoch.
    """
    geoms = open_geometries(obstore_store(store_url))
    geom = next(iter(geoms.values()))
    one_chunk = int(np.prod(geom.slot_shape(0))) * geom.dtype.itemsize
    pool = ChunkPool(geoms, budget_bytes=one_chunk)
    sched = Scheduler(store_url, geoms, pool, SchedulerConfig(max_inflight=2))

    sched.start(np.arange(geom.n_chunks), geom.sample_chunk_size)
    for _ in range(500):  # wait for the driver to park, not for a wall-clock guess
        if sched._tasks:
            break
        threading.Event().wait(0.01)
    assert sched._tasks, "driver never registered itself for shutdown"

    sched.close()
    assert not [t for t in sched._tasks if not t.done()], "close() left our tasks running"


def test_borrowed_loop_is_not_stopped_or_closed(store_url):
    """``close()`` leaves zarr's loop running and usable -- it is not ours to close.

    Note there is nothing to "tidy up" afterwards: the loop belongs to zarr and lives for
    the process. An earlier draft of this test "cleaned up" by closing the loop itself,
    which stopped zarr's process-global loop and hung every subsequent test in the session
    on ``zarr.core.sync.sync()``. That is exactly the failure mode this contract exists to
    prevent, and it is one line away at all times.
    """
    sched = _sched(store_url)
    loop = sched._loop
    assert loop is _get_loop(), "the scheduler must run on zarr's loop, not one of its own"

    sched.close()

    assert not loop.is_closed(), "close() closed a loop it does not own"
    assert loop.is_running(), "close() stopped a loop it does not own"
    # ...and it still runs work afterwards, which is the property that actually matters.
    assert asyncio.run_coroutine_threadsafe(asyncio.sleep(0, result=7), loop).result(timeout=5) == 7
