"""Residency: a batch must fit the resident window, and starvation must raise.

Two guards over the pool's byte budget, both regressions of the same defect --
a batch is cut over the whole epoch order, so it draws from every shuffle-block
it spans, and those blocks are released only *after* the gather. The residency
floor covers the current block plus one read-ahead, so a batch wider than a
block needs more blocks co-resident than the floor provides and admission parks
forever.

1. The block is widened to hold a batch (:class:`InSituDataset`), which restores
   the "at most two blocks co-resident" invariant the scheduler is written to.
2. Should the working set ever be under-estimated anyway (windowed or
   non-uniform chunking, where the floor is an estimate), a starved admission
   *raises* rather than hanging -- a hang is the worst failure mode, and this one
   cost a night of looking at the storage layer.

The failure mode under test is a permanent hang, so every end-to-end case runs
under the ``run_by`` deadline: a regression must fail the suite, not block it.
Pre-fix, the wedge is the driver parked in ``Scheduler._admit`` and the consumer
parked in ``ChunkPool.wait_ready``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
import zarr

from insitubatch import ensure_local_dir, obstore_store, open_geometries, split_by_chunk
from insitubatch.pool import ChunkPool
from insitubatch.scheduler import Scheduler, SchedulerConfig
from insitubatch.source import InSituDataset

DEADLINE = 30.0  # generous: only a genuine deadlock should ever reach it


def _dataset(url: str, **kwargs: Any) -> InSituDataset:
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    return InSituDataset(obstore_store(url), manifest, **kwargs)


# -- 1. the block is widened to hold a batch --------------------------------


def test_block_widens_to_hold_a_batch(write_zarr) -> None:
    # One sample per chunk (the SDSS multi-plate / Hubble geometry): a 64-sample batch
    # would span four 16-chunk blocks, but the residency floor covers two.
    url, _ = write_zarr(n=128, spc=1)
    ds = _dataset(url, batch_size=64, block_chunks=16)

    assert ds.block_chunks == 64  # ceil(batch_size / sample_chunk_size)


def test_block_request_is_a_floor_not_a_ceiling(write_zarr) -> None:
    # A block that already holds a batch is left exactly as asked -- block_chunks is a
    # shuffle-span knob, and widening it further would silently cost residency.
    url, _ = write_zarr(n=128, spc=8)
    ds = _dataset(url, batch_size=16, block_chunks=16)

    assert ds.block_chunks == 16


def test_block_capped_at_the_chunk_count(write_zarr) -> None:
    # A batch wider than the whole array cannot span more blocks than there are chunks;
    # capping keeps the derived budget from ballooning past the data.
    url, _ = write_zarr(n=16, spc=1)
    ds = _dataset(url, batch_size=256, block_chunks=4)

    assert ds.block_chunks == 16  # the array's chunk count, not 256


def test_budget_covers_every_block_a_batch_spans(write_zarr) -> None:
    # The floor is two blocks; with the block widened to a batch, that is two batches'
    # worth of chunks -- the invariant the scheduler's release bookkeeping assumes.
    url, _ = write_zarr(n=128, spc=1, inner=(2, 2))
    ds = _dataset(url, batch_size=64, block_chunks=16)

    chunk_bytes = 1 * 2 * 2 * 4  # spc * inner * f4
    assert ds.cache_budget_bytes >= 2 * 64 * chunk_bytes


@pytest.mark.parametrize("shuffle", [False, True])
def test_epoch_completes_with_one_sample_per_chunk(write_zarr, run_by, shuffle: bool) -> None:
    # The SDSS multi-plate regression, offline: 128 chunks of one sample, batch_size=64.
    # Pre-fix this delivers no batch at all, in either draw order.
    url, src = write_zarr(n=128, spc=1)
    ds = _dataset(url, batch_size=64, block_chunks=16, shuffle=shuffle)

    def epoch() -> list[np.ndarray]:
        return [b.sample_indices.copy() for b in ds.train]

    seen = np.concatenate(run_by(DEADLINE, epoch))
    np.testing.assert_array_equal(np.sort(seen), np.arange(128))  # every sample, once
    ds.close()


def test_delivered_samples_are_correct_with_one_sample_per_chunk(write_zarr, run_by) -> None:
    # Widening the block must not disturb what a batch actually contains.
    url, src = write_zarr(n=128, spc=1)
    ds = _dataset(url, batch_size=64, block_chunks=16, shuffle=True)

    def epoch() -> dict[int, np.ndarray]:
        got: dict[int, np.ndarray] = {}
        for batch in ds.train:
            for k, idx in enumerate(batch.sample_indices):
                got[int(idx)] = batch.arrays["t2m"][k]
        return got

    got = run_by(DEADLINE, epoch)
    for i in range(128):
        np.testing.assert_array_equal(got[i], src["t2m"][i])
    ds.close()


# -- 2. a starved admission raises instead of hanging -----------------------


@pytest.fixture
def small_store(tmp_path):
    """8 chunks of one sample -- small enough to starve a 2-chunk budget."""
    url = f"file://{tmp_path}/starve.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    arr = group.create_array("t2m", shape=(8, 2, 2), chunks=(1, 2, 2), dtype="f4")
    arr[:] = np.random.default_rng(0).standard_normal((8, 2, 2)).astype("f4")
    return url


def test_starved_admission_raises_instead_of_hanging(small_store, run_by) -> None:
    # The provably-terminal state: nothing in flight (so no scatter can complete), the
    # budget full of pinned slots (so no eviction can free one), and the consumer blocked
    # waiting on a chunk that was never admitted (so no unpin can come). Only the
    # scheduler can break the tie, and it must do so by raising.
    geoms = open_geometries(obstore_store(small_store))
    geom = geoms["t2m"]
    chunk_bytes = 2 * 2 * 4
    pool = ChunkPool(geoms, budget_bytes=2 * chunk_bytes)  # room for two of eight chunks

    with Scheduler(obstore_store(small_store), geoms, pool, SchedulerConfig()) as sched:
        sched.start(list(range(geom.n_chunks)), geom.sample_chunk_size)
        # Wait on the last chunk and never unpin the earlier ones.
        with pytest.raises(RuntimeError, match="residency budget"):
            run_by(DEADLINE, lambda: pool.wait_ready("t2m", 7))


def test_slow_consumer_is_not_mistaken_for_starvation(small_store, run_by) -> None:
    # False-positive guard: a consumer that is merely slow still unpins, so admission is
    # starved but *not* terminal. It must wait, however long the pause -- the detector
    # keys on the deadlock being provable, never on elapsed time.
    geoms = open_geometries(obstore_store(small_store))
    geom = geoms["t2m"]
    chunk_bytes = 2 * 2 * 4
    pool = ChunkPool(geoms, budget_bytes=2 * chunk_bytes)

    def drain(sched: Scheduler) -> int:
        n = 0
        for cid in range(geom.n_chunks):
            pool.wait_ready("t2m", cid)
            time.sleep(0.25)  # longer than the starvation poll interval
            sched.unpin_block({(geom.path, cid)})
            n += 1
        return n

    with Scheduler(obstore_store(small_store), geoms, pool, SchedulerConfig()) as sched:
        sched.start(list(range(geom.n_chunks)), geom.sample_chunk_size)
        assert run_by(DEADLINE, lambda: drain(sched)) == geom.n_chunks
