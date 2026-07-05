"""Scheduler: end-to-end fetch+decode+scatter, in-flight/residency bounds, errors.

These drive the real :class:`Scheduler` against a local zarr store (the spike's
two layouts incl. partial edges). A tiny in-line consumer plays the role
``source.py`` will: wait on slot readiness, gather, evict. We assert the filled
pool reconstructs ``arr[:]`` (under both unbounded and bounded residency), that
the single ``max_inflight`` budget is honored, and that a driver failure poisons
the pool instead of hanging a waiter.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from insitubatch import ensure_local_dir, obstore_store, open_geometries, valid_anchor_range
from insitubatch.pool import ChunkPool
from insitubatch.scheduler import Scheduler, SchedulerConfig
from insitubatch.types import ArrayGeometry

SHAPE = (5, 9, 7)
LAYOUTS = {"single_inner": (2, 9, 7), "spatial": (2, 4, 4)}


@pytest.fixture
def tiled_store(tmp_path):
    url = f"file://{tmp_path}/sched.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    data = np.random.default_rng(1).standard_normal(SHAPE).astype("f4")
    srcs = {}
    for var, chunks in LAYOUTS.items():
        arr = group.create_array(var, shape=SHAPE, chunks=chunks, dtype="f4")
        arr[:] = data
        srcs[var] = np.asarray(arr[:])
    return url, srcs


def _make(url, geoms, *, max_inflight=4, budget=None, **store_kwargs) -> Scheduler:
    pool = ChunkPool(geoms, budget_bytes=budget)
    return Scheduler(url, geoms, pool, SchedulerConfig(max_inflight=max_inflight), **store_kwargs)


def _drain_in_order(sched: Scheduler, geom: ArrayGeometry, var: str, *, unpin: bool) -> np.ndarray:
    """Consumer stand-in: wait+gather every outer chunk in order, optionally unpinning."""
    spc = geom.sample_chunk_size
    out = []
    for cid in range(geom.n_chunks):
        sched.pool.wait_ready(var, cid)
        n0 = len(geom.samples_in_chunk(cid))
        rows = np.array([[cid, w] for w in range(n0)], dtype=np.int64)
        out.append(sched.pool.gather(rows, [var], spc).arrays[var])
        if unpin:
            sched.unpin_block({(geom.path, cid)})
    return np.concatenate(out, axis=0)


@pytest.mark.parametrize("var", list(LAYOUTS))
@pytest.mark.parametrize("bounded", [False, True])  # unbounded, then a 2-chunk budget
def test_scheduler_fills_pool_matches_array(tiled_store, var, bounded):
    url, srcs = tiled_store
    geoms = open_geometries(obstore_store(url), variables=[var])
    geom = geoms[var]
    budget = None
    if bounded:
        full = geom.sample_chunk_size * int(np.prod(geom.inner_shape)) * geom.dtype.itemsize
        budget = 2 * full  # holds two full chunks; admission must evict to fit the 3rd

    with _make(url, geoms, budget=budget) as sched:
        fut = sched.start(range(geom.n_chunks), geom.sample_chunk_size)
        got = _drain_in_order(sched, geom, var, unpin=bounded)
        fut.result(timeout=30)  # surface any driver error
        assert sched.inflight_peak <= 4  # the single max_inflight budget holds
        if bounded:
            assert sched.pool.max_resident <= 2  # the byte budget bounds residency

    assert np.array_equal(got, srcs[var])


def test_scheduler_inflight_saturates_to_budget(tiled_store):
    """With many tiles and a small budget, in-flight peaks at exactly max_inflight."""
    url, _ = tiled_store
    geoms = open_geometries(
        obstore_store(url), variables=["spatial"]
    )  # 3 chunks x 6 tiles = 18 reads
    with _make(url, geoms, max_inflight=4) as sched:
        fut = sched.start(range(geoms["spatial"].n_chunks), geoms["spatial"].sample_chunk_size)
        _drain_in_order(sched, geoms["spatial"], "spatial", unpin=False)
        fut.result(timeout=30)
        assert sched.inflight_peak == 4


def test_scheduler_close_closes_the_loop(tiled_store):
    """close() must close the loop, not leave it for GC.

    An unclosed event loop is re-closed by ``BaseEventLoop.__del__`` during garbage
    collection, which raises ``ValueError: Invalid file descriptor: -1`` on the
    already-gone self-pipe socket -- a noisy unraisable first seen under
    free-threaded 3.13t (where finalizer timing exposes the latent leak).
    """
    url, _ = tiled_store
    geoms = open_geometries(obstore_store(url), variables=["single_inner"])
    sched = _make(url, geoms)
    sched.start(range(geoms["single_inner"].n_chunks), geoms["single_inner"].sample_chunk_size)
    sched.close()
    assert sched._loop.is_closed()


def test_scheduler_windowed_views_decode_once_and_lead(tiled_store):
    """Two windowed views of one array (same path, different offset) open the array
    once and share one set of decoded slots (decode-once), and ``next = now.shift(1)``
    reads one sample ahead of ``now`` at each anchor.

    Anchors are restricted to ``valid_anchor_range`` so every ``anchor + offset`` stays
    on the array (the producer drops edge anchors; here we do it explicitly).
    """
    url, srcs = tiled_store
    base = open_geometries(obstore_store(url), variables=["single_inner"])["single_inner"]
    geoms = {"now": base, "next": base.shift(1)}  # one array, two views
    spc = base.sample_chunk_size
    src = srcs["single_inner"]
    lo, hi = valid_anchor_range([0, 1], base.n_samples)  # (0, T-1): drop the last anchor
    anchors = np.arange(lo, hi)
    rows = np.stack([anchors // spc, anchors % spc], axis=1).astype(np.int64)

    with _make(url, geoms) as sched:
        fut = sched.start(range(base.n_chunks), base.sample_chunk_size)
        for cid in range(base.n_chunks):
            sched.pool.wait_ready(base.path, cid)  # slots are keyed by path
        batch = sched.pool.gather(rows, ["now", "next"], spc)
        fut.result(timeout=30)  # surface any driver error
        assert len(sched._arrays) == 1  # opened once: deduped by path
        assert len(sched.pool._slots) == base.n_chunks  # one array's slots, not two

    assert np.array_equal(batch.sample_indices, anchors)  # provenance is the anchor
    assert np.array_equal(batch.arrays["now"], src[anchors])  # now reads array[anchor]
    assert np.array_equal(batch.arrays["next"], src[anchors + 1])  # next leads by one


def test_scheduler_poisons_pool_on_driver_failure(tiled_store):
    """A variable absent from the store fails array-open; the pool poison unblocks waiters."""
    url, _ = tiled_store
    ghost = ArrayGeometry(path="ghost", shape=(4, 2, 2), chunks=(2, 2, 2), dtype=np.dtype("f4"))
    with _make(url, {"ghost": ghost}) as sched:
        fut = sched.start([0, 1], ghost.sample_chunk_size)
        with pytest.raises(Exception):  # noqa: B017 - zarr surfaces a store-specific error type
            fut.result(timeout=30)
        with pytest.raises(Exception):  # noqa: B017 - same error re-raised to the consumer
            sched.pool.wait_ready("ghost", 0)
