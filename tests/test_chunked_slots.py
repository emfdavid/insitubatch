"""The pool's buffer unit is the stored chunk: residency, edges, and the budget.

A slot adopts decoded tiles by reference instead of copying them into one assembled
array. Two consequences need pinning, because neither is visible on a grid that divides
evenly -- and every probe store we had divided evenly (720/45, 1440/90, 2048/256):

* a stored chunk is decoded **whole**, padding included, and we keep it whole, so
  residency is ``n_tiles * chunk_shape`` and can EXCEED the assembled ``slot_shape``;
* the budget must charge that, or a pool told N bytes quietly resides more.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from insitubatch import ensure_local_dir, obstore_store, open_geometries
from insitubatch.pool import ChunkPool
from insitubatch.scheduler import Scheduler, SchedulerConfig
from insitubatch.transforms import StandardScaler


def _store(tmp_path, shape, chunks, name="ragged.zarr"):
    url = f"file://{tmp_path}/{name}"
    ensure_local_dir(url)
    g = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    a = g.create_array("v", shape=shape, chunks=chunks, dtype="f4")
    a[:] = np.random.default_rng(0).standard_normal(shape).astype("f4")
    return url, np.asarray(a[:])


def test_ragged_grid_charges_the_padding_it_actually_holds(tmp_path):
    """721 rows chunked at 180 stores 900 rows. The budget must say so.

    Charging ``slot_shape`` would report 3.96 MiB while holding 4.94 MiB -- a pool told
    "2048 MiB" would resident ~2560. The ratio here is the ERA5 one, 1.248x.
    """
    url, _ = _store(tmp_path, (2, 721, 1440), (1, 180, 360))
    geoms = open_geometries(obstore_store(url))
    geom = geoms["v"]
    assert geom.n_inner_chunks(0) == 5 * 4, "grid must be ragged for this test to mean anything"

    pool = ChunkPool(geoms)
    pool.try_admit("v", 0, pool.new_owner())

    assembled = int(np.prod(geom.slot_shape(0))) * geom.dtype.itemsize
    held = geom.n_inner_chunks(0) * int(np.prod(geom.chunks)) * geom.dtype.itemsize
    assert held > assembled, "a ragged grid stores more than the array covers"
    assert pool.resident_bytes == held
    assert pool.resident_bytes != assembled
    assert held / assembled == pytest.approx(1.248, abs=0.005)


def test_short_final_chunk_charges_the_full_tiles(tmp_path):
    """A short final OUTER chunk compounds it: tiles are spc-deep regardless."""
    url, _ = _store(tmp_path, (10, 8, 8), (4, 8, 8), name="short.zarr")
    geoms = open_geometries(obstore_store(url))
    geom = geoms["v"]
    last = geom.n_chunks - 1
    assert len(geom.samples_in_chunk(last)) == 2, "final chunk must be short"

    pool = ChunkPool(geoms)
    pool.try_admit("v", last, pool.new_owner())
    assembled = int(np.prod(geom.slot_shape(last))) * geom.dtype.itemsize
    held = geom.n_inner_chunks(last) * int(np.prod(geom.chunks)) * geom.dtype.itemsize
    assert pool.resident_bytes == held == 2 * assembled


def test_tiles_are_adopted_not_copied(tmp_path):
    """The decoded tile IS the resident buffer -- the fill path has no memcpy left."""
    url, _ = _store(tmp_path, (1, 8, 8), (1, 4, 4), name="adopt.zarr")
    geoms = open_geometries(obstore_store(url))
    pool = ChunkPool(geoms)
    pool.try_admit("v", 0, pool.new_owner())
    tile = np.arange(16, dtype="f4").reshape(1, 4, 4)
    pool.deliver_tile("v", 0, (0, 0), tile)
    assert pool._slots[("v", 0)].tiles[(0, 0)] is tile, "the pool copied instead of adopting"


@pytest.mark.parametrize("chunks", [(1, 180, 360), (1, 721, 1440)])
def test_ragged_gather_matches_the_array(tmp_path, chunks):
    """End to end: edge tiles carry padding, and none of it reaches the batch."""
    url, src = _store(tmp_path, (3, 721, 1440), chunks, name=f"g{chunks[1]}.zarr")
    geoms = open_geometries(obstore_store(url))
    geom = geoms["v"]
    with Scheduler(obstore_store(url), geoms, ChunkPool(geoms), SchedulerConfig()) as sched:
        sched.start(range(geom.n_chunks), geom.sample_chunk_size)
        got = []
        for cid in range(geom.n_chunks):
            sched.pool.wait_ready("v", cid, sched.owner)
            rows = np.array([[cid, 0]], dtype=np.int64)
            got.append(sched.pool.gather(rows, ["v"], geom.sample_chunk_size).arrays["v"])
    np.testing.assert_array_equal(np.concatenate(got), src)


def test_a_transformed_chunk_is_a_one_tile_slot(tmp_path):
    """A transform needs a whole array, so its slot republishes as a single tile.

    That is what keeps `gather` on one path: it never asks "assembled or tiled?", because
    a whole-array slot is just a 1x1 grid.
    """
    url, src = _store(tmp_path, (2, 8, 8), (1, 4, 4), name="tx.zarr")
    geoms = open_geometries(obstore_store(url))
    geom = geoms["v"]
    pool = ChunkPool(geoms, chunk_transforms=[StandardScaler(mean={"v": 0.0}, std={"v": 2.0})])
    with Scheduler(obstore_store(url), geoms, pool, SchedulerConfig()) as sched:
        sched.start(range(geom.n_chunks), geom.sample_chunk_size)
        sched.pool.wait_ready("v", 0, sched.owner)
        slot = pool._slots[("v", 0)]
        assert len(slot.tiles) == 1, "a transformed chunk must publish as one tile"
        rows = np.array([[0, 0]], dtype=np.int64)
        got = pool.gather(rows, ["v"], geom.sample_chunk_size).arrays["v"]
    np.testing.assert_allclose(got[0], src[0] / 2.0, rtol=1e-6)


def test_user_code_never_runs_on_the_shared_event_loop(tmp_path):
    """A ``chunk_transform`` must not execute on zarr's process-global loop.

    Publishing an assembled chunk runs the assembly memcpy, the user transform and the
    mmap write-back. The scheduler now shares zarr's loop with the whole process, so doing
    any of that on the loop thread stalls every other zarr caller -- the exact shared-fate
    failure the loop consolidation is meant to avoid. Regression: an earlier draft
    delivered inline unconditionally and ran user code on ``zarr_io``.
    """
    import threading

    ran_on: list[str] = []

    class Spy:
        def __call__(self, chunk):
            ran_on.append(threading.current_thread().name)
            return chunk

    url, _ = _store(tmp_path, (2, 8, 8), (1, 4, 4), name="thread.zarr")
    geoms = open_geometries(obstore_store(url))
    geom = geoms["v"]
    pool = ChunkPool(geoms, chunk_transforms=[Spy()])
    with Scheduler(obstore_store(url), geoms, pool, SchedulerConfig()) as sched:
        sched.start(range(geom.n_chunks), geom.sample_chunk_size)
        for cid in range(geom.n_chunks):
            sched.pool.wait_ready("v", cid, sched.owner)

    assert ran_on, "the transform never ran"
    assert all(t.startswith("insitu-dec") for t in ran_on), (
        f"chunk_transform ran on {sorted(set(ran_on))}; it must run on the decode pool, "
        "never on the shared event loop"
    )
