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


# -- persistence: tile-major on disk, one path in memory -----------------------------


def _fill(url, geoms, **poolkw):
    """Drive one full pass so every chunk is fetched, assembled and persisted."""
    geom = geoms["v"]
    pool = ChunkPool(geoms, **poolkw)
    with Scheduler(obstore_store(url), geoms, pool, SchedulerConfig()) as sched:
        sched.start(range(geom.n_chunks), geom.sample_chunk_size)
        for cid in range(geom.n_chunks):
            sched.pool.wait_ready("v", cid, sched.owner)
    return pool


def test_cache_dir_no_longer_assembles(tmp_path):
    """Persisting is not a reason to assemble -- only a chunk_transform is."""
    url, _ = _store(tmp_path, (2, 8, 8), (1, 4, 4), name="p1.zarr")
    geoms = open_geometries(obstore_store(url))
    assert ChunkPool(geoms, backing_dir=tmp_path / "c1", persist=True).assembles is False
    assert ChunkPool(geoms).assembles is False


def test_persisted_chunk_is_stored_tile_major(tmp_path):
    """One .npy per chunk, shaped (n_tiles, *tile_shape), each tile contiguous."""
    url, src = _store(tmp_path, (2, 8, 8), (1, 4, 4), name="p2.zarr")
    geoms = open_geometries(obstore_store(url))
    geom = geoms["v"]
    cache = tmp_path / "c2"
    _fill(url, geoms, backing_dir=cache, persist=True)

    files = sorted(cache.glob("*.npy"))
    assert len(files) == geom.n_chunks, "file count per chunk must not change"
    on_disk = np.lib.format.open_memmap(files[0], mode="r")
    assert on_disk.shape == (geom.n_inner_chunks(0), *geom.tile_shape())
    # every tile is contiguous in the file, in inner_index order
    for coord in geom.inner_coords():
        proj = geom.tile_placement(0, coord)
        np.testing.assert_array_equal(
            on_disk[geom.inner_index(coord)][proj.chunk_selection], src[0:1][proj.out_selection]
        )


def test_revived_tiles_are_zero_copy_views_of_one_mapping(tmp_path):
    """A cross-run hit maps the file and slices it -- it does not read it into memory."""
    url, src = _store(tmp_path, (2, 8, 8), (1, 4, 4), name="p3.zarr")
    geoms = open_geometries(obstore_store(url))
    geom = geoms["v"]
    cache = tmp_path / "c3"
    _fill(url, geoms, backing_dir=cache, persist=True)

    revived = ChunkPool(geoms, backing_dir=cache, persist=True)
    # A cross-run hit is revived by `pin_if_ready` -- the driver's "can I skip the fetch?"
    # question -- not by `try_admit`, which is the miss path that allocates for a fetch.
    assert revived.pin_if_ready("v", 0, revived.new_owner())
    slot = revived._slots[("v", 0)]
    assert slot.state.name == "READY", "a persisted chunk must revive ready, not refetch"
    assert len(slot.tiles) == geom.n_inner_chunks(0), "revive must come back TILED, not assembled"
    for tile in slot.tiles.values():
        assert tile.base is slot.backing, "revived tiles must be views of the one mapping"


def test_persist_round_trips_the_data(tmp_path):
    """The real gate: a cross-run cache hit gathers byte-identical to a cold read."""
    url, src = _store(tmp_path, (4, 721, 180), (1, 180, 90), name="p4.zarr")
    geoms = open_geometries(obstore_store(url))
    geom = geoms["v"]
    assert geom.n_inner_chunks(0) == 5 * 2, "ragged, so edge padding is exercised"
    cache = tmp_path / "c4"

    def read(pool):
        out = []
        for cid in range(geom.n_chunks):
            rows = np.array([[cid, 0]], dtype=np.int64)
            out.append(pool.gather(rows, ["v"], geom.sample_chunk_size).arrays["v"])
        return np.concatenate(out)

    cold = _fill(url, geoms, backing_dir=cache, persist=True)
    np.testing.assert_array_equal(read(cold), src)
    cold.close()  # run 1 ends here: one writer at a time holds the cache dir's lock

    warm = ChunkPool(geoms, backing_dir=cache, persist=True)
    owner = warm.new_owner()
    for cid in range(geom.n_chunks):
        assert warm.pin_if_ready("v", cid, owner), "every chunk should revive from disk"
    np.testing.assert_array_equal(read(warm), src)
    assert warm.misses == 0, "a fully persisted cache should serve every chunk"


def test_a_version_2_cache_is_rejected_not_misread(tmp_path):
    """The layout changed, so an old cache must be reset -- never reinterpreted.

    A v2 ``.npy`` holds one assembled ``slot_shape`` array; read as tile-major it would be
    plausible-looking garbage. The manifest version is what stops that.
    """
    import json

    url, _ = _store(tmp_path, (2, 8, 8), (1, 4, 4), name="p5.zarr")
    geoms = open_geometries(obstore_store(url))
    cache = tmp_path / "c5"
    _fill(url, geoms, backing_dir=cache, persist=True).close()  # run 1 ends; lock released

    log = next(cache.glob("*.log"), None) or next(cache.glob("*.jsonl"), None)
    assert log is not None, f"no cache log found in {sorted(cache.iterdir())}"
    lines = log.read_text().splitlines()
    header = json.loads(lines[0])
    header["format_version"] = 2  # pretend the cache predates the tile-major layout
    log.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")

    # Loud by design: a stale cache is a user decision, not something to silently discard.
    with pytest.raises(ValueError, match="stale .log format changed"):
        ChunkPool(geoms, backing_dir=cache, persist=True)

    # ...and with consent it resets and refetches rather than reinterpreting v2 bytes.
    reset = ChunkPool(geoms, backing_dir=cache, persist=True, reset_stale_cache=True)
    assert reset.manifest_entries == 0
    assert not reset.pin_if_ready("v", 0, reset.new_owner()), "a reset cache must not revive"


def test_a_transform_never_sees_stored_chunk_padding(tmp_path):
    """User code gets the logical chunk: no edge padding, no fill values, no masking.

    721 rows chunked at 180 occupy 900 rows of storage. The pool holds those tiles whole,
    so the padding is real -- but assembly clips every tile before the transform runs, and
    `DecodedChunk.data` documents that guarantee.
    """
    seen: list[tuple] = []

    class Spy:
        def __call__(self, chunk):
            seen.append((chunk.data.shape, float(np.nanmin(chunk.data)), chunk.data.base))
            return chunk

    url, _ = _store(tmp_path, (2, 721, 360), (1, 180, 360), name="pad.zarr")
    geoms = open_geometries(obstore_store(url))
    geom = geoms["v"]
    assert geom.inner_grid()[0].stop * 180 > geom.inner_shape[0], "grid must overhang the array"

    pool = ChunkPool(geoms, chunk_transforms=[Spy()])
    with Scheduler(obstore_store(url), geoms, pool, SchedulerConfig()) as sched:
        sched.start(range(geom.n_chunks), geom.sample_chunk_size)
        for cid in range(geom.n_chunks):
            sched.pool.wait_ready("v", cid, sched.owner)

    assert seen, "the transform never ran"
    for shape, _mn, base in seen:
        assert shape == geom.slot_shape(0), f"transform saw {shape}, not the logical chunk"
        assert base is None, "the transform must get a real array, not a view over tiles"


def test_an_assembling_slot_is_charged_its_fill_time_residency(tmp_path):
    """A transform pool holds SOURCE tiles for its whole fill, then collapses to one.

    On a ragged grid the tiles are the larger shape, so charging only the assembled output
    under-reports the entire fill window -- 1.248x on this geometry.
    """
    url, _ = _store(tmp_path, (2, 721, 1440), (1, 180, 360), name="charge.zarr")
    geoms = open_geometries(obstore_store(url))
    geom = geoms["v"]

    tiles = geom.n_inner_chunks(0) * int(np.prod(geom.tile_shape())) * geom.dtype.itemsize
    assembled = int(np.prod(geom.slot_shape(0))) * geom.dtype.itemsize
    assert tiles > assembled

    class Scale:
        def __call__(self, chunk):
            return chunk

    for transforms in ([], [Scale()]):
        pool = ChunkPool(geoms, chunk_transforms=transforms)
        pool.try_admit("v", 0, pool.new_owner())
        assert pool.resident_bytes == tiles, (
            f"charged {pool.resident_bytes} but holds {tiles} bytes of tiles during fill"
        )
