"""Stored-chunk plan + ChunkPool scatter/gather parity.

The risky mechanic -- fetch a stored tile, decode it, scatter it into its outer
chunk's slot -- must reconstruct *exactly* what ``arr[:]`` returns, for
single-inner and spatially-chunked arrays including partial edge chunks (this is
what ``bench/spike_v2_decode.py`` proved at the zarr level; here it is a unit
test driving the real :class:`ChunkPool`). We also drive the scatter from many
threads at once -- two tiles of the same outer chunk concurrently -- to exercise
the free-threading invariant (disjoint lock-free copy + lock-published readiness).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
import zarr
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype

from insitubatch import ensure_local_dir, open_geometries, store_from_url
from insitubatch.plan import build_stored_chunk_reads
from insitubatch.pool import ChunkPool
from insitubatch.types import ArrayGeometry, StoredChunkRead

# shape with partial edge chunks on every axis; two layouts: single-inner + spatial
SHAPE = (5, 9, 7)
LAYOUTS = {"single_inner": (2, 9, 7), "spatial": (2, 4, 4)}


@pytest.fixture
def tiled_store(tmp_path):
    """Write the spike's two-layout store; return (url, {var: source array})."""
    url = f"file://{tmp_path}/tiled.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=store_from_url(url, read_only=False), mode="w")
    data = np.random.default_rng(0).standard_normal(SHAPE).astype("f4")
    srcs = {}
    for var, chunks in LAYOUTS.items():
        arr = group.create_array(var, shape=SHAPE, chunks=chunks, dtype="f4")
        arr[:] = data
        srcs[var] = np.asarray(arr[:])
    return url, srcs


async def _decode_tiles(url: str, var: str) -> dict[tuple[int, ...], np.ndarray]:
    """Fetch + decode every stored tile of ``var`` (the spike's zarr-internals path)."""
    aa = zarr.open_array(store=store_from_url(url), path=var, mode="r")._async_array
    proto = default_buffer_prototype()
    spec = ArraySpec(
        shape=aa.metadata.chunks,
        dtype=aa.metadata.data_type,
        fill_value=aa.metadata.fill_value,
        config=aa.config,
        prototype=proto,
    )
    grid = [range(-(-s // c)) for s, c in zip(aa.metadata.shape, aa.metadata.chunks, strict=True)]
    tiles: dict[tuple[int, ...], np.ndarray] = {}
    for coords in itertools.product(*grid):
        key = aa.store_path.path + "/" + aa.metadata.chunk_key_encoding.encode_chunk_key(coords)
        buf = await aa.store_path.store.get(key, prototype=proto)
        [tile] = list(await aa.codec_pipeline.decode([(buf, spec)]))
        tiles[coords] = tile.as_numpy_array()
    return tiles


def test_build_stored_chunk_reads_expands_and_dedups(tiled_store):
    url, _ = tiled_store
    geoms = open_geometries(url)
    reads = build_stored_chunk_reads([0, 1, 0], geoms)  # repeat 0 -> must dedup

    # spatial expands to its inner grid (3x2=6), single_inner to 1; x2 outer chunks.
    per_outer = sum(g.n_inner_chunks(0) for g in geoms.values())
    assert len(reads) == per_outer * 2  # chunks 0 and 1, the repeat collapsed
    assert len(set(reads)) == len(reads)  # no duplicate stored-chunk reads
    # priority order is chunk-major: every chunk-0 read precedes every chunk-1 read.
    first_c1 = next(i for i, r in enumerate(reads) if r.chunk_index == 1)
    assert all(r.chunk_index == 0 for r in reads[:first_c1])


def test_pool_aliased_labels_decode_once(tiled_store):
    """Two variable *labels* backed by one underlying array (``geom.name``) share a single
    decoded slot per chunk -- decode-once across offsets (the M-W pool path-keying).

    The pool must key slots on the array name, not the dict label, and ``gather`` must map
    each label back to its array. Today the label *is* the name, so the engine can't
    express two views (e.g. ``t2m_now`` / ``t2m_next``) of one array sharing a decode.
    """
    url, srcs = tiled_store
    array = "single_inner"
    base = open_geometries(url, variables=[array])[array]  # base.name == array
    geoms = {"now": base, "next": base}  # two labels, one underlying array
    tiles = asyncio.run(_decode_tiles(url, array))
    reads = build_stored_chunk_reads(range(base.n_chunks), {array: base})

    pool = ChunkPool(geoms)
    for cid in range(base.n_chunks):
        pool.try_admit(array, cid)  # admit by array name
    for read in reads:
        pool.scatter(read.array, read.chunk_index, read.inner_coord, tiles[read.coords])

    spc = base.sample_chunk_size
    for cid in range(base.n_chunks):
        pool.wait_ready(array, cid)
    n0 = len(base.samples_in_chunk(0))
    rows = np.array([[0, w] for w in range(n0)], dtype=np.int64)
    batch = pool.gather(rows, ["now", "next"], spc)

    assert len(pool._slots) == base.n_chunks  # one slot per chunk, not one per (label, chunk)
    np.testing.assert_array_equal(batch.arrays["now"], batch.arrays["next"])
    np.testing.assert_array_equal(batch.arrays["now"], srcs[array][:n0])


@pytest.mark.parametrize("var", list(LAYOUTS))
def test_pool_scatter_reconstructs_array(tiled_store, var):
    """Concurrent tile scatter assembles each outer chunk == arr[:] (FT stress)."""
    url, srcs = tiled_store
    geoms = open_geometries(url, variables=[var])
    geom = geoms[var]
    tiles = asyncio.run(_decode_tiles(url, var))
    reads = build_stored_chunk_reads(range(geom.n_chunks), geoms)

    pool = ChunkPool(geoms)
    for cid in range(geom.n_chunks):
        pool.try_admit(var, cid)

    # Scatter every tile from a thread pool: two tiles of one slot land at once,
    # so this exercises the disjoint-lock-free-copy / lock-published-ready rule.
    def scatter(read: StoredChunkRead) -> None:
        pool.scatter(read.array, read.chunk_index, read.inner_coord, tiles[read.coords])

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(scatter, reads))

    spc = geom.sample_chunk_size
    for cid in range(geom.n_chunks):
        pool.wait_ready(var, cid)
        n0 = len(geom.samples_in_chunk(cid))
        rows = np.array([[cid, w] for w in range(n0)], dtype=np.int64)
        batch = pool.gather(rows, [var], spc)
        expected = srcs[var][cid * spc : cid * spc + n0]
        assert np.array_equal(batch.arrays[var], expected)
        assert np.array_equal(batch.sample_indices, np.arange(cid * spc, cid * spc + n0))


def test_pool_concurrent_scatter_is_race_free() -> None:
    """Stress the lock-free disjoint writes: 64 tiles, 32 threads, repeated rounds.

    Passes (serialized) under the GIL build; under ``PYTHON_GIL=0`` on free-threaded
    3.13t it is a genuine data-race probe -- the FT CI job runs it GIL-free. A fine
    8x8 inner grid means many threads write disjoint regions of one slot at once,
    which is exactly the scatter concurrency the design relies on. No zarr/network:
    tiles are exact-multiple slices of a reference array (edge clipping is covered
    by the parity test), so this isolates the pool's threading.
    """
    geom = ArrayGeometry("v", (4, 64, 64), (4, 8, 8), np.dtype("f4"))  # 8x8 = 64 tiles
    geoms = {"v": geom}
    ref = np.random.default_rng(0).standard_normal((4, 64, 64)).astype("f4")
    coords = list(geom.inner_coords())

    for _round in range(8):  # repeated admit -> concurrent scatter -> gather (fresh pool)
        pool = ChunkPool(geoms)
        pool.try_admit("v", 0)

        def scatter(ic, pool=pool, geom=geom, ref=ref) -> None:  # default-bind per round
            dst, _src = geom.tile_placement(0, ic)
            pool.scatter("v", 0, ic, ref[dst].copy())  # full chunk-shaped tile

        with ThreadPoolExecutor(max_workers=32) as ex:
            list(ex.map(scatter, coords))

        pool.wait_ready("v", 0)
        rows = np.array([[0, w] for w in range(4)], dtype=np.int64)
        got = pool.gather(rows, ["v"], geom.sample_chunk_size).arrays["v"]
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("var", list(LAYOUTS))
def test_pool_mmap_backing_parity_and_cleanup(tiled_store, tmp_path, var):
    """mmap-backed slots reconstruct == arr[:], live as files on disk, and the
    files are freed on close()."""
    url, srcs = tiled_store
    geoms = open_geometries(url, variables=[var])
    geom = geoms[var]
    tiles = asyncio.run(_decode_tiles(url, var))
    backing = tmp_path / "slots"

    pool = ChunkPool(geoms, backing_dir=backing)
    for cid in range(geom.n_chunks):
        pool.try_admit(var, cid)
        for inner in geom.inner_coords():
            pool.scatter(var, cid, inner, tiles[(cid, *inner)])

    assert list(backing.glob("*.npy")), "slots should live as .npy files on disk"
    spc = geom.sample_chunk_size
    for cid in range(geom.n_chunks):
        pool.wait_ready(var, cid)
        n0 = len(geom.samples_in_chunk(cid))
        rows = np.array([[cid, w] for w in range(n0)], dtype=np.int64)
        got = pool.gather(rows, [var], spc).arrays[var]
        assert np.array_equal(got, srcs[var][cid * spc : cid * spc + n0])

    pool.close()
    assert not list(backing.glob("*.npy")), "close() must unlink the slot files"


def test_pool_mmap_backing_with_transform(tiled_store, tmp_path):
    """A shape-preserving chunk_transform's output is written back into the memmap."""
    url, srcs = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))

    def scale(chunk):  # new array, same shape
        chunk.data = chunk.data * 2.0
        return chunk

    pool = ChunkPool(geoms, backing_dir=tmp_path / "s", chunk_transforms=[scale])
    pool.try_admit("single_inner", 0)
    for inner in geom.inner_coords():
        pool.scatter("single_inner", 0, inner, tiles[(0, *inner)])
    pool.wait_ready("single_inner", 0)

    spc = geom.sample_chunk_size
    rows = np.array([[0, w] for w in range(spc)], dtype=np.int64)
    got = pool.gather(rows, ["single_inner"], spc).arrays["single_inner"]
    np.testing.assert_allclose(got, srcs["single_inner"][:spc] * 2.0)


class MeanLastAxis:
    """A *reshaping* chunk_transform: mean over the last inner axis (and an optional
    dtype recast). Declares its output inner geometry via ``output_inner`` so the engine
    can size the cache slot and gather buffer at the post-transform shape."""

    def __init__(self, out_dtype: str | np.dtype = "f4") -> None:
        self.out_dtype = np.dtype(out_dtype)

    def __call__(self, chunk):
        chunk.data = chunk.data.mean(axis=-1).astype(self.out_dtype)  # (n,9,7) -> (n,9)
        return chunk

    def output_inner(self, geom: ArrayGeometry) -> tuple[tuple[int, ...], np.dtype]:
        return geom.inner_shape[:-1], self.out_dtype


def _gather_chunk(pool, var, cid, geom, spc):
    n0 = len(geom.samples_in_chunk(cid))
    rows = np.array([[cid, w] for w in range(n0)], dtype=np.int64)
    return pool.gather(rows, [var], spc).arrays[var]


def test_pool_reshaping_transform_heap_gather(tiled_store):
    """A reshaping chunk_transform's output shape flows through gather on heap backing.

    Today gather allocates at the *source* inner_shape and assigns from the post-transform
    slot (output shape) -> broadcast error. The output geometry must drive gather."""
    url, srcs = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    spc = geom.sample_chunk_size

    pool = ChunkPool(geoms, chunk_transforms=[MeanLastAxis()])
    for cid in range(geom.n_chunks):
        _fill_chunk(pool, "single_inner", cid, geom, tiles)
        got = _gather_chunk(pool, "single_inner", cid, geom, spc)
        n0 = len(geom.samples_in_chunk(cid))
        expected = srcs["single_inner"][cid * spc : cid * spc + n0].mean(axis=-1)
        assert got.shape == (n0, geom.inner_shape[0])  # (n, 9), reshaped from (n, 9, 7)
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_pool_reshaping_transform_dtype_recast(tiled_store):
    """A dtype-changing transform (f4 -> f8) propagates its dtype through gather."""
    url, srcs = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    spc = geom.sample_chunk_size

    pool = ChunkPool(geoms, chunk_transforms=[MeanLastAxis(out_dtype="f8")])
    _fill_chunk(pool, "single_inner", 0, geom, tiles)
    got = _gather_chunk(pool, "single_inner", 0, geom, spc)
    assert got.dtype == np.dtype("f8")
    np.testing.assert_allclose(got, srcs["single_inner"][:spc].mean(axis=-1), rtol=1e-12)


def test_pool_reshaping_transform_mmap_persist_roundtrip(tiled_store, tmp_path):
    """The fenced path: a reshaping transform on mmap backing with persist must write the
    output-shaped result into the (output-sized) slot, survive close(), and revive as a
    ready hit on reopen -- re-decoding zero chunks -- reconstructing the regridded data."""
    url, srcs = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    spc = geom.sample_chunk_size
    backing = tmp_path / "cache"

    pool = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[MeanLastAxis()])
    for cid in range(geom.n_chunks):
        _fill_chunk(pool, "single_inner", cid, geom, tiles)
    assert pool.misses == geom.n_chunks and pool.hits == 0
    pool.close()
    assert list(backing.glob("*.npy")) and (backing / "insitu_cache.json").exists()

    pool2 = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[MeanLastAxis()])
    assert pool2.manifest_entries == geom.n_chunks
    for cid in range(geom.n_chunks):
        assert pool2.pin_if_ready("single_inner", cid), "regridded chunk must revive as a hit"
        n0 = len(geom.samples_in_chunk(cid))
        got = _gather_chunk(pool2, "single_inner", cid, geom, spc)
        expected = srcs["single_inner"][cid * spc : cid * spc + n0].mean(axis=-1)
        assert got.shape == (n0, geom.inner_shape[0])
        np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert pool2.hits == geom.n_chunks and pool2.misses == 0
    pool2.close()


def test_pool_reshaping_transform_structural_mismatch_is_miss(tiled_store, tmp_path):
    """A persisted output-shaped entry whose *output* geometry no longer matches (the
    transform now yields a different inner shape) is invalidated structurally -- a miss."""
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    backing = tmp_path / "cache"

    class MeanBothAxes(MeanLastAxis):  # output inner () vs (9,) -> structural drift
        def __call__(self, chunk):
            chunk.data = chunk.data.mean(axis=(-2, -1)).astype(self.out_dtype)
            return chunk

        def output_inner(self, geom):
            return (), self.out_dtype

    # Share a cache_key so the transform *fingerprint* matches across the two runs; only
    # the declared output inner shape differs ((9,) -> ()). That isolates the structural
    # backstop: the persisted .npy header no longer matches the current output geometry.
    t_old = MeanLastAxis()
    t_old.cache_key = "k"  # type: ignore[attr-defined]
    pool = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[t_old])
    _fill_chunk(pool, "single_inner", 0, geom, tiles)
    pool.close()

    t_new = MeanBothAxes()
    t_new.cache_key = "k"  # type: ignore[attr-defined]
    pool2 = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[t_new])
    assert pool2.manifest_entries == 1, "same fingerprint -> entry is consulted"
    assert not pool2.pin_if_ready("single_inner", 0), "output-geometry drift must invalidate"
    assert pool2.revive_mismatch == 1 and pool2.hits == 0
    pool2.close()


def test_pool_wait_ready_raises_on_failure(tiled_store):
    """A poisoned tile surfaces on the waiting consumer (fail-fast), not a hang."""
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["spatial"])
    pool = ChunkPool(geoms)
    pool.try_admit("spatial", 0)
    pool.fail("spatial", 0, RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        pool.wait_ready("spatial", 0)


def test_pool_lru_evicts_unpinned_under_budget(tiled_store):
    """Admission evicts unpinned-LRU to make room; pinned chunks are never dropped,
    and a still-resident chunk is retained (the cross-epoch-reuse mechanic)."""
    url, srcs = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    spc = geom.sample_chunk_size
    full_nbytes = spc * int(np.prod(geom.inner_shape)) * geom.dtype.itemsize
    pool = ChunkPool(geoms, budget_bytes=2 * full_nbytes)  # holds two full chunks

    def fill(cid):
        assert pool.try_admit("single_inner", cid)
        for inner in geom.inner_coords():
            pool.scatter("single_inner", cid, inner, tiles[(cid, *inner)])
        pool.wait_ready("single_inner", cid)

    fill(0)
    fill(1)
    assert pool.resident_chunks == 2
    # Budget full of *referenced* slots -> a third miss can't be admitted yet.
    assert not pool.try_admit("single_inner", 2)
    pool.unpin_keys({("single_inner", 0), ("single_inner", 1)})  # release -> LRU-evictable
    assert pool.try_admit("single_inner", 2)  # evicts chunk 0
    for inner in geom.inner_coords():
        pool.scatter("single_inner", 2, inner, tiles[(2, *inner)])
    pool.wait_ready("single_inner", 2)

    assert not pool.is_ready("single_inner", 0)  # LRU victim, gone
    assert pool.is_ready("single_inner", 1)  # retained
    rows = np.array([[1, w] for w in range(spc)], dtype=np.int64)
    kept = pool.gather(rows, ["single_inner"], spc).arrays["single_inner"]
    np.testing.assert_array_equal(kept, srcs["single_inner"][spc : 2 * spc])


def test_pool_pin_is_reference_counted_not_boolean(tiled_store):
    """A chunk read by several windowed blocks is pinned several times and must stay
    resident until the *last* release -- a single unpin after a double-pin must not free
    it (the boolean-set bug). Pins are counts: N pins need N unpins."""
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    full = geom.sample_chunk_size * int(np.prod(geom.inner_shape)) * geom.dtype.itemsize
    pool = ChunkPool(geoms, budget_bytes=full)  # holds exactly one chunk

    pool.try_admit("single_inner", 0)  # admission references it -> refcount 1
    for inner in geom.inner_coords():
        pool.scatter("single_inner", 0, inner, tiles[(0, *inner)])
    pool.wait_ready("single_inner", 0)

    pool.pin_keys({("single_inner", 0)})  # a second block references it -> 2
    assert pool._pinned[("single_inner", 0)] == 2
    pool.unpin_keys({("single_inner", 0)})  # one block done -> 1, still pinned
    assert not pool.try_admit("single_inner", 1)  # budget full, chunk 0 not evictable
    pool.unpin_keys({("single_inner", 0)})  # last block done -> 0, now evictable
    assert ("single_inner", 0) not in pool._pinned
    assert pool.try_admit("single_inner", 1)  # evicts chunk 0
    assert ("single_inner", 0) not in pool._slots


def test_pool_unpin_all_drops_abandoned_partial_keeps_ready(tiled_store):
    """Epoch-boundary reset (unpin_all): clear every refcount, drop a not-ready
    (abandoned) partial -- a half-scattered chunk a cancelled epoch left behind can
    never be a valid cache entry -- but retain a fully ready chunk for cross-epoch
    reuse, with the byte accounting reclaimed."""
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["spatial"])
    geom = geoms["spatial"]
    tiles = asyncio.run(_decode_tiles(url, "spatial"))
    inner = list(geom.inner_coords())
    assert len(inner) > 1  # need a multi-tile chunk to leave a real partial

    pool = ChunkPool(geoms)
    # chunk 0: only the first tile scattered -> not ready (abandoned partial)
    pool.try_admit("spatial", 0)
    pool.scatter("spatial", 0, inner[0], tiles[(0, *inner[0])])
    assert not pool.is_ready("spatial", 0)
    # chunk 1: fully scattered -> ready (a valid cache entry)
    pool.try_admit("spatial", 1)
    for ic in inner:
        pool.scatter("spatial", 1, ic, tiles[(1, *ic)])
    pool.wait_ready("spatial", 1)
    bytes_ready = pool._slots[("spatial", 1)].nbytes

    pool.unpin_all()

    assert pool._pinned == {}  # every refcount cleared
    assert ("spatial", 0) not in pool._slots  # abandoned partial dropped
    assert pool.is_ready("spatial", 1)  # ready chunk retained (cross-epoch reuse)
    assert pool._bytes == bytes_ready  # the partial's bytes were reclaimed


def test_pool_persist_requires_cache_dir(tiled_store):
    """persist=True only makes sense with a backing dir to keep files in -- fail fast."""
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    with pytest.raises(ValueError, match="cache_dir|backing_dir"):
        ChunkPool(geoms, persist=True)


def _fill_chunk(pool, var, cid, geom, tiles):
    pool.try_admit(var, cid)
    for inner in geom.inner_coords():
        pool.scatter(var, cid, inner, tiles[(cid, *inner)])
    pool.wait_ready(var, cid)


def test_pool_cross_run_persistence(tiled_store, tmp_path):
    """A persistent cache survives process exit: a new pool over the same dir serves
    each chunk as a ready hit (no re-scatter), reconstructing the source exactly."""
    url, srcs = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    spc = geom.sample_chunk_size
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    backing = tmp_path / "cache"

    # Run 1: populate + close. persist must KEEP the files and write a manifest.
    pool = ChunkPool(geoms, backing_dir=backing, persist=True)
    for cid in range(geom.n_chunks):
        _fill_chunk(pool, "single_inner", cid, geom, tiles)
    assert pool.misses == geom.n_chunks and pool.hits == 0  # cold: every chunk a miss
    pool.close()
    assert list(backing.glob("*.npy")), "persist=True must keep slot files after close()"
    assert (backing / "insitu_cache.json").exists(), "persist must write a manifest"

    # Run 2: same dir -> every chunk a ready hit, no fetch/scatter.
    pool2 = ChunkPool(geoms, backing_dir=backing, persist=True)
    assert pool2.manifest_entries == geom.n_chunks
    for cid in range(geom.n_chunks):
        assert pool2.pin_if_ready("single_inner", cid), "persisted chunk must be a hit"
        n0 = len(geom.samples_in_chunk(cid))
        rows = np.array([[cid, w] for w in range(n0)], dtype=np.int64)
        got = pool2.gather(rows, ["single_inner"], spc).arrays["single_inner"]
        assert np.array_equal(got, srcs["single_inner"][cid * spc : cid * spc + n0])
    assert pool2.hits == geom.n_chunks and pool2.misses == 0  # warm: every chunk a hit
    pool2.close()


def test_pool_persist_invalidates_on_geometry_mismatch(tiled_store, tmp_path):
    """A persisted entry whose stored shape/dtype no longer matches the current
    geometry is ignored (a miss), not served -- the structural fingerprint check.
    The dataset/pipeline identity is the cache_dir path; the user versions that."""
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    backing = tmp_path / "cache"

    pool = ChunkPool(geoms, backing_dir=backing, persist=True)
    _fill_chunk(pool, "single_inner", 0, geom, tiles)
    pool.close()

    # Reopen with a structurally different dtype for the same array -> mismatch -> miss.
    drifted = {"single_inner": replace(geom, dtype=np.dtype("f8"))}
    pool2 = ChunkPool(drifted, backing_dir=backing, persist=True)
    assert not pool2.pin_if_ready("single_inner", 0), "geometry drift must invalidate"
    assert pool2.revive_mismatch == 1 and pool2.hits == 0  # the mismatch is counted
    pool2.close()


def test_pool_persist_invalidates_on_transform_change(tiled_store, tmp_path):
    """The manifest carries a chunk_transform fingerprint: the same transform reopens as
    a hit; a changed transform discards the whole cache (cold start), not a false hit."""
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    backing = tmp_path / "cache"

    def scale_a(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    pool = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[scale_a])
    _fill_chunk(pool, "single_inner", 0, geom, tiles)
    pool.close()

    same = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[scale_a])
    assert same.manifest_entries == 1 and same.pin_if_ready("single_inner", 0)  # same fp -> hit
    same.close()

    def scale_b(chunk):  # different body -> different fingerprint
        chunk.data = chunk.data * 3.0
        return chunk

    changed = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[scale_b])
    assert changed.manifest_entries == 0  # stale cache discarded, not loaded
    assert not changed.pin_if_ready("single_inner", 0)
    changed.close()


def test_pool_persist_cache_key_overrides_fingerprint(tiled_store, tmp_path):
    """An explicit transform.cache_key is authoritative: distinct function objects sharing
    a key reopen as a hit; bumping the key invalidates."""
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    backing = tmp_path / "cache"

    def t1(chunk):
        return chunk

    t1.cache_key = "v1"  # type: ignore[attr-defined]
    pool = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[t1])
    _fill_chunk(pool, "single_inner", 0, geom, tiles)
    pool.close()

    def t2(chunk):  # a different object/body, same declared key -> treated as identical
        return chunk

    t2.cache_key = "v1"  # type: ignore[attr-defined]
    same = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[t2])
    assert same.pin_if_ready("single_inner", 0)
    same.close()

    t2.cache_key = "v2"  # type: ignore[attr-defined]  # bump the key -> invalidate
    changed = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[t2])
    assert changed.manifest_entries == 0
    changed.close()


def test_pool_persist_cloudpickle_catches_closure_change(tiled_store, tmp_path):
    """cloudpickle's stronger guarantee: two closures with identical source but a different
    closed-over constant get different fingerprints (a source hash would falsely match)."""
    pytest.importorskip("cloudpickle")
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    backing = tmp_path / "cache"

    def make_scaler(factor):  # identical source text; only the closure cell differs
        def scale(chunk):
            chunk.data = chunk.data * factor
            return chunk

        return scale

    pool = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[make_scaler(2.0)])
    _fill_chunk(pool, "single_inner", 0, geom, tiles)
    pool.close()

    changed = ChunkPool(
        geoms, backing_dir=backing, persist=True, chunk_transforms=[make_scaler(3.0)]
    )
    assert changed.manifest_entries == 0  # closure-value change caught -> cache discarded
    changed.close()


def test_pool_persist_without_cloudpickle_warns_and_falls_back(
    tiled_store, tmp_path, monkeypatch, caplog
):
    """No cloudpickle + no cache_key -> a best-effort source fingerprint, with a one-time
    warning. The source hash still reopens an unchanged transform as a hit."""
    monkeypatch.setattr("insitubatch.pool.cloudpickle", None)
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    backing = tmp_path / "cache"

    def scale(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    with caplog.at_level(logging.WARNING, logger="insitubatch.pool"):
        pool = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[scale])
    assert any("best-effort" in r.message for r in caplog.records), caplog.text
    _fill_chunk(pool, "single_inner", 0, geom, tiles)
    pool.close()

    same = ChunkPool(geoms, backing_dir=backing, persist=True, chunk_transforms=[scale])
    assert same.pin_if_ready("single_inner", 0)  # source hash matches -> hit
    same.close()


def test_pool_spill_unlinks_without_persist(tiled_store, tmp_path):
    """Without persist, a backing dir is scratch: files are unlinked on close and no
    manifest is written (the persistence machinery is persist-only)."""
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    geom = geoms["single_inner"]
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    backing = tmp_path / "spill"

    pool = ChunkPool(geoms, backing_dir=backing)
    _fill_chunk(pool, "single_inner", 0, geom, tiles)
    pool.close()
    assert not list(backing.glob("*.npy")), "spill must unlink slot files on close()"
    assert not (backing / "insitu_cache.json").exists(), "no manifest without persist"
