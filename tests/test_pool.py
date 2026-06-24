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
from concurrent.futures import ThreadPoolExecutor

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
    # Budget full of *pinned* slots -> a third miss can't be admitted yet.
    assert not pool.try_admit("single_inner", 2)
    pool.unpin({0, 1})  # release -> now LRU-evictable (0 is LRU)
    assert pool.try_admit("single_inner", 2)  # evicts chunk 0
    for inner in geom.inner_coords():
        pool.scatter("single_inner", 2, inner, tiles[(2, *inner)])
    pool.wait_ready("single_inner", 2)

    assert not pool.is_ready("single_inner", 0)  # LRU victim, gone
    assert pool.is_ready("single_inner", 1)  # retained
    rows = np.array([[1, w] for w in range(spc)], dtype=np.int64)
    kept = pool.gather(rows, ["single_inner"], spc).arrays["single_inner"]
    np.testing.assert_array_equal(kept, srcs["single_inner"][spc : 2 * spc])
