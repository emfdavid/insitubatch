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
from insitubatch.types import StoredChunkRead

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
        pool.allocate(var, cid)

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


def test_pool_wait_ready_raises_on_failure(tiled_store):
    """A poisoned tile surfaces on the waiting consumer (fail-fast), not a hang."""
    url, _ = tiled_store
    geoms = open_geometries(url, variables=["spatial"])
    pool = ChunkPool(geoms)
    pool.allocate("spatial", 0)
    pool.fail("spatial", 0, RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        pool.wait_ready("spatial", 0)


def test_pool_evict_frees_residency(tiled_store):
    url, srcs = tiled_store
    geoms = open_geometries(url, variables=["single_inner"])
    tiles = asyncio.run(_decode_tiles(url, "single_inner"))
    pool = ChunkPool(geoms)
    for cid in range(geoms["single_inner"].n_chunks):
        pool.allocate("single_inner", cid)
        for inner in geoms["single_inner"].inner_coords():
            pool.scatter("single_inner", cid, inner, tiles[(cid, *inner)])
    resident = pool.resident_chunks
    assert resident == geoms["single_inner"].n_chunks
    dropped = pool.evict({0, 1})
    assert dropped == 2
    assert pool.resident_chunks == resident - 2
