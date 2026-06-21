"""Caches: byte-bounded LRU of prepped chunks (memory + disk).

These are standalone unit tests of the cache classes. In B1 the V2 scheduler ships
cache-off (read-once); dataset-level cross-epoch reuse returns in B2, where the
ChunkPool subsumes these backings behind the same ``ChunkCache`` protocol -- so the
classes (and this coverage) stay live, but the InSituDataset ``cache=`` intercept
is gone for now.
"""

from __future__ import annotations

import numpy as np

from insitubatch import DiskCache, MemoryCache
from insitubatch.types import ChunkRead, DecodedChunk


def _chunk(cid: int) -> DecodedChunk:
    return DecodedChunk(
        read=ChunkRead("t2m", cid),
        data=np.full((2, 2), cid, dtype="f4"),  # 16 bytes
        sample_offset=cid * 2,
    )


# -- MemoryCache unit -------------------------------------------------------


def test_memory_cache_byte_lru() -> None:
    c = MemoryCache(max_bytes=32)  # holds two 16-byte chunks
    c.put("t2m", 0, _chunk(0))
    c.put("t2m", 1, _chunk(1))
    assert len(c) == 2
    assert c.get("t2m", 0) is not None  # touch 0 -> 1 is LRU
    c.put("t2m", 2, _chunk(2))  # over budget -> evict 1
    assert c.get("t2m", 1) is None
    assert c.get("t2m", 0) is not None
    assert c.get("t2m", 2) is not None
    assert len(c) == 2


def test_memory_cache_hit_miss_stats() -> None:
    c = MemoryCache(max_bytes=64)
    assert c.get("t2m", 0) is None  # miss
    c.put("t2m", 0, _chunk(0))
    assert c.get("t2m", 0) is not None  # hit
    assert (c.hits, c.misses) == (1, 1)


# -- DiskCache unit ---------------------------------------------------------


def test_disk_cache_roundtrip_and_offset(tmp_path) -> None:
    c = DiskCache(tmp_path / "cache", max_bytes=1_000)
    src = np.arange(4, dtype="f4").reshape(2, 2)
    c.put("t2m", 7, DecodedChunk(ChunkRead("t2m", 7), src, sample_offset=70))
    got = c.get("t2m", 7)
    assert got is not None
    np.testing.assert_array_equal(np.asarray(got.data), src)  # mmap'd value matches
    assert got.sample_offset == 70  # metadata preserved


def test_disk_cache_byte_lru_unlinks(tmp_path) -> None:
    d = tmp_path / "cache"
    c = DiskCache(d, max_bytes=32)  # holds two 16-byte chunks
    c.put("t2m", 0, _chunk(0))
    c.put("t2m", 1, _chunk(1))
    assert c.get("t2m", 0) is not None  # touch 0 -> 1 is LRU
    c.put("t2m", 2, _chunk(2))  # evict 1
    assert c.get("t2m", 1) is None
    assert len(c) == 2
    assert len(list(d.glob("*.npy"))) == 2  # evicted file actually unlinked
