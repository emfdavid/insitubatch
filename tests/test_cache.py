"""Caches: byte-LRU (memory + disk) and cross-epoch reuse of prepped chunks."""

from __future__ import annotations

import collections

import numpy as np
import pytest

from insitubatch import (
    DiskCache,
    MemoryCache,
    SplitName,
    open_geometries,
    split_by_chunk,
)
from insitubatch.source import InSituDataset
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


# -- cross-epoch reuse through InSituDataset (memory + disk) ----------------


@pytest.mark.parametrize("kind", ["memory", "disk"])
def test_cache_reuses_prepped_chunks_across_epochs(write_zarr, tmp_path, kind) -> None:
    url, _ = write_zarr(n=160, spc=8)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))
    cache = MemoryCache(10_000_000) if kind == "memory" else DiskCache(tmp_path / "c", 10_000_000)

    transformed: collections.Counter[int] = collections.Counter()

    def count(chunk: DecodedChunk) -> DecodedChunk:
        transformed[chunk.read.chunk_index] += 1
        return chunk

    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        batch_size=4,
        block_chunks=4,
        cache=cache,
        to_tensor=False,
        chunk_transforms=[count],
    )
    for epoch in range(2):
        ds.set_epoch(epoch)
        for _ in ds:
            pass

    assert set(transformed) == set(manifest.chunks[SplitName.TRAIN.value])
    assert all(v == 1 for v in transformed.values()), dict(transformed)  # decoded once
    assert cache.hits > 0


def test_no_cache_by_default(write_zarr) -> None:
    url, _ = write_zarr(n=80, spc=8)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))
    ds = InSituDataset(url, manifest, split=SplitName.TRAIN, to_tensor=False)
    assert ds.cache is None
