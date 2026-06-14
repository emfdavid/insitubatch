"""Chunk cache: LRU unit behaviour + cross-epoch reuse of prepped chunks."""

from __future__ import annotations

import collections

import numpy as np
import zarr

from insitubatch import (
    SplitName,
    ensure_local_dir,
    open_geometries,
    split_by_chunk,
    store_from_url,
)
from insitubatch.cache import ChunkCache
from insitubatch.source import InSituDataset
from insitubatch.types import ChunkRead, DecodedChunk


def _chunk(cid: int) -> DecodedChunk:
    return DecodedChunk(
        read=ChunkRead("t2m", cid),
        data=np.zeros((2, 2), dtype="f4"),
        sample_offset=cid * 2,
    )


def _write(tmp_path, *, n=160, spc=8, inner=(2, 2)) -> str:
    url = f"file://{tmp_path}/d.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=store_from_url(url, read_only=False), mode="w")
    arr = group.create_array("t2m", shape=(n, *inner), chunks=(spc, *inner), dtype="f4")
    arr[:] = np.arange(n * int(np.prod(inner)), dtype="f4").reshape(n, *inner)
    return url


# -- unit: LRU + stats ------------------------------------------------------


def test_chunk_cache_lru_eviction() -> None:
    c = ChunkCache(capacity_chunks=2)
    c.put("t2m", 0, _chunk(0))
    c.put("t2m", 1, _chunk(1))
    assert c.get("t2m", 0) is not None  # touch 0 -> 0 becomes MRU, 1 is LRU
    c.put("t2m", 2, _chunk(2))  # over capacity -> evict LRU (1)
    assert c.get("t2m", 1) is None
    assert c.get("t2m", 0) is not None
    assert c.get("t2m", 2) is not None
    assert len(c) == 2


def test_chunk_cache_hit_miss_stats() -> None:
    c = ChunkCache(capacity_chunks=2)
    assert c.get("t2m", 0) is None  # miss
    c.put("t2m", 0, _chunk(0))
    assert c.get("t2m", 0) is not None  # hit
    assert (c.hits, c.misses) == (1, 1)


# -- integration: cross-epoch reuse ----------------------------------------


def test_cache_reuses_prepped_chunks_across_epochs(tmp_path) -> None:
    url = _write(tmp_path)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))

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
        cache_chunks=100,  # > train chunk count -> no eviction between epochs
        to_tensor=False,
        chunk_transforms=[count],
    )

    for epoch in range(2):
        ds.set_epoch(epoch)
        for _ in ds:
            pass

    train_chunks = set(manifest.chunks[SplitName.TRAIN.value])
    assert set(transformed) == train_chunks
    # Each chunk decoded + transformed exactly once across BOTH epochs (cached).
    assert all(v == 1 for v in transformed.values()), dict(transformed)
    assert ds.cache is not None and ds.cache.hits > 0


def test_no_cache_by_default(tmp_path) -> None:
    url = _write(tmp_path)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))
    ds = InSituDataset(url, manifest, split=SplitName.TRAIN, to_tensor=False)
    assert ds.cache is None
