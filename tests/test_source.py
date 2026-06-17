"""InSituDataset: the equal-chunking invariant and shuffle on/off."""

from __future__ import annotations

import collections

import numpy as np
import pytest
import zarr

from insitubatch import (
    SplitName,
    ensure_local_dir,
    open_geometries,
    split_by_chunk,
    store_from_url,
)
from insitubatch.shuffle import block_shuffled_order
from insitubatch.source import InSituDataset, _partition_blocks


def test_partition_blocks_covers_disjoint_contiguous_blocks() -> None:
    # 10 chunks x 4 samples, block_chunks=3 -> blocks of 3,3,3,1 chunks.
    chunk_ids = np.arange(10, dtype=np.int64)
    order = block_shuffled_order(chunk_ids, 4, 40, block_chunks=3, seed=0, epoch=0)
    blocks = _partition_blocks(order, block_chunks=3)

    assert [len(b[2]) for b in blocks] == [3, 3, 3, 1]  # chunk counts per block
    # Row ranges tile the whole order contiguously, and chunk sets are disjoint.
    assert blocks[0][0] == 0 and blocks[-1][1] == len(order)
    seen: set[int] = set()
    prev_stop = 0
    for rstart, rstop, chunks in blocks:
        assert rstart == prev_stop
        prev_stop = rstop
        ids = {int(c) for c in chunks}
        assert seen.isdisjoint(ids)
        # every row in this block draws only from this block's chunks
        assert set(int(c) for c in order[rstart:rstop, 0]) == ids
        seen |= ids
    assert seen == set(range(10))


def test_partition_blocks_empty() -> None:
    assert _partition_blocks(np.empty((0, 2), dtype=np.int64), block_chunks=4) == []


def test_unequal_chunking_raises(tmp_path) -> None:
    url = f"file://{tmp_path}/uv.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=store_from_url(url, read_only=False), mode="w")
    group.create_array("u10", shape=(40, 2, 2), chunks=(8, 2, 2), dtype="f4")
    group.create_array("v10", shape=(40, 2, 2), chunks=(4, 2, 2), dtype="f4")  # different spc
    geoms = open_geometries(url)
    manifest = split_by_chunk(geoms["u10"], fractions=(0.8, 0.1, 0.1))

    with pytest.raises(ValueError, match="same sample-axis"):
        InSituDataset(url, manifest, geometries=geoms, to_tensor=False)


def test_shuffle_false_is_in_order(write_zarr) -> None:
    url, _ = write_zarr(n=40, spc=8)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        shuffle=False,
        to_tensor=False,
        batch_size=5,
        block_chunks=2,
    )
    ds.set_epoch(0)
    idx = np.concatenate([b.sample_indices for b in ds])
    assert idx.tolist() == list(range(40))  # strictly in order


def test_shuffle_true_covers_but_reorders(write_zarr) -> None:
    url, _ = write_zarr(n=40, spc=8)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        url,
        manifest,
        shuffle=True,
        seed=1,
        to_tensor=False,
        batch_size=5,
        block_chunks=4,
    )
    ds.set_epoch(0)
    idx = np.concatenate([b.sample_indices for b in ds])
    assert sorted(idx.tolist()) == list(range(40))  # full coverage
    assert idx.tolist() != list(range(40))  # but not in order


def test_chunk_decoded_once_per_epoch_without_cache(write_zarr) -> None:
    # block_chunks >> batch_size + shuffle scatters each chunk's samples across
    # non-consecutive batches. With the cache off, last-use eviction must still
    # decode each chunk exactly once per epoch (naive per-batch eviction re-reads).
    url, _ = write_zarr(n=80, spc=4)  # 20 chunks
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))

    decoded: collections.Counter[int] = collections.Counter()

    def count(chunk):
        decoded[chunk.read.chunk_index] += 1
        return chunk

    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        batch_size=1,
        block_chunks=20,
        shuffle=True,
        seed=0,
        to_tensor=False,
        chunk_transforms=[count],
    )
    ds.set_epoch(0)
    for _ in ds:
        pass

    assert set(decoded) == set(manifest.chunks[SplitName.TRAIN.value])
    assert all(v == 1 for v in decoded.values()), dict(decoded)  # decoded once, no re-reads


def test_buffer_residency_is_bounded_by_block(write_zarr) -> None:
    # Last-use eviction holds at most one shuffle block (+ a straddling-window
    # margin), independent of epoch length -- NOT the whole split.
    url, _ = write_zarr(n=160, spc=4)  # 40 chunks
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    block_chunks, batch_size = 8, 4
    ds = InSituDataset(
        url,
        manifest,
        batch_size=batch_size,
        block_chunks=block_chunks,
        shuffle=True,
        seed=0,
        to_tensor=False,
    )
    ds.set_epoch(0)
    for _ in ds:
        pass
    assert 1 <= ds.buffer_peak <= block_chunks + batch_size  # ~one block, not all 40
