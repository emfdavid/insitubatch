"""InSituDataset: the equal-chunking invariant and shuffle on/off."""

from __future__ import annotations

import collections
import logging

import numpy as np
import pytest
import zarr

from insitubatch import (
    SplitName,
    ensure_local_dir,
    obstore_store,
    open_geometries,
    split_by_chunk,
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


def test_unequal_sample_length_raises(tmp_path) -> None:
    url = f"file://{tmp_path}/uv.zarr"
    # Different sample-axis *length* is genuinely unsupported (samples aren't paired).
    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    group.create_array("u10", shape=(40, 2, 2), chunks=(8, 2, 2), dtype="f4")
    group.create_array("v10", shape=(32, 2, 2), chunks=(8, 2, 2), dtype="f4")  # different length
    geoms = open_geometries(obstore_store(url))
    manifest = split_by_chunk(geoms["u10"], fractions=(0.8, 0.1, 0.1))

    with pytest.raises(ValueError, match="sample-axis length"):
        InSituDataset(obstore_store(url), manifest, geometries=geoms)


def test_shuffle_false_is_in_order(write_zarr) -> None:
    url, _ = write_zarr(n=40, spc=8)
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        obstore_store(url),
        manifest,
        shuffle=False,
        batch_size=5,
        block_chunks=2,
    )
    ds.set_epoch(0)
    idx = np.concatenate([b.sample_indices for b in ds.train])
    assert idx.tolist() == list(range(40))  # strictly in order


def test_shuffle_true_covers_but_reorders(write_zarr) -> None:
    url, _ = write_zarr(n=40, spc=8)
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        obstore_store(url),
        manifest,
        shuffle=True,
        seed=1,
        batch_size=5,
        block_chunks=4,
    )
    ds.set_epoch(0)
    idx = np.concatenate([b.sample_indices for b in ds.train])
    assert sorted(idx.tolist()) == list(range(40))  # full coverage
    assert idx.tolist() != list(range(40))  # but not in order


def test_chunk_decoded_once_per_epoch_without_cache(write_zarr) -> None:
    # block_chunks >> batch_size + shuffle scatters each chunk's samples across
    # non-consecutive batches. With the cache off, last-use eviction must still
    # decode each chunk exactly once per epoch (naive per-batch eviction re-reads).
    url, _ = write_zarr(n=80, spc=4)  # 20 chunks
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))

    decoded: collections.Counter[int] = collections.Counter()

    def count(chunk):
        decoded[chunk.read.chunk_index] += 1
        return chunk

    ds = InSituDataset(
        obstore_store(url),
        manifest,
        batch_size=1,
        block_chunks=20,
        shuffle=True,
        seed=0,
        chunk_transforms=[count],
    )
    ds.set_epoch(0)
    for _ in ds.train:
        pass

    assert set(decoded) == set(manifest.chunks[SplitName.TRAIN.value])
    assert all(v == 1 for v in decoded.values()), dict(decoded)  # decoded once, no re-reads


def test_residency_is_bounded_by_block(write_zarr) -> None:
    # The default (no cache) budget = the working set: the current block plus one
    # read-ahead block (2*block_chunks), independent of epoch length -- NOT the whole
    # split. The lower bound is one block: a batch may draw across a whole block,
    # so the block must be co-resident to gather.
    url, _ = write_zarr(n=160, spc=4)  # 40 chunks
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    block_chunks, batch_size = 8, 4
    ds = InSituDataset(
        obstore_store(url),
        manifest,
        batch_size=batch_size,
        block_chunks=block_chunks,
        shuffle=True,
        seed=0,
    )
    ds.set_epoch(0)
    for _ in ds.train:
        pass
    assert block_chunks <= ds.resident_peak <= 2 * block_chunks  # ~two blocks, not all 40


@pytest.mark.parametrize("kind", ["heap", "mmap"])
def test_cache_decode_once_across_epochs(write_zarr, tmp_path, kind) -> None:
    # A cache budget large enough to hold the whole split -> every chunk is decoded
    # + transformed exactly ONCE across two epochs; epoch 2 is served from the pool
    # (cross-epoch reuse, the B2 win). Both backings: heap and mmap'd-on-NVMe.
    url, _ = write_zarr(n=160, spc=8)  # 20 chunks
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))

    transformed: collections.Counter[int] = collections.Counter()

    def count(chunk):  # runs once per outer chunk, on the miss (decode) path
        transformed[chunk.read.chunk_index] += 1
        return chunk

    ds = InSituDataset(
        obstore_store(url),
        manifest,
        batch_size=4,
        block_chunks=4,
        shuffle=True,
        seed=0,
        chunk_transforms=[count],
        cache_dir=str(tmp_path / "cache") if kind == "mmap" else None,
        cache_budget_bytes=10_000_000,  # >> the 20-chunk split -> nothing evicted
    )
    try:
        for epoch in range(2):
            ds.set_epoch(epoch)
            for _ in ds.train:
                pass
    finally:
        ds.close()

    assert set(transformed) == set(manifest.chunks[SplitName.TRAIN.value])
    assert all(v == 1 for v in transformed.values()), dict(transformed)  # once, not twice


def test_cache_persists_across_runs(write_zarr, tmp_path) -> None:
    # persist=True keeps the mmap cache + manifest past close(); a fresh dataset over the
    # same cache_dir serves every chunk from disk -- no re-decode (the chunk transform
    # never runs in run 2) -- and yields byte-identical batches in the same draw order.
    url, _ = write_zarr(n=160, spc=8)  # 20 chunks
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    cache = str(tmp_path / "cache")

    decoded: collections.Counter[int] = collections.Counter()

    def count(chunk):  # runs once per outer chunk, on the miss (decode) path only
        decoded[chunk.read.chunk_index] += 1
        return chunk

    def make() -> InSituDataset:
        return InSituDataset(
            obstore_store(url),
            manifest,
            batch_size=4,
            block_chunks=4,
            shuffle=True,
            seed=0,
            chunk_transforms=[count],
            cache_dir=cache,
            persist=True,
            cache_budget_bytes=10_000_000,  # >> the split -> nothing evicted
        )

    n_train = len(manifest.chunks[SplitName.TRAIN.value])
    ds = make()
    try:
        ds.set_epoch(0)
        run1 = [b.arrays["t2m"].copy() for b in ds.train]
    finally:
        ds.close()
    assert set(decoded) == set(manifest.chunks[SplitName.TRAIN.value])  # all decoded in run 1
    assert ds.cache_misses == n_train and ds.cache_hits == 0  # cold run: all misses

    decoded.clear()
    ds2 = make()  # fresh process-equivalent: new pool over the same persisted cache_dir
    try:
        ds2.set_epoch(0)
        run2 = [b.arrays["t2m"].copy() for b in ds2.train]
    finally:
        ds2.close()

    assert decoded == collections.Counter()  # nothing re-decoded: every chunk a cross-run hit
    assert ds2.cache_hits == n_train and ds2.cache_misses == 0  # warm run: all hits
    assert len(run1) == len(run2)
    for a, b in zip(run1, run2, strict=True):
        assert np.array_equal(a, b)


def test_persist_stale_transform_raises_then_rebuilds_with_flag(write_zarr, tmp_path) -> None:
    # A changed chunk_transform makes the cache stale. Default: constructing the dataset RAISES
    # (a stale cache is almost never intended). reset_stale_cache=True instead wipes it and
    # rebuilds cold -- yielding correct batches again.
    url, _ = write_zarr(n=80, spc=8)  # 10 chunks
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    cache = str(tmp_path / "cache")

    def scale_a(chunk):
        chunk.data = chunk.data * 2.0
        return chunk

    def scale_b(chunk):
        chunk.data = chunk.data * 3.0
        return chunk

    def make(transform, reset=False) -> InSituDataset:
        return InSituDataset(
            obstore_store(url),
            manifest,
            batch_size=4,
            block_chunks=4,
            shuffle=False,
            chunk_transforms=[transform],
            cache_dir=cache,
            persist=True,
            reset_stale_cache=reset,
            cache_budget_bytes=10_000_000,
        )

    ds = make(scale_a)
    ds.set_epoch(0)
    _ = [b for b in ds.train]
    ds.close()

    # Default: a different transform -> stale -> raise at construction (fail fast).
    with pytest.raises(ValueError, match="stale|reset_stale_cache"):
        make(scale_b)

    # Opt in: wipe + rebuild, batches correct against the new transform.
    ds2 = make(scale_b, reset=True)
    try:
        ds2.set_epoch(0)
        got = np.concatenate([b.arrays["t2m"] for b in ds2.train])
    finally:
        ds2.close()
    assert ds2.cache_misses > 0 and ds2.cache_hits == 0  # rebuilt cold, no false hits
    assert np.isfinite(got).all()


def test_persist_warns_when_cache_unusable(write_zarr, tmp_path, caplog) -> None:
    # persist=True but the cache is consulted and serves nothing (here: the .npy files
    # were deleted under a surviving manifest) -> one loud WARNING per epoch, and the
    # chunks are still re-fetched correctly (a miss, never a raise).
    url, _ = write_zarr(n=80, spc=8)  # 10 chunks
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    cache = tmp_path / "cache"

    def make() -> InSituDataset:
        return InSituDataset(
            obstore_store(url),
            manifest,
            batch_size=4,
            block_chunks=4,
            shuffle=False,
            cache_dir=str(cache),
            persist=True,
            cache_budget_bytes=10_000_000,
        )

    ds = make()
    try:
        ds.set_epoch(0)
        want = np.concatenate([b.arrays["t2m"] for b in ds.train])
    finally:
        ds.close()

    # Break the cache: keep the manifest, delete the data files it points at.
    for f in cache.glob("*.npy"):
        f.unlink()

    ds2 = make()
    ds2.set_epoch(0)
    with caplog.at_level(logging.WARNING, logger="insitubatch.source"):
        got = np.concatenate([b.arrays["t2m"] for b in ds2.train])
    ds2.close()

    assert ds2.cache_hits == 0  # nothing served from the (broken) cache
    assert np.array_equal(got, want)  # but the data is correct -- re-fetched, not raised
    assert any("persist=True but 0" in r.message for r in caplog.records), caplog.text


def test_sample_range_subsets_what_the_dataset_reads(write_zarr) -> None:
    # sample_range restricts the manifest to the covering chunks; the dataset then
    # yields exactly those samples (chunk-aligned).
    url, _ = write_zarr(n=80, spc=8)  # 10 chunks
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0), sample_range=(16, 40))
    ds = InSituDataset(obstore_store(url), manifest, shuffle=False, batch_size=8)
    ds.set_epoch(0)
    idx = np.concatenate([b.sample_indices for b in ds.train])
    assert set(idx.tolist()) == set(range(16, 40))  # chunks 2,3,4 -> samples 16..39


def test_split_views_share_one_pool_and_cover_disjoint(write_zarr) -> None:
    """train / val / test / all are views on *one* dataset sharing *one* pool; the splits
    are disjoint and ``all`` is their union."""
    url, _ = write_zarr(n=80, spc=8)  # 10 chunks
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(0.6, 0.2, 0.2))
    ds = InSituDataset(obstore_store(url), manifest, batch_size=8, block_chunks=4)
    pool = ds._pool  # the one pool every view shares

    seen: dict[str, set[int]] = {}
    for name in ("train", "val", "test"):
        ds.set_epoch(0)
        idx = np.concatenate([b.sample_indices for b in getattr(ds, name)])
        seen[name] = set(idx.tolist())
        assert ds._pool is pool  # iterating a view doesn't rebuild the pool

    assert seen["train"].isdisjoint(seen["val"])
    assert seen["train"].isdisjoint(seen["test"])
    assert seen["val"].isdisjoint(seen["test"])
    all_idx = set(np.concatenate([b.sample_indices for b in ds.all]).tolist())
    assert all_idx == seen["train"] | seen["val"] | seen["test"]


def test_val_view_is_deterministic_train_shuffles(write_zarr) -> None:
    url, _ = write_zarr(n=80, spc=8)
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(0.6, 0.2, 0.2))
    ds = InSituDataset(
        obstore_store(url), manifest, batch_size=8, block_chunks=4, shuffle=True, seed=1
    )
    ds.set_epoch(0)
    val0 = np.concatenate([b.sample_indices for b in ds.val])
    ds.set_epoch(5)  # different epoch
    val5 = np.concatenate([b.sample_indices for b in ds.val])
    np.testing.assert_array_equal(val0, val5)  # val ignores epoch (no shuffle)
    np.testing.assert_array_equal(val0, np.sort(val0))  # and is in order
