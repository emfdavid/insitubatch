"""InSituDataset: the equal-chunking invariant and shuffle on/off."""

from __future__ import annotations

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
from insitubatch.source import InSituDataset


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
