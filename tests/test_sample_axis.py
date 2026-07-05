"""Arbitrary sample axis — a sample is a slice along any single physical axis.

A sample is a slice along any *physical* axis, not just axis 0 -- the OME-NGFF
``(T,C,Z,Y,X)`` "sample over Z" case. shape/chunks stay in physical order; the engine
works sample-first via one moveaxis confined to the scheduler.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from insitubatch import ensure_local_dir, obstore_store, open_geometries, split_by_chunk
from insitubatch.source import InSituDataset
from insitubatch.types import ArrayGeometry


def _write(url: str, shape: tuple[int, ...], chunks: tuple[int, ...]) -> np.ndarray:
    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    arr = group.create_array("field", shape=shape, chunks=chunks, dtype="f4")
    data = np.arange(int(np.prod(shape)), dtype="f4").reshape(shape)
    arr[:] = data
    return data


# -- geometry unit level -----------------------------------------------------


def test_geometry_selects_sample_axis() -> None:
    # physical (C=2, Z=6, Y=8, X=8), sample over Z (axis 1), chunks (2,1,4,4).
    g = ArrayGeometry(
        path="field", shape=(2, 6, 8, 8), chunks=(2, 1, 4, 4), dtype=np.dtype("f4"), sample_axis=1
    )
    assert g.n_samples == 6
    assert g.sample_chunk_size == 1
    assert g.inner_shape == (2, 8, 8)  # C, Y, X (physical order, Z dropped)
    assert g.inner_chunks == (2, 4, 4)
    assert g.slot_shape(0) == (1, 2, 8, 8)  # sample-first (logical)
    # logical (chunk_index=3, inner_coord over C,Y,X) -> physical (c, 3, y, x)
    assert g.physical_chunk_coord(3, (0, 1, 1)) == (0, 3, 1, 1)


def test_sample_axis_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="sample_axis"):
        ArrayGeometry(path="f", shape=(2, 6), chunks=(2, 6), dtype=np.dtype("f4"), sample_axis=2)


def test_sample_axis_zero_is_identity() -> None:
    g = ArrayGeometry(path="f", shape=(6, 8), chunks=(2, 8), dtype=np.dtype("f4"))
    assert g.sample_axis == 0
    assert g.physical_chunk_coord(3, (0,)) == (3, 0)  # unchanged from the old (cid, *inner)


# -- end to end through the real engine --------------------------------------


def test_end_to_end_values_sampling_over_z(tmp_path) -> None:
    url = f"file://{tmp_path}/z.zarr"
    src = _write(url, shape=(2, 6, 8, 8), chunks=(2, 1, 4, 4))  # C,Z,Y,X
    geoms = open_geometries(obstore_store(url), sample_axis=1)
    g = geoms["field"]
    assert g.n_samples == 6 and g.inner_shape == (2, 8, 8)

    manifest = split_by_chunk(g, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(obstore_store(url), manifest, geometries=geoms, batch_size=2, shuffle=False)

    got = np.concatenate([b.arrays["field"] for b in ds.all], axis=0)
    # logical view: Z to the front -> (Z, C, Y, X)
    expected = np.moveaxis(src, 1, 0)
    assert got.shape == expected.shape == (6, 2, 8, 8)
    np.testing.assert_array_equal(got, expected)
    ds.close()


def test_windowing_composes_with_sample_axis(tmp_path) -> None:
    # A forecast-style shift along a non-zero sample axis: target leads input by 1 Z-slice.
    url = f"file://{tmp_path}/zw.zarr"
    src = _write(url, shape=(2, 6, 4, 4), chunks=(2, 1, 4, 4))
    base = open_geometries(obstore_store(url), sample_axis=1)["field"]
    geoms = {"x": base, "y": base.shift(1)}

    manifest = split_by_chunk(base, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(obstore_store(url), manifest, geometries=geoms, batch_size=2, shuffle=False)

    logical = np.moveaxis(src, 1, 0)  # (Z, C, Y, X)
    for b in ds.all:
        for i, anchor in enumerate(b.sample_indices):
            np.testing.assert_array_equal(b.arrays["x"][i], logical[anchor])
            np.testing.assert_array_equal(b.arrays["y"][i], logical[anchor + 1])
    ds.close()
