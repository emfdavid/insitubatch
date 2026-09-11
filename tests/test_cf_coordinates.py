"""``open_geometries(store)`` over a CF/xarray-written group (#56).

A CF store's group holds more than data variables: 1-D coordinate arrays, and a 0-D
``spatial_ref`` whose *attributes* carry the CRS. Taking every array in the group makes the
0-D one raise ("sample_axis 0 out of range for 0-D array") and, worse, returns ``time`` and
``latitude`` as if they were variables to batch.
"""

from __future__ import annotations

import numpy as np
import pytest

from insitubatch import InSituDataset, obstore_store, open_geometries, split_by_chunk


@pytest.mark.parametrize("v3", [True, False], ids=["v3-dimension_names", "v2-_ARRAY_DIMENSIONS"])
def test_coordinates_and_grid_mapping_are_not_variables(write_cf_zarr, v3):
    """Both dimension-metadata spellings: v3 carries ``dimension_names`` in metadata, v2
    carries xarray's ``_ARRAY_DIMENSIONS`` attribute."""
    url, srcs = write_cf_zarr(variables=("t2m", "u10"), v3=v3)
    geoms = open_geometries(obstore_store(url))
    assert sorted(geoms) == ["t2m", "u10"]


def test_the_quickstart_call_works_on_a_cf_store(write_cf_zarr):
    """``open_geometries(store)`` with no ``variables`` must survive contact with a store
    written by xarray: it is the shortest form, so it is the one tried first."""
    url, srcs = write_cf_zarr()
    store = obstore_store(url)
    geoms = open_geometries(store)
    manifest = split_by_chunk(geoms["t2m"], fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(store, manifest, geometries=geoms, batch_size=4, shuffle=False)
    for batch in ds.all:
        np.testing.assert_array_equal(batch.arrays["t2m"], srcs["t2m"][batch.sample_indices])
    ds.close()


def test_an_explicit_coordinate_name_is_still_honoured(write_cf_zarr):
    """Skipping coordinates is a default, not a prohibition: asking for one by name gives it
    to you (a 1-D series is a legitimate thing to batch)."""
    url, _ = write_cf_zarr()
    geoms = open_geometries(obstore_store(url), variables=["time"])
    assert geoms["time"].n_samples == 32


def test_a_group_with_no_batchable_array_raises_and_says_what_to_do(write_cf_zarr):
    """When inference leaves nothing, say what was found, why each was skipped, and name the
    explicit route -- rather than returning {} and failing later with something unrelated."""
    url, _ = write_cf_zarr(variables=())
    with pytest.raises(ValueError) as exc:
        open_geometries(obstore_store(url))
    msg = str(exc.value)
    assert "variables=" in msg, "the error must name the explicit route"
    assert "spatial_ref" in msg and "time" in msg, "and what it skipped"


def test_unknown_variable_still_raises_plainly(write_cf_zarr):
    """An explicit name that is not there is a typo, not an inference problem."""
    url, _ = write_cf_zarr()
    with pytest.raises(KeyError):
        open_geometries(obstore_store(url), variables=["t2m_typo"])
