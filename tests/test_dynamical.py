"""Downscaling example: deep sample chunks, a wrapping region crop, and the model learns.

One fixture builds a small synthetic irradiance archive with the three properties that make
dynamical.org's GFS analysis interesting -- many samples inside one stored chunk, a field tiled
across several stored chunks, and a tile grid that does not divide the field. The tests assert
the engine batches off that geometry, that the region crop wraps the prime meridian, and that
the torch model beats the bilinear baseline it can only beat by reading spatial context.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.dynamical.data import (
    COARSEN,
    GFS_VAR,
    LABEL,
    REGION,
    bilinear_upsample,
    block_mean,
    downscaling_dataset,
    inputs_and_targets,
    make_irradiance_store,
    region_index,
    rmse,
)
from insitubatch import obstore_store, open_geometries

N_STEPS, NLAT, NLON, SAMPLE_CHUNK, TILE = 192, 91, 180, 64, 32


@pytest.fixture
def irradiance_store(tmp_path) -> str:
    """A small synthetic archive with GFS-shaped chunking (fast: short, coarse grid)."""
    url = f"file://{tmp_path}/irradiance.zarr"
    make_irradiance_store(
        url, n_steps=N_STEPS, nlat=NLAT, nlon=NLON, sample_chunk=SAMPLE_CHUNK, tile=TILE, seed=0
    )
    return url


def test_geometry_is_deep_and_tiled(irradiance_store) -> None:
    geom = open_geometries(obstore_store(irradiance_store), variables=[GFS_VAR])[GFS_VAR]
    # Many samples behind one stored chunk -- the property the whole example is about.
    assert geom.sample_chunk_size == SAMPLE_CHUNK
    assert geom.n_chunks == N_STEPS // SAMPLE_CHUNK
    # The field spans several stored chunks, on a tile grid that does not divide it, so the
    # slot carries padding: residency exceeds the logical chunk, exactly as it does on GFS.
    assert geom.n_inner_chunks(0) == -(-NLAT // TILE) * -(-NLON // TILE)
    assert geom.chunk_bytes > SAMPLE_CHUNK * NLAT * NLON * 4


def test_region_crop_wraps_the_prime_meridian() -> None:
    rows, cols = region_index(NLAT, NLON, REGION)
    assert rows.size % COARSEN == 0 and cols.size % COARSEN == 0
    # The European window runs west of 0 and east of it, so the columns wrap and are not
    # monotonic -- a plain slice would silently take the long way round the globe instead.
    assert cols[0] > cols[-1]
    assert (np.diff(cols) != 1).sum() == 1  # exactly one discontinuity: the wrap itself
    assert cols.max() < NLON and cols.min() >= 0


def test_bilinear_baseline_reconstructs_a_smooth_field_exactly() -> None:
    # Validate the instrument before the subject: on a field that is linear in both axes,
    # bilinear upsampling of the block mean is exact, so any RMSE the eval reports later is
    # the model's or the data's -- not the baseline's arithmetic.
    y, x = np.mgrid[0:16, 0:24].astype("f4")
    field = (2.0 * y + 3.0 * x)[None]
    restored = bilinear_upsample(block_mean(field, COARSEN), COARSEN)
    assert restored.shape == field.shape
    # Cell-centre bilinear is exact on a linear ramp away from the clamped outer half-cells.
    inner = (slice(None), slice(COARSEN, -COARSEN), slice(COARSEN, -COARSEN))
    assert rmse(restored[inner], field[inner]) < 1e-3


def test_batches_carry_the_whole_field_and_crop_to_the_region(irradiance_store) -> None:
    ds = downscaling_dataset(
        obstore_store(irradiance_store), batch_size=8, block_chunks=1, shuffle=False
    )
    ds.set_epoch(0)
    batch = next(iter(ds.train))
    assert set(batch.arrays) == {LABEL}
    assert batch.arrays[LABEL].shape == (8, NLAT, NLON)  # field axes carried whole

    x, target = inputs_and_targets(batch)
    rows, cols = region_index(NLAT, NLON)
    assert x.shape == target.shape == (8, rows.size, cols.size)
    assert x.dtype == target.dtype == np.dtype("f4")
    assert target.min() >= 0.0  # irradiance is non-negative
    # The baseline leaves real headroom: the sub-grid cloud structure is gone from x.
    assert rmse(x, target) > 1.0


def test_torch_beats_bilinear(irradiance_store) -> None:
    pytest.importorskip("torch")
    from examples.dynamical.train_torch import train

    ds = downscaling_dataset(obstore_store(irradiance_store), batch_size=8, block_chunks=1)
    model_rmse, bilinear_rmse = train(ds, epochs=8)
    assert model_rmse < bilinear_rmse
