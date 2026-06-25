"""Advected-field forecast example: windowed-data correctness + each framework learns.

The data layer is framework-neutral, so one fixture builds the store and three tests --
torch (run in CI), JAX, TF (each ``importorskip``) -- train the *same* tiny model on the
*same* dataset and assert it beats the persistence baseline (i.e. it learned the wind-
driven advection that windowed, multi-variable, no-reshard sampling makes available).
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.advection.data import (
    SYNTH_HORIZON,
    forecast_dataset,
    inputs_and_targets,
    make_advection_store,
    rmse,
)


@pytest.fixture
def synth_store(tmp_path) -> str:
    """A small advected-field store (fast: short trajectory, small grid)."""
    url = f"file://{tmp_path}/adv.zarr"
    make_advection_store(url, n_steps=288, size=24, seed=0)
    return url


def _dataset(url: str):
    return forecast_dataset(url, batch_size=32)


def test_forecast_dataset_is_windowed_and_multivariable(synth_store) -> None:
    ds = forecast_dataset(synth_store, batch_size=32, shuffle=False)
    ds.set_epoch(0)
    batch = next(iter(ds.train))

    assert set(batch.arrays) == {"t2m", "u10", "v10", "target"}
    # target is t2m read `horizon` steps ahead -- two views of one in-place array, no reshard
    assert batch.offsets == {"t2m": 0, "u10": 0, "v10": 0, "target": SYNTH_HORIZON}
    np.testing.assert_array_equal(
        batch.read_indices("target"), batch.sample_indices + SYNTH_HORIZON
    )
    x, persistence, target = inputs_and_targets(batch)
    assert x.shape[1] == 3  # three input channels stacked (Batch.stack)
    assert persistence.shape == target.shape
    # the field advects over 24 h, so persistence has real error a model can beat
    assert rmse(persistence, target) > 0.3


def test_torch_beats_persistence(synth_store) -> None:
    pytest.importorskip("torch")
    from examples.advection.train_torch import train

    model_rmse, persistence_rmse = train(_dataset(synth_store), epochs=8)
    assert model_rmse < persistence_rmse


def test_jax_beats_persistence(synth_store) -> None:
    pytest.importorskip("flax")
    from examples.advection.train_jax import train

    model_rmse, persistence_rmse = train(_dataset(synth_store), epochs=8)
    assert model_rmse < persistence_rmse


def test_tf_beats_persistence(synth_store) -> None:
    pytest.importorskip("tensorflow")
    from examples.advection.train_tf import train

    model_rmse, persistence_rmse = train(_dataset(synth_store), epochs=8)
    assert model_rmse < persistence_rmse
