"""Advected-field forecast example: windowed-data correctness + each framework learns.

The data layer is framework-neutral, so one fixture builds the store and three tests --
torch (run in CI), JAX, TF (each ``importorskip``) -- train the *same* tiny model on the
*same* dataset and assert it beats the persistence baseline (i.e. it learned the wind-
driven advection that windowed, multi-variable, no-reshard sampling makes available).
"""

from __future__ import annotations

import re

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


def test_torch_metrics_collects_stall_ceiling_and_writes_jsonl(synth_store, tmp_path) -> None:
    pytest.importorskip("torch")
    import json

    from examples._forecast_metrics import MetricsLog
    from examples.advection.train_torch_metrics import train

    out = tmp_path / "metrics.jsonl"
    log = MetricsLog(str(out))
    model_rmse, persistence_rmse = train(_dataset(synth_store), epochs=3, ceiling=True, log=log)
    log.flush()

    assert model_rmse > 0 and persistence_rmse > 0  # a real training run happened (skill: sibling)

    runs: dict[str, list] = {}
    for m in log.rows:
        runs.setdefault(m.run, []).append(m)
    assert set(runs) == {"insitu", "ceiling"}  # --ceiling ran the compute-only baseline too
    assert len(runs["insitu"]) == len(runs["ceiling"]) == 3  # one row per epoch per run

    for m in log.rows:
        assert 0.0 <= m.data_stall_fraction <= 1.0
        assert m.n_batches > 0 and m.n_samples > 0 and m.samples_per_s > 0
        assert m.wall_s == pytest.approx(m.data_wait_s + m.compute_s)  # wall is the split's sum
        assert m.data_stall_fraction == pytest.approx(m.data_wait_s / m.wall_s)

    # val skill is patched onto the insitu run's final row only (known after the eval pass)
    assert runs["insitu"][-1].val_model_rmse == model_rmse
    assert np.isnan(runs["ceiling"][-1].val_model_rmse)

    # the JSONL mirrors the in-memory rows one line each
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(written) == len(log.rows) == 6
    assert {r["run"] for r in written} == {"insitu", "ceiling"}


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


# -- a store too small to be scored is refused before it is trained on -------


@pytest.mark.parametrize(
    "n_steps,why",
    [
        (192, "the 10% split rounds to zero chunks"),
        (256, "the split gets a chunk, but the horizon drops every anchor in it"),
    ],
    ids=["no-chunks", "no-drawable-anchors"],
)
def test_a_store_too_small_to_score_is_refused_at_setup(tmp_path, n_steps, why) -> None:
    """Both ways a small store runs out of evaluation data, caught before training.

    Neither is worth an epoch to discover: the run would train, validate, and only then
    reach `evaluate` with nothing to score, having kept none of it. Both are decided by the
    arguments, so both are answered from them.
    """
    url = f"file://{tmp_path}/small.zarr"
    make_advection_store(url, n_steps=n_steps, size=16, seed=0)

    with pytest.raises(ValueError, match="the val split has nothing to draw") as exc:
        forecast_dataset(url, batch_size=8)

    msg = str(exc.value)
    assert f"{n_steps} samples" in msg, why
    assert "Use at least" in msg, "naming the size that works is what makes it actionable"


def test_the_size_the_refusal_names_actually_works(tmp_path) -> None:
    """The control, and the one that keeps the advice honest.

    A remedy nobody checked is how a clear error message becomes a second wrong turn. The
    number is computed against the same split and window arithmetic the run uses, so this
    pins that the two agree.
    """
    url = f"file://{tmp_path}/small.zarr"
    make_advection_store(url, n_steps=192, size=16, seed=0)
    with pytest.raises(ValueError) as exc:
        forecast_dataset(url, batch_size=8)
    named = int(re.search(r"Use at least (\d+) samples", str(exc.value)).group(1))

    bigger = f"file://{tmp_path}/big.zarr"
    make_advection_store(bigger, n_steps=named, size=16, seed=0)
    ds = forecast_dataset(bigger, batch_size=8)
    try:
        ds.set_epoch(0)
        assert sum(int(b.arrays["t2m"].shape[0]) for b in ds.val) > 0
    finally:
        ds.close()
