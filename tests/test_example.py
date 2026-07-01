"""Smoke: the WB2-parallel examples run on synthetic data and crop the subregion."""

from __future__ import annotations

import pytest

from examples.wb2_dataloader import run_demo


def test_wb2_example_crops_subregion(tmp_path) -> None:
    summary = run_demo(
        url=None,  # synthetic, no network
        subregion=(8, 8),
        batch_size=8,
        block_chunks=4,
        num_epochs=1,
        verbose=False,
    )
    assert summary["samples"] > 0
    assert summary["sample_shape"] == (8, 8)  # the batch_transform cropped to subregion
    assert summary["ttfb_ms"] >= 0.0


def test_wb2_xbatcher_example_spawns(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("xbatcher")
    from examples.wb2_xbatcher import run_xbatcher_demo

    summary = run_xbatcher_demo(
        url=None,  # synthetic, no network
        subregion=(8, 8),
        batch_size=8,
        num_workers=2,
        mp_mode="spawn",  # portable: forkserver is Linux-only here
        num_epochs=1,
        verbose=False,
    )
    assert summary["samples"] > 0
    assert summary["sample_shape"] == (8, 8)
    assert summary["ttfb_ms"] > 0.0


def test_transforms_example_runs_both_stages(tmp_path) -> None:
    from examples.transforms import run_demo

    summary = run_demo(url=None, verbose=False)  # synthetic, no network
    # the cross-variable derived field proves the batch_transform ran
    assert summary["variables"] == ["t2m", "u10", "v10", "windspeed"]
    assert -50.0 < summary["t2m_mean_c"] < 50.0  # chunk_transform converted K -> C
    assert summary["windspeed_mean"] > 0.0
    assert summary["windspeed_nonneg"]
    assert summary["samples"] > 0
    # the reshaping chunk_transform (Coarsen factor=2) halved the spatial grid
    src_lat, src_lon = summary["source_inner"]
    assert summary["sample_shape"] == (src_lat // 2, src_lon // 2)


def test_fit_scaler_example_partial_fit(tmp_path) -> None:
    pytest.importorskip("sklearn")  # bench extra
    from examples.fit_scaler import run_demo

    summary = run_demo(url=None, cache_dir=str(tmp_path / "cache"), verbose=False)
    assert summary["samples"] > 0
    assert summary["stat_max_err"] < 1e-3  # partial_fit over the loader == true global stats
    assert abs(summary["scaled_mean"]) < 1e-2  # batch-stage scaling -> standardized output
    assert abs(summary["scaled_std"] - 1.0) < 0.05
