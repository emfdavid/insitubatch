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
