"""Smoke: the bench JSONL renders to Plotly HTML graphs."""

from __future__ import annotations

import pytest

from bench.run import run_suite


def test_plot_smoke(tmp_path) -> None:
    pytest.importorskip("plotly")
    from bench.plot import build_figures, load, write_figures

    out = tmp_path / "r.jsonl"
    run_suite(
        out=out,
        data_dir=tmp_path / "d",
        chunk_sizes=(1, 4),  # so the chunk-size axis varies (G1)
        engines=("naive", "insitu", "memory"),
        caches=("none", "memory"),
        n_samples=48,
        inner=(4, 4),
        batch_size=8,
        block_chunks=4,
        num_workers=0,
        epochs=2,  # so the cache cold/warm graph has data (G4)
        verbose=False,
    )

    figs = build_figures(load(out))
    assert "g1_throughput_vs_chunk" in figs
    assert "g4_cache_epochs" in figs
    paths = write_figures(figs, tmp_path / "fig")
    assert paths
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
