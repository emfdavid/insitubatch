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
        chunk_sizes=(1, 4),  # chunk-size axis varies (G1)
        engines=("naive", "insitu", "memory"),
        caches=("none", "memory"),
        n_samples=48,
        inner=(4, 4),
        batch_size=8,
        block_chunks=4,
        worker_sweep=(0,),
        compute_ms_sweep=(0.0, 2.0),  # compute axis varies (G3)
        epochs=2,  # cache cold/warm (G4)
        verbose=False,
    )

    figs = build_figures(load(out))
    assert {"g1_throughput_vs_chunk", "g3_throughput_vs_compute", "g4_cache_epochs"} <= set(figs)
    paths = write_figures(figs, tmp_path / "fig")
    assert paths
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_g7_worker_tuning_from_frame() -> None:
    pytest.importorskip("plotly")
    import pandas as pd

    from bench.plot import build_figures

    rows = [
        dict(
            engine="workers",
            cache="none",
            storage="file",
            sample_chunk=4,
            n_samples=40,
            epoch=0,
            batch_size=8,
            block_chunks=8,
            prefetch_depth=2,
            num_workers=nw,
            compute_ms=0.0,
            seconds=1.0,
            samples_per_s=100.0 * nw,
            mb_per_s=1.0,
            ttfb_ms=5.0,
            peak_rss_mb=100.0,
        )
        for nw in (2, 4)
    ]
    df = pd.DataFrame(rows)
    df["engine_label"] = "workers"
    assert "g7_worker_tuning" in build_figures(df)
