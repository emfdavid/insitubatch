"""Smoke test: the one-command suite runs each engine locally and logs JSONL."""

from __future__ import annotations

from bench.run import run_suite


def test_suite_smoke(tmp_path) -> None:
    out = tmp_path / "suite.jsonl"
    results = run_suite(
        out=out,
        data_dir=tmp_path / "data",
        chunk_sizes=(1, 4),
        engines=("naive", "workers", "insitu", "memory"),
        caches=("none", "memory", "disk"),
        n_samples=64,
        inner=(4, 4),
        batch_size=8,
        block_chunks=4,
        num_workers=0,  # single-process DataLoader -> fast + deterministic in CI
        epochs=1,
        verbose=False,
    )

    assert out.exists()
    assert results, "suite produced no rows"
    engines = {r.engine for r in results}
    # naive/insitu/memory are torch-free and must run; workers needs torch (may skip).
    assert {"naive", "insitu", "memory"} <= engines
    assert all(r.samples_per_s > 0 for r in results)
    assert all(r.n_samples > 0 for r in results)
    # insitu ran with each cache backend
    insitu_caches = {r.cache for r in results if r.engine == "insitu"}
    assert {"none", "memory", "disk"} <= insitu_caches
