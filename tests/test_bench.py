"""Smoke tests: the one-command suite runs each engine locally and logs JSONL."""

from __future__ import annotations

import pytest

from bench.engines import Cfg, run
from bench.make_dataset import make_dataset
from bench.run import run_suite


def test_suite_smoke(tmp_path) -> None:
    out = tmp_path / "suite.jsonl"
    results = run_suite(
        out=out,
        data_dir=tmp_path / "data",
        chunk_sizes=(1, 4),
        engines=("naive", "workers", "xbatcher", "insitu", "memory"),
        caches=("none", "memory", "disk"),
        n_samples=64,
        inner=(4, 4),
        batch_size=8,
        block_chunks=4,
        worker_sweep=(0,),  # single-process DataLoader -> fast + deterministic in CI
        cache_dir=tmp_path / "cache",  # exercise the explicit DiskCache scratch dir
        epochs=1,
        verbose=False,
    )

    assert out.exists()
    assert results, "suite produced no rows"
    engines = {r.engine for r in results}
    # torch-free engines must always run; workers/xbatcher need optional deps.
    assert {"naive", "insitu", "memory"} <= engines
    assert all(r.samples_per_s > 0 for r in results)
    assert all(r.n_samples > 0 for r in results)
    insitu_caches = {r.cache for r in results if r.engine == "insitu"}
    assert {"none", "memory", "disk"} <= insitu_caches


def test_xbatcher_engine(tmp_path) -> None:
    pytest.importorskip("xbatcher")  # the B2 baseline (bench extra)
    url = f"file://{tmp_path}/x.zarr"
    make_dataset(url, n_samples=40, inner=(3, 3), sample_chunk=8, variables=["t2m"])
    cfg = Cfg(
        engine="xbatcher",
        url=url,
        storage="file",
        sample_chunk=8,
        batch_size=8,
        num_workers=0,
        epochs=1,
    )
    rows = run(cfg)
    assert rows and rows[0].samples_per_s > 0
    assert rows[0].n_samples > 0


def test_run_suite_compute_sweep(tmp_path) -> None:
    out = tmp_path / "s.jsonl"
    res = run_suite(
        out=out,
        data_dir=tmp_path / "d",
        chunk_sizes=(4,),
        engines=("insitu",),
        caches=("none",),
        n_samples=48,
        inner=(4, 4),
        batch_size=8,
        block_chunks=4,
        worker_sweep=(0,),
        compute_ms_sweep=(0.0, 2.0),
        epochs=1,
        verbose=False,
    )
    assert {r.compute_ms for r in res} == {0.0, 2.0}  # the compute sweep produced both


def test_workers_engine_spawns(tmp_path) -> None:
    # Guards the worker-pickling regression: the worker dataset must be a top-level,
    # picklable class (num_workers>0 starts a worker via forkserver/spawn, which
    # re-imports + unpickles it).
    pytest.importorskip("torch")
    url = f"file://{tmp_path}/w.zarr"
    make_dataset(url, n_samples=32, inner=(3, 3), sample_chunk=8, variables=["t2m"])
    cfg = Cfg(
        engine="workers",
        url=url,
        storage="file",
        sample_chunk=8,
        batch_size=8,
        num_workers=1,
        epochs=1,
    )
    rows = run(cfg)
    assert rows and rows[0].n_samples > 0
