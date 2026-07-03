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
        caches=("none",),  # B1 is read-once; insitu cache returns in B2 on the ChunkPool
        n_samples=64,
        inner=(4, 4),
        batch_size=8,
        block_chunks_sweep=(4,),
        worker_sweep=(0,),  # single-process DataLoader -> fast + deterministic in CI
        cache_dir=tmp_path / "cache",
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
    assert insitu_caches == {"none"}  # B1 read-once: only the cache-off path runs


def test_xbatcher_engine(tmp_path) -> None:
    pytest.importorskip("xbatcher")  # the B2 baseline (bench extra)
    pytest.importorskip("torch")  # _run_xbatcher wraps in a torch DataLoader
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
        block_chunks_sweep=(4,),
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


def test_run_fsspec_backend_threads_to_row(tmp_path) -> None:
    # The M-GCS A/B: an engine must read through the fsspec backend and stamp it on the
    # JSONL row so obstore vs fsspec rows are distinguishable. file:// exercises the whole
    # dispatch without cloud (FsspecStore auto-wraps the sync LocalFileSystem).
    pytest.importorskip("fsspec")
    url = f"file://{tmp_path}/f.zarr"
    make_dataset(url, n_samples=40, inner=(3, 3), sample_chunk=8, variables=["t2m"])
    cfg = Cfg(
        engine="insitu",
        url=url,
        storage="file",
        backend="fsspec",
        sample_chunk=8,
        batch_size=8,
        epochs=1,
    )
    rows = run(cfg)
    assert rows and rows[0].n_samples > 0
    assert rows[0].backend == "fsspec"


def test_make_dataset_fsspec_backend_round_trips(tmp_path) -> None:
    # make_dataset --backend fsspec is the only writer that reaches GCS Rapid (gRPC).
    # Prove a full write->read round-trip through fsspec; file:// exercises it locally,
    # which requires fsspec_store's auto_mkdir default (LocalFileSystem won't create the
    # nested chunk dirs zarr writes, unlike obstore's LocalStore).
    pytest.importorskip("fsspec")
    url = f"file://{tmp_path}/w.zarr"
    make_dataset(
        url, n_samples=32, inner=(3, 3), sample_chunk=8, variables=["t2m"], backend="fsspec"
    )
    cfg = Cfg(
        engine="naive",
        url=url,
        storage="file",
        backend="fsspec",
        sample_chunk=8,
        batch_size=8,
        epochs=1,
    )
    # 32 samples / chunk 8 = 4 chunks; run() splits (0.8, 0.1, 0.1) -> train = 3 chunks,
    # and naive reads only the train split, so exactly 24 samples come back.
    rows = run(cfg)
    assert rows and rows[0].n_samples == 24


def test_run_forwards_store_kwargs(tmp_path) -> None:
    # Regression: engines build the store from store_kwargs and hand a Store to
    # open_geometries/InSituDataset -- the kwargs must NOT be splatted into those (the
    # Store-only migration break). The suite smoke tests use empty store_kwargs, so only
    # a non-empty dict exercises the path (obstore ignores skip_signature on file://).
    url = f"file://{tmp_path}/k.zarr"
    make_dataset(url, n_samples=40, inner=(3, 3), sample_chunk=8, variables=["t2m"])
    cfg = Cfg(engine="insitu", url=url, storage="file", sample_chunk=8, batch_size=8, epochs=1)
    rows = run(cfg, store_kwargs={"skip_signature": True})
    assert rows and rows[0].n_samples > 0
