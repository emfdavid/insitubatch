"""Smoke tests: the one-command suite runs each engine locally and logs JSONL."""

from __future__ import annotations

import pytest

from bench.advection_sweep import PersistenceCheck, store_key
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


def test_persistence_check_flags_the_drift_that_hid_the_double_lend() -> None:
    # Replays the real 2026-07-31 numbers. The corrupt pinned arm scored +13.6% forecast
    # skill -- better than real ERA5's honest +11.4% -- so skill could not catch it. What
    # gave it away is that persistence RMSE, which involves no model, moved between repeats
    # of one config while the healthy arm reproduced its value exactly.
    check = PersistenceCheck()
    for r, healthy in enumerate((0.682, 0.682, 0.682)):
        check.observe("synth128", healthy, f"synth128 repeat {r}")
    assert check.report()  # bit-identical across repeats -> silent

    check = PersistenceCheck()
    for r, corrupt in enumerate((0.682, 0.934, 0.627)):
        check.observe("synth128", corrupt, f"synth128 repeat {r}")
    assert not check.report()
    assert len(check.drifted) == 2  # both departures from the first value


def test_persistence_check_ties_every_batch_size_to_one_store() -> None:
    # Batch size does not change the val set: measured, all of 32/64/128 returned
    # 0.588030696 over the synth256 store. Keying on the store (not the geom) is what makes
    # a payload sweep cross-check its own arms instead of each config trusting itself.
    check = PersistenceCheck()
    for bs in (32, 64, 128):
        check.observe("synth256", 0.588030696, f"synth256b{bs}")
    assert check.report()

    check.observe("synth256", 0.588030696 * (1 + 1e-9), "fp jitter")  # under RTOL -> ignored
    assert check.report()
    check.observe("synth256", 0.594, "the buggy reuse arm")
    assert not check.report()


def test_persistence_check_separates_stores_that_differ_by_chunking() -> None:
    # Regression: the sweep writes one store per (store, sample_chunk, inner_chunk) --
    # `..._synth128_c256_i128.zarr` vs `..._synth128_c4_i128.zarr` -- and splits are
    # chunk-aligned, so a different sample_chunk yields a different val set and a legitimately
    # different persistence RMSE. Keying the invariant on `store` alone collapsed all four
    # configs of `--sweeps chunk` onto one baseline and failed a healthy run.
    check = PersistenceCheck()
    for spc, value in ((256, 0.588), (64, 0.612), (16, 0.640), (4, 0.671)):
        check.observe(
            store_key({"store": "synth128", "size": 128, "sample_chunk": spc, "inner_chunk": None}),
            value,
            f"synth128 c{spc}",
        )
    assert check.report(), "configs reading different stores must not share a baseline"

    # ...while a real drift within one store is still caught.
    key = store_key({"store": "synth128", "size": 128, "sample_chunk": 64, "inner_chunk": None})
    check.observe(key, 0.934, "the corrupt arm")
    assert not check.report()


def test_persistence_check_still_ties_inner_chunk_variants_by_their_own_store() -> None:
    # The `inner` sweep varies spatial tiling only, which cannot move a sample-axis split --
    # but each variant is still its own zarr, so it gets its own baseline rather than being
    # asserted equal to the others by accident.
    keys = {
        store_key({"store": "synth128", "size": 128, "sample_chunk": 64, "inner_chunk": ic})
        for ic in (128, 64, 32)
    }
    assert len(keys) == 3
