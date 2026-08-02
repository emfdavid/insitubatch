"""Smoke tests: the one-command suite runs each engine locally and logs JSONL."""

from __future__ import annotations

import numpy as np
import pytest

import bench.probe_batch_buffers as probe
import insitubatch.buffers as core_buffers
from bench.advection_sweep import PersistenceCheck, store_key
from bench.engines import Cfg, run
from bench.make_dataset import make_dataset
from bench.probe_batch_buffers import ARMS, CASES, Case, JaxCheck, Roundtrip
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


# --- probe_batch_buffers -----------------------------------------------------------------
#
# The probe is what decided the pool's design, and it is the instrument we go back to when a
# measurement is disputed -- so it has to keep measuring the thing the core actually does. It
# predates the implementation, and carried its own copies of `aligned_empty`, `XLA_ALIGN` and
# the numpy->torch dtype mapping; a change to any of those in `src/` would have left the probe
# quietly reporting the old design's numbers. These tests pin the coupling (identity, not
# equality: an equal-looking reimplementation is exactly the drift being prevented) and run the
# CPU-reachable arms end to end so an API rename breaks CI rather than the next GPU session.

# One tiny case: the arms are exercised for wiring, not timed. The real CASES sweep to 256 MiB
# per buffer times four source chunks, which is a benchmark, not a test.
_TOY = (Case("toy", 4, (2, 3), np.dtype("f4")),)


def test_probe_shares_the_cores_buffer_primitives() -> None:
    assert probe.aligned_empty is core_buffers.aligned_empty
    assert probe.XLA_ALIGN is core_buffers.XLA_ALIGN


def test_probe_alloc_arm_times_both_sides() -> None:
    fresh_ms, reuse_ms = probe.probe_alloc(_TOY[0], iters=3)
    assert fresh_ms > 0 and reuse_ms > 0


def test_probe_main_runs_the_host_arm(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """The default invocation from the module docstring, on a toy payload."""
    monkeypatch.setattr(probe, "CASES", _TOY)
    monkeypatch.setattr("sys.argv", ["probe_batch_buffers", "--arms", "alloc", "--iters", "2"])
    probe.main()
    out = capsys.readouterr().out
    assert "fresh np.empty vs reused buffer" in out and "toy" in out


def test_probe_rejects_an_unknown_arm(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("sys.argv", ["probe_batch_buffers", "--arms", "aloc"])
    with pytest.raises(SystemExit, match="unknown arm"):
        probe.main()


def test_probe_cases_are_dtypes_the_core_can_allocate() -> None:
    """Every swept case must survive the pool's own allocator and torch's own mapping.

    The GPU arms build their tensors with `_torch_dtype`, the same converter `pinned_allocator`
    uses, so a case whose dtype the core cannot serve is a probe measuring nothing -- and it
    would only show up on a GPU box. Checked here, on CPU, for the real CASES.
    """
    torch = pytest.importorskip("torch")
    from insitubatch.frameworks import _torch_dtype

    for case in CASES:
        assert probe.aligned_empty((1, *case.inner), case.dtype).dtype == case.dtype
        assert torch.empty(0, dtype=_torch_dtype(case.dtype)).numpy().dtype == case.dtype


def test_probe_jax_arm_runs_without_a_gpu() -> None:
    """The soundness question `to_jax` rests on, exercised on whatever backend is present.

    Without a GPU the two device-transfer checks are untestable and must report *unknown*
    rather than *safe* -- a probe that returns "ok" on a CPU box would retire the question it
    exists to keep open.
    """
    pytest.importorskip("jax")
    check = probe.probe_jax(trials=2)
    assert check.keeps_alive, "jnp.from_dlpack must hold the array it was given"
    assert check.aligned_zero_copy == 2, "128-byte alignment is what makes zero-copy reliable"
    if not [d for d in __import__("jax").devices() if d.platform == "gpu"]:
        assert (check.race_observed, check.holds_source, check.poll_insufficient) == (
            None,
            None,
            None,
        )


@pytest.mark.parametrize(
    ("race", "holds", "insufficient"),
    [(True, False, True), (True, True, False), (False, False, False), (None, None, None)],
)
def test_jax_verdict_needs_both_an_async_copy_and_an_unheld_source(
    race: bool | None, holds: bool | None, insufficient: bool | None
) -> None:
    # Only the pairing condemns reuse: an async transfer JAX holds across is guarded by the
    # pool's own poll, and a synchronous one has no window at all.
    check = JaxCheck(
        backend="cpu",
        device="cpu",
        trials=1,
        default_zero_copy=1,
        aligned_zero_copy=1,
        keeps_alive=True,
        race_observed=race,
        holds_source=holds,
    )
    assert check.poll_insufficient is insufficient


@pytest.mark.parametrize(
    ("shares", "full", "prefix", "ms", "verdict"),
    [
        (False, True, True, 1.0, "BROKEN: .numpy() copied"),
        (True, True, True, 1.0, "ok"),
        (True, False, True, 1.0, "fast but flag says pageable"),
        (True, True, True, 9.0, "DEAD: degrades to pageable"),
        (True, True, False, 1.0, "fast but flag says pageable"),
    ],
)
def test_roundtrip_verdict_requires_the_flag_and_the_clock_to_agree(
    shares: bool, full: bool, prefix: bool, ms: float, verdict: str
) -> None:
    # `is_pinned()` alone is not trustworthy: the failure that matters is a non_blocking copy
    # silently degrading to a synchronous pageable one, which shows up in the clock.
    rt = Roundtrip(
        shares=shares,
        full_pinned=full,
        prefix_pinned=prefix,
        ms=ms,
        pageable_ms=10.0,
        pinned_ms=1.0,
    )
    assert rt.verdict == verdict


def test_every_declared_arm_is_reachable_from_main() -> None:
    # `--arms all` expands to ARMS, so an arm added to the tuple without a branch in main()
    # (or renamed in one place only) is a silent no-op.
    source = probe.main.__code__.co_consts
    branches = {c for c in source if isinstance(c, str)}
    assert set(ARMS) <= branches
