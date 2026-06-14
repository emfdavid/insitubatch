"""Read-plan dedup behaviour across the fat-chunk <-> degenerate spectrum."""

from __future__ import annotations

import numpy as np

from insitubatch import ArrayGeometry, build_read_plan, dedup_ratio


def _geom(chunk: int) -> ArrayGeometry:
    return ArrayGeometry("t2m", shape=(1000, 4, 4), chunks=(chunk, 4, 4), dtype=np.dtype("f4"))


def test_fat_chunks_dedup_collapses_reads() -> None:
    # 100-sample chunks: a contiguous batch of 50 samples lives in one chunk.
    geom = _geom(100)
    plan = build_read_plan(list(range(50)), {"t2m": geom})
    assert plan.n_reads == 1  # all 50 samples share chunk 0
    assert len(plan.gathers["t2m"]) == 50
    assert dedup_ratio(plan) == 50.0


def test_degenerate_one_sample_per_chunk() -> None:
    # GRIB-per-timestep: chunk size 1 -> every sample is its own read.
    geom = _geom(1)
    plan = build_read_plan([0, 5, 9, 100], {"t2m": geom})
    assert plan.n_reads == 4
    assert dedup_ratio(plan) == 1.0


def test_gather_recovers_within_offsets() -> None:
    geom = _geom(10)
    samples = [3, 7, 12, 25]  # chunks 0,0,1,2
    plan = build_read_plan(samples, {"t2m": geom})
    assert plan.n_reads == 3
    withins = [g.within for g in plan.gathers["t2m"]]
    assert withins == [3, 7, 2, 5]


def test_multi_variable_coscheduled() -> None:
    a = ArrayGeometry("a", shape=(100, 2), chunks=(10, 2), dtype=np.dtype("f4"))
    b = ArrayGeometry("b", shape=(100, 2), chunks=(10, 2), dtype=np.dtype("f4"))
    plan = build_read_plan([0, 1, 2], {"a": a, "b": b})
    # one chunk per variable -> two reads, three gathers each.
    assert plan.n_reads == 2
    assert len(plan.gathers["a"]) == len(plan.gathers["b"]) == 3
