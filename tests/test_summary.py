"""``describe()`` is a promise about what will happen before anything runs, so the tests
that matter are the ones that catch it *disagreeing with the engine*: a report that
predicts a budget the loader does not use, or stays quiet about a layout that will cost
2x, is worse than no report at all.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import zarr

from insitubatch import ensure_local_dir, obstore_store, open_geometries, split_by_chunk
from insitubatch.source import InSituDataset
from insitubatch.summary import (
    ALLOCATOR_RETENTION,
    MIN_GATHER_RUN_BYTES,
    describe,
    gather_run_bytes,
    print_summary,
    working_set_bytes,
)
from insitubatch.types import ArrayGeometry, SplitName


def _geom(shape, chunks, *, dtype="f4", sample_axis=0):
    return ArrayGeometry(
        path="v",
        shape=shape,
        chunks=chunks,
        dtype=np.dtype(dtype),
        sample_axis=sample_axis,
    )


def _write_tiled(tmp_path, *, n, spc, inner, inner_chunks, variables=("t2m",)):
    """A store whose *inner* axes are chunked too -- what the fixture cannot make, and the
    only shape in which a ragged grid or a multi-tile chunk exists at all."""
    url = f"file://{tmp_path}/tiled.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    for var in variables:
        arr = group.create_array(var, shape=(n, *inner), chunks=(spc, *inner_chunks), dtype="f4")
        arr[:] = np.zeros((n, *inner), dtype="f4")
    store = obstore_store(url)
    geoms = open_geometries(store, list(variables))
    return store, geoms, split_by_chunk(geoms[variables[0]])


def _dataset(write_zarr, *, variables=("t2m",), **kw):
    url, srcs = write_zarr(
        variables=variables, **{k: kw.pop(k) for k in ("n", "spc", "inner") if k in kw}
    )
    store = obstore_store(url)
    geoms = open_geometries(store, list(variables))
    manifest = split_by_chunk(geoms[variables[0]])
    return InSituDataset(store, manifest, geometries=geoms, **kw), srcs


# --- the run length is set by the innermost extent, which is the whole advice ---------


def test_gather_run_is_the_whole_tile_when_every_inner_axis_spans():
    g = _geom((8, 721, 1440), (1, 721, 1440))
    assert gather_run_bytes(g) == 721 * 1440 * 4


def test_gather_run_of_a_full_width_slab_is_the_row_block():
    # (k, W) tiles keep the last axis whole, so the run is k rows, not one.
    assert gather_run_bytes(_geom((8, 720, 1440), (1, 180, 1440))) == 180 * 1440 * 4


def test_gather_run_of_a_square_tile_is_one_row_of_the_tile():
    # The same bytes per tile as the slab above, and 1/180th the run: this is the pair
    # the short-run warning exists to separate.
    assert gather_run_bytes(_geom((8, 720, 1440), (1, 180, 360))) == 360 * 4


def test_gather_run_includes_the_first_partial_axis_when_everything_inside_it_spans():
    # 10 consecutive planes, each whole: the tile is one contiguous block of the batch,
    # so the run is the partial axis's chunk width too -- it stops *after* it, not before.
    g = _geom((8, 100, 100, 100), (1, 10, 100, 100))
    assert gather_run_bytes(g) == 10 * 100 * 100 * 4


def test_gather_run_follows_the_sample_axis_not_axis_zero():
    g = _geom((721, 1440, 8), (721, 360, 1), sample_axis=2)
    assert gather_run_bytes(g) == 360 * 4


# --- one formula, called twice: the report must not predict a budget the engine ignores


def test_report_reproduces_the_budget_the_engine_actually_chose(write_zarr):
    ds, _ = _dataset(write_zarr, n=96, spc=4, inner=(6, 6), block_chunks=2, batch_size=4)
    report = ds.describe()
    assert report["memory"]["working_set_bytes"] == ds.cache_budget_bytes
    assert report["memory"]["budget_is_automatic"] is True


def test_an_explicit_budget_is_not_reported_as_automatic(write_zarr):
    ds, _ = _dataset(
        write_zarr,
        n=96,
        spc=4,
        inner=(6, 6),
        block_chunks=2,
        batch_size=4,
        cache_budget_bytes=64 << 20,
    )
    report = ds.describe()
    assert report["memory"]["residency_bytes"] == 64 << 20
    assert report["memory"]["budget_is_automatic"] is False


def test_working_set_charges_the_padded_stored_chunk_not_the_logical_one():
    # 7x7 field on a 4x4 grid: four stored chunks, 64 elements held for 49 wanted.
    g = _geom((32, 7, 7), (4, 4, 4))
    manifest = split_by_chunk(g)
    padded = working_set_bytes(
        [g], [g], manifest, block_chunks=2, ref_spc=4, shuffle=True, assembles=False
    )
    assert padded == 2 * 2 * (4 * 64 * 4)


# --- the findings ---------------------------------------------------------------------


def _codes(report, subject=None):
    return [n["code"] for n in report["notes"] if subject in (None, n["subject"])]


def test_short_gather_run_warns_on_a_square_tile(write_zarr):
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(6, 6), batch_size=4, block_chunks=2)
    report = ds.describe()
    (note,) = [n for n in report["notes"] if n["code"] == "short-gather-run"]
    assert note["severity"] == "warn"
    assert note["subject"] == "t2m"
    assert report["variables"]["t2m"]["gather_run_bytes"] < MIN_GATHER_RUN_BYTES


def test_a_long_run_is_not_flagged(write_zarr):
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(4, 512), batch_size=4, block_chunks=2)
    assert "short-gather-run" not in _codes(ds.describe())


def test_ragged_grid_is_reported_with_the_multiplier_it_costs(tmp_path):
    # 7x7 field on a 4x4 grid: four stored chunks, 64 elements held for the 49 wanted.
    store, geoms, manifest = _write_tiled(tmp_path, n=32, spc=4, inner=(7, 7), inner_chunks=(4, 4))
    ds = InSituDataset(store, manifest, geometries=geoms, batch_size=4, block_chunks=2)
    report = ds.describe()
    v = report["variables"]["t2m"]
    assert v["stored_chunks_per_chunk"] == 4
    assert v["ragged_multiplier"] == pytest.approx(64 / 49)
    assert v["slot_bytes"] == 4 * 64 * 4
    (note,) = [n for n in report["notes"] if n["code"] == "ragged-grid"]
    assert note["severity"] == "note"
    assert "1.306x" in note["message"]


def test_a_dividing_grid_is_not_called_ragged(tmp_path):
    store, geoms, manifest = _write_tiled(tmp_path, n=32, spc=4, inner=(8, 8), inner_chunks=(4, 4))
    ds = InSituDataset(store, manifest, geometries=geoms, batch_size=4, block_chunks=2)
    report = ds.describe()
    assert report["variables"]["t2m"]["ragged_multiplier"] == 1.0
    assert "ragged-grid" not in _codes(report)
    assert "single-inner-fat-chunk" not in _codes(report)


def test_one_stored_chunk_per_outer_chunk_is_reported_at_any_size(write_zarr):
    # Small field: still coupled concurrency and memory, so it is still worth saying.
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(4, 4), batch_size=4, block_chunks=2)
    report = ds.describe()
    assert report["variables"]["t2m"]["stored_chunks_per_chunk"] == 1
    assert "single-inner-fat-chunk" in _codes(report)


def test_block_chunks_widened_is_reported_against_what_was_asked_for(write_zarr):
    ds, _ = _dataset(write_zarr, n=128, spc=2, inner=(4, 4), batch_size=16, block_chunks=1)
    report = ds.describe()
    assert report["config"]["block_chunks_requested"] == 1
    assert report["config"]["block_chunks"] > 1
    assert "block-chunks-widened" in _codes(report, subject="config")


def test_variable_findings_come_before_global_ones(write_zarr):
    ds, _ = _dataset(write_zarr, n=128, spc=2, inner=(6, 6), batch_size=16, block_chunks=1)
    subjects = [n["subject"] for n in ds.describe()["notes"]]
    assert "t2m" in subjects and "config" in subjects
    assert subjects.index("config") > max(i for i, s in enumerate(subjects) if s == "t2m")


def test_the_always_true_facts_are_not_notes(write_zarr):
    # They hold for every dataset; a note that always fires trains people to skip the
    # section, so they are footnotes on the memory block instead.
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(6, 6), batch_size=4, block_chunks=2)
    codes = _codes(ds.describe())
    assert "allocator-retention" not in codes
    assert "budget-sized-for-one-iteration" not in codes


# --- the memory block -------------------------------------------------------------------


def test_estimated_peak_applies_the_measured_retention_factor(write_zarr):
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(6, 6), batch_size=4, block_chunks=2)
    mem = ds.describe()["memory"]
    assert mem["accounted_bytes"] == (
        mem["residency_bytes"] + mem["inflight_bytes"] + mem["queue_bytes"] + mem["scratch_bytes"]
    )
    assert mem["estimated_peak_bytes"] == int(mem["accounted_bytes"] * ALLOCATOR_RETENTION)
    assert mem["estimated_peak_bytes"] > mem["accounted_bytes"]


def test_concurrent_iterations_multiply_residency_only(write_zarr):
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(6, 6), batch_size=4, block_chunks=2)
    one, two = ds.describe()["memory"], ds.describe(iterations=2)["memory"]
    assert two["residency_bytes"] == 2 * one["residency_bytes"]
    assert two["inflight_bytes"] == one["inflight_bytes"]
    assert two["queue_bytes"] == one["queue_bytes"]


def test_iterations_must_be_at_least_one(write_zarr):
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(6, 6), batch_size=4, block_chunks=2)
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        ds.describe(iterations=0)


def test_describe_opens_nothing(write_zarr, monkeypatch):
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(6, 6), batch_size=4, block_chunks=2)

    def boom(*a, **k):
        raise AssertionError("describe() must not touch the store")

    monkeypatch.setattr(type(ds.store), "get", boom, raising=False)
    ds.describe()


# --- rendering ---------------------------------------------------------------------------


def _render(ds, **kw):
    buf = io.StringIO()
    ds.print_summary(file=buf, **kw)
    return buf.getvalue()


def test_rendering_carries_the_numbers_and_the_footnotes(write_zarr):
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(6, 6), batch_size=4, block_chunks=2)
    text = _render(ds)
    assert "variables" in text and "t2m" in text
    assert "ESTIMATED" in text
    assert f"accounted x {ALLOCATOR_RETENTION}" in text
    assert "Sized for ONE iteration" in text
    assert "warnings" in text  # the short gather run on a 6x6 field


def test_the_one_iteration_footnote_goes_away_when_you_asked_for_more(write_zarr):
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(6, 6), batch_size=4, block_chunks=2)
    assert "Sized for ONE iteration" not in _render(ds, iterations=3)
    assert "3 concurrent iteration" in _render(ds, iterations=3)


def test_short_runs_render_in_bytes_and_big_ones_in_binary_units(write_zarr):
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(6, 6), batch_size=4, block_chunks=2)
    assert "gather run 144 B" in _render(ds)
    big, _ = _dataset(write_zarr, n=64, spc=1, inner=(512, 512), batch_size=4, block_chunks=2)
    assert "gather run 1.0 MiB" in _render(big)


def test_print_summary_takes_a_report(write_zarr):
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(6, 6), batch_size=4, block_chunks=2)
    buf = io.StringIO()
    print_summary(describe(ds), file=buf)
    assert buf.getvalue() == _render(ds)


def test_two_variables_each_get_their_own_block(write_zarr):
    ds, _ = _dataset(
        write_zarr,
        variables=("t2m", "u10"),
        n=64,
        spc=4,
        inner=(6, 6),
        batch_size=4,
        block_chunks=2,
    )
    text = _render(ds)
    assert "  t2m\n" in text and "  u10\n" in text
    assert ds.describe()["config"]["n_chunks"] == {"t2m": 16, "u10": 16}


def test_split_chunk_counts_are_reported(write_zarr):
    ds, _ = _dataset(write_zarr, n=64, spc=4, inner=(6, 6), batch_size=4, block_chunks=2)
    cfg = ds.describe()["config"]
    assert sum(cfg["split_chunks"].values()) == 16
    assert cfg["split_chunks"][SplitName.TRAIN.value] > 0
