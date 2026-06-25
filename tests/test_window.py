"""M-W phase 1: offset variables (`ArrayGeometry.shift`) + anchor range validity.

Pure coordinate-space behavior -- no engine. A variable is ``(path, offset)``; a window
is a set of offsets around a shared anchor; the only validity the engine enforces is that
every read ``anchor + offset`` stays on the array.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from insitubatch import open_geometries, split_by_chunk, valid_anchor_range
from insitubatch.source import InSituDataset
from insitubatch.types import ArrayGeometry, Batch


def _geom(n: int = 20, offset: int = 0) -> ArrayGeometry:
    return ArrayGeometry(
        path="t2m", shape=(n, 4, 4), chunks=(4, 4, 4), dtype=np.dtype("f4"), offset=offset
    )


def test_offset_defaults_zero_and_shift_composes() -> None:
    g = _geom()
    assert g.offset == 0
    s = g.shift(1)
    assert s.offset == 1
    # same array, only the offset moves -- two views that will share decoded slots
    assert (s.path, s.shape, s.chunks, s.dtype) == (g.path, g.shape, g.chunks, g.dtype)
    assert s.shift(1).offset == 2  # composes relatively
    assert g.shift(-3).offset == -3
    assert g.offset == 0  # frozen: the original is untouched


def test_valid_anchor_range_drops_edge_anchors() -> None:
    T = 20
    assert valid_anchor_range([0], T) == (0, T)  # no window -> every anchor
    assert valid_anchor_range([0, 1], T) == (0, T - 1)  # +1 target -> drop last
    assert valid_anchor_range([-1, 0], T) == (1, T)  # -1 history -> drop first
    assert valid_anchor_range([-1, 0, 1], T) == (1, T - 1)  # both ends
    assert valid_anchor_range([-2, 3], T) == (2, T - 3)  # asymmetric
    assert valid_anchor_range([], T) == (0, T)  # no offsets


def test_valid_anchor_range_empty_when_window_exceeds_array() -> None:
    lo, hi = valid_anchor_range([0, 25], 20)  # window wider than the array
    assert lo >= hi  # no anchor can satisfy it


# -- Batch offsets metadata + projection helpers -------------------------------


def test_batch_read_indices_and_stack() -> None:
    arrays = {
        "t_m1": np.arange(6).reshape(3, 2),
        "t_0": np.arange(6, 12).reshape(3, 2),
        "t_1": np.arange(12, 18).reshape(3, 2),
    }
    batch = Batch(
        arrays=arrays,
        sample_indices=np.array([5, 6, 7]),
        offsets={"t_m1": -1, "t_0": 0, "t_1": 1},
    )
    # read_indices = anchor + offset (label provenance); absent label defaults to 0.
    np.testing.assert_array_equal(batch.read_indices("t_0"), [5, 6, 7])
    np.testing.assert_array_equal(batch.read_indices("t_m1"), [4, 5, 6])
    np.testing.assert_array_equal(batch.read_indices("t_1"), [6, 7, 8])
    np.testing.assert_array_equal(batch.read_indices("missing"), [5, 6, 7])

    # stack assembles a multi-step window along a new axis, in the given label order.
    window = batch.stack(["t_m1", "t_0", "t_1"])
    assert window.shape == (3, 3, 2)
    for k, label in enumerate(["t_m1", "t_0", "t_1"]):
        np.testing.assert_array_equal(window[:, k], arrays[label])


# -- end-to-end windowed sampling through the real dataset ---------------------


def _forecast_dataset(write_zarr, *, n, spc, shuffle, **kw):
    """A dataset with two views of one array: input ``x`` at the anchor, target ``y``
    one step ahead (``shift(1)``) -- the canonical forecast setup, no reshard."""
    url, srcs = write_zarr(n=n, spc=spc)
    geom = open_geometries(url)["t2m"]
    geoms = {"x": geom, "y": geom.shift(1)}
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(url, manifest, geometries=geoms, shuffle=shuffle, **kw)
    return ds, srcs["t2m"]


@pytest.mark.parametrize("shuffle", [False, True])
def test_windowed_forecast_target_leads_input(write_zarr, shuffle) -> None:
    """Each row pairs input x[t] with target y[t+1] from the same in-place array, and
    every batch holds that relation regardless of shuffle (gather is anchor-indexed)."""
    ds, src = _forecast_dataset(write_zarr, n=40, spc=8, shuffle=shuffle, batch_size=7, seed=3)
    ds.set_epoch(0)
    seen = []
    for batch in ds.train:
        anchor = batch.sample_indices
        np.testing.assert_array_equal(batch.arrays["x"], src[anchor])  # x = array[t]
        np.testing.assert_array_equal(batch.arrays["y"], src[anchor + 1])  # y = array[t+1]
        seen.append(anchor)

    anchors = np.concatenate(seen)
    # Edge anchor (t = T-1, whose +1 read is off the array) is dropped; all others
    # are covered exactly once.
    assert sorted(anchors.tolist()) == list(range(39))


def test_windowed_batch_carries_offsets(write_zarr) -> None:
    """The Batch records each label's offset, so a label's true read indices and a
    stacked window can be recovered downstream (the Phase 3 metadata + helpers)."""
    ds, src = _forecast_dataset(write_zarr, n=40, spc=8, shuffle=False, batch_size=7, seed=0)
    ds.set_epoch(0)
    batch = list(ds.train)[0]

    assert batch.offsets == {"x": 0, "y": 1}
    np.testing.assert_array_equal(batch.read_indices("x"), batch.sample_indices)
    np.testing.assert_array_equal(batch.read_indices("y"), batch.sample_indices + 1)
    # the stacked window aligns with the per-label arrays it was built from
    window = batch.stack(["x", "y"])
    np.testing.assert_array_equal(window[:, 0], batch.arrays["x"])
    np.testing.assert_array_equal(window[:, 1], batch.arrays["y"])


def test_windowed_history_and_target(write_zarr) -> None:
    """A three-view window {-1, 0, +1} drops both edge anchors and reads each offset."""
    url, srcs = write_zarr(n=24, spc=4)
    src = srcs["t2m"]
    geom = open_geometries(url)["t2m"]
    geoms = {"prev": geom.shift(-1), "now": geom, "next": geom.shift(1)}
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(url, manifest, geometries=geoms, shuffle=True, batch_size=5, seed=1)
    ds.set_epoch(0)

    anchors = []
    for batch in ds.train:
        a = batch.sample_indices
        np.testing.assert_array_equal(batch.arrays["prev"], src[a - 1])
        np.testing.assert_array_equal(batch.arrays["now"], src[a])
        np.testing.assert_array_equal(batch.arrays["next"], src[a + 1])
        anchors.append(a)
    # valid anchors for {-1,0,1} over T=24 are [1, 23): both ends dropped.
    assert sorted(np.concatenate(anchors).tolist()) == list(range(1, 23))


def test_windowed_eviction_and_cross_epoch_reuse(write_zarr) -> None:
    """Windowed forecast with a budget far smaller than the split, over several epochs:
    chunks are admitted, evicted, and (cross-epoch) reused under churn. Every batch must
    still pair x[t] with y[t+1] -- the regression guard for the cross-epoch eviction race
    where a chunk freed for a miss was gathered as stale memory."""
    url, srcs = write_zarr(n=160, spc=8)  # 20 chunks
    src = srcs["t2m"]
    geom = open_geometries(url)["t2m"]
    geoms = {"x": geom, "y": geom.shift(1)}
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    # block_chunks=2, no override: the windowed working set holds only a few blocks, so
    # 20 chunks force repeated eviction. shuffle=False keeps the read-union local (the
    # tight budget is viable) while still exercising churn and cross-epoch reuse.
    ds = InSituDataset(url, manifest, geometries=geoms, shuffle=False, batch_size=5, block_chunks=2)
    for epoch in range(3):
        ds.set_epoch(epoch)
        anchors = []
        for batch in ds.train:
            a = batch.sample_indices
            np.testing.assert_array_equal(batch.arrays["x"], src[a])
            np.testing.assert_array_equal(batch.arrays["y"], src[a + 1])
            anchors.append(a)
        assert sorted(np.concatenate(anchors).tolist()) == list(range(159))  # drop t=159


def test_windowed_partial_iteration_then_clean_epoch(write_zarr) -> None:
    """Abort a windowed epoch mid-stream, then run a full one.

    An early break leaves pinned and in-flight (not-ready) chunks in the persistent
    pool. The next epoch must reconstruct the forecast correctly -- no deadlock, no
    stale data -- and afterwards the pool's counters and slots must be clean: every
    reference released and no abandoned not-ready partial lingering.
    """
    url, srcs = write_zarr(n=160, spc=8)  # 20 chunks
    src = srcs["t2m"]
    geom = open_geometries(url)["t2m"]
    geoms = {"x": geom, "y": geom.shift(1)}
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        url,
        manifest,
        geometries=geoms,
        shuffle=True,
        batch_size=4,
        block_chunks=2,
        prefetch_depth=2,
        seed=2,
    )

    # epoch 0: pull one batch, let the producer pin + read ahead (in-flight), then abort.
    ds.set_epoch(0)
    it = iter(ds.train)
    next(it)
    time.sleep(0.2)
    it.close()  # GeneratorExit -> teardown; leaves pins/partials in the persistent pool

    # epoch 1: full pass must be correct despite the leftovers.
    ds.set_epoch(1)
    anchors = []
    for batch in ds.train:
        a = batch.sample_indices
        np.testing.assert_array_equal(batch.arrays["x"], src[a])
        np.testing.assert_array_equal(batch.arrays["y"], src[a + 1])
        anchors.append(a)
    assert sorted(np.concatenate(anchors).tolist()) == list(range(159))

    # counters/slots clean: all references released, no abandoned not-ready slot.
    assert ds._pool._pinned == {}
    assert all(slot.ready for slot in ds._pool._slots.values())
    ds.close()


def test_windowed_val_correct_after_train_shared_pool(write_zarr) -> None:
    """The shared pool across split views, with windowing: iterate train (which warms --
    and via windowed spill *pollutes* -- the pool with chunks near the train/val boundary),
    then val. Val's forecast pairs must still be exact (boundary chunks are reused, not
    corrupted) -- the cross-split overlap the single shared pool is meant to exploit."""
    url, srcs = write_zarr(n=120, spc=8)  # 15 chunks
    src = srcs["t2m"]
    geom = open_geometries(url)["t2m"]
    geoms = {"x": geom, "y": geom.shift(1)}  # forecast: input x[t], target y[t]=x[t+1]
    manifest = split_by_chunk(geom, fractions=(0.7, 0.3, 0.0))
    ds = InSituDataset(url, manifest, geometries=geoms, batch_size=6, block_chunks=2)

    ds.set_epoch(0)
    for _ in ds.train:  # warm/pollute the shared pool (windowed reads spill across the split)
        pass
    for batch in ds.val:
        a = batch.sample_indices
        np.testing.assert_array_equal(batch.arrays["x"], src[a])
        np.testing.assert_array_equal(batch.arrays["y"], src[a + 1])
