"""A windowed shuffled pass releases its spill per block, and says that it did.

Under retention a chunk is held from its first use to its last. With a chunk permutation
those are most of an epoch apart, so a windowed shuffled pass keeps close to the whole
split resident (#66) -- measured at 46-56% of the split, and 97% for a four-lead
configuration.

Releasing each block's chunks when that block drains returns residency to the three-block
floor. The cost is admitting a chunk again when a later block reads it, which is only a good
trade when that second admission is served from already-decoded bytes -- hence the rule
rather than a knob.

The rule keys on `persist`, not on `cache_dir`. Only a persisted slot survives its own
eviction as a revivable file; with `cache_dir` alone the backing is unlinked when the slot is
evicted, so the re-read refetches and re-decodes. Measured on the 256-chunk windowed split
below, the same 249 re-reads were served 2% from cache without `persist` and 49% with it.

The re-read count is not decoration. A re-read served from the cache counts as a *hit*, so
the hit rate alone cannot distinguish "the working set fits" from "the budget is churning";
these tests pin that the number is reported.
"""

from __future__ import annotations

import logging
import tempfile

import numpy as np
import pytest

from insitubatch import obstore_store, open_geometries, split_by_chunk
from insitubatch.plan import build_stored_chunk_reads
from insitubatch.source import InSituDataset
from insitubatch.types import SplitName

N, SPC = 256, 4
_TMP = tempfile.mkdtemp()


@pytest.fixture
def windowed(write_zarr):
    url, _ = write_zarr(n=N, spc=SPC, inner=(4, 4))
    geometries = open_geometries(obstore_store(url))
    base = geometries["t2m"]
    geoms = {"now": base, "next": base.shift(SPC * 3)}  # a lead several chunks away
    manifest = split_by_chunk(base, fractions=(1.0, 0.0, 0.0))
    return url, geoms, manifest


def _drain(ds: InSituDataset) -> int:
    ds.set_epoch(0)
    return sum(int(b.arrays["now"].shape[0]) for b in ds.train)


def _dataset(windowed, **kw) -> InSituDataset:
    url, geoms, manifest = windowed
    return InSituDataset(
        obstore_store(url),
        manifest,
        geometries=geoms,
        batch_size=8,
        block_chunks=2,
        shuffle=True,
        seed=0,
        **kw,
    )


def test_the_rule_needs_a_persisted_cache(windowed, tmp_path) -> None:
    """Releasing early is the better trade only when the re-read is a local revive.

    `cache_dir` on its own is not enough and the middle case is the point: it gives the pool
    an mmap tier, but an evicted slot's backing is unlinked, so a re-read pays the fetch and
    the decode again -- the opposite of the trade this policy assumes.
    """
    assert _dataset(windowed).reread_spill is False
    assert _dataset(windowed, cache_dir=str(tmp_path / "a")).reread_spill is False
    assert _dataset(windowed, cache_dir=str(tmp_path / "b"), persist=True).reread_spill is True


def test_an_unwindowed_pass_keeps_retaining(windowed, tmp_path, write_zarr) -> None:
    """The control: without offsets there is no spill, so nothing changes.

    A chunk belongs to one block, is released when that block drains, and is never read
    again -- the policy would be a no-op, and turning it on would only cost the dedup.
    """
    url, _ = write_zarr(n=N, spc=SPC, inner=(4, 4))
    geometries = open_geometries(obstore_store(url))
    manifest = split_by_chunk(geometries["t2m"], fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        obstore_store(url),
        manifest,
        geometries=geometries,
        batch_size=8,
        block_chunks=2,
        shuffle=True,
        cache_dir=str(tmp_path / "c"),
        persist=True,
    )

    assert ds.reread_spill is False


def test_releasing_per_block_holds_far_less(windowed, tmp_path) -> None:
    """The point of the change: peak residency, retained versus released."""
    retained = _dataset(windowed)
    retained_samples = _drain(retained)
    retained_peak = retained._pool.max_resident
    retained.close()

    released = _dataset(windowed, cache_dir=str(tmp_path / "c"), persist=True)
    released_samples = _drain(released)
    released_peak = released._pool.max_resident
    released.close()

    # Fewer than N: a windowed view drops the edge anchors whose lead runs off the array.
    # What matters is that the policy does not change which samples arrive.
    assert retained_samples == released_samples > 0

    assert released_peak < retained_peak, (
        f"released {released_peak} vs retained {retained_peak}: releasing per block must "
        "hold less, or the policy is buying nothing"
    )


def test_the_epoch_line_reports_re_reads_and_evictions(windowed, tmp_path, caplog) -> None:
    """A re-read from the cache counts as a hit, so the hit rate alone hides the churn."""
    ds = _dataset(windowed, cache_dir=str(tmp_path / "c"), persist=True)
    with caplog.at_level(logging.INFO, logger="insitubatch"):
        _drain(ds)
    ds.close()

    lines = [r.message for r in caplog.records if "chunks" in r.message and "hit" in r.message]
    assert lines, "the per-epoch summary must be emitted"
    assert any("re-read" in line for line in lines), (
        f"re-reads must be visible when the spill is released: {lines}"
    )


def test_retained_passes_report_no_re_reads(windowed, caplog) -> None:
    """The control: without the policy a chunk is taken once, so the count stays silent.

    If this ever prints a re-read, either the plan stopped deduplicating or a chunk is
    being admitted twice for a reason nobody asked for.
    """
    ds = _dataset(windowed)
    with caplog.at_level(logging.INFO, logger="insitubatch"):
        _drain(ds)
    ds.close()

    assert not any("re-read" in r.message for r in caplog.records)


def test_the_floor_drops_from_the_split_to_three_blocks(windowed, tmp_path) -> None:
    """The point of the change, in the number a caller is actually sized by.

    Releasing costs nothing if the budget still has to hold the split: a released chunk is
    only *evictable*, and with a floor sized for retention nothing is ever evicted. So the
    floor has to know the policy, and this is the assertion that it does.
    """

    def floor_chunks(ds: InSituDataset) -> int:
        return ds.describe()["memory"]["residency_bytes"] // ds.geometries["now"].chunk_bytes

    retained = _dataset(windowed)
    released = _dataset(windowed, cache_dir=str(tmp_path / "c"), persist=True)
    try:
        split_chunks = len(retained.manifest.chunks["train"])
        assert floor_chunks(retained) >= split_chunks, "retention holds the split, by design"
        assert floor_chunks(released) <= 3 * released.block_chunks * 2, (
            "a released spill is sized by three blocks across two views, not by the split"
        )
    finally:
        retained.close()
        released.close()


def test_a_re_read_is_served_from_the_cache_rather_than_refetched(windowed, tmp_path) -> None:
    """The premise of the trade, asserted rather than assumed.

    A re-read that refetches and re-decodes is strictly worse than having retained the
    chunk, so the policy is only sound while the cache answers. The control is the same
    pass with `cache_dir` but no `persist`: identical re-read count, almost none of it
    served.
    """
    served = _dataset(windowed, cache_dir=str(tmp_path / "p"), persist=True)
    _drain(served)
    hit_rate = served._pool.hits / max(served._pool.hits + served._pool.misses, 1)
    rereads = served._pool.rereads
    served.close()

    assert rereads > 0, "the fixture must actually re-read, or this asserts nothing"
    assert hit_rate > 0.3, f"re-reads must come from the cache, not the store: {hit_rate:.0%}"


# -- a pass may only stand on its own outstanding fetches --------------------


def test_a_pass_does_not_skip_a_fetch_on_another_passs_writer(write_zarr) -> None:
    """Two iterations share a pool, and one may be abandoned at any moment.

    Skipping a fetch because *some* writer holds the slot reads a shared counter as if it
    were a promise. It is not: a pass abandoned mid-fetch unwinds without delivering, and
    the slot is left FILLING with nothing coming -- while the other pass, having skipped,
    waits on it forever. `ChunkPool` records how many writers a slot has and not whose they
    are, so the only sound basis for skipping is a scheduler's own outstanding work.

    Driven through the pool directly: the window is a cancellation between one pass's write
    starting and its delivery, and racing a real scheduler into it is exactly the flakiness
    these tests avoid.
    """
    from insitubatch.pool import ChunkPool, SlotState
    from insitubatch.types import ArrayGeometry

    geom = ArrayGeometry(path="t2m", shape=(32, 4, 4), chunks=(4, 4, 4), dtype=np.dtype("f4"))
    pool = ChunkPool({"t2m": geom})
    abandoned, live = pool.new_owner(), pool.new_owner()

    assert pool.try_admit("t2m", 3, abandoned)
    writing = pool.tile_write("t2m", 3, ())
    writing.__enter__()  # the abandoned pass's tile task is in flight
    assert pool.try_admit("t2m", 3, live)  # the live pass references the same chunk
    writing.__exit__(None, None, None)  # cancelled: unwinds without delivering
    pool.release_owner(abandoned)  # its teardown, after `Scheduler.close` drained it

    slot = pool._slots.get(("t2m", 3))
    assert slot is not None and slot.state is SlotState.FILLING, (
        "the live pass still references it, so it is not reclaimed"
    )
    assert slot.writers == 0 and slot.pending > 0, "nothing is coming for this slot"
    assert not pool.is_ready("t2m", 3)

    # The live pass must therefore be the one to fill it. Nothing on the pool may tell it
    # that a delivery is on the way, because none is.
    assert not hasattr(pool, "delivery_underway"), (
        "a pool-wide 'a writer holds this' query cannot answer 'will it be delivered' -- "
        "the writer may belong to a pass that is being torn down"
    )


# -- a block boundary is an admission boundary -------------------------------


def test_a_chunk_ending_one_block_and_starting_the_next_is_admitted_twice(
    write_zarr, tmp_path
) -> None:
    """The driver's per-chunk admission must not merge across a block boundary.

    Reads are chunk-major, so `_drive` decides admission once per run of tiles belonging to
    one chunk. Under a released spill the plan deliberately emits a chunk once per block that
    reads it -- and when a chunk is the *last* read of one block and the *first* of the next,
    those two emissions are adjacent. Merged, they take one permit and one reference where the
    consumer will release two: the first block's release unpins the chunk out from under the
    second, whose wait can then never be satisfied.

    The geometry is the one that produces the collision: a lead of exactly three chunks with
    two-chunk blocks, so a block's spill lands on the chunk the next block opens with. Counted
    from the plan rather than driven, so it does not depend on how the threads interleave.
    """
    url, _ = write_zarr(n=800, spc=8, inner=(4, 4))
    base = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(base, fractions=(0.8, 0.1, 0.1))
    ds = InSituDataset(
        obstore_store(url),
        manifest,
        geometries={"now": base, "next": base.shift(24)},
        batch_size=16,
        block_chunks=2,
        shuffle=True,
        seed=0,
        cache_dir=str(tmp_path / "c"),
        persist=True,
    )
    try:
        assert ds.reread_spill, "the fixture must be the released-spill case"
        ds.set_epoch(0)
        spc = ds._ref_spc
        blocks = ds._blocks(ds._draw_order(SplitName.TRAIN, True), spc)
        plan = build_stored_chunk_reads(
            [int(c) for b in blocks for c in b.chunk_ids],
            ds.geometries,
            spc,
            groups=[len(b.chunk_ids) for b in blocks],
        )
        # Count admissions the way `_drive` does: one per run of tiles naming the same
        # chunk, with a run ended by a block boundary whatever chunk it names.
        starts, runs, prev = set(plan.block_starts), 0, None
        for index, r in enumerate(plan.reads):
            key = (r.array, r.chunk_index)
            if index in starts:
                prev = None
            if key != prev:
                runs += 1
                prev = key
        owed = sum(len(b.read_keys) for b in blocks)
        assert runs == owed, (
            f"the driver sees {runs} chunk-runs but the consumer releases {owed} keys: "
            f"{owed - runs} admission(s) merged across a block boundary"
        )
    finally:
        ds.close()


def test_a_lead_landing_on_the_next_blocks_first_chunk_still_drains(write_zarr, tmp_path) -> None:
    """The same geometry, end to end: it must deliver every sample rather than wedge.

    Reported rather than hung (the fetch-ahead wait is bounded), but a run that stops is a
    run that stops.
    """
    url, _ = write_zarr(n=800, spc=8, inner=(4, 4))
    base = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(base, fractions=(0.8, 0.1, 0.1))
    ds = InSituDataset(
        obstore_store(url),
        manifest,
        geometries={"now": base, "next": base.shift(24)},
        batch_size=16,
        block_chunks=2,
        shuffle=True,
        seed=0,
        cache_dir=str(tmp_path / "c"),
        persist=True,
    )
    try:
        ds.set_epoch(0)
        drawn = sum(int(b.arrays["now"].shape[0]) for b in ds.train)
        assert drawn == 640
    finally:
        ds.close()
