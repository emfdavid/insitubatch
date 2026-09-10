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

import numpy as np
import pytest

from insitubatch import obstore_store, open_geometries, split_by_chunk
from insitubatch.source import InSituDataset

N, SPC = 256, 4


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


# -- the predicate that stops a second block refetching tiles already on their way ------


def _pool_with_one_chunk():
    from insitubatch.pool import ChunkPool
    from insitubatch.types import ArrayGeometry

    geom = ArrayGeometry(path="t2m", shape=(32, 4, 4), chunks=(4, 4, 4), dtype=np.dtype("f4"))
    return ChunkPool({"t2m": geom}), geom


def test_delivery_underway_is_false_for_a_slot_nobody_has_admitted() -> None:
    pool, _ = _pool_with_one_chunk()
    assert pool.delivery_underway("t2m", 3) is False


def test_delivery_underway_is_true_while_a_writer_holds_the_slot() -> None:
    """The case the policy needs: block i's tiles are in flight when block i+1 admits."""
    pool, geom = _pool_with_one_chunk()
    owner = pool.new_owner()
    assert pool.try_admit("t2m", 3, owner)
    with pool.tile_write("t2m", 3, ()):
        assert pool.delivery_underway("t2m", 3) is True


def test_delivery_underway_is_true_once_the_slot_is_ready() -> None:
    pool, geom = _pool_with_one_chunk()
    owner = pool.new_owner()
    assert pool.try_admit("t2m", 3, owner)
    pool.deliver_tile("t2m", 3, (), np.zeros(geom.tile_shape(), dtype="f4"))
    assert pool.delivery_underway("t2m", 3) is True


def test_an_abandoned_partial_is_not_underway(caplog) -> None:
    """The control, and the one that would hang if it were wrong.

    A slot left FILLING by a cancelled pass has no writer and will never complete on its
    own. Reporting it as underway would make the next pass skip the fetch and then wait
    for a delivery nobody is going to make.
    """
    pool, _ = _pool_with_one_chunk()
    owner = pool.new_owner()
    assert pool.try_admit("t2m", 3, owner)  # allocated, never written

    assert pool.delivery_underway("t2m", 3) is False


def test_a_failed_slot_is_not_underway() -> None:
    """A poisoned slot is dropped so a later pass can refetch; it is not a delivery."""
    pool, _ = _pool_with_one_chunk()
    owner = pool.new_owner()
    assert pool.try_admit("t2m", 3, owner)
    pool.fail("t2m", 3, RuntimeError("bad chunk"))

    assert pool.delivery_underway("t2m", 3) is False
