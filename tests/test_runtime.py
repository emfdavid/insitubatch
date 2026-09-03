"""Runtime observability: the counters, and the stage they accuse.

The attribution rule is tested on constructed :class:`PassStats` rather than by trying to
provoke each bottleneck against a real store: a test that has to make the store slow to
assert "store-bound" is testing the fixture. The integration tests below check the wiring
-- that the numbers arrive, are attributed to the right thread, and survive teardown.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace

import numpy as np
import pytest

from insitubatch import (
    Depths,
    InSituDataset,
    PassStats,
    StageTimes,
    bottleneck,
    obstore_store,
    open_geometries,
    split_by_chunk,
)
from insitubatch.runtime import (
    DOMINANT,
    FULL_ENOUGH,
    SEPARATION,
    STARVED_ENOUGH,
    StatsCollector,
    format_pass,
)


def _stats(**kw) -> PassStats:
    """A PassStats with sane defaults, so each test states only what it is about."""
    depths = Depths(**kw.pop("depths", {}))
    times = StageTimes(**kw.pop("times", {}))
    return PassStats(split="train", epoch=0, depths=depths, times=times, **kw)


# -- the attribution rule ---------------------------------------------------------


def test_no_samples_is_unknown_not_a_guess():
    stage, why = bottleneck(_stats())
    assert stage == "unknown"
    assert "no consumer samples" in why


def test_a_fed_queue_blames_the_consumer():
    # The desired steady state: the loader kept up, so the training step is the constraint.
    stats = _stats(
        depths={
            "batch_queue_capacity": 2,
            "batch_queue_samples": 100,
            "batch_queue_full_enough": 90,
            "batch_queue_empty": 2,
        }
    )
    stage, why = bottleneck(stats)
    assert stage == "consumer"
    assert "desired steady state" in why


def test_rare_starvation_is_noise_not_a_bottleneck():
    stats = _stats(
        depths={"batch_queue_capacity": 2, "batch_queue_samples": 100, "batch_queue_empty": 5},
        times={"fetch_wait_s": 99.0},
    )
    # Even with a huge fetch total, 5% starvation does not justify accusing the store.
    assert bottleneck(stats)[0] == "unknown"


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("admission_parked_s", "residency"),
        ("fetch_wait_s", "store"),
        ("decode_s", "decode"),
        ("assemble_s", "decode"),
        ("gather_s", "gather"),
        ("batch_transform_s", "gather"),
    ],
)
def test_a_starved_queue_accuses_the_dominant_stage(field, expected):
    starved = {"batch_queue_capacity": 2, "batch_queue_samples": 100, "batch_queue_empty": 90}
    stats = _stats(depths=starved, times={field: 10.0})
    assert bottleneck(stats)[0] == expected


def test_residency_starvation_is_distinguishable_from_slow_storage():
    """The row with no counterpart in LanceDB's model, and the one we most need.

    A budget-starved loader and slow storage present identically -- an empty batch queue --
    and want opposite fixes. Only `admission_parked_s` separates them.
    """
    starved = {"batch_queue_capacity": 2, "batch_queue_samples": 100, "batch_queue_empty": 90}
    storage = _stats(depths=starved, times={"fetch_wait_s": 10.0, "admission_parked_s": 0.1})
    budget = _stats(depths=starved, times={"fetch_wait_s": 0.1, "admission_parked_s": 10.0})
    assert bottleneck(storage)[0] == "store"
    assert bottleneck(budget)[0] == "residency"
    assert "cache_budget_bytes" in bottleneck(budget)[1]


def test_a_spread_cost_admits_it_rather_than_picking_one():
    """No stage dominant -> "unknown". A confident wrong answer tunes the wrong knob."""
    starved = {"batch_queue_capacity": 2, "batch_queue_samples": 100, "batch_queue_empty": 90}
    even = dict.fromkeys(("fetch_wait_s", "decode_s", "gather_s"), 1.0)
    stage, why = bottleneck(_stats(depths=starved, times=even))
    assert stage == "unknown"
    assert "spread" in why


def test_two_near_tied_stages_are_reported_as_a_tie():
    """Both clear DOMINANT and are 2% apart, so neither is *the* answer.

    Ranking on dominance alone, the winner is decided by tie-break order -- a coin flip
    wearing a diagnosis. (Residency is the one exception, and has its own test: parking is
    backpressure, so it yields rather than ties.)
    """
    starved = {"batch_queue_capacity": 2, "batch_queue_samples": 100, "batch_queue_empty": 100}
    stats = _stats(depths=starved, times={"fetch_wait_s": 10.2, "decode_s": 10.0})
    stage, why = bottleneck(stats)
    assert stage == "unknown"
    assert "store" in why and "decode" in why
    assert "Address them together" in why


def test_a_stage_does_not_lose_to_itself():
    """decode and gather each have two timers; they must be summed before ranking."""
    starved = {"batch_queue_capacity": 2, "batch_queue_samples": 100, "batch_queue_empty": 90}
    # Split across gather's two timers, each individually below the store's single one.
    stats = _stats(
        depths=starved, times={"gather_s": 3.0, "batch_transform_s": 3.0, "fetch_wait_s": 4.0}
    )
    assert bottleneck(stats)[0] == "gather"


def test_thresholds_are_the_documented_ones():
    # Guards the constants against a silent retune: they are load-bearing for every verdict.
    assert (FULL_ENOUGH, STARVED_ENOUGH, DOMINANT, SEPARATION) == (0.5, 0.2, 0.4, 0.1)


# -- the collector ----------------------------------------------------------------


def test_queue_sampling_classifies_empty_and_fed():
    c = StatsCollector(queue_capacity=4)
    for depth in (0, 0, 1, 2, 4):
        c.sample_queue(depth)
    assert c.queue_samples == 5
    assert c.queue_empty == 2
    assert c.queue_full_enough == 2  # 2/4 and 4/4 are both >= FULL_ENOUGH
    assert c.queue_peak == 4


def test_decode_depth_tracks_its_peak_not_its_last_value():
    c = StatsCollector()
    c.decode_enter()
    c.decode_enter()
    c.decode_enter()
    c.decode_exit()
    c.decode_exit()
    c.decode_exit()
    assert c.decode_inflight == 0
    assert c.decode_queue_peak == 3


def test_collector_takes_no_lock():
    """The single-writer design is the reason there is no lock; keep it that way.

    If someone adds one, they have either introduced contention on the hot path or papered
    over a broken ownership rule -- both worth failing a test over.
    """
    assert not any("lock" in s.lower() for s in StatsCollector.__slots__)


# -- wiring, against a real store --------------------------------------------------


def test_a_pass_reports_itself(write_zarr):
    url, _ = write_zarr(n=80, spc=8)
    store = obstore_store(url)
    geoms = open_geometries(store)
    manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
    ds = InSituDataset(store, manifest, batch_size=8, block_chunks=4)

    assert ds.last_pass is None  # nothing has run yet
    batches = list(ds.train)

    st = ds.last_pass
    assert st is not None
    assert st.split == "train"
    assert st.batches == len(batches)
    assert st.wall_s > 0
    # Every tile is fetched and decoded, so both stages must have registered something.
    assert st.times.fetch_wait_s > 0
    assert st.depths.inflight_peak > 0  # previously unreachable: lived and died on Scheduler
    assert st.depths.max_inflight == ds.scheduler_config.max_inflight
    assert st.depths.batch_queue_capacity == ds.prefetch_depth
    assert st.depths.batch_queue_samples > 0
    assert st.depths.resident_peak > 0
    assert st.as_dict()["limiting_stage"] == st.limiting_stage


def test_each_pass_gets_its_own_report(write_zarr):
    url, _ = write_zarr(n=80, spc=8)
    store = obstore_store(url)
    geoms = open_geometries(store)
    manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
    ds = InSituDataset(store, manifest, batch_size=8, block_chunks=4)

    list(ds.train)
    train = ds.last_pass
    list(ds.val)
    val = ds.last_pass

    assert train is not None and val is not None
    assert (train.split, val.split) == ("train", "val")
    # Not accumulated across passes: val reads its own, smaller split.
    assert val.batches < train.batches


def test_gather_is_billed_to_the_producer_not_the_loop(write_zarr):
    """gather runs on the producer thread, so its cost must land in `gather_s`.

    Guards the ownership rule the collector depends on: if gather were ever moved onto the
    loop, this number would keep incrementing while the field's documented single writer
    changed underneath it.
    """
    url, _ = write_zarr(n=160, spc=8, inner=(16, 16))
    store = obstore_store(url)
    geoms = open_geometries(store)
    manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
    ds = InSituDataset(store, manifest, batch_size=8, block_chunks=4)
    list(ds.train)
    assert ds.last_pass is not None
    assert ds.last_pass.times.gather_s > 0


def test_the_epoch_line_names_a_stage(write_zarr, caplog):
    url, _ = write_zarr(n=80, spc=8)
    store = obstore_store(url)
    geoms = open_geometries(store)
    manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
    ds = InSituDataset(store, manifest, batch_size=8, block_chunks=4)
    with caplog.at_level(logging.INFO, logger="insitubatch"):
        list(ds.train)
    assert "limited by:" in caplog.text


def test_format_pass_survives_an_unbounded_budget():
    # budget_bytes is None for an unbounded pool; the formatter must not divide by it.
    line = format_pass(_stats(depths={"batch_queue_samples": 1, "budget_bytes": None}))
    assert "unbounded" in line


def test_an_abandoned_pass_still_reports(write_zarr):
    """A `break` out of the loop must still leave a report behind.

    Found by writing the tuning-guide example. The report used to be built *after* the
    `with`, and an early break throws GeneratorExit at the yield: it unwinds through the
    `with` (so the scheduler closed correctly) but skipped everything that merely followed
    it, leaving `last_pass` as None. Backwards -- a pass you abandoned is usually one you
    abandoned *because* it was slow, which is exactly when you want the diagnosis.
    """
    url, _ = write_zarr(n=160, spc=8)
    store = obstore_store(url)
    geoms = open_geometries(store)
    manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
    ds = InSituDataset(store, manifest, batch_size=8, block_chunks=4)

    for i, _ in enumerate(ds.train):
        if i == 1:
            break

    st = ds.last_pass
    assert st is not None
    assert st.batches == 2  # only what the consumer actually took
    assert st.depths.max_inflight == ds.scheduler_config.max_inflight


def test_residency_loses_a_near_tie_because_parking_is_backpressure():
    """Parking inflates behind ANY slow stage, so it must not win a tie.

    Measured twice on real passes: `max_inflight=1` gave residency 49% / store 45%, and a
    heavy `batch_transform` gave residency 49% / gather 45%. Both times the other stage was
    the cause and raising the budget would have fixed neither -- chunks stay pinned while a
    slow stage drains them, so the budget fills as a *consequence*.
    """
    starved = {"batch_queue_capacity": 2, "batch_queue_samples": 100, "batch_queue_empty": 100}
    stats = _stats(depths=starved, times={"admission_parked_s": 0.58, "batch_transform_s": 0.52})
    stage, why = bottleneck(stats)
    assert stage == "gather"
    assert "backpressure" in why


def test_residency_still_wins_outright_when_nothing_downstream_explains_it():
    starved = {"batch_queue_capacity": 2, "batch_queue_samples": 100, "batch_queue_empty": 100}
    stats = _stats(depths=starved, times={"admission_parked_s": 10.0, "fetch_wait_s": 0.5})
    assert bottleneck(stats)[0] == "residency"


# -- end-to-end: does the verdict actually turn up? --------------------------------


def _dataset(write_zarr, **kw):
    # 64x64 fields, not the default 2x2: a transform has to cost more than the local
    # read for the verdict to be about the transform, and 2x2 arrays are all call overhead.
    url, _ = write_zarr(n=512, spc=16, inner=(64, 64))
    store = obstore_store(url)
    geoms = open_geometries(store)
    manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
    return InSituDataset(store, manifest, batch_size=16, block_chunks=4, **kw)


def _burn(a):
    for _ in range(400):
        a = np.sqrt(np.abs(a)) * 1.000001
    return np.ascontiguousarray(a, dtype="f4")


def test_a_slow_consumer_is_named_as_the_bottleneck(write_zarr):
    ds = _dataset(write_zarr)
    for i, _ in enumerate(ds.train):
        time.sleep(0.02)  # a "training step": the loader gets ahead and stays ahead
        if i == 9:
            break
    assert ds.last_pass is not None
    assert ds.last_pass.limiting_stage == "consumer"
    assert ds.last_pass.depths.fed_frac > 0.5


def test_a_heavy_chunk_transform_is_attributed_to_decode(write_zarr):
    ds = _dataset(write_zarr, chunk_transforms=(lambda c: replace(c, data=_burn(c.data)),))
    for i, _ in enumerate(ds.train):
        if i == 9:
            break
    assert ds.last_pass is not None
    assert ds.last_pass.limiting_stage == "decode"


def test_a_heavy_batch_transform_is_attributed_to_gather(write_zarr):
    def heavy(batch):
        for k, v in batch.arrays.items():
            batch.arrays[k] = _burn(v)
        return batch

    ds = _dataset(write_zarr, batch_transforms=(heavy,))
    for i, _ in enumerate(ds.train):
        if i == 9:
            break
    assert ds.last_pass is not None
    assert ds.last_pass.limiting_stage == "gather"


def test_assemble_time_is_not_lost_across_the_await(write_zarr):
    """Regression: `x += await f()` loses updates between concurrent tile tasks.

    Augmented assignment loads the target *before* evaluating the right-hand side, so the
    await suspends between the load and the store and each task writes back a value read
    before the others ran. It cost ~8x of the measured cost (0.21s recorded against 1.62s
    actually spent) and it is invisible to a correctness test -- only the total is wrong.
    Single-writer means no await between load and store, not merely one thread.
    """
    spent = []

    def timed(chunk):
        t0 = time.thread_time()
        out = _burn(chunk.data)
        spent.append(time.thread_time() - t0)
        return replace(chunk, data=out)

    ds = _dataset(write_zarr, chunk_transforms=(timed,))
    for i, _ in enumerate(ds.train):
        if i == 9:
            break
    assert ds.last_pass is not None
    recorded, actual = ds.last_pass.times.assemble_s, sum(spent)
    assert actual > 0
    # Generous: assemble_s also covers the assembly memcpy, so it is >= the transform. The
    # bug made it a *fraction*, which is what this catches.
    assert recorded >= actual * 0.8, f"lost {1 - recorded / actual:.0%} of the transform"
