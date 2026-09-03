"""Runtime observability: the counters, and the stage they accuse.

The attribution rule is tested on constructed :class:`PassStats` rather than by trying to
provoke each bottleneck against a real store: a test that has to make the store slow to
assert "store-bound" is testing the fixture. The integration tests below check the wiring
-- that the numbers arrive, are attributed to the right thread, and survive teardown.
"""

from __future__ import annotations

import logging

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
    """The real numbers that caught this: both clear DOMINANT, 2.6% apart.

    From an actual under-configured pass (max_inflight=1, chunk-per-sample store). Ranking
    on DOMINANT alone, both stages clear it and the winner is decided by tie-break order --
    a coin flip wearing a diagnosis. Fixing either alone just moves the pass to the other.
    """
    starved = {"batch_queue_capacity": 2, "batch_queue_samples": 100, "batch_queue_empty": 100}
    stats = _stats(
        depths=starved,
        times={
            "admission_parked_s": 23.26,
            "fetch_wait_s": 22.05,
            "decode_s": 1.16,
            "gather_s": 0.35,
        },
    )
    stage, why = bottleneck(stats)
    assert stage == "unknown"
    assert "residency" in why and "store" in why
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
