"""What did this pass actually do? -- runtime counters, and the stage they accuse.

:mod:`~insitubatch.summary` predicts from geometry before anything runs; this module
reports what happened after it did. The two are deliberately separate surfaces: a
prediction that had to touch the store would be useless in the situation it exists for,
and a measurement that could be computed from configuration would not be a measurement.

The payload is :class:`PassStats`, and the part worth having is :func:`bottleneck` -- a
rule mapping observed pressure to the *one* stage to go fix. Depth counters alone tell you
the batch queue was empty; they do not tell you whether that was the store, the decode
pool, or a residency budget too small to admit the next chunk. Those three want opposite
responses, and telling them apart is the whole job.

**The measurement rule, which is sharper than the obvious one.** Wall-clock around a
thread hop measures GIL wait, not work. That is how our scatter memcpy once read as 51% of
the hot path when its real share is 7.7-10.1%. So:

* in-thread **cost** uses :func:`time.thread_time` -- CPU actually burned by that thread;
* **waiting** uses :func:`time.perf_counter`, because there the waiting *is* the quantity.

A stage timer that over-attributes is worse than no timer: it sends people to optimize a
stage that was never the problem.

**Wait totals are summed across concurrent tasks and will exceed wall time.** Many tiles
are in flight at once, so `fetch_wait_s` is task-seconds, not a share of the pass. Divide
by :attr:`Depths.inflight_peak` for an order-of-magnitude per-tile feel, and read the
ratios between stages rather than any absolute against the clock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Stage = Literal["consumer", "store", "decode", "residency", "gather", "unknown"]

# A queue sampled at or above this fraction of capacity is "full enough": the consumer is
# the constraint and every producer stage is keeping up. Not 1.0 -- a healthy pipeline
# dips as the consumer takes an item, and demanding a permanently brimming queue would
# report GPU-bound as I/O-bound on every well-tuned run.
FULL_ENOUGH = 0.5

# Below this share of samples-with-an-empty-queue we do not accuse anything. Starvation
# that rare is noise, and naming a stage for it is how a report loses its authority.
STARVED_ENOUGH = 0.2

# A stage must own this share of the accounted producer time before it is named. Two
# stages within this of each other is a genuine tie, and saying so beats a coin flip
# dressed as a diagnosis.
DOMINANT = 0.4


@dataclass(frozen=True)
class StageTimes:
    """Cumulative seconds by stage. See the module docstring for the units rule."""

    fetch_wait_s: float = 0.0  # perf_counter, summed over tasks: awaiting the store
    admission_parked_s: float = 0.0  # perf_counter: awaiting residency budget
    decode_s: float = 0.0  # thread_time on the decode pool: codec work
    assemble_s: float = 0.0  # thread_time: tile assembly + chunk_transform
    gather_s: float = 0.0  # thread_time in the producer: batch assembly
    batch_transform_s: float = 0.0  # thread_time in the producer: user batch stage

    @property
    def producer_s(self) -> float:
        """Total accounted producer-side cost, the denominator for a stage's share."""
        return (
            self.fetch_wait_s
            + self.admission_parked_s
            + self.decode_s
            + self.assemble_s
            + self.gather_s
            + self.batch_transform_s
        )


@dataclass(frozen=True)
class Depths:
    """Sampled pressure. Depths come from the consumer thread; peaks from their owners."""

    batch_queue_capacity: int = 0
    batch_queue_peak: int = 0
    batch_queue_samples: int = 0
    batch_queue_empty: int = 0  # samples that found nothing queued
    batch_queue_full_enough: int = 0  # samples at >= FULL_ENOUGH of capacity
    inflight_peak: int = 0
    max_inflight: int = 0
    decode_queue_peak: int = 0  # tiles submitted to the decode pool but not yet done
    decode_threads: int = 0
    resident_peak: int = 0  # distinct outer chunks held at once
    resident_peak_bytes: int = 0
    budget_bytes: int | None = None

    @property
    def starved_frac(self) -> float:
        """Share of consumer samples that found the batch queue empty."""
        if not self.batch_queue_samples:
            return 0.0
        return self.batch_queue_empty / self.batch_queue_samples

    @property
    def fed_frac(self) -> float:
        """Share of consumer samples that found the queue at least half full."""
        if not self.batch_queue_samples:
            return 0.0
        return self.batch_queue_full_enough / self.batch_queue_samples


@dataclass(frozen=True)
class PassStats:
    """One pass over one split: counters, depths, times, and the stage they accuse."""

    split: str
    epoch: int
    batches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    bad_chunks: int = 0
    wall_s: float = 0.0
    times: StageTimes = field(default_factory=StageTimes)
    depths: Depths = field(default_factory=Depths)

    @property
    def limiting_stage(self) -> Stage:
        """The stage to go fix, by :func:`bottleneck`."""
        return bottleneck(self)[0]

    def as_dict(self) -> dict[str, object]:
        """A plain dict, for logging as JSON or asserting in a test."""
        d = asdict(self)
        d["limiting_stage"], d["advice"] = bottleneck(self)
        return d


def bottleneck(stats: PassStats) -> tuple[Stage, str]:
    """Which stage limited this pass, and what to do about it.

    The rule reads the batch queue first, because a full queue settles the question: every
    producer stage kept up and the consumer is the constraint, which is the state you want.
    Only once the queue is *repeatedly* empty is there a producer stage to accuse, and then
    the accusation comes from which one spent the time -- not from which one feels slow.

    Returns ``("unknown", ...)`` rather than guessing when the evidence does not separate
    the candidates: no samples, or no stage clearly dominant. A confident wrong answer here
    costs more than an admission, because it sends someone to tune the wrong knob.
    """
    d, t = stats.depths, stats.times
    if not d.batch_queue_samples:
        return "unknown", "no consumer samples: the pass ended before any batch was taken."
    if d.fed_frac >= FULL_ENOUGH:
        return (
            "consumer",
            "the batch queue stayed fed, so the loader kept up and your training step is "
            "the constraint. This is the desired steady state -- nothing to tune here.",
        )
    if d.starved_frac < STARVED_ENOUGH:
        return (
            "unknown",
            f"the queue was empty on only {d.starved_frac:.0%} of samples, which is noise "
            "rather than a bottleneck.",
        )

    total = t.producer_s
    if total <= 0:
        return "unknown", "the queue ran empty but no stage time was accounted."
    ranked = sorted(
        (
            (t.admission_parked_s, "residency"),
            (t.fetch_wait_s, "store"),
            (t.decode_s, "decode"),
            (t.assemble_s, "decode"),
            (t.gather_s, "gather"),
            (t.batch_transform_s, "gather"),
        ),
        reverse=True,
    )
    top_s, top = ranked[0]
    if top_s / total < DOMINANT:
        return (
            "unknown",
            f"the queue ran empty but no stage owned more than {top_s / total:.0%} of "
            "accounted time: the cost is spread, so there is no single knob to turn.",
        )
    advice: dict[str, str] = {
        "residency": (
            "admission parked on the residency budget: the pool could not allocate the "
            "next chunk. Raise cache_budget_bytes, or lower batch_size, block_chunks, or "
            "the number of concurrent iterations. This is the state that otherwise "
            "presents as slow storage."
        ),
        "store": (
            "the loader waited on the store. Raise max_inflight, and check the store is "
            "in-region and the backend is the fast one for it."
        ),
        "decode": (
            "codec decode and chunk_transform dominated. Raise decode_threads, and check "
            "the chunk_transform is vectorized numpy that releases the GIL -- a "
            "per-element Python transform serializes the whole decode pool."
        ),
        "gather": (
            "batch assembly dominated. Check the gather run length (see describe()) and "
            "any batch_transform, which runs per batch on the producer thread."
        ),
    }
    return top, advice[top]  # type: ignore[return-value]


class StatsCollector:
    """Mutable accumulator behind :class:`PassStats`. **Lock-free by ownership.**

    There is no lock here, and that is a design constraint rather than an omission: a
    float ``+=`` is three bytecodes and is not atomic under the GIL, let alone off it. The
    counters are safe because **every field has exactly one writing thread**:

    ==============================================  ====================================
    field                                           sole writer
    ==============================================  ====================================
    ``fetch_wait_s``, ``admission_parked_s``        the event loop
    ``decode_s``, ``assemble_s``, decode depth      the event loop
    ``gather_s``, ``batch_transform_s``             the producer thread
    batch-queue depth samples                       the consumer thread
    ==============================================  ====================================

    Decode is the case that has to be *made* to fit: it runs on the decode pool's threads.
    So the pool thread measures its own ``thread_time`` and **returns** the delta, and the
    coroutine awaiting it does the add back on the loop. The decode threads never touch
    this object -- which is also why ``decode_s`` is honest CPU time rather than the GIL
    wait a wall clock around the hop would have recorded.

    Reads happen once, at pass teardown, behind edges that already exist: ``producer.join()``
    publishes the producer's fields and the scheduler's ``close()`` publishes the loop's.
    Snapshotting before either would be a race for the sake of a slightly shorter function.
    """

    __slots__ = (
        "admission_parked_s",
        "assemble_s",
        "batch_transform_s",
        "decode_inflight",
        "decode_queue_peak",
        "decode_s",
        "fetch_wait_s",
        "gather_s",
        "queue_capacity",
        "queue_empty",
        "queue_full_enough",
        "queue_peak",
        "queue_samples",
    )

    def __init__(self, queue_capacity: int = 0) -> None:
        self.fetch_wait_s = 0.0
        self.admission_parked_s = 0.0
        self.decode_s = 0.0
        self.assemble_s = 0.0
        self.gather_s = 0.0
        self.batch_transform_s = 0.0
        self.decode_inflight = 0
        self.decode_queue_peak = 0
        self.queue_capacity = queue_capacity
        self.queue_samples = 0
        self.queue_empty = 0
        self.queue_full_enough = 0
        self.queue_peak = 0

    # -- loop thread -------------------------------------------------------------

    def decode_enter(self) -> None:
        """A tile was handed to the decode pool. Loop thread only."""
        self.decode_inflight += 1
        if self.decode_inflight > self.decode_queue_peak:
            self.decode_queue_peak = self.decode_inflight

    def decode_exit(self) -> None:
        """A tile left the decode pool, by any path including failure. Loop thread only."""
        self.decode_inflight -= 1

    # -- consumer thread ---------------------------------------------------------

    def sample_queue(self, depth: int) -> None:
        """Record the batch queue's depth as the consumer found it. Consumer thread only.

        Sampled at the moment of the ``get``, which is the only moment whose answer means
        anything: a depth read from another thread describes a queue nobody was asking of.
        """
        self.queue_samples += 1
        if depth == 0:
            self.queue_empty += 1
        elif self.queue_capacity and depth / self.queue_capacity >= FULL_ENOUGH:
            self.queue_full_enough += 1
        if depth > self.queue_peak:
            self.queue_peak = depth

    # -- teardown ----------------------------------------------------------------

    def times(self) -> StageTimes:
        """The stage totals. Call after ``join()`` and ``close()``; see the class docstring."""
        return StageTimes(
            fetch_wait_s=self.fetch_wait_s,
            admission_parked_s=self.admission_parked_s,
            decode_s=self.decode_s,
            assemble_s=self.assemble_s,
            gather_s=self.gather_s,
            batch_transform_s=self.batch_transform_s,
        )


def format_pass(stats: PassStats) -> str:
    """The one-line-per-stage human view, for the per-epoch log."""
    d, t = stats.depths, stats.times
    stage, advice = bottleneck(stats)
    budget = "unbounded" if d.budget_bytes is None else f"{d.budget_bytes / 2**20:.0f} MiB"
    return (
        f"epoch {stats.epoch} ({stats.split}): {stats.batches} batches in {stats.wall_s:.1f}s"
        f" | queue {d.batch_queue_peak}/{d.batch_queue_capacity}"
        f" (empty {d.starved_frac:.0%}, fed {d.fed_frac:.0%})"
        f" | inflight peak {d.inflight_peak}/{d.max_inflight}"
        f" | decode peak {d.decode_queue_peak}/{d.decode_threads}"
        f" | resident {d.resident_peak} chunks, {d.resident_peak_bytes / 2**20:.0f} MiB"
        f" of {budget}"
        f" | wait fetch {t.fetch_wait_s:.2f}s, parked {t.admission_parked_s:.2f}s"
        f" | cpu decode {t.decode_s:.2f}s, assemble {t.assemble_s:.2f}s,"
        f" gather {t.gather_s:.2f}s, batch_tf {t.batch_transform_s:.2f}s"
        f" | limited by: {stage} -- {advice}"
    )
