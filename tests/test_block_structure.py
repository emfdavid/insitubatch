"""Structural invariants of a shuffle-block: what a block *is*, checked against an oracle.

The engine's block bookkeeping is invisible to every value-level test in the suite.
Retention pins a chunk from its first use to its last, so a block that claims the wrong
chunks still delivers exactly the right samples -- the batches are correct and the
structure underneath them is not. Nothing here asserts on data; these are assertions about
the plan.

**Where the answers come from.** An invariant is only worth asserting if its oracle is
something other than the code under test, or the test says `f(x) == f(x)` in a costume. Two
sources here are outside the engine's vectorized path:

* **A per-sample walk.** ``_reference_read_keys`` resolves each drawn anchor one at a time,
  and resolves ``sample -> chunk`` by *inverting* :meth:`ArrayGeometry.samples_in_chunk`
  rather than by the floor-division the engine uses. Different direction, different
  primitive. It stays independent permanently, and not by anyone's discipline: "the Python
  hot path is O(chunks), not O(samples)" is a load-bearing invariant, so the engine can
  never converge on this implementation. What we give up for throughput we get back as a
  free oracle. ``samples_in_chunk`` is itself pinned against the ``zarr-indexing`` package
  in `test_zarr_indexing_parity.py`, so the chain bottoms out outside this repository.
* **An explicit live-set simulation.** ``_reference_peak`` holds a set and measures it,
  where :func:`read_ahead_bound` does open/close arithmetic over first- and last-use maps.

Both survive the unification these tests exist to guide (#68). When one definition of "the
chunks this block reads" replaces the present two, an assertion that the two agree becomes
an assertion that a value equals itself -- so no such assertion is made here. Every check
below relates the engine to an oracle, or the block structure to the draw order it came
from, and both of those still have content afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from insitubatch import obstore_store, open_geometries, split_by_chunk
from insitubatch.plan import build_stored_chunk_reads
from insitubatch.shuffle import DrawOrder, chunk_permutation
from insitubatch.source import InSituDataset, _Block, read_ahead_bound
from insitubatch.types import ArrayGeometry, SplitName

Key = tuple[str, int]


# -- the oracles ---------------------------------------------------------------


def _sample_to_chunk(geom: ArrayGeometry) -> list[int]:
    """``sample -> chunk`` by inverting ``samples_in_chunk``, not by dividing.

    The engine computes this the other way round and in bulk. Building the table from the
    chunk's own declared extent keeps the reference honest about *which* function it is
    trusting: the one the zarr-indexing parity test already pins.
    """
    table = [-1] * geom.n_samples
    for chunk in range(geom.n_chunks):
        for sample in geom.samples_in_chunk(chunk):
            table[sample] = chunk
    assert -1 not in table, "samples_in_chunk does not tile the sample axis"
    return table


def _reference_read_keys(
    rows: np.ndarray, geometries: dict[str, ArrayGeometry], ref_spc: int
) -> set[Key]:
    """The ``(path, chunk)`` slots ``rows`` read, resolved one drawn sample at a time."""
    tables = {name: _sample_to_chunk(geom) for name, geom in geometries.items()}
    keys: set[Key] = set()
    for chunk_id, within in rows.tolist():
        anchor = int(chunk_id) * ref_spc + int(within)
        for name, geom in geometries.items():
            keys.add((geom.path, tables[name][anchor + geom.offset]))
    return keys


def _reference_peak(block_keys: list[set[Key]]) -> int:
    """Peak concurrent residency, by holding a live set rather than counting transitions.

    A key is live from the block that first reads it through the block that last does; the
    driver is additionally allowed to be working on block ``b+1`` while the consumer
    gathers ``b``. That is the same rule :func:`read_ahead_bound` encodes, arrived at by
    simulation instead of by arithmetic over first/last-use maps.
    """
    last_use = {key: bi for bi, keys in enumerate(block_keys) for key in keys}
    live: set[Key] = set()
    peak = 0
    for bi, keys in enumerate(block_keys):
        live |= keys
        ahead = block_keys[bi + 1] if bi + 1 < len(block_keys) else set()
        peak = max(peak, len(live | ahead))
        live = {key for key in live if last_use[key] > bi}
    return max(peak, 1)


# -- the sweep -----------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One geometry. ``offsets`` are windowed views of the single stored array."""

    id: str
    n: int
    spc: int
    block_chunks: int
    offsets: tuple[int, ...]
    shuffle: bool = True
    fractions: tuple[float, float, float] = (1.0, 0.0, 0.0)


CASES = [
    # -- controls: no window, so no anchor is ever dropped --------------------
    Case("plain", n=256, spc=4, block_chunks=2, offsets=(0,)),
    Case("plain-sequential", n=256, spc=4, block_chunks=2, offsets=(0,), shuffle=False),
    Case("plain-wide-blocks", n=256, spc=4, block_chunks=8, offsets=(0,)),
    Case("plain-split", n=256, spc=4, block_chunks=2, offsets=(0,), fractions=(0.8, 0.1, 0.1)),
    Case("plain-ragged-tail", n=250, spc=4, block_chunks=3, offsets=(0,)),
    # -- windowed: `valid_anchor_range` drops edge anchors ---------------------
    Case("lead-1", n=256, spc=4, block_chunks=2, offsets=(0, 1)),
    Case("lead-one-chunk", n=256, spc=4, block_chunks=2, offsets=(0, 4)),
    Case("lead-three-chunks", n=256, spc=4, block_chunks=2, offsets=(0, 12)),
    Case("lead-sequential", n=256, spc=4, block_chunks=2, offsets=(0, 12), shuffle=False),
    Case("history-and-lead", n=256, spc=4, block_chunks=2, offsets=(-4, 0, 4)),
    Case("sparse-leads", n=256, spc=4, block_chunks=4, offsets=(0, 12, 60)),
    Case("wide-blocks-windowed", n=256, spc=4, block_chunks=8, offsets=(0, 12)),
    Case("one-sample-chunks", n=64, spc=1, block_chunks=4, offsets=(0, 3)),
    Case("fat-chunks", n=256, spc=32, block_chunks=2, offsets=(0, 32)),
    Case("ragged-tail-windowed", n=250, spc=4, block_chunks=3, offsets=(0, 12)),
    Case(
        "windowed-split", n=256, spc=4, block_chunks=2, offsets=(0, 12), fractions=(0.8, 0.1, 0.1)
    ),
]

WINDOWED = [c for c in CASES if any(o != 0 for o in c.offsets)]


@dataclass
class Pass:
    """One epoch's draw order and the blocks the engine walks it in."""

    ds: InSituDataset
    order: DrawOrder
    blocks: list[_Block]
    ref_spc: int

    def rows(self, bi: int) -> np.ndarray:
        block = self.blocks[bi]
        return self.order.rows[block.start : block.stop]

    def declared(self, bi: int) -> set[Key]:
        """The slots the *driver* plans for block ``bi`` -- expanded from its chunk ids."""
        reads = build_stored_chunk_reads(
            self.blocks[bi].chunk_ids, self.ds.geometries, self.ref_spc
        )
        return {(r.array, r.chunk_index) for r in reads}

    def awaited(self, bi: int) -> set[Key]:
        """The slots the *consumer* pins and releases for block ``bi``."""
        return self.blocks[bi].read_keys


@pytest.fixture
def build_pass(write_zarr):
    """Open a dataset for one case and expose its epoch-0 draw order and blocks."""
    made: list[InSituDataset] = []

    def _build(case: Case) -> Pass:
        url, _ = write_zarr(n=case.n, spc=case.spc, inner=(2, 2))
        base = open_geometries(obstore_store(url))["t2m"]
        geoms = {f"o{i}": base.shift(o) for i, o in enumerate(case.offsets)}
        manifest = split_by_chunk(base, fractions=case.fractions)
        ds = InSituDataset(
            obstore_store(url),
            manifest,
            geometries=geoms,
            batch_size=4,
            block_chunks=case.block_chunks,
            shuffle=case.shuffle,
            seed=0,
        )
        made.append(ds)
        ds.set_epoch(0)
        order = ds._draw_order(SplitName.TRAIN, shuffle=case.shuffle)
        spc = base.sample_chunk_size
        return Pass(ds, order, ds._blocks(order, spc), spc)

    yield _build
    for ds in made:
        ds.close()


# -- validate the instrument ---------------------------------------------------


def test_the_reference_resolves_a_hand_computed_case() -> None:
    """Point the oracle at an answer worked out by hand before trusting it on a sweep."""
    geom = ArrayGeometry("t2m", (8, 2), (4, 2), np.dtype("f4"))
    assert _sample_to_chunk(geom) == [0, 0, 0, 0, 1, 1, 1, 1]

    # Anchors 2 and 3 with a lead of 2 read samples {2,3} for "now" and {4,5} for "next":
    # chunk 0 for the first, chunk 1 for the second.
    rows = np.array([[0, 2], [0, 3]], dtype=np.int64)
    geoms = {"now": geom, "next": geom.shift(2)}
    assert _reference_read_keys(rows, geoms, 4) == {("t2m", 0), ("t2m", 1)}


def test_the_reference_agrees_with_the_engine_where_nothing_is_dropped(build_pass) -> None:
    """The control for the oracle: with no window the engine's own math is known-good."""
    p = build_pass(Case("plain", n=256, spc=4, block_chunks=2, offsets=(0,)))
    everything = set().union(*(p.awaited(bi) for bi in range(len(p.blocks))))
    assert _reference_read_keys(p.order.rows, p.ds.geometries, p.ref_spc) == everything


def test_the_peak_simulation_matches_a_hand_computed_schedule() -> None:
    """Worked by hand: a key shared across blocks stays live, and one block runs ahead."""
    a, b, c, d = ("x", 0), ("x", 1), ("x", 2), ("x", 3)

    # b0={a,b} b1={b} b2={c}: draining b0 holds {a,b} while fetching b1's {b} -- already
    # held, so 2. Draining b1 holds {b} and fetches {c}: also 2.
    assert _reference_peak([{a, b}, {b}, {c}]) == 2

    # Widen the block ahead and the peak follows it: {b} live, {c,d} arriving.
    assert _reference_peak([{a, b}, {b}, {c, d}]) == 3

    assert _reference_peak([]) == 1


# -- the invariants ------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_blocks_tile_the_draw_order(build_pass, case: Case) -> None:
    """Row ranges partition the order: contiguous, gapless, covering every drawn row."""
    p = build_pass(case)
    assert p.blocks, "a non-empty split must produce at least one block"
    assert p.blocks[0].start == 0
    assert p.blocks[-1].stop == len(p.order)
    for block, nxt in zip(p.blocks, p.blocks[1:], strict=False):
        assert block.stop == nxt.start


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_each_drawn_chunk_belongs_to_exactly_one_block(build_pass, case: Case) -> None:
    """No chunk's rows straddle a boundary, and every drawn chunk is in some block.

    Asserting instead that a block's chunk ids equal its own rows' chunks would be
    tautological -- one is derived from the other. What is *not* given is that the
    boundaries fall where no chunk spans two of them: a chunk in two blocks is pinned by
    both and released by whichever drains last, which is how a shared residency becomes a
    reference nobody returns.
    """
    p = build_pass(case)
    seen: set[int] = set()
    for bi, block in enumerate(p.blocks):
        ids = {int(c) for c in block.chunk_ids}
        assert seen.isdisjoint(ids), (
            f"block {bi} shares chunks {sorted(seen & ids)} with an earlier block"
        )
        seen |= ids
    assert seen == {int(c) for c in p.order.rows[:, 0]}


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_blocks_are_slices_of_the_epoch_chunk_permutation(build_pass, case: Case) -> None:
    """A block is `block_chunks` consecutive chunks of this epoch's permutation.

    That is the definition -- `block_shuffled_order` groups the permutation and shuffles
    within each group -- and it is the property internal consistency cannot see. A recovery
    that re-groups the surviving chunks into fresh runs can be perfectly self-consistent
    while describing blocks the shuffle never laid down, which changes both what is held
    resident together and how far a chunk's residency has to stretch.

    The oracle is :func:`chunk_permutation`, keyed on (seed, epoch) alone and tested in
    `test_shuffle.py` -- a separate function from the one under test, and one that stays
    separate however the block structure comes to be reported.
    """
    p = build_pass(case)
    ids = np.asarray(p.ds.manifest.chunks["train"], dtype=np.int64)
    laid_down = chunk_permutation(ids, seed=p.ds.seed, epoch=0) if case.shuffle else ids
    drawn = {int(c) for c in p.order.rows[:, 0]}  # edge-anchor drop can empty a chunk
    # Group the permutation *first*, then remove what the drop took. Filtering before
    # grouping would repack the survivors into fresh runs -- which is the defect, not the
    # specification: a block that loses a chunk narrows, it does not borrow from its
    # neighbour, and a block that loses all of them disappears.
    groups = [
        set(int(c) for c in laid_down[k : k + case.block_chunks]) & drawn
        for k in range(0, len(laid_down), case.block_chunks)
    ]
    expected = [g for g in groups if g]
    got = [{int(c) for c in block.chunk_ids} for block in p.blocks]
    # Membership, not order within a block: the rows are shuffled together, so which of a
    # block's chunks is named first carries no meaning and the engine reports them by first
    # appearance. The sequence *of* blocks does carry meaning and is compared.
    assert got == expected, (
        f"{len(got)} blocks against {len(expected)} in the permutation this epoch laid down"
    )


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_the_awaited_keys_match_a_per_sample_walk(build_pass, case: Case) -> None:
    """The vectorized residency set equals the one a per-sample walk resolves."""
    p = build_pass(case)
    for bi in range(len(p.blocks)):
        assert p.awaited(bi) == _reference_read_keys(p.rows(bi), p.ds.geometries, p.ref_spc), (
            f"block {bi}: the O(chunks) read-key math disagrees with an O(samples) walk"
        )


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_the_driver_plans_every_chunk_its_consumer_will_wait_on(build_pass, case: Case) -> None:
    """A consumer must never block on a slot the driver did not plan to fill.

    The driver expands each block's chunk ids; the consumer waits on the keys its rows
    Containment is the weaker half of what release-and-re-read needs -- #66 wants equality,
    so that every permit taken comes back.

    It costs nothing today, and the reason is worth stating so nobody weakens it by
    accident: the planner expands each anchor into *every* chunk that anchor reads, so a
    windowed view's spill chunk is fetched alongside the anchor needing it whatever block
    the anchor was attributed to, and the pass-wide dedup plus retention covers the rest.
    Measured over the failing geometries below, no block is fetch-complete later than its
    own anchors -- 0 of 77 on `lead-three-chunks`, against 0 of 32 on the control. The
    property is latent, which is precisely why it needs an assertion rather than a
    benchmark: nothing observable will regress until per-block release makes it fatal.
    """
    p = build_pass(case)
    for bi in range(len(p.blocks)):
        missing = p.awaited(bi) - p.declared(bi)
        assert not missing, (
            f"block {bi}: the consumer waits on {sorted(missing)[:4]}, "
            f"which the driver never planned for this block"
        )


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_every_drawn_chunk_is_planned_exactly_once(build_pass, case: Case) -> None:
    """Across the pass, the driver's plan covers what the consumer reads, without repeats.

    Repeats matter as much as omissions: today's plan dedups over the whole pass, so a
    chunk admitted twice is a reference nobody returns.
    """
    p = build_pass(case)
    ordered = [int(c) for block in p.blocks for c in block.chunk_ids]
    assert len(ordered) == len(set(ordered)), "a chunk is planned by more than one block"
    planned = {
        (r.array, r.chunk_index)
        for r in build_stored_chunk_reads(ordered, p.ds.geometries, p.ref_spc)
    }
    assert _reference_read_keys(p.order.rows, p.ds.geometries, p.ref_spc) <= planned


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_read_ahead_bound_covers_the_simulated_peak(build_pass, case: Case) -> None:
    """The computed bound must be at least the peak a live-set simulation observes.

    Below the peak the pass deadlocks against its own limit; the scheduler now reports that
    rather than hanging, but reporting it is not the same as not doing it.
    """
    p = build_pass(case)
    block_keys = [p.awaited(bi) for bi in range(len(p.blocks))]
    last_use = {key: bi for bi, keys in enumerate(block_keys) for key in keys}
    assert read_ahead_bound(block_keys, last_use) >= _reference_peak(block_keys)
