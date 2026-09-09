"""The fetch-ahead bound is computed, and every term in it is load-bearing.

Bounded read-ahead only works if the bound is right: below it the pass deadlocks against
its own limit (the driver cannot admit the chunk its consumer is blocked on), and above it
admission is bounded only by the pool -- which is what lets one iteration claim the whole
budget and starve another.

Each case here is a configuration that falsified a simpler rule. They are written against
the pure function rather than a running pass because each was originally found as a *hang*,
and a test whose failure mode is "the suite stops" is a bad trade for CI. The one test that
does drive a real pass is the harness at the bottom: it exists to show the guard can still
catch the broken case, since a bound that is never wrong is indistinguishable from a bound
that is never checked.
"""

from __future__ import annotations

import threading

from insitubatch import obstore_store, open_geometries, split_by_chunk
from insitubatch.scheduler import SchedulerConfig
from insitubatch.source import InSituDataset, read_ahead_bound


def _last_use(block_keys: list[set[tuple[str, int]]]) -> dict[tuple[str, int], int]:
    last: dict[tuple[str, int], int] = {}
    for bi, keys in enumerate(block_keys):
        for key in keys:
            last[key] = bi
    return last


def _blocks(*groups: list[int]) -> list[set[tuple[str, int]]]:
    return [{("t2m", c) for c in group} for group in groups]


def test_plain_blocks_need_the_block_plus_one_ahead() -> None:
    """Disjoint blocks of 4: the consumer holds one while the driver works the next."""
    block_keys = _blocks([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11])

    assert read_ahead_bound(block_keys, _last_use(block_keys)) == 8


def test_a_chunk_read_by_several_blocks_stays_counted_until_its_last() -> None:
    """A windowed read spans blocks; its permit is held from first admission to last use.

    Chunk 99 is read by blocks 0 and 2, so it is live across block 1 as well -- a rule
    counting only two blocks' worth would not see it.
    """
    block_keys = _blocks([0, 1, 99], [2, 3], [4, 5, 99])
    bound = read_ahead_bound(block_keys, _last_use(block_keys))

    naive_two_blocks = 3 + 2  # |block 0| + |block 1|
    assert bound > naive_two_blocks, "the chunk held across block 1 must be counted"


def test_a_finer_chunked_variable_contributes_more_than_block_chunks() -> None:
    """`block_chunks` is the knob, not the count.

    With a variable chunked finer than the reference grid, one anchor chunk maps onto
    several of its chunks, so any per-variable multiple of `block_chunks` under-counts.
    """
    block_keys = [
        {("coarse", 0), ("fine", 0), ("fine", 1), ("fine", 2), ("fine", 3)},
        {("coarse", 1), ("fine", 4), ("fine", 5), ("fine", 6), ("fine", 7)},
    ]
    bound = read_ahead_bound(block_keys, _last_use(block_keys))

    block_chunks, n_variables = 1, 2
    assert bound > block_chunks * n_variables * 2


def test_the_bound_is_never_below_one_block_and_its_successor() -> None:
    """The floor, asserted directly: whatever else is live, both blocks must fit."""
    block_keys = _blocks([0, 1, 2, 3, 4], [5, 6])

    assert read_ahead_bound(block_keys, _last_use(block_keys)) >= 5 + 2


def test_a_single_block_still_admits_something() -> None:
    """Degenerate plans must not produce a zero bound -- that admits nothing, forever."""
    assert read_ahead_bound([], {}) >= 1
    assert read_ahead_bound(_blocks([0]), _last_use(_blocks([0]))) >= 1


# -- the harness: can we still catch a wrong bound? --------------------------


def test_an_undersized_bound_raises_instead_of_wedging(write_zarr) -> None:
    """A permit deficit must be diagnosable, not a silent hang.

    Every other wait in the scheduler is bounded and proves itself terminal before giving
    up -- that is what turns a stall into `residency budget exhausted` with a reason. The
    fetch-ahead permit is a second thing a driver can wait on forever, so it needs the same
    treatment: without it, any miscount in the permit accounting presents as three idle
    threads and no message, which is the failure mode the starvation detector exists to
    replace.

    `read_ahead_chunks` is not a user knob -- there is no value a caller could choose that
    beats the computed one. It stays settable so this test can supply a deficit at all.
    """
    url, _ = write_zarr(n=128, spc=4, inner=(4, 4))
    geometries = open_geometries(obstore_store(url))
    manifest = split_by_chunk(geometries["t2m"], fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        obstore_store(url),
        manifest,
        geometries=geometries,
        batch_size=8,
        block_chunks=4,
        shuffle=True,
        seed=0,
    )
    ds.scheduler_config = SchedulerConfig(
        max_inflight=ds.scheduler_config.max_inflight, read_ahead_chunks=2
    )
    ds.set_epoch(0)

    result: dict[str, object] = {}

    def body() -> None:
        try:
            for _batch in ds.train:
                pass
            result["outcome"] = "completed"
        except Exception as exc:  # noqa: BLE001 - the outcome is what is under test
            result["outcome"] = exc

    worker = threading.Thread(target=body, daemon=True)
    worker.start()
    worker.join(30)

    assert not worker.is_alive(), "an undersized bound hung instead of reporting"
    assert isinstance(result["outcome"], Exception), f"expected a raise, got {result['outcome']}"
    assert "read-ahead" in str(result["outcome"]), (
        f"the error must name the fetch-ahead bound, not something else: {result['outcome']}"
    )
