"""Store reads are a function of the plan -- never of ``batch_size`` or shuffle.

This is the "O(chunks), not O(samples)" invariant made falsifiable. It is the whole
performance thesis, and until now nothing in the suite would have caught its loss: a
per-sample read path is not slower on a fixture this small, it is just *wrong*, and
throughput tests on 8 chunks cannot see it.

The assertion is deliberately **not** "amplification is 1.0x", which is false in both
directions. A warm epoch reads *fewer* chunks than the plan names, because the pool
retains them across the epoch boundary (measured: 8, then 6, then 5). A sharded array
may take several byte-range calls for one shard key without anything being wrong.
What survives both is an *independence* property, asserted three ways below, plus a
control proving the instrument can see the failure it is looking for.

Held fixed, because each one would legitimately change the count:

- **local store** -- a remote retry is an honest extra read
- **one variable** -- reads are counted per array, not globally
- **unsharded** -- count distinct keys, not calls, once #57 lands partial shard reads
- **no cache_dir** -- persistence makes a second epoch read the local cache, not the
  store; that is a different claim and deserves its own test
- **single epoch** for the first two tests; v1 sample geometry has no cross-chunk
  samples, so windowing does not enter yet -- it would if that ever changes
"""

from __future__ import annotations

import collections
from typing import Any

import pytest
import zarr
from zarr.storage import WrapperStore

from insitubatch import obstore_store, open_geometries, split_by_chunk
from insitubatch.source import InSituDataset

N, SPC = 64, 8
CHUNKS = N // SPC

# Module-level on purpose. Opening a store read-only makes zarr *copy* it, so a counter
# kept on the instance the test holds would stay empty forever -- and an empty counter
# reads as "no redundant reads at all", which is indistinguishable from a pass. That trap
# is why ``test_the_control_amplifies`` exists.
READS: collections.Counter[str] = collections.Counter()


class CountingStore(WrapperStore):  # type: ignore[type-arg]
    """Counts chunk reads by key.

    Overrides only the async ``get``, which is the seam zarr reads through today. If a
    future zarr moves these reads to the sync seam this wrapper goes blind -- and
    ``test_the_control_amplifies`` is what turns that into a loud failure rather than a
    suite full of vacuous passes.
    """

    async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
        READS[key] += 1
        return await self._store.get(key, prototype, byte_range=byte_range)


def chunk_reads() -> int:
    return sum(v for k, v in READS.items() if "/c/" in k)


@pytest.fixture
def counted(write_zarr):
    """A one-variable store whose chunk reads are counted, plus its split manifest."""
    url, _ = write_zarr(n=N, spc=SPC)

    def _open() -> Any:
        return CountingStore(obstore_store(url))

    manifest = split_by_chunk(open_geometries(_open())["t2m"], fractions=(1.0, 0.0, 0.0))
    return _open, manifest


def _epoch(ds: InSituDataset) -> int:
    return sum(int(b.arrays["t2m"].shape[0]) for b in ds.train)


# -- 1. the hand-computed case ------------------------------------------------


def test_a_cold_epoch_reads_each_chunk_exactly_once(counted) -> None:
    """64 samples, 8 per chunk, so 8 chunks and 8 reads.

    The 8 is computed by hand, not asked of the planner: a test that takes both sides
    of the comparison from the code under test asserts only that it agrees with itself.
    """
    open_store, manifest = counted
    ds = InSituDataset(open_store(), manifest, batch_size=8, shuffle=False, block_chunks=2)
    READS.clear()
    assert _epoch(ds) == N
    assert chunk_reads() == CHUNKS


# -- 2. the property that survives config changes -----------------------------


@pytest.mark.parametrize("batch_size", [1, 3, 8, 16, 64])
@pytest.mark.parametrize("shuffle", [False, True])
def test_reads_do_not_scale_with_batch_size_or_shuffle(counted, batch_size, shuffle) -> None:
    """The invariant with no magic number in it -- and the one that catches a
    per-sample read path the moment it appears.

    ``batch_size=64`` is included because it exceeds one block (2 chunks = 16 samples)
    and forces the block to widen; that changes residency, and must not change reads.
    """
    open_store, manifest = counted
    ds = InSituDataset(
        open_store(), manifest, batch_size=batch_size, shuffle=shuffle, block_chunks=2
    )
    READS.clear()
    assert _epoch(ds) == N
    assert chunk_reads() == CHUNKS


# -- 3. reuse may only help ---------------------------------------------------


def test_a_warm_epoch_never_reads_more_than_a_cold_one(counted) -> None:
    """An inequality, not an equality: chunks still resident from the previous epoch
    are reused, so a warm epoch legitimately reads *fewer* than the plan names.
    Asserting equality here would go red on a correctly working pool.
    """
    open_store, manifest = counted
    ds = InSituDataset(open_store(), manifest, batch_size=8, shuffle=True, block_chunks=2)

    ds.set_epoch(0)
    READS.clear()
    _epoch(ds)
    cold = chunk_reads()

    warm = []
    for epoch in (1, 2):
        ds.set_epoch(epoch)
        READS.clear()
        _epoch(ds)
        warm.append(chunk_reads())

    assert cold == CHUNKS
    assert all(w <= cold for w in warm), f"a warm epoch read more than the cold one: {warm}"


# -- 4. the control: can this instrument see amplification at all? ------------


def test_the_control_amplifies(counted) -> None:
    """Drive a case *known* to amplify through the same counter.

    Reading one sample at a time straight from zarr must cost one chunk read per
    sample -- ``SPC`` times the ideal. If this ever reports 1.0x, the counter is not
    wired to the reads and every other assertion in this file is vacuous. Starving the
    pool is no longer available as a control: an under-floor ``cache_budget_bytes``
    raises at construction (#56), so the amplifying case cannot be built through the
    public API any more.
    """
    open_store, _ = counted
    arr = zarr.open_group(open_store(), mode="r")["t2m"]

    READS.clear()
    for i in range(N):
        _ = arr[i]

    assert chunk_reads() == N, "the counter is not seeing chunk reads"
    assert chunk_reads() == SPC * CHUNKS
