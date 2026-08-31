"""Slot lifecycle: "safe to take away" must be true, not approximately true.

Regression tests for #33/#34/#35 -- three ways the pool's eviction predicate used to
lie. They are grouped in one file because they shared a root cause: "may this slot be
taken away" was spread across five loosely-coupled fields (``remaining``, ``ready``,
``claimed``, ``error``, ``_pinned``), and each issue was a different field lying.

The predicate now lives in one place -- ``ChunkPool._evictable`` over one ``SlotState``
plus two counters -- so these tests ask *it* rather than re-implementing it, and a
future change that reintroduces a second lever fails here.

They are written to fail *loudly and for the stated reason* -- a wrong answer or a
stale raise, never a hang -- so a regression is legible. End-to-end cases run under the
``run_by`` deadline for that reason.

Why byte fingerprints rather than "it ran": the failure mode these guard against is a
consumer reading a slot that was recycled or evicted underneath it. That produces
*plausible* data, which throughput, shapes, and smoke tests all pass. Only comparing
delivered bytes against a single-producer reference catches it.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pytest

from insitubatch import obstore_store, open_geometries, split_by_chunk
from insitubatch.pool import ChunkPool
from insitubatch.source import InSituDataset

DEADLINE = 30.0
CHUNK_BYTES = 8 * 2 * 2 * 4  # spc * inner * f4, for the write_zarr defaults used here


def _dataset(url: str, **kwargs: Any) -> InSituDataset:
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    return InSituDataset(obstore_store(url), manifest, **kwargs)


def _fingerprint(batches: list) -> str:
    """SHA-256 over delivered bytes, in sample order (shuffle-independent)."""
    idx = np.concatenate([b.sample_indices for b in batches])
    out = np.concatenate([b.arrays["t2m"] for b in batches])[np.argsort(idx)]
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(out).tobytes())
    h.update(np.sort(idx).astype(np.int64).tobytes())
    return h.hexdigest()


def _chunk_path(url: str, index: int, inner_dims: int) -> Path:
    return Path(urlparse(url).path).joinpath("t2m", "c", str(index), *(["0"] * inner_dims))


# -- #33: fail() marks a slot ready while tile tasks are still writing ------------


def test_failed_chunk_does_not_poison_the_next_epoch(write_zarr, run_by) -> None:
    """A transient bad chunk must not wedge every later epoch.

    ``fail()`` sets ``ready = True`` (``pool.py:651-656``) so a waiter stops hanging.
    But ``ready`` is also "safe to cache", so the poisoned slot survives the epoch
    boundary -- ``unpin_all`` drops only ``not slot.ready`` (``pool.py:531``). Next
    epoch ``pin_if_ready`` rejects it (``:420``) but ``try_admit`` takes the resident
    branch (``:375``) and returns True *without refetching*, so ``wait_ready``
    re-raises the stale error forever.

    Here the corruption is repaired between epochs, so epoch 1 has nothing wrong with
    it and must succeed. Pre-fix it re-raises epoch 0's error.
    """
    url, srcs = write_zarr(n=40, spc=8, inner=(2, 2))
    chunk = _chunk_path(url, 1, 2)
    assert chunk.exists(), f"chunk file not found: {chunk}"
    good_bytes = chunk.read_bytes()
    chunk.write_bytes(b"\x00\x01\x02\x03")

    ds = _dataset(url, shuffle=False, batch_size=8)
    ds.set_epoch(0)
    with pytest.raises(Exception):  # noqa: B017 - codec-specific decode failure
        list(ds.train)

    chunk.write_bytes(good_bytes)  # the store is healthy again

    # Discriminator: a FRESH pool reads the repaired store fine, so any failure below
    # is the stale slot, not the data. Without this the test could pass or fail for
    # the wrong reason (a repair that silently did not take).
    fresh = _dataset(url, shuffle=False, batch_size=8)
    fresh.set_epoch(0)
    assert run_by(DEADLINE, lambda: list(fresh.train)), "repair did not take"

    ds.set_epoch(1)
    try:
        batches = run_by(DEADLINE, lambda: list(ds.train))
    except Exception as exc:  # noqa: BLE001 - the regression IS the stale raise
        pytest.fail(
            "epoch 1 re-raised the poisoned slot's stored error instead of refetching "
            f"the (now repaired) chunk: {exc!r}"
        )
    assert ds.cache_misses > 0, (
        "epoch 1 served the poisoned chunk from the resident slot without refetching: "
        "try_admit took the resident branch (pool.py:375) and wait_ready re-raised the "
        "cached exception object"
    )
    idx = np.concatenate([b.sample_indices for b in batches])
    out = np.concatenate([b.arrays["t2m"] for b in batches])[np.argsort(idx)]
    np.testing.assert_array_equal(out, srcs["t2m"])


def test_failed_slot_is_not_evictable_while_tiles_are_outstanding(write_zarr) -> None:
    """``fail()`` must not make a half-written slot an eviction candidate.

    Unit-level statement of the invariant: after ``fail()``, sibling tile tasks may
    still be scattering into the slot (``scheduler.py:429-459``), so it is not
    quiescent and must not be selectable by ``_make_room``. Today ``ready`` is set
    regardless of ``remaining``, so it is selectable.
    """
    geoms = open_geometries(obstore_store(write_zarr(n=40, spc=8, inner=(2, 2))[0]))
    pool = ChunkPool(geoms, budget_bytes=None)
    owner = pool.new_owner()
    key = ("t2m", 0)
    pool.try_admit("t2m", 0, owner)
    pool.unpin_keys({key}, owner)  # drop the admit ref: only quiescence protects it now
    slot = pool._slots[key]

    # An open write scope IS a sibling tile task still running: `writers` counts STARTED
    # tasks, so the scope is what makes the slot non-quiescent.
    coord = next(iter(geoms["t2m"].inner_coords()))
    with pool.tile_write("t2m", 0, coord):
        assert slot.writers > 0, "the open scope must register an outstanding writer"
        pool.fail("t2m", 0, ValueError("a sibling tile failed"))
        # Ask the real predicate, not a copy of it: the whole point of this work is that
        # "may this be taken away" is answered in exactly one place.
        assert not pool._evictable(key, slot), (
            "a failed slot with an outstanding tile task is quiescent-unsafe: the "
            "sibling is still writing into it, so it must not be evictable"
        )
    assert slot.error is not None, "the error must be recorded so a waiter can re-raise"
    assert not pool.is_ready(*key), "a poisoned chunk must never be published"


# -- #34/#35: references are per owner ------------------------------------------


def test_one_owners_teardown_leaves_another_owners_pins_intact(write_zarr) -> None:
    """Releasing one iteration's references must not touch another's.

    The old ``unpin_all()`` cleared the whole map, so with two producers over one pool
    -- a supported configuration (``buffers.py:238-247``) -- one iteration's epoch
    boundary stripped the other's pins and its in-use chunks became eviction
    candidates mid-gather.

    Stated at the pool rather than end to end on purpose. The end-to-end version is a
    race: a live iteration's producer keeps draining in the background and releases
    those same pins on its own, so "A's pins vanished" cannot distinguish the bug from
    A's normal progress (measured: they vanish within ~0.3 s whether or not B ever
    starts). The byte-level statement is the fingerprint test below.
    """
    geoms = open_geometries(obstore_store(write_zarr(n=40, spc=8, inner=(2, 2))[0]))
    pool = ChunkPool(geoms, budget_bytes=None)
    a, b = pool.new_owner(), pool.new_owner()
    shared, a_only, b_only = ("t2m", 0), ("t2m", 1), ("t2m", 2)

    pool.pin_keys({shared, a_only}, a)
    pool.pin_keys({shared, b_only}, b)
    assert pool._refs(shared) == 2, "both owners reference the shared chunk"

    pool.release_owner(b)

    assert pool._refs(a_only) == 1, "A's exclusive pin must survive B's teardown"
    assert pool._refs(shared) == 1, "only B's reference to the shared chunk may go"
    assert pool._refs(b_only) == 0, "B's own pins must be released"

    pool.release_owner(a)
    assert pool._pinned == {}, "the last owner's teardown clears the map"


def test_another_owners_reference_does_not_satisfy_this_owners_wait(write_zarr) -> None:
    """#35: ``claimed`` was one bool, so one iteration's claim satisfied another's wait.

    ``wait_ready`` must be satisfied only by *this* owner's reference. With a shared
    flag A could gather a chunk it never referenced -- and A's later release then
    decremented B's count, making B's in-use chunk evictable underneath it.
    """
    geoms = open_geometries(obstore_store(write_zarr(n=40, spc=8, inner=(2, 2))[0]))
    geom = geoms["t2m"]
    pool = ChunkPool(geoms, budget_bytes=None)
    a, b = pool.new_owner(), pool.new_owner()

    pool.try_admit("t2m", 0, b)  # only B fills and references chunk 0
    for ic in geom.inner_coords():
        pool.deliver_tile("t2m", 0, ic, np.zeros(geom.chunks, dtype=geom.dtype))
    pool.wait_ready("t2m", 0, b)  # B's own wait is satisfied

    waited = threading.Event()

    def wait_as_a() -> None:
        pool.wait_ready("t2m", 0, a)
        waited.set()

    t = threading.Thread(target=wait_as_a, daemon=True)
    t.start()
    assert not waited.wait(timeout=0.5), (
        "A's wait_ready returned on a chunk A never referenced: B's claim satisfied it, "
        "so A would gather a chunk the driver has not pinned for A"
    )
    pool.pin_keys({("t2m", 0)}, a)  # now A references it too
    assert waited.wait(timeout=5), "A's own reference must satisfy A's wait"
    t.join(timeout=5)


# -- #34 + #35 end to end --------------------------------------------------------


def test_concurrent_iterations_each_deliver_correct_data(write_zarr, run_by) -> None:
    """Two interleaved iterations over one pool must each deliver correct bytes.

    The end-to-end statement of #34 (global unpin) and #35 (``claimed`` is one bool,
    so one iteration's claim satisfies another's ``wait_ready``). The budget is tight
    enough that admissions must evict, which is what turns a lost pin into a slot
    recycled underneath a live reader.
    """
    url, _ = write_zarr(n=256, spc=8, inner=(2, 2))  # 32 chunks: room to evict

    ref_ds = _dataset(url, shuffle=False, batch_size=8, block_chunks=2)
    ref_ds.set_epoch(0)
    expected = _fingerprint(list(ref_ds.train))

    # Tight enough that admissions must evict (32 chunks total), loose enough to hold
    # both iterations' working sets (~4 chunks each) so a starvation raise cannot
    # masquerade as the bug under test.
    ds = _dataset(
        url,
        shuffle=False,
        batch_size=8,
        block_chunks=2,
        # Sized for BOTH iterations: two owners each need their working set resident at
        # once. The old global unpin hid this by spuriously freeing the other's pins, so
        # a budget that "worked" before was relying on the bug. Measured: two concurrent
        # iterations over this store deadlock below 32 chunks; one iteration does not.
        cache_budget_bytes=32 * CHUNK_BYTES,
    )
    ds.set_epoch(0)

    def interleaved() -> tuple[str, str]:
        a, b = iter(ds.train), iter(ds.train)
        interleaved._a, interleaved._b = a, b  # so teardown can close them deterministically
        got_a, got_b = [], []
        while True:
            done = 0
            for it, sink in ((a, got_a), (b, got_b)):
                try:
                    sink.append(next(it))
                except StopIteration:
                    done += 1
            if done == 2:
                break
        return _fingerprint(got_a), _fingerprint(got_b)

    try:
        fp_a, fp_b = run_by(DEADLINE, interleaved)
    finally:
        for it in (getattr(interleaved, "_a", None), getattr(interleaved, "_b", None)):
            if it is not None:
                it.close()
    assert fp_a == expected, "first iteration delivered different bytes when interleaved"
    assert fp_b == expected, "second iteration delivered different bytes when interleaved"
