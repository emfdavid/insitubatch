"""Page-locked batch buffers and the owned H2D transfer that makes recycling them safe.

Pinning is only worth anything because a page-locked source can DMA straight to the device
instead of being staged through a driver bounce buffer -- which is also precisely what makes
the copy genuinely asynchronous, and therefore what makes recycling the source dangerous. So
the two halves are tested together: that buffers really are pinned, and that a batch whose
copy is still in flight is still held.

Most of this needs a CUDA device. What runs anywhere is the budget behaviour, which is where
the interesting policy decision lives: exhausting the budget must *degrade to pageable*, never
block (a consumer legitimately holding batches could deadlock the producer) and never raise
(a performance feature must not become a crash).
"""

from __future__ import annotations

import numpy as np
import pytest

from insitubatch.buffers import XLA_ALIGN, BatchBuffers, aligned_empty
from insitubatch.frameworks import _InFlight

torch = pytest.importorskip("torch")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _ptr(a: np.ndarray) -> int:
    return int(a.__array_interface__["data"][0])


@needs_cuda
def test_pool_buffers_are_actually_pinned() -> None:
    """The whole point: buffers must reach torch still page-locked.

    The path is long -- ``torch.empty(pin_memory=True)`` -> ``.numpy()`` -> pool base ->
    ``base[:k]`` view -> DLPack -> tensor -- and a failure anywhere in it is silent, degrading
    ``non_blocking`` to a synchronous copy with no error and no signal.
    """
    from insitubatch.frameworks import pinned_allocator

    pool = BatchBuffers(allocator=pinned_allocator())
    view = pool.take(4, (3, 2), np.dtype("f4"))
    assert torch.from_dlpack(view).is_pinned()


@needs_cuda
def test_a_ragged_prefix_is_still_pinned() -> None:
    """The short final batch takes a prefix view, which must not lose page-locking."""
    from insitubatch.frameworks import pinned_allocator

    pool = BatchBuffers(allocator=pinned_allocator())
    full = pool.take(8, (3,), np.dtype("f4"))
    del full
    assert torch.from_dlpack(pool.take(3, (3,), np.dtype("f4"))).is_pinned()


@needs_cuda
def test_two_live_pinned_batches_never_share_a_buffer() -> None:
    """Reuse must stay liveness-aware when the buffers are page-locked.

    ``torch.empty(pin_memory=True).numpy()`` hands back an array that *is* its own data owner,
    unlike the aligned-heap default which views one. That difference decides which references
    are live when the pool records a buffer's idle refcount, and getting it wrong makes every
    pinned buffer look permanently free -- so the pool lends one allocation to every batch and
    each overwrites the last, silently, on the pinned path only. The CPU-side analogue is
    ``tests/test_buffers.py::test_liveness_holds_however_the_allocator_owns_its_memory``.
    """
    from insitubatch.frameworks import pinned_allocator

    pool = BatchBuffers(allocator=pinned_allocator())
    first = pool.take(4, (16,), np.dtype("f4"))
    first[:] = 1.0
    second = pool.take(4, (16,), np.dtype("f4"))
    second[:] = 2.0

    assert _ptr(first) != _ptr(second), "the pinned pool lent one buffer to two live batches"
    assert np.all(first == 1.0), "a live pinned batch was overwritten by the next one"


@needs_cuda
def test_in_flight_transfer_holds_the_source_buffer() -> None:
    """A batch whose async copy is still draining must not be recycled underneath it.

    Refcounts cannot see an in-flight DMA, so ``to_torch(device=...)`` holds the source until
    its event fires. Dropping every *visible* reference must therefore still not free the
    buffer for reuse.
    """
    from insitubatch.frameworks import pinned_allocator, to_torch
    from insitubatch.types import Batch

    pool = BatchBuffers(allocator=pinned_allocator())
    view = pool.take(64, (256, 256), np.dtype("f4"))  # big enough to still be in flight
    view[:] = 1.0
    addr = _ptr(view)

    to_torch(Batch(arrays={"x": view}), device="cuda")
    del view  # the caller's references are gone; the transfer's hold is not

    assert _ptr(pool.take(64, (256, 256), np.dtype("f4"))) != addr


@needs_cuda
def test_transferred_batch_has_the_right_values() -> None:
    """Owning the copy must not change what lands on the device."""
    from insitubatch.frameworks import pinned_allocator, to_torch
    from insitubatch.types import Batch

    pool = BatchBuffers(allocator=pinned_allocator())
    view = pool.take(4, (3,), np.dtype("f4"))
    view[:] = np.arange(12, dtype="f4").reshape(4, 3)

    out = to_torch(Batch(arrays={"x": view}), device="cuda")
    torch.cuda.synchronize()
    assert out["x"].device.type == "cuda"
    assert np.array_equal(out["x"].cpu().numpy(), view)


@needs_cuda
def test_budget_exhaustion_degrades_to_pageable(caplog: pytest.LogCaptureFixture) -> None:
    """Past the budget: keep serving, unpinned, and say so once."""
    from insitubatch.frameworks import pinned_allocator

    inner, dtype = (1024,), np.dtype("f4")
    one = 4 * int(np.prod(inner)) * dtype.itemsize  # exactly one buffer's worth
    pool = BatchBuffers(allocator=pinned_allocator(budget_bytes=one))

    first = pool.take(4, inner, dtype)
    second = pool.take(4, inner, dtype)  # over budget -> pageable, not an error
    assert torch.from_dlpack(first).is_pinned()
    assert not torch.from_dlpack(second).is_pinned()
    assert "budget exhausted" in caplog.text


def test_budget_of_zero_never_pins_and_still_serves(caplog: pytest.LogCaptureFixture) -> None:
    """The degrade path itself, testable without a GPU: allocation must still succeed."""
    from insitubatch.frameworks import pinned_allocator

    pool = BatchBuffers(allocator=pinned_allocator(budget_bytes=0))
    view = pool.take(4, (3, 2), np.dtype("f4"))
    view[:] = 3.0  # writable, correctly shaped, simply not pinned
    assert view.shape == (4, 3, 2) and np.all(view == 3.0)
    assert _ptr(view) % XLA_ALIGN == 0  # the fallback keeps XLA alignment
    assert "budget exhausted" in caplog.text


def test_warning_is_emitted_once_not_per_batch(caplog: pytest.LogCaptureFixture) -> None:
    """A per-batch warning would bury a training log."""
    from insitubatch.frameworks import pinned_allocator

    pool = BatchBuffers(allocator=pinned_allocator(budget_bytes=0))
    for _ in range(5):
        del pool._pools  # force a fresh miss each time
        pool._pools = {}
        pool.take(4, (3, 2), np.dtype("f4"))
    assert caplog.text.count("budget exhausted") == 1


def test_to_torch_with_a_cpu_device_needs_no_cuda() -> None:
    """``device="cpu"`` must work on a machine with no driver.

    The guard only exists for an async DMA, and ``torch.cuda.Event()`` raises outright where
    there is no NVIDIA driver -- so a CPU target has to skip it rather than construct one.
    Examples default to ``--device cpu``, which is how this surfaced.
    """
    from insitubatch.frameworks import to_torch
    from insitubatch.types import Batch

    pool = BatchBuffers()
    view = pool.take(4, (3,), np.dtype("f4"))
    view[:] = 2.0
    out = to_torch(Batch(arrays={"x": view}), device="cpu")
    assert out["x"].device.type == "cpu"
    assert np.array_equal(out["x"].numpy(), view)


def test_set_allocator_drops_old_buffers_and_uses_the_new_one() -> None:
    """Swapping allocators must not leave the pool half pinned and half not.

    Asserted by what the pool owns and which allocator it calls -- not by comparing
    addresses, since a freed block is very often handed straight back by ``malloc``.
    """
    calls: list[tuple[int, ...]] = []

    def spy(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        calls.append(shape)
        return aligned_empty(shape, dtype)

    pool = BatchBuffers()
    pool.take(4, (3, 2), np.dtype("f4"))
    assert pool.nbytes > 0

    pool.set_allocator(spy)
    assert pool.nbytes == 0  # the un-swapped buffers are gone, not mixed in
    pool.take(4, (3, 2), np.dtype("f4"))
    assert calls == [(4, 3, 2)]


def test_concurrent_holds_never_drop_a_live_one() -> None:
    """Two threads converting batches at once must not lose each other's holds.

    ``_InFlight`` is process-wide -- two DataLoaders, or ``zip(ds.train, ds.val)``, run this
    concurrently -- and prune-then-append is a read-modify-write. Dropping a hold releases a
    buffer whose DMA is still reading it, so the pool can recycle it mid-copy: silent
    corruption, exactly the class the pool's own lock exists to prevent.

    This is a guard, not a reproducer, and the distinction is worth stating. The unguarded
    window is only the few bytecodes between the comprehension and the slice assignment, and
    it does not reproduce unaided at any thread count -- the comprehension iterates the live
    list by index, so a concurrent append lands where it is still picked up. Widening that
    exact gap with a sleep loses a live hold 20/20. So this test asserts the invariant under
    real contention; the lock is what makes it hold by construction rather than by luck.
    """
    import threading

    class NeverDone:
        """An event whose copy is still in flight -- its hold must never be retired."""

        def query(self) -> bool:
            return False

    flight = _InFlight()
    n = 8
    start = threading.Barrier(n)

    def hold(name: str) -> None:
        start.wait()
        flight.hold({name: np.empty(1, dtype="float32")}, NeverDone())

    threads = [threading.Thread(target=hold, args=(f"b{i}",)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    held = {name for arrays, _ in flight._pending for name in arrays}
    expected = {f"b{i}" for i in range(n)}
    assert held == expected, f"lost holds: {sorted(expected - held)}"


def test_finished_holds_are_retired_so_the_set_stays_bounded() -> None:
    # The other half of the invariant: completed copies must not accumulate, or a long run
    # would pin every batch it ever transferred. One deferred entry is the intended tail.
    class Done:
        def query(self) -> bool:
            return True

    flight = _InFlight()
    for i in range(50):
        flight.hold({f"b{i}": np.empty(1, dtype="float32")}, Done())
    assert len(flight._pending) == 1  # the newest, deliberately retained until the next hold
