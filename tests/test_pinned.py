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
