"""Batch-output buffer reuse: liveness-polled hand-out, and the guarantees it rests on.

:class:`BatchBuffers` recycles a batch's output array instead of allocating a fresh one per
batch, which is only safe because a buffer is handed back *as a view* and reclaimed only once
that view is unreferenced. Two properties carry the whole design and are asserted here: a
dropped view returns its buffer, and a **retained** one never does -- consumers hold batches
for perfectly ordinary reasons (gradient accumulation, ``[b for b in ds]``) and recycling
underneath them would corrupt training data silently.

The framework tests at the bottom guard facts that are load-bearing but not contractual:
``torch.from_dlpack`` and ``jax.device_put`` both keep the source alive, which is what makes
the poll sufficient for an exported tensor. JAX documents in-place mutation of a ``from_dlpack``
buffer as *undefined behaviour*, so if that ever regresses we want a red test here rather than
corrupted batches in a training run.
"""

from __future__ import annotations

import numpy as np
import pytest

from insitubatch.buffers import XLA_ALIGN, BatchBuffers, aligned_empty


def _ptr(a: np.ndarray) -> int:
    """Address of an array's first byte -- identity of the underlying allocation."""
    return int(a.__array_interface__["data"][0])


def test_dropped_view_is_reused() -> None:
    """The point of the pool: back-to-back batches share one allocation."""
    pool = BatchBuffers()
    first = pool.take(4, (3, 2), np.dtype("f4"))
    addr = _ptr(first)
    del first
    assert _ptr(pool.take(4, (3, 2), np.dtype("f4"))) == addr


def test_retained_view_is_never_reused() -> None:
    """A held batch must keep its memory -- this is the invariant that makes reuse safe."""
    pool = BatchBuffers()
    held = pool.take(4, (3, 2), np.dtype("f4"))
    held[:] = 1.0
    second = pool.take(4, (3, 2), np.dtype("f4"))
    second[:] = 2.0
    assert _ptr(second) != _ptr(held)
    assert np.all(held == 1.0)  # the retained batch is untouched


def test_a_derived_view_keeps_its_buffer() -> None:
    """A *slice* of a lent batch must hold the buffer, not just the lent array itself.

    numpy collapses base chains: ``lent[..., i:j].base`` is the ultimate data owner, **not**
    ``lent``. So tracking liveness by the lent object alone frees the buffer the moment a
    consumer keeps a crop and drops the original -- which is precisely what a cropping
    ``batch_transform`` does (``examples/wb2_arraylake.py``, and the guidance in
    docs/architecture.md), and it corrupts training data silently.
    """
    pool = BatchBuffers()
    lent = pool.take(4, (8, 8), np.dtype("f4"))
    lent[:] = 1.0
    crop = lent[..., 0:4, 0:4]  # a transform's output, aliasing the buffer
    del lent  # the consumer keeps only the crop

    other = pool.take(4, (8, 8), np.dtype("f4"))
    other[:] = 9.0
    assert np.all(crop == 1.0), "the pool recycled a buffer a live view still aliases"


def test_a_derived_view_of_a_ragged_prefix_keeps_its_buffer() -> None:
    """Same, for the short final batch -- a prefix view is where the tail lives."""
    pool = BatchBuffers()
    lent = pool.take(3, (8, 8), np.dtype("f4"))
    lent[:] = 2.0
    crop = lent[..., 2:6, 2:6]
    del lent

    pool.take(3, (8, 8), np.dtype("f4"))[:] = 7.0
    assert np.all(crop == 2.0)


def test_pool_grows_to_the_in_flight_count_then_stops() -> None:
    """No depth parameter: allocate-on-miss converges on however many are live at once."""
    pool = BatchBuffers()
    live = [pool.take(4, (3, 2), np.dtype("f4")) for _ in range(3)]
    addrs = {_ptr(a) for a in live}
    assert len(addrs) == 3  # three live at once -> three buffers
    del live
    # Nothing is live now, so the next three hand-outs must come from those same three.
    again = [pool.take(4, (3, 2), np.dtype("f4")) for _ in range(3)]
    assert {_ptr(a) for a in again} == addrs


def test_ragged_tail_reuses_the_full_buffer_as_a_prefix() -> None:
    """A short final batch is a prefix view of a full-size buffer, not a new allocation."""
    pool = BatchBuffers()
    full = pool.take(8, (3,), np.dtype("f4"))
    addr = _ptr(full)
    del full
    tail = pool.take(3, (3,), np.dtype("f4"))
    assert tail.shape == (3, 3)
    assert _ptr(tail) == addr  # same allocation, shorter view


def test_a_larger_request_does_not_alias_a_smaller_buffer() -> None:
    """Growing past a buffer's capacity must allocate, never over-run it."""
    pool = BatchBuffers()
    small = pool.take(2, (3,), np.dtype("f4"))
    addr = _ptr(small)
    del small
    big = pool.take(8, (3,), np.dtype("f4"))
    assert big.shape == (8, 3)
    assert _ptr(big) != addr


@pytest.mark.parametrize(
    ("inner", "dtype"),
    [((3, 2), "f4"), ((3,), "f4"), ((3, 2), "f8")],
)
def test_shape_and_dtype_do_not_collide(inner: tuple[int, ...], dtype: str) -> None:
    """Buffers are keyed by geometry: a batch never gets another variable's memory."""
    pool = BatchBuffers()
    a = pool.take(4, (3, 2), np.dtype("f4"))
    b = pool.take(4, inner, np.dtype(dtype))
    assert b.shape == (4, *inner) and b.dtype == np.dtype(dtype)
    if (inner, dtype) != ((3, 2), "f4"):
        assert _ptr(a) != _ptr(b)


def test_buffers_are_xla_aligned() -> None:
    """128-byte alignment is what makes ``to_jax`` reliably zero-copy rather than lucky."""
    pool = BatchBuffers()
    for _ in range(8):  # several allocations: default numpy alignment passes only by chance
        assert _ptr(pool.take(4, (3, 2), np.dtype("f4"))) % XLA_ALIGN == 0


def test_aligned_empty_is_a_real_writable_array() -> None:
    """The alignment trick must not hand back a read-only or mis-shaped view."""
    a = aligned_empty((4, 3), np.dtype("f4"))
    a[:] = 7.0
    assert a.shape == (4, 3) and a.dtype == np.dtype("f4") and np.all(a == 7.0)
    assert _ptr(a) % XLA_ALIGN == 0


def test_a_custom_allocator_is_used_for_every_buffer() -> None:
    """The injection point pinning arrives through (the torch adapter supplies a pinned one)."""
    calls: list[tuple[tuple[int, ...], np.dtype]] = []

    def spy(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        calls.append((shape, dtype))
        return aligned_empty(shape, dtype)

    pool = BatchBuffers(allocator=spy)
    pool.take(4, (3, 2), np.dtype("f4"))
    assert calls == [((4, 3, 2), np.dtype("f4"))]


def test_iteration_recycles_buffers_and_still_yields_correct_data(write_zarr) -> None:  # type: ignore[no-untyped-def]
    """End to end: a real epoch reuses a bounded set of buffers *and* stays correct.

    Correctness is the half that matters -- recycling is only worth anything if a reused
    buffer is fully overwritten, so every batch is checked against the source array. The
    address count proves the wiring actually reached ``gather``; asserting merely that the
    data is right would pass just as well with the old fresh-allocation path.
    """
    from insitubatch import InSituDataset, obstore_store, open_geometries, split_by_chunk

    url, srcs = write_zarr(n=40, spc=8)
    geom = open_geometries(obstore_store(url))["t2m"]
    ds = InSituDataset(
        obstore_store(url),
        split_by_chunk(geom, fractions=(1.0, 0.0, 0.0)),
        shuffle=False,
        batch_size=5,
        block_chunks=2,
    )
    ds.set_epoch(0)

    addrs = set()
    for batch in ds.train:
        addrs.add(_ptr(batch.arrays["t2m"]))
        expected = srcs["t2m"][batch.sample_indices]
        assert np.array_equal(batch.arrays["t2m"], expected)
    # 8 batches drawn, but only the handful actually in flight at once get allocated.
    assert 0 < len(addrs) < 8


def test_a_retained_batch_survives_later_iteration(write_zarr) -> None:  # type: ignore[no-untyped-def]
    """Keeping batches while iterating -- ``[b for b in ds]`` -- must not corrupt them."""
    from insitubatch import InSituDataset, obstore_store, open_geometries, split_by_chunk

    url, srcs = write_zarr(n=40, spc=8)
    geom = open_geometries(obstore_store(url))["t2m"]
    ds = InSituDataset(
        obstore_store(url),
        split_by_chunk(geom, fractions=(1.0, 0.0, 0.0)),
        shuffle=False,
        batch_size=5,
        block_chunks=2,
    )
    ds.set_epoch(0)

    kept = list(ds.train)  # every batch held at once -> no buffer may be recycled
    for batch in kept:
        assert np.array_equal(batch.arrays["t2m"], srcs["t2m"][batch.sample_indices])
    assert len({_ptr(b.arrays["t2m"]) for b in kept}) == len(kept)  # all distinct


def test_early_break_leaves_no_buffer_stranded(write_zarr) -> None:  # type: ignore[no-untyped-def]
    """Abandoning an epoch mid-iteration must not strand buffers as permanently 'lent'.

    This is the shape of bug that forced ``unpin_all()`` for *chunk* residency pins, where an
    early ``break`` leaks the read-ahead's pins into the next epoch. Liveness-based reclaim is
    immune by construction -- an abandoned batch is an unreferenced view, so its buffer is free
    again with no epoch-boundary reset to remember -- and this pins that down.
    """
    from insitubatch import InSituDataset, obstore_store, open_geometries, split_by_chunk

    url, srcs = write_zarr(n=40, spc=8)
    geom = open_geometries(obstore_store(url))["t2m"]
    ds = InSituDataset(
        obstore_store(url),
        split_by_chunk(geom, fractions=(1.0, 0.0, 0.0)),
        shuffle=False,
        batch_size=5,
        block_chunks=2,
    )

    ds.set_epoch(0)
    for _ in ds.train:  # take one batch, walk away
        break

    for epoch in range(1, 5):  # further epochs stay correct and do not accumulate
        ds.set_epoch(epoch)
        for batch in ds.train:
            assert np.array_equal(batch.arrays["t2m"], srcs["t2m"][batch.sample_indices])

    # 8 batches per epoch, 4 epochs since the break: a pool that stranded anything would
    # have grown with the iteration. It is bounded by what is in flight instead.
    assert ds._pool._buffers.nbytes < 8 * (5 * 2 * 2 * 4)


def test_exported_torch_tensor_holds_the_buffer() -> None:
    """A live DLPack export must block reuse -- torch keeps a strong ref to the source."""
    torch = pytest.importorskip("torch")
    pool = BatchBuffers()
    view = pool.take(4, (3, 2), np.dtype("f4"))
    addr = _ptr(view)
    tensor = torch.from_dlpack(view)
    tensor[:] = 5.0
    del view  # only the tensor holds it now
    assert _ptr(pool.take(4, (3, 2), np.dtype("f4"))) != addr
    assert torch.all(tensor == 5.0)


def test_exported_jax_array_holds_the_buffer() -> None:
    """Same for JAX, where recycling under a live array is documented undefined behaviour."""
    jnp = pytest.importorskip("jax.numpy")
    pool = BatchBuffers()
    view = pool.take(4, (3, 2), np.dtype("f4"))
    view[:] = 5.0
    addr = _ptr(view)
    arr = jnp.from_dlpack(view)
    del view
    assert _ptr(pool.take(4, (3, 2), np.dtype("f4"))) != addr
    assert bool((arr == 5.0).all())
