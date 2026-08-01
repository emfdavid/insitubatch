"""Reusable batch-output buffers.

``pool.gather`` used to allocate a fresh array per variable per batch. The cost is not the
allocation call -- ``np.empty`` only reserves address space -- but what glibc does with a
request above its 32 MiB dynamic ``mmap`` threshold: it ``mmap``s the batch and ``munmap``s it
on free, **every batch**. A syscall trace confirms it directly (128 MiB batches, 12-batch
epoch): 12 ``mmap``/``munmap`` pairs of the full buffer without the pool, 2 with it, and the
count tracks batches rather than dataset size. Paid per batch, that is a fresh page set to
fault in plus ``munmap``'s TLB work.

Below the threshold the freed block is recycled on the heap instead and reuse buys nothing --
it costs about 2-3%. End to end (``bench/batch_buffer_sweep.py``) the pool is *flat* across a
16x payload range while fresh allocation falls off a cliff at 32 MiB: 8 MiB -2.4%, 32 MiB
+16%, 64 MiB +13%, 128 MiB +10%. So this removes a cliff rather than adding speed. Weather
batches sit well below it; ViT and microscopy batches sit above.

**Hand-out is by view, reclaim is by liveness.** A buffer is lent as a view of a base array the
pool owns forever, and comes back only once that view is unreferenced. The check happens
lazily, inside :meth:`take` -- the latest moment it can, and exactly the moment the answer
matters -- so there is no separate reclaim pass and nothing for a caller to remember to call.

Three consequences worth stating, because they are what make this safe rather than clever:

* **Retaining a batch is retaining its buffer**, automatically. A consumer that keeps a batch
  (gradient accumulation, ``[b for b in ds]``, an exported tensor) keeps its view alive, so the
  pool simply allocates elsewhere. There is deliberately no ``retain()`` API: holding a
  reference already does it, and a ``retain()`` someone forgets is a silent-corruption bug
  where a held reference is safe by construction.
* **There is no depth parameter.** Allocate-on-miss converges on however many buffers are
  genuinely in flight, so nobody has to derive ``prefetch_depth`` + n correctly.
* **A short final batch is a prefix view** (``base[:k]``) of a full-size buffer -- the ragged
  tail costs neither an allocation nor a copy.

Buffers are 128-byte aligned because XLA:CPU zero-copies a DLPack import only from that
boundary and silently copies otherwise; numpy guarantees 16, which made ``to_jax`` zero-copy
by allocator luck (measured 25/50) rather than by construction. Alignment is also free for the
pinned path, since page-locked memory is page-aligned.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable

import numpy as np

XLA_ALIGN = 128
"""Alignment XLA:CPU requires to import a DLPack buffer without copying."""

HostAllocator = Callable[[tuple[int, ...], np.dtype], np.ndarray]
"""How the pool gets host memory. The default is aligned heap; the torch adapter swaps in a
page-locked one, which is how pinning arrives without the core importing a framework."""


def aligned_empty(shape: tuple[int, ...], dtype: np.dtype, align: int = XLA_ALIGN) -> np.ndarray:
    """``np.empty(shape, dtype)`` whose first byte is a multiple of ``align``.

    Over-allocate a byte buffer, skip to the next boundary, then view and reshape. numpy makes
    no alignment promise past its own 16-byte floor, so this is the only way to get the
    guarantee rather than the coin flip.
    """
    nbytes = int(np.prod(shape)) * dtype.itemsize
    raw = np.empty(nbytes + align, dtype=np.uint8)
    offset = (-int(raw.__array_interface__["data"][0])) % align
    return raw[offset : offset + nbytes].view(dtype).reshape(shape)


class _Buffer:
    """One allocation, tracked by how many arrays still reference its data owner."""

    __slots__ = ("base", "owner", "_idle_refs")

    def __init__(self, base: np.ndarray) -> None:
        self.base = base
        # Every array viewing this memory -- the batch we lend, a transform's crop of it, a
        # torch/JAX tensor's hold on either -- keeps a strong reference to the object that
        # *owns* the data. numpy collapses base chains to that owner rather than chaining, so
        # a lent view is NOT referenced by a slice taken from it; the owner is. Probing with a
        # throwaway slice is what identifies it, since it differs by allocator (heap arrays
        # own their own data; a page-locked one is a view of a torch tensor's buffer).
        probe = base[:1]
        self.owner = base if probe.base is None else probe.base
        del probe  # the probe is itself a reference; the idle baseline must not include it
        self._idle_refs = 0  # provisional -- calibrate() sets the real one

    def calibrate(self) -> None:
        """Record the refcount that means *nothing is lent*, and check the buffer is trackable.

        Split from ``__init__`` because the baseline is only meaningful once the allocation's
        transient references are gone. Where the owner is the array itself -- which is what
        ``torch.empty(pin_memory=True).numpy()`` gives us, so this is the *pinned* path, not a
        corner case -- the allocator's return value is still on the caller's stack during
        ``__init__``. Counting it makes the baseline permanently too high, every buffer then
        looks free forever, and the pool hands one allocation to every batch at once, each
        overwriting the last. So the caller drops its reference first, then calls this.

        The probe below is the guard that keeps that class of bug loud: a buffer whose lent
        view does not raise the owner's refcount cannot be tracked at all, and reusing it would
        corrupt batches silently. Once per allocation, never per batch.
        """
        self._idle_refs = self._refs()
        lent = self.base[:1]
        trackable = not self.free
        del lent
        if not trackable:
            raise RuntimeError(
                "batch buffer liveness is untrackable: a view of this allocation does not "
                "reference the data owner, so the pool cannot tell when a batch is still in "
                "use. The host allocator must return an array that either owns its data or "
                "directly views the object that does (one level, as numpy collapses base "
                f"chains) -- got {type(self.base).__name__} over {type(self.owner).__name__}."
            )

    def _refs(self) -> int:
        """References to the data owner. Must be reached through exactly one call frame so
        the idle baseline and the live count are measured identically."""
        return sys.getrefcount(self.owner)

    @property
    def free(self) -> bool:
        """True when nothing outside this pool references the memory, so it is ours to rewrite.

        Counting references to the *owner* rather than weak-referencing the lent array is what
        makes this survive a derived view: CPython drops each array on its last decref, and
        every zero-copy export we support (``torch.from_dlpack``, ``jnp.from_dlpack``) holds
        the array it was given, which in turn holds the owner. A crop the consumer kept after
        discarding the batch therefore still counts.
        """
        return self._refs() <= self._idle_refs


class BatchBuffers:
    """Pool of reusable batch-output arrays, keyed by ``(inner_shape, dtype)``.

    **Hand-out is locked because there can be more than one producer.** Each active iteration
    gets its own producer thread (``source._iterate``) over the one shared ``ChunkPool``, so
    ``zip(ds.train, ds.val)`` or two ``DataLoader``s over one dataset run two of them into this
    pool. Deciding a buffer is free and lending it must therefore be one atomic step: otherwise
    both producers observe the same buffer as free and write their batches into it. The lock is
    held only across that decision -- never across a gather -- so it costs one uncontended
    acquire per variable per batch, against a gather that copies megabytes.

    Writing a lent buffer stays lock-free and always was: a buffer is lent to exactly one
    producer, which is the only thread that touches it.
    """

    def __init__(self, *, allocator: HostAllocator | None = None) -> None:
        self._alloc: HostAllocator = allocator or aligned_empty
        self._pools: dict[tuple[tuple[int, ...], np.dtype], list[_Buffer]] = {}
        self._lock = threading.Lock()

    def take(self, n: int, inner_shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Lend an ``(n, *inner_shape)`` array of ``dtype``, reusing a free buffer if there is
        one. The returned view's contents are undefined -- the caller overwrites every row.
        """
        key = (tuple(inner_shape), dtype)
        with self._lock:
            buffers = self._pools.setdefault(key, [])
            for buf in buffers:
                # A free buffer big enough along the batch axis serves this batch; a short
                # final batch takes a prefix of a full-size one. The lend must happen under the
                # lock too: it is what makes the buffer *look* taken to the next caller (the
                # returned view is what raises the owner's refcount), so releasing between the
                # check and the slice would hand the same memory to two producers.
                if buf.free and buf.base.shape[0] >= n:
                    return self._lend(buf, n)
            allocated = self._alloc((n, *inner_shape), dtype)
            buf = _Buffer(allocated)
            del allocated  # load-bearing: the idle baseline must not count this reference
            buf.calibrate()
            buffers.append(buf)
            return self._lend(buf, n)

    @staticmethod
    def _lend(buf: _Buffer, n: int) -> np.ndarray:
        """Hand out a view of ``buf``'s first ``n`` rows.

        Slicing is what lets a ragged final batch and a full one share an allocation; the
        returned array references the buffer's owner, which is how :attr:`_Buffer.free`
        sees it.
        """
        return buf.base[:n]

    @property
    def nbytes(self) -> int:
        """Total host memory the pool owns. Converges once the pipeline reaches steady state."""
        with self._lock:  # a concurrent take may be appending to one of these lists
            return sum(buf.base.nbytes for pool in self._pools.values() for buf in pool)

    def set_allocator(self, allocator: HostAllocator) -> None:
        """Swap where buffers come from, dropping any already allocated.

        The one injection point for page-locked memory: the core cannot call
        ``torch.empty(pin_memory=True)`` without importing a framework, so the torch adapter
        supplies it. Existing buffers are dropped rather than mixed, so the pool does not end
        up half pinned and half not for no visible reason. Call before iterating -- swapping
        mid-epoch is legal (outstanding batches keep their own memory by refcount) but wastes
        whatever the pool had already warmed.
        """
        with self._lock:
            self._alloc = allocator
            self._pools.clear()

    def clear(self) -> None:
        """Drop every owned buffer (pool teardown).

        Safe with batches still outstanding: a consumer's view keeps its own memory alive by
        refcount, it just stops being reusable. Buffers are otherwise held for the pool's
        lifetime *on purpose* -- that is what makes the next epoch free -- so this belongs at
        close, never at an epoch boundary.
        """
        with self._lock:
            self._pools.clear()
