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

import weakref
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
    """One owned base array plus a weak handle on the view most recently lent from it."""

    __slots__ = ("base", "lent")

    def __init__(self, base: np.ndarray) -> None:
        self.base = base
        self.lent: weakref.ref[np.ndarray] | None = None

    @property
    def free(self) -> bool:
        """True when nobody holds the last view we lent -- so the memory is ours to rewrite.

        ``lent() is None`` is the whole safety argument: CPython drops a view on its last
        decref, and every zero-copy export we support (``torch.from_dlpack``,
        ``jnp.from_dlpack``) takes a strong reference to that view, so a live tensor keeps
        this False.
        """
        return self.lent is None or self.lent() is None


class BatchBuffers:
    """Pool of reusable batch-output arrays, keyed by ``(inner_shape, dtype)``.

    Not thread-safe by itself: it is driven from the single producer thread that assembles
    batches (``ChunkPool.gather``), which is also the only thread that may write a lent buffer.
    """

    def __init__(self, *, allocator: HostAllocator | None = None) -> None:
        self._alloc: HostAllocator = allocator or aligned_empty
        self._pools: dict[tuple[tuple[int, ...], np.dtype], list[_Buffer]] = {}

    def take(self, n: int, inner_shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Lend an ``(n, *inner_shape)`` array of ``dtype``, reusing a free buffer if there is
        one. The returned view's contents are undefined -- the caller overwrites every row.
        """
        key = (tuple(inner_shape), dtype)
        buffers = self._pools.setdefault(key, [])
        for buf in buffers:
            # A free buffer big enough along the batch axis serves this batch; a short final
            # batch takes a prefix of a full-size one.
            if buf.free and buf.base.shape[0] >= n:
                return self._lend(buf, n)
        buf = _Buffer(self._alloc((n, *inner_shape), dtype))
        buffers.append(buf)
        return self._lend(buf, n)

    @staticmethod
    def _lend(buf: _Buffer, n: int) -> np.ndarray:
        """Hand out a fresh view of ``buf`` and record a weak handle on it.

        The view must be a new object each time -- that is what the weakref tracks. Slicing
        always builds one, and slicing the full length is what lets a ragged batch and a full
        batch share the same base.
        """
        view = buf.base[:n]
        buf.lent = weakref.ref(view)
        return view

    @property
    def nbytes(self) -> int:
        """Total host memory the pool owns. Converges once the pipeline reaches steady state."""
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
        self._alloc = allocator
        self.clear()

    def clear(self) -> None:
        """Drop every owned buffer (pool teardown).

        Safe with batches still outstanding: a consumer's view keeps its own memory alive by
        refcount, it just stops being reusable. Buffers are otherwise held for the pool's
        lifetime *on purpose* -- that is what makes the next epoch free -- so this belongs at
        close, never at an epoch boundary.
        """
        self._pools.clear()
