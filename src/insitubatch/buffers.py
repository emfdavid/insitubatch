"""Reusable batch-output buffers.

``pool.gather`` used to allocate a fresh array per variable per batch. ``np.empty`` itself is
nearly free -- it reserves address space without touching pages -- so the cost was never the
allocation but the **first-touch page faults** on the scatter-write that follows, plus the
``mmap``/``munmap`` churn glibc falls back to above its 32 MiB dynamic threshold. Below that
threshold the freed block is recycled on the heap and reuse buys nothing; above it, reuse saves
22-33% of assembly time (``bench/probe_batch_buffers.py --arms alloc``). Weather-sized batches
are far below; ViT and microscopy batches are far above.

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

    def clear(self) -> None:
        """Drop every owned buffer (pool teardown).

        Safe with batches still outstanding: a consumer's view keeps its own memory alive by
        refcount, it just stops being reusable. Buffers are otherwise held for the pool's
        lifetime *on purpose* -- that is what makes the next epoch free -- so this belongs at
        close, never at an epoch boundary.
        """
        self._pools.clear()
