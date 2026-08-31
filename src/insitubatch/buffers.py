"""Reusable batch-output buffers.

``pool.gather`` used to allocate a fresh array per variable per batch. The cost is not the
allocation call -- ``np.empty`` only reserves address space -- but what glibc does with a
request above its 32 MiB dynamic ``mmap`` threshold: it ``mmap``s the batch and ``munmap``s it
on free, **every batch**. A syscall trace confirms it directly (128 MiB batches, 12-batch
epoch): 12 ``mmap``/``munmap`` pairs of the full buffer without the pool, 2 with it, and the
count tracks batches rather than dataset size. Paid per batch, that is a fresh page set to
fault in plus ``munmap``'s TLB work.

Below the threshold the freed block is recycled on the heap instead and reuse buys nothing --
it costs about 2-3%.

**What this is worth is a range, and where a workload lands in it is set by one thing: how
much of the loader's work the compute step is already hiding.** The range is bounded at both
ends, and both ends are measured.

*Upper bound -- allocation-bound* (``bench/batch_buffer_sweep.py``, the producer's cost fully
exposed): the pool is flat across a 16x payload range while fresh allocation falls off a
cliff at 32 MiB. 8 MiB -2.4%, 32 MiB +16%, 64 MiB +13%, 128 MiB +10%.

*Lower bound -- compute-bound: exactly zero*, and the GPU advection sweep is that case, not a
midpoint. A/B'd against no-reuse over 8/16/32/64 MiB per buffer it came back at
+0.21/+0.20/-0.18/-0.29 points of ceiling -- sign flipping, and at 64 MiB the two arms agree
to 0.04% absolute. That loop runs at 98.7-100% of its compute ceiling, so producer-side
allocation has nowhere to surface *even where the* ``mmap`` *path demonstrably engages*.
Prefetch hiding this is prefetch working.

So the honest claim is not "faster training" -- it is that the cliff cannot bite you, at a
cost of 2-3% below the threshold. It pays where the loader is not already free: shallower
compute per byte, inference rather than training, a cold cache, an IO-bound epoch. Weather
batches sit below the threshold anyway; ViT and microscopy batches sit above it.

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

import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

XLA_ALIGN = 128
"""Alignment XLA:CPU requires to import a DLPack buffer without copying."""

NO_REUSE_ENV = "INSITUBATCH_NO_BUFFER_REUSE"
"""Set truthy to allocate a fresh buffer per batch, bypassing reuse and its liveness tracking.

An escape hatch, deliberately not a tuning parameter. Reuse is safe by construction against
everything the transform API can do -- a transform that returns a derived array frees its
buffer early, one that returns a view keeps it tracked, one that retains simply makes the pool
allocate elsewhere -- so there is no knob here worth asking a user to reason about. What this
covers is the residue: memory escaping Python object semantics (``arr.ctypes.data`` handed to
a C extension that stores the pointer), a platform where ``sys.getrefcount`` does not mean
what we assume, and the possibility that we are wrong. One sentence of documentation:
*if batches contain data from a previous batch, set this and open an issue.*

It is also the A/B control. Reuse was first measured against a second checkout with the
``gather`` call site hand-reverted -- a worktree, a branch and a divergence risk, to answer a
one-line question. This flag is the better instrument: identical code on both sides, only the
reuse decision differs.

Costs, when set: above glibc's 32 MiB ``mmap`` threshold every batch is ``mmap``ed and
``munmap``ed again (which is the cliff reuse exists to remove), and under a *pinned* allocator
every batch becomes its own ``cudaHostAlloc`` -- far more expensive than the pooled case, and
loud about it (:meth:`BatchBuffers.set_allocator` warns). This is a diagnostic, not a supported
production configuration.
"""


_FALSY = ("", "0", "false", "no", "off")
"""Values of :data:`NO_REUSE_ENV` that leave reuse on. ``off`` belongs here: the variable is
named for the thing it *enables*, so ``=off`` reads as "no-reuse off" and must not disable
reuse -- the inverted-intent bug that omitting it creates is silent apart from the log stamp."""


def _reuse_enabled() -> bool:
    """Read :data:`NO_REUSE_ENV` once, at pool construction -- never on the hot path."""
    return os.environ.get(NO_REUSE_ENV, "").strip().lower() in _FALSY


HostAllocator = Callable[[tuple[int, ...], np.dtype], np.ndarray]
"""How the pool gets host memory. The default is aligned heap; the torch adapter swaps in a
page-locked one, which is how pinning arrives without the core importing a framework."""


class BufferStats(NamedTuple):
    """One consistent reading of the pool, for the per-epoch log line."""

    n_buffers: int
    nbytes: int
    kind: str  # "heap" or "pinned" -- allocator intent, see BatchBuffers.kind
    reuse: bool
    lends: int
    allocations: int

    def summary(self) -> str:
        """The log fragment: ``17 x pinned = 544.0 MiB, 100 lent, 0 allocated``.

        Formatted here rather than in the caller so the field names in a docstring and the
        words in the line cannot drift, and so the reuse-off case reads as a sentence instead
        of an interjection inside the arithmetic.
        """
        line = (
            f"{self.n_buffers} x {self.kind} = {self.nbytes / 2**20:.1f} MiB, "
            f"{self.lends} lent, {self.allocations} allocated"
        )
        return line if self.reuse else f"{line} (reuse OFF via {NO_REUSE_ENV})"


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

        Dropping *below* the baseline is a different thing entirely, and it raises. Every
        reference counted at calibration is structural -- ``self.owner``, and the base chain
        ``self.base`` holds -- so all of them outlive the buffer and the count can only go up.
        A lower reading means the baseline itself over-counted a transient, which is not a
        near-miss: it makes ``free`` permanently true, so the pool lends one allocation to
        every batch at once and each silently overwrites the last. That is exactly the defect
        :meth:`calibrate` exists to prevent, and it cost a night to find by its symptoms. The
        comparison is one integer against another on a path that already copies megabytes, so
        checking forever is free; the only reachable way to hit it is a bug in this file or a
        platform where ``sys.getrefcount`` does not mean what we assume, and both are better
        as a loud stop than as corrupted training data. :data:`NO_REUSE_ENV` is the way past
        it if it ever fires.
        """
        refs = self._refs()
        if refs < self._idle_refs:
            raise RuntimeError(
                f"batch buffer liveness baseline is wrong: the data owner now has {refs} "
                f"references, below the {self._idle_refs} recorded when nothing was lent. "
                "The baseline must count only references that outlive the buffer, so it can "
                "never be exceeded from below; a higher one makes every buffer look free and "
                "lends one allocation to every batch at once. Set "
                f"{NO_REUSE_ENV}=1 to bypass reuse, and please open an issue."
            )
        return refs <= self._idle_refs


class BatchBuffers:
    """Pool of reusable batch-output arrays, keyed by ``(inner_shape, dtype)``.

    **Hand-out is locked because there can be more than one producer.** Each active iteration
    gets its own producer thread (``source._iterate``) over the one shared ``ChunkPool``, so
    ``zip(ds.train, ds.val)`` or two ``DataLoader``s over one dataset run two of them into this
    pool. Deciding a buffer is free and lending it must therefore be one atomic step: otherwise
    both producers observe the same buffer as free and write their batches into it. The lock is
    held only across that decision -- never across a gather -- so it costs one uncontended
    acquire per variable per batch, against a gather that copies megabytes.

    Running two iterations also costs *residency*: each holds its own chunk references, so the
    pool's budget must cover both working sets. The auto-sized default covers one -- see
    docs/tuning.md, "Several iterations at once multiply the budget".

    Writing a lent buffer stays lock-free and always was: a buffer is lent to exactly one
    producer, which is the only thread that touches it.
    """

    def __init__(self, *, allocator: HostAllocator | None = None) -> None:
        self._alloc: HostAllocator = allocator or aligned_empty
        self._pools: dict[tuple[tuple[int, ...], np.dtype], list[_Buffer]] = {}
        self._lock = threading.Lock()
        self._reuse = _reuse_enabled()
        # Per-epoch observability, reset at the epoch boundary like the chunk cache's
        # hits/misses. `allocations` is the one that matters: it should fall to zero once the
        # pool has converged on the in-flight count, so a nonzero count in a later epoch means
        # buffers are not coming back -- retained batches, or a batch geometry that changes.
        self.lends = 0
        self.allocations = 0

    def take(self, n: int, inner_shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Lend an ``(n, *inner_shape)`` array of ``dtype``, reusing a free buffer if there is
        one. The returned view's contents are undefined -- the caller overwrites every row.
        """
        if not self._reuse:
            # The escape hatch: pre-pool behaviour exactly -- one fresh allocation per call,
            # no _Buffer, no liveness poll, nothing to reclaim. Allocated outside the lock,
            # which is only guarding the counters here.
            with self._lock:
                self.lends += 1
                self.allocations += 1
            return self._alloc((n, *inner_shape), dtype)
        key = (tuple(inner_shape), dtype)
        with self._lock:
            self.lends += 1
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
            self.allocations += 1
            return self._lend(buf, n)

    @staticmethod
    def _lend(buf: _Buffer, n: int) -> np.ndarray:
        """Hand out a view of ``buf``'s first ``n`` rows.

        Slicing is what lets a ragged final batch and a full one share an allocation; the
        returned array references the buffer's owner, which is how :attr:`_Buffer.free`
        sees it.
        """
        return buf.base[:n]

    def stats(self) -> BufferStats:
        """A consistent snapshot of what the pool holds, under one lock acquisition.

        Reading the properties individually takes the lock once each, so a producer allocating
        between them yields a line whose count and byte total disagree. Cheap to avoid, and a
        report that cannot be self-contradictory is worth more than three getters.
        """
        with self._lock:  # a concurrent take may be appending to one of these lists
            return BufferStats(
                n_buffers=sum(len(pool) for pool in self._pools.values()),
                nbytes=sum(buf.base.nbytes for pool in self._pools.values() for buf in pool),
                kind=str(getattr(self._alloc, "label", "heap")),
                reuse=self._reuse,
                lends=self.lends,
                allocations=self.allocations,
            )

    @property
    def nbytes(self) -> int:
        """Total host memory the pool owns. Converges once the pipeline reaches steady state."""
        with self._lock:  # a concurrent take may be appending to one of these lists
            return sum(buf.base.nbytes for pool in self._pools.values() for buf in pool)

    @property
    def n_buffers(self) -> int:
        """How many buffers the pool owns -- the in-flight count it converged on."""
        with self._lock:
            return sum(len(pool) for pool in self._pools.values())

    @property
    def kind(self) -> str:
        """What the buffers are made of, for the epoch summary: ``"pinned"`` or ``"heap"``.

        Read off a label the allocator carries rather than inspected from the memory, because
        the core cannot ask CUDA anything. It reports *intent*: a pinned allocator that has
        exhausted its budget hands out pageable buffers and says so through its own warning.

        Reports the allocator alone; whether reuse is switched off is a separate field, which
        :meth:`BufferStats.summary` renders alongside it -- the epoch line is where someone
        reads back what a run actually did, and a benchmark taken with the escape hatch set
        must not be mistakable for a normal one afterwards.
        """
        return str(getattr(self._alloc, "label", "heap"))

    def reset_counters(self) -> None:
        """Zero the per-epoch counters (called at the epoch boundary, with the cache's)."""
        with self._lock:
            self.lends = 0
            self.allocations = 0

    def set_allocator(self, allocator: HostAllocator) -> None:
        """Swap where buffers come from, dropping any already allocated.

        The one injection point for page-locked memory: the core cannot call
        ``torch.empty(pin_memory=True)`` without importing a framework, so the torch adapter
        supplies it. Existing buffers are dropped rather than mixed, so the pool does not end
        up half pinned and half not for no visible reason. Call before iterating -- swapping
        mid-epoch is legal (outstanding batches keep their own memory by refcount) but wastes
        whatever the pool had already warmed.

        Pinning with reuse switched off warns rather than raises. The combination is bad
        enough to be worth saying out loud -- every batch becomes its own ``cudaHostAlloc``,
        and the pinned allocator's budget is a monotone counter sized for a pool that
        converges, so once it is spent the run quietly finishes on pageable memory while
        :attr:`kind` still reports ``pinned`` intent. But it must stay *reachable*: the
        double-lend that motivated :data:`NO_REUSE_ENV` lived on the pinned path and nowhere
        else, so refusing this pairing would remove the one control that isolates reuse from
        pinning, and leave a user suspecting corruption no way to test it without changing
        two things at once.
        """
        if not self._reuse and str(getattr(allocator, "label", "heap")) != "heap":
            logger.warning(
                "buffer reuse is off (%s) under a %s allocator: every batch takes its own "
                "page-locked allocation, and the pin budget does not recycle, so it will be "
                "spent after a few batches and the rest of the run will be pageable. Fine as "
                "an A/B control against reuse; not a configuration to benchmark or train on.",
                NO_REUSE_ENV,
                str(getattr(allocator, "label", "heap")),
            )
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
