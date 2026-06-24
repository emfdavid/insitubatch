"""ChunkPool: the batch-assembly buffer *and* the cache -- one machinery.

The scheduler fetches at *stored chunk* (tile) granularity and scatters each
decoded tile into its outer chunk's pre-allocated array. The pool owns those
arrays -- the "slots" -- across their whole life: admit (size to the assembled
chunk, evicting unpinned-LRU for room) -> scatter tiles (disjoint writes from the
decode pool) -> mark ready (when the last tile lands) -> gather batches (coalesced,
on the consumer thread) -> unpin (when a shuffle-block is fully drained).

Buffer and cache differ only in budget + backing, so they are the same code:

* a **byte budget** with **pin/unpin + LRU** eviction. A chunk is pinned while the
  current epoch needs it; unpinned chunks stay resident (retained) until budget
  pressure evicts them in LRU order. A small budget is read-once (unpinned chunks
  evicted promptly); a large budget retains drained chunks so a still-resident
  prepped chunk is a hit next epoch -- decode-once reuse.
* a **backing**: heap (``np.empty``) or mmap'd ``.npy`` on NVMe (``open_memmap``),
  the scatter writing straight into the slot either way; mmap keeps the working set
  as reclaimable page cache rather than anon heap.

See [docs/architecture.md] for where this sits in the pipeline.

Thread-safety / free-threading (the load-bearing invariant)
-----------------------------------------------------------
Scatter runs on many decode-pool threads at once -- including two tiles of the
*same* outer chunk concurrently (inner-grid parallelism). We never lock the data
copy; we lock only the Python-level bookkeeping. Two rules make this correct
under both the GIL build *and* free-threaded 3.13t (where the GIL is no longer an
incidental barrier):

1. **Disjoint, fixed-shape writes are lock-free.** Tiles cover disjoint regions
   of a slot whose shape never changes, so the memcpy ``slot[dst] = tile`` races
   nothing.
2. **Readiness is published *through the lock*, after the copy.** Each scatter
   does its memcpy, *then* takes the lock to decrement the completion counter.
   The consumer observes ``ready`` under the same lock. That lock release ->
   acquire pair is the happens-before edge guaranteeing the consumer sees every
   completed copy -- we do not lean on the GIL for it.

So the GIL build is just the serialized (slower) case; free threading is upside.
"""

from __future__ import annotations

import contextlib
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import ArrayGeometry, Batch, ChunkRead, DecodedChunk

ChunkTransform = Callable[[DecodedChunk], DecodedChunk]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


@dataclass(slots=True)
class _Slot:
    """One outer chunk's assembly buffer plus its completion bookkeeping."""

    data: np.ndarray
    remaining: int  # inner tiles not yet scattered; 0 => fully assembled
    nbytes: int  # slot size, charged to the budget (fixed at admit)
    ready: bool = False
    error: BaseException | None = None
    claimed: bool = False  # the driver has referenced it *this epoch* (see wait_ready)


class ChunkPool:
    """Byte-budgeted pool of outer-chunk slots, keyed ``(array, chunk_index)``.

    The pool is the assembly buffer *and* the cache. A slot is **pinned** while the
    current epoch needs it (in-flight or block-not-yet-drained) and **unpinned** once
    its block is drained;
    unpinned slots stay resident (retained for cross-epoch reuse) until budget
    pressure evicts them in LRU order. ``budget_bytes`` is the single knob:

    * small (~``2*block_chunks`` worth) -> read-once (unpinned evicted promptly);
    * large + persistent across epochs -> a decode-once cache (a still-resident
      prepped chunk is a hit, skipping fetch + decode + transform).

    Eviction targets unpinned-LRU only; a slot is never unpinned before it is
    ready+drained, so an in-flight or in-use chunk is never dropped. Backing is heap
    or mmap (see ``backing_dir``); ``chunk_transforms`` run once per outer chunk on
    the *assembled* array, so a hit reflects decode + transform.
    """

    def __init__(
        self,
        geometries: dict[str, ArrayGeometry],
        *,
        chunk_transforms: Sequence[ChunkTransform] = (),
        backing_dir: str | Path | None = None,
        budget_bytes: int | None = None,
    ) -> None:
        self._geom = geometries  # label -> geometry (a label is one (array, offset) view)
        # Slots are keyed by the underlying array *path*, not the variable label, so two
        # views of one array (e.g. t2m_now / t2m_next) share a single decode. One
        # representative geometry per path suffices for slot sizing (aliases share shape).
        self._by_path = {g.path: g for g in geometries.values()}
        self._chunk_transforms = tuple(chunk_transforms)
        # backing: heap (np.empty) or mmap'd .npy under backing_dir (point at NVMe).
        # The scatter writes straight into the slot either way; mmap keeps the working
        # set as reclaimable page cache rather than anon heap. Default heap: scattering
        # into mmap is NVMe write traffic even when never reused, so reach for it to
        # spill a working set past RAM or for cross-epoch reuse, not for plain streaming.
        self._dir = Path(backing_dir) if backing_dir is not None else None
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
        self._budget = budget_bytes  # None => unbounded (never self-evicts)
        self._bytes = 0
        # OrderedDict in recency order (LRU front -> MRU back). Eviction targets only
        # a *ready, unreferenced* slot: a not-ready slot is an in-flight fetch, and a
        # refcounted slot is held by a live block (windows let one chunk be read by
        # several blocks at once, so pins are reference-counted, not a boolean).
        self._slots: OrderedDict[tuple[str, int], _Slot] = OrderedDict()
        self._pinned: dict[tuple[str, int], int] = {}  # key -> refcount (>0 == pinned)
        self._cv = threading.Condition(threading.Lock())
        self._error: BaseException | None = None  # global poison (driver death)
        self.max_resident = 0  # peak distinct outer chunk positions held at once

    # -- observability ------------------------------------------------------

    def _positions(self) -> set[int]:
        """Distinct outer chunk indices currently resident (call under the lock)."""
        return {cid for _array, cid in self._slots}

    @property
    def resident_chunks(self) -> int:
        with self._cv:
            return len(self._positions())

    @property
    def resident_bytes(self) -> int:
        with self._cv:
            return self._bytes

    # -- admission / pinning / eviction -------------------------------------

    def try_admit(self, array: str, chunk_index: int) -> bool:
        """Reserve + allocate + reference one outer-chunk slot, evicting ready-LRU for room.

        Admission takes one reference (incref) and *claims* the slot for this epoch (so
        the consumer's :meth:`wait_ready` won't gather it until the driver has referenced
        it -- see there), so the slot stays resident from its in-flight fetch through to
        the consumer's release. The driver fetches each chunk once per epoch, so an
        eviction before consume could not be re-fetched and would deadlock the waiter.
        The consumer releases each chunk at its *last* use (:meth:`unpin_keys`); windowed
        reads let one chunk be referenced by several blocks, hence reference counts, not a
        boolean. Idempotent if already resident (incref only). Returns ``False`` only when
        the budget is full of in-flight or referenced slots -- the caller awaits a release.
        """
        key = (array, chunk_index)
        geom = self._by_path[array]
        nbytes = int(np.prod(geom.slot_shape(chunk_index), dtype=np.int64)) * geom.dtype.itemsize
        with self._cv:
            if key in self._slots:
                self._pin(key)  # already resident (in-flight or ready hit) -> incref, reuse
                self._slots[key].claimed = True
                self._cv.notify_all()  # a ready hit may now satisfy a waiter
                return True
            if not self._make_room(nbytes):
                return False
            self._slots[key] = _Slot(
                data=self._alloc(array, chunk_index, geom.slot_shape(chunk_index), geom.dtype),
                remaining=geom.n_inner_chunks(chunk_index),
                nbytes=nbytes,
                claimed=True,
            )
            self._bytes += nbytes
            self._pin(key)
            self.max_resident = max(self.max_resident, len(self._positions()))
            return True

    def is_ready(self, array: str, chunk_index: int) -> bool:
        """True if the chunk is resident, fully assembled, and not failed (a hit)."""
        with self._cv:
            slot = self._slots.get((array, chunk_index))
            return slot is not None and slot.ready and slot.error is None

    def pin_if_ready(self, array: str, chunk_index: int) -> bool:
        """Incref + return ``True`` iff the chunk is resident, ready, and not failed.

        A cross-epoch cache hit the driver can skip fetching -- but it must still be
        referenced so it stays resident through the consumer's use (released at last
        use like an admitted chunk), else it could be evicted before the waiter gathers
        it and, since the driver fetches each chunk once, deadlock. One lock so the
        check and the incref cannot race an eviction in between.
        """
        with self._cv:
            key = (array, chunk_index)
            slot = self._slots.get(key)
            if slot is not None and slot.ready and slot.error is None:
                self._pin(key)
                slot.claimed = True  # publish the claim so a waiter may now proceed
                self._cv.notify_all()
                return True
            return False

    def pin_keys(self, keys: set[tuple[str, int]]) -> None:
        """Reference (incref) a set of ``(path, chunk_index)`` slots for a live block.

        Windows make one chunk readable by several concurrent blocks, so pins are
        reference-counted: each block that needs a slot increfs it on entry and
        decrefs it on drain (:meth:`unpin_keys`). A slot with refcount > 0 is never
        evicted. Pinning a not-yet-allocated key is fine -- the count is recorded and
        the slot, once admitted, inherits it.
        """
        with self._cv:
            for key in keys:
                self._pin(key)

    def unpin_keys(self, keys: set[tuple[str, int]]) -> None:
        """Release (decref) a block's ``(path, chunk_index)`` references.

        A slot dropping to refcount 0 becomes LRU-evictable (retained for cross-epoch
        reuse until budget pressure drops it), not dropped now. Wakes any admit parked
        on a full budget.
        """
        with self._cv:
            for key in keys:
                self._unpin_one(key)
            self._cv.notify_all()

    def unpin_all(self) -> None:
        """Epoch boundary reset: clear every pin and drop abandoned partials.

        A pin is per-epoch working state, not cache membership -- ready chunks stay
        resident (unpinned) for cross-epoch reuse. No pin may survive an epoch: an
        aborted epoch (early ``break``) leaves its read-ahead and un-drained current
        block referenced, which would shrink the next epoch's budget until admission
        can free no room and the driver deadlocks. A *not-ready* slot at this boundary
        is an abandoned partial (its fetch was cancelled mid-flight) -- it can never be
        a valid cache entry, so drop it; that also restores the in-epoch invariant
        "not ready => in flight" that protects fetches from eviction. Called at the
        next epoch's start, when the prior scheduler is fully closed (no race).
        """
        with self._cv:
            self._pinned.clear()
            for key in [k for k, slot in self._slots.items() if not slot.ready]:
                self._drop(key)
            for slot in self._slots.values():
                slot.claimed = False  # a retained chunk must be re-claimed next epoch
            self._cv.notify_all()

    def _pin(self, key: tuple[str, int]) -> None:  # call under the lock
        self._pinned[key] = self._pinned.get(key, 0) + 1
        if key in self._slots:
            self._slots.move_to_end(key)  # MRU (the slot may not be allocated yet)

    def _unpin_one(self, key: tuple[str, int]) -> None:  # call under the lock
        n = self._pinned.get(key, 0)
        if n <= 1:
            self._pinned.pop(key, None)
        else:
            self._pinned[key] = n - 1

    def _make_room(self, nbytes: int) -> bool:  # call under the lock
        if self._budget is None:
            return True
        while self._bytes + nbytes > self._budget:
            # Evict the LRU slot that is ready *and* unreferenced; a not-ready slot is
            # an in-flight fetch and a refcounted slot is held by a live block.
            victim = next(
                (k for k, s in self._slots.items() if k not in self._pinned and s.ready), None
            )
            if victim is None:  # everything resident is in-flight or pinned -> no room
                return False
            self._drop(victim)
        return True

    def _drop(self, key: tuple[str, int]) -> None:  # call under the lock
        slot = self._slots.pop(key)
        self._pinned.pop(key, None)  # no stale refcount if dropping a (rare) pinned partial
        self._bytes -= slot.nbytes
        self._free(slot)

    def _alloc(
        self, array: str, chunk_index: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
        if self._dir is None:
            return np.empty(shape, dtype=dtype)
        path = self._dir / f"{_safe(array)}__{chunk_index}.npy"
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    def scatter(
        self, array: str, chunk_index: int, inner_coord: tuple[int, ...], tile: np.ndarray
    ) -> None:
        """Copy one decoded tile into its slot; complete the chunk if it was the last.

        The memcpy happens *before* the lock (rule 1); the completion counter and
        ``ready`` flip happen *under* the lock (rule 2). Completion (chunk
        transforms on the assembled array) runs outside the lock -- no other thread
        touches the slot once ``remaining`` hits 0 -- then ``ready`` is published.
        """
        key = (array, chunk_index)
        slot = self._slots[key]  # allocated by the scheduler before any scatter
        dst, src = self._by_path[array].tile_placement(chunk_index, inner_coord)
        slot.data[dst] = tile[src]  # disjoint, fixed-shape: lock-free (rule 1)

        with self._cv:
            slot.remaining -= 1
            last = slot.remaining == 0
        if not last:
            return

        # Sole owner now: assemble-stage transforms on the whole outer chunk.
        prepped = self._apply_transforms(array, chunk_index, slot.data)
        with self._cv:
            slot.data = self._persist(slot.data, prepped)
            slot.ready = True
            self._cv.notify_all()

    def _persist(self, current: np.ndarray, prepped: np.ndarray) -> np.ndarray:
        """Land the prepped (post-transform) array in the slot's backing.

        No transform (``prepped is current``) is a no-op -- the scatter already
        wrote the slot. With a transform, heap just holds the new array; mmap writes
        it back into the slot's file so the cached chunk stays on NVMe (a shape-
        changing transform like regrid would need a re-sized memmap -- deferred).
        """
        if prepped is current or self._dir is None:
            return prepped
        if prepped.shape != current.shape:
            raise NotImplementedError(
                "mmap backing with a reshaping chunk_transform (e.g. regrid) is not "
                "supported yet; use heap backing for that path."
            )
        current[:] = prepped  # write the transformed result into the memmap
        return current

    def fail(self, array: str, chunk_index: int, error: BaseException) -> None:
        """Mark a slot failed so a waiting consumer re-raises instead of hanging.

        Fail-fast: a fetch/decode error on any tile poisons its outer chunk; the
        consumer's ``wait_ready`` surfaces it on the main thread.
        """
        with self._cv:
            slot = self._slots.get((array, chunk_index))
            if slot is not None:
                slot.error = error
                slot.ready = True
                self._cv.notify_all()

    def set_error(self, error: BaseException) -> None:
        """Poison the whole pool (the fetch driver died) so every waiter re-raises.

        Unlike :meth:`fail` (one chunk), this unblocks consumers waiting on chunks
        that may never be allocated -- the driver failed before reaching them. The
        first error wins (later failures are usually cascade noise).
        """
        with self._cv:
            if self._error is None:
                self._error = error
            self._cv.notify_all()

    def _apply_transforms(self, array: str, chunk_index: int, data: np.ndarray) -> np.ndarray:
        if not self._chunk_transforms:
            return data
        offset = chunk_index * self._by_path[array].sample_chunk_size
        chunk = DecodedChunk(read=ChunkRead(array, chunk_index), data=data, sample_offset=offset)
        for transform in self._chunk_transforms:  # vectorized numpy -> GIL released
            chunk = transform(chunk)
        return chunk.data

    # -- consume: wait / gather ---------------------------------------------

    def wait_ready(self, array: str, chunk_index: int) -> None:
        """Block until the outer chunk is assembled *and claimed this epoch* (or raise).

        Waiting on ``claimed`` (set by the driver's admit / pin_if_ready) closes a
        cross-epoch race: a chunk still resident-and-ready from the prior epoch would
        otherwise be gathered before the driver references it, letting the consumer's
        last-use release land before the driver's pin -- a lost release that leaks a
        reference (and, worse, lets the driver evict a chunk mid-gather). Requiring the
        claim orders pin-before-consume-before-release. Wakes on: ready+claimed, the
        chunk failed (:meth:`fail`), or the pool was poisoned (:meth:`set_error`, covering
        a driver death before this chunk was allocated).
        """
        key = (array, chunk_index)
        with self._cv:
            self._cv.wait_for(
                lambda: (
                    self._error is not None
                    or (key in self._slots and self._slots[key].ready and self._slots[key].claimed)
                )
            )
            if self._error is not None:
                raise self._error
            error = self._slots[key].error
        if error is not None:
            raise error

    def gather(self, rows: np.ndarray, variables: list[str], sample_chunk_size: int) -> Batch:
        """Assemble one batch from ``[chunk_id, within]`` *anchor* draw rows.

        Each row is one sample anchor ``t = chunk_id*spc + within``; each variable
        reads its array at ``t + offset`` (offset 0 is the plain non-windowed case).
        Output is in anchor-row order: row ``i`` of every variable is the same anchor,
        ``sample_indices[i] == t_i``. Per variable the reads are grouped by the
        variable's *own* (offset-shifted) chunk -- one coalesced fancy-index per chunk,
        never a Python per-sample loop. The caller must have waited every referenced
        ``(path, offset-shifted chunk)`` ready.
        """
        spc = sample_chunk_size
        anchor = rows[:, 0].astype(np.int64) * spc + rows[:, 1].astype(np.int64)  # (N,)
        n = anchor.shape[0]

        arrays: dict[str, np.ndarray] = {}
        for var in variables:
            geom = self._geom[var]
            sample = anchor + geom.offset  # this view reads array[anchor + offset]
            read_cid = sample // spc
            within = sample % spc
            out = np.empty((n, *geom.inner_shape), dtype=geom.dtype)
            for cid in np.unique(read_cid):
                mask = read_cid == cid  # rows that read this chunk -> one coalesced index
                out[mask] = self._slots[(geom.path, int(cid))].data[within[mask]]
            arrays[var] = out
        offsets = {var: self._geom[var].offset for var in variables}
        return Batch(arrays=arrays, sample_indices=anchor, offsets=offsets)

    def _free(self, slot: _Slot) -> None:
        """Release a slot's backing: a no-op for heap, flush+close+unlink for mmap.

        Dropping a slot drops its cached bytes (intra-run); cross-*run* persistence
        (keep the file + a content-keyed index) is the deferred follow-up.
        """
        mmap = getattr(slot.data, "_mmap", None)
        if mmap is not None:
            fname = getattr(slot.data, "filename", None)
            mmap.close()
            if fname:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(fname)

    def close(self) -> None:
        """Free every remaining slot (mmap files included) -- e.g. on teardown."""
        with self._cv:
            for k in list(self._slots):
                self._free(self._slots.pop(k))
            self._bytes = 0
            self._pinned.clear()
