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
        self._geom = geometries
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
        # OrderedDict in recency order (LRU front -> MRU back); pinned keys are
        # exempt from eviction.
        self._slots: OrderedDict[tuple[str, int], _Slot] = OrderedDict()
        self._pinned: set[tuple[str, int]] = set()
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
        """Reserve + allocate + pin one outer chunk, evicting unpinned-LRU for room.

        Returns ``True`` once the slot exists and is pinned (idempotent if already
        resident -- e.g. another variable's tile admitted this position). Returns
        ``False`` only when the budget is full of *pinned* slots: the working set
        exceeds the budget, so the caller must wait for the consumer to unpin one.
        """
        key = (array, chunk_index)
        geom = self._geom[array]
        nbytes = int(np.prod(geom.slot_shape(chunk_index), dtype=np.int64)) * geom.dtype.itemsize
        with self._cv:
            if key in self._slots:
                # Only reached on a miss (a ready chunk is taken by pin_if_ready), so
                # any resident slot here is a stale partial left by a cancelled epoch
                # -- drop it and rebuild fresh rather than reuse a half-filled buffer.
                self._drop(key)
            if not self._make_room(nbytes):
                return False
            self._slots[key] = _Slot(
                data=self._alloc(array, chunk_index, geom.slot_shape(chunk_index), geom.dtype),
                remaining=geom.n_inner_chunks(chunk_index),
                nbytes=nbytes,
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
        """Atomically pin a resident, ready chunk (a cache hit) and touch its LRU.

        Returns ``True`` on a hit (caller skips fetch+decode+transform), ``False``
        if the chunk is absent or still assembling (a miss -> ``try_admit``). One
        lock so the check and the pin can't race an eviction in between.
        """
        with self._cv:
            slot = self._slots.get((array, chunk_index))
            if slot is not None and slot.ready and slot.error is None:
                self._pin((array, chunk_index))
                return True
            return False

    def unpin(self, chunk_ids: set[int]) -> None:
        """Release the given drained outer chunks: now LRU-evictable, not yet dropped.

        Replaces eviction -- the pool decides *when* to drop (lazily, under budget
        pressure), so unpinned chunks linger for cross-epoch reuse. Wakes any admit
        parked on a full budget.
        """
        with self._cv:
            for key in [k for k in self._slots if k[1] in chunk_ids]:
                self._pinned.discard(key)
            self._cv.notify_all()

    def _pin(self, key: tuple[str, int]) -> None:  # call under the lock
        self._pinned.add(key)
        self._slots.move_to_end(key)  # MRU

    def _make_room(self, nbytes: int) -> bool:  # call under the lock
        if self._budget is None:
            return True
        while self._bytes + nbytes > self._budget:
            victim = next((k for k in self._slots if k not in self._pinned), None)
            if victim is None:  # everything resident is pinned -> no room
                return False
            self._drop(victim)
        return True

    def _drop(self, key: tuple[str, int]) -> None:  # call under the lock
        slot = self._slots.pop(key)
        self._pinned.discard(key)  # no stale pin if dropping a (rare) pinned partial
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
        dst, src = self._geom[array].tile_placement(chunk_index, inner_coord)
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
        offset = chunk_index * self._geom[array].sample_chunk_size
        chunk = DecodedChunk(read=ChunkRead(array, chunk_index), data=data, sample_offset=offset)
        for transform in self._chunk_transforms:  # vectorized numpy -> GIL released
            chunk = transform(chunk)
        return chunk.data

    # -- consume: wait / gather ---------------------------------------------

    def wait_ready(self, array: str, chunk_index: int) -> None:
        """Block until the outer chunk is fully assembled (or raise its error).

        Wakes on three conditions: the chunk is ready, the chunk failed
        (:meth:`fail`), or the whole pool was poisoned (:meth:`set_error`) -- the
        last covers a driver death before this chunk was even allocated, so a
        consumer never waits forever on a chunk that will never arrive.
        """
        key = (array, chunk_index)
        with self._cv:
            self._cv.wait_for(
                lambda: self._error is not None or (key in self._slots and self._slots[key].ready)
            )
            if self._error is not None:
                raise self._error
            error = self._slots[key].error
        if error is not None:
            raise error

    def gather(self, rows: np.ndarray, variables: list[str], sample_chunk_size: int) -> Batch:
        """Assemble one batch from ``[chunk_id, within]`` draw rows.

        Caller must have waited for every referenced outer chunk to be ready.
        Draws are grouped by chunk so each slot is touched once (one coalesced
        fancy-index, never a Python per-sample loop); ``data`` and ``sample_indices``
        come out in the same grouped order, so row ``i`` of every variable refers to
        the same sample.
        """
        chunk_ids = rows[:, 0]
        within = rows[:, 1]
        uniq = np.unique(chunk_ids)

        out: dict[str, list[np.ndarray]] = {v: [] for v in variables}
        idx_pieces: list[np.ndarray] = []
        for cid in uniq:
            w = within[chunk_ids == cid]
            idx_pieces.append(cid * sample_chunk_size + w)
            for var in variables:
                out[var].append(self._slots[(var, int(cid))].data[w])

        arrays = {var: np.concatenate(pieces, axis=0) for var, pieces in out.items()}
        return Batch(arrays=arrays, sample_indices=np.concatenate(idx_pieces))

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
