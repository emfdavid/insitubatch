"""ChunkPool: the batch-assembly buffer (and, in B2, the cache).

The scheduler fetches at *stored chunk* (tile) granularity and scatters each
decoded tile into its outer chunk's pre-allocated array. The pool owns those
arrays -- the "slots" -- across their whole life: allocate (size to the assembled
chunk) -> scatter tiles (disjoint writes from the decode pool) -> mark ready (when
the last tile lands) -> gather batches (coalesced, on the consumer thread) ->
evict (when a shuffle-block is fully drained). It plays the residency +
coalesced-gather role of the old shuffle-block buffer, generalized to
tile-at-a-time fill; B2 swaps the slot *backing* (heap -> mmap'd ``.npy``) and the
*policy* (read-once -> LRU) to subsume the chunk cache.

Implements the assembly half of the M1.6 decoupled fetch scheduler (DESIGN.md).

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

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .types import ArrayGeometry, Batch, ChunkRead, DecodedChunk

ChunkTransform = Callable[[DecodedChunk], DecodedChunk]


@dataclass(slots=True)
class _Slot:
    """One outer chunk's assembly buffer plus its completion bookkeeping."""

    data: np.ndarray
    remaining: int  # inner tiles not yet scattered; 0 => fully assembled
    ready: bool = False
    error: BaseException | None = None


class ChunkPool:
    """Bounded pool of assembling/assembled outer-chunk slots, keyed ``(array, chunk_index)``.

    Residency is bounded by the *caller* (the scheduler admits new outer chunks,
    the consumer evicts drained ones); the pool exposes ``resident_chunks`` /
    ``max_resident`` for that policy and observability but does not itself block on
    a cap -- admission/backpressure lives in the scheduler so it never parks a
    decode worker mid-flight. ``chunk_transforms`` run once per outer chunk on the
    *assembled* array (the v1 per-chunk contract; per-tile fusion is a later
    opt-in), so a hit reflects decode + transform exactly like the old buffer.
    """

    def __init__(
        self,
        geometries: dict[str, ArrayGeometry],
        *,
        chunk_transforms: Sequence[ChunkTransform] = (),
    ) -> None:
        self._geom = geometries
        self._chunk_transforms = tuple(chunk_transforms)
        self._slots: dict[tuple[str, int], _Slot] = {}
        self._cv = threading.Condition(threading.Lock())
        self.max_resident = 0  # peak distinct outer chunk positions held at once

    # -- residency observability (caller owns the policy) -------------------

    def _positions(self) -> set[int]:
        """Distinct outer chunk indices currently resident (call under the lock)."""
        return {cid for _array, cid in self._slots}

    @property
    def resident_chunks(self) -> int:
        with self._cv:
            return len(self._positions())

    # -- allocate / scatter / complete --------------------------------------

    def allocate(self, array: str, chunk_index: int) -> None:
        """Create the slot for one outer chunk if absent (idempotent, non-blocking).

        The scheduler calls this when it first admits an outer chunk, before
        scattering any of its tiles. Sized to ``slot_shape`` so a short final
        chunk fits exactly.
        """
        key = (array, chunk_index)
        geom = self._geom[array]
        with self._cv:
            if key in self._slots:
                return
            self._slots[key] = _Slot(
                data=np.empty(geom.slot_shape(chunk_index), dtype=geom.dtype),
                remaining=geom.n_inner_chunks(chunk_index),
            )
            self.max_resident = max(self.max_resident, len(self._positions()))

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
            slot.data = prepped
            slot.ready = True
            self._cv.notify_all()

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

    def _apply_transforms(self, array: str, chunk_index: int, data: np.ndarray) -> np.ndarray:
        if not self._chunk_transforms:
            return data
        offset = chunk_index * self._geom[array].sample_chunk_size
        chunk = DecodedChunk(read=ChunkRead(array, chunk_index), data=data, sample_offset=offset)
        for transform in self._chunk_transforms:  # vectorized numpy -> GIL released
            chunk = transform(chunk)
        return chunk.data

    # -- consume: wait / gather / evict -------------------------------------

    def wait_ready(self, array: str, chunk_index: int) -> None:
        """Block until the outer chunk is fully assembled (or raise its error)."""
        key = (array, chunk_index)
        with self._cv:
            self._cv.wait_for(lambda: key in self._slots and self._slots[key].ready)
            error = self._slots[key].error
        if error is not None:
            raise error

    def gather(self, rows: np.ndarray, variables: list[str], sample_chunk_size: int) -> Batch:
        """Assemble one batch from ``[chunk_id, within]`` draw rows.

        Caller must have waited for every referenced outer chunk to be ready.
        Draws are grouped by chunk so each slot is touched once (one coalesced
        fancy-index); ``data`` and ``sample_indices`` come out in the same grouped
        order, so row ``i`` of every variable refers to the same sample. The
        grouped-by-chunk gather math is unchanged from the shuffle-block buffer;
        only the source (assembled slot vs buffered whole chunk) differs.
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

    def evict(self, chunk_ids: set[int]) -> int:
        """Drop every variable's slot for the given drained outer chunks.

        Frees the residency the chunk occupied and wakes the scheduler, which may
        be parked waiting to admit the next block. Returns slots dropped.
        """
        with self._cv:
            drop = [k for k in self._slots if k[1] in chunk_ids]
            for k in drop:
                del self._slots[k]
            if drop:
                self._cv.notify_all()
            return len(drop)
