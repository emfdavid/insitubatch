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

With ``persist=True`` (requires a ``backing_dir``) the mmap tier becomes a **cross-run
cache**: slot files survive ``close`` (only budget eviction removes them), an append-only
``insitu_cache.jsonl`` log records each completed entry *as it lands* (so a killed process
still leaves a usable cache -- crash recovery, no re-decode of what finished), and a new
pool over the same dir revives them as ready hits -- no fetch/decode. The ``backing_dir``
path *is* the dataset identity (bury a version in it; the store URL is not in the key).
The log header carries a **chunk_transform fingerprint**: a changed transform (or a format
bump) is a **stale cache**, which by default *raises* (``reset_stale_cache=True`` deletes
and rebuilds it). Revive additionally does a **shape/dtype** check per entry; that mismatch
is a miss (recompute + overwrite), never an error. The fingerprint uses cloudpickle when
present (``--extra cache``; captures closures/globals), else a best-effort source hash
(warned), and always honors an explicit ``transform.cache_key``. Without ``persist`` a
``backing_dir`` is ephemeral spill (unlinked on close).

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
import hashlib
import inspect
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import numpy as np

from .types import ArrayGeometry, Batch, ChunkRead, DecodedChunk

try:  # optional: stronger transform fingerprint (closures + globals). `--extra cache`.
    import cloudpickle
except ImportError:  # pragma: no cover - exercised by the no-cloudpickle fallback path
    cloudpickle = None

logger = logging.getLogger(__name__)

ChunkTransform = Callable[[DecodedChunk], DecodedChunk]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def output_geometry(geom: ArrayGeometry, transforms: Sequence[ChunkTransform]) -> ArrayGeometry:
    """The geometry a chunk has *after* the chunk_transform pipeline.

    A reshaping transform (regrid / dtype recast) declares ``output_inner(geom) ->
    (inner_shape, dtype)``; the pipeline folds them so each transform sees the geometry
    produced by the ones before it. Only the *inner* dims and dtype may change -- the
    sample axis (``shape[0]`` / ``chunks[0]``) is spliced straight back from the source, so
    a transform can neither move nor reshape it (the sample-geometry invariant). A transform
    without ``output_inner`` is identity; with none reshaping, the source geometry returns
    unchanged. Output ``chunks`` are set to the inner shape (the cache slot is one assembled
    buffer, never inner-tiled), keeping the geometry self-consistent."""
    inner, dt = geom.inner_shape, geom.dtype
    for t in transforms:
        declare = getattr(t, "output_inner", None)
        if declare is None:
            continue
        current = replace(
            geom, shape=(geom.shape[0], *inner), chunks=(geom.chunks[0], *inner), dtype=dt
        )
        inner, dt = declare(current)
        inner, dt = tuple(int(s) for s in inner), np.dtype(dt)
    return replace(geom, shape=(geom.shape[0], *inner), chunks=(geom.chunks[0], *inner), dtype=dt)


def _transform_token(fn: ChunkTransform) -> str:
    """A stable identity for one chunk_transform, for the cross-run cache fingerprint.

    Precedence: an explicit ``fn.cache_key`` (user-owned, strongest) -> a cloudpickle
    hash if available (captures closure cells + referenced globals) -> a best-effort
    source/qualname hash (catches an edited body, but NOT a changed closed-over constant
    or a called helper -- the same blind spot joblib has). The method is encoded in the
    token, so toggling cloudpickle on/off changes the fingerprint (honest re-compute
    rather than a false match).

    On the source path a *class-based* transform (a callable instance, e.g. a dataclass
    like :class:`StandardScaler`) is hashed by its **class** source + qualname, not the
    instance: ``inspect.getsource(instance)`` raises and the default ``object`` repr embeds
    the object's memory address, so the token would be unstable across runs (a spurious
    cache miss on every reopen). A stable, non-default ``__repr__`` (dataclasses, partials)
    is folded in so instance config still affects the token; the address-bearing default is not.
    """
    key = getattr(fn, "cache_key", None)
    if key is not None:
        return f"key:{key}"
    if cloudpickle is not None:
        with contextlib.suppress(Exception):  # unpicklable -> fall through to source
            return "pickle:" + hashlib.sha256(cloudpickle.dumps(fn)).hexdigest()
    # Hash the routine itself, or the class of a callable instance (never the instance --
    # its default repr carries an unstable address).
    target = fn if inspect.isroutine(fn) else type(fn)
    try:
        src = inspect.getsource(target)
    except (OSError, TypeError):
        src = ""  # C funcs / REPL: no source -> lean on the qualname alone
    name = getattr(target, "__qualname__", repr(target))
    # A class-defined repr (dataclass/partial) is stable and captures config; skip the
    # default object repr, which would reintroduce the address.
    config = (
        repr(fn) if not inspect.isroutine(fn) and type(fn).__repr__ is not object.__repr__ else ""
    )
    return "src:" + hashlib.sha256(f"{name}\n{src}\n{config}".encode()).hexdigest()


def _has_weak_token(transforms: Sequence[ChunkTransform]) -> bool:
    """True if any transform falls back to the source hash (no cache_key, no cloudpickle)."""
    return cloudpickle is None and any(getattr(t, "cache_key", None) is None for t in transforms)


@dataclass(slots=True)
class _Slot:
    """One outer chunk's cache slot plus its completion bookkeeping.

    ``data`` is the cache slot, sized at the transform's *output* geometry. ``scratch`` is
    a transient *source*-shaped assembly buffer, present only when a reshaping transform
    means the slot cannot also be the assembly target; tiles scatter into it and the
    transformed result lands in ``data`` (then ``scratch`` is dropped). With no reshaping
    transform ``scratch`` is ``None`` and tiles scatter straight into ``data`` -- the slot
    is buffer and cache in one, no extra copy (the common path)."""

    data: np.ndarray
    remaining: int  # inner tiles not yet scattered; 0 => fully assembled
    nbytes: int  # slot size, charged to the budget (fixed at admit)
    ready: bool = False
    error: BaseException | None = None
    claimed: bool = False  # the driver has referenced it *this epoch* (see wait_ready)
    scratch: np.ndarray | None = None  # source-shaped assembly buffer (reshaping path only)


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

    _MANIFEST_NAME = "insitu_cache.jsonl"
    _MANIFEST_FORMAT = 2

    def __init__(
        self,
        geometries: dict[str, ArrayGeometry],
        *,
        chunk_transforms: Sequence[ChunkTransform] = (),
        backing_dir: str | Path | None = None,
        budget_bytes: int | None = None,
        persist: bool = False,
        reset_stale_cache: bool = False,
    ) -> None:
        self._geom = geometries  # label -> geometry (a label is one (array, offset) view)
        # Slots are keyed by the underlying array *path*, not the variable label, so two
        # views of one array (e.g. t2m_now / t2m_next) share a single decode. One
        # representative geometry per path suffices for slot sizing (aliases share shape).
        self._by_path = {g.path: g for g in geometries.values()}
        self._chunk_transforms = tuple(chunk_transforms)
        # Output geometry after the chunk_transform pipeline. A reshaping transform (regrid /
        # dtype recast) makes the cached chunk differ from the source, so everything
        # *downstream of assembly* -- slot sizing, the cache budget, gather, the revive
        # structural check -- is sized at the OUTPUT geometry, while tile assembly stays at
        # the SOURCE geometry (in scratch). With no reshaping transform out == source.
        self._out_geom = {
            label: output_geometry(g, self._chunk_transforms) for label, g in geometries.items()
        }
        self._out_by_path = {g.path: self._out_geom[label] for label, g in geometries.items()}
        self._reshapes = {
            p: (
                out.inner_shape != self._by_path[p].inner_shape
                or out.dtype != self._by_path[p].dtype
            )
            for p, out in self._out_by_path.items()
        }
        # backing: heap (np.empty) or mmap'd .npy under backing_dir (point at NVMe).
        # The scatter writes straight into the slot either way; mmap keeps the working
        # set as reclaimable page cache rather than anon heap. Default heap: scattering
        # into mmap is NVMe write traffic even when never reused, so reach for it to
        # spill a working set past RAM or for cross-epoch reuse, not for plain streaming.
        self._dir = Path(backing_dir) if backing_dir is not None else None
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
        # Observability. hits/misses (+ the revive failure breakdown) are per-epoch --
        # reset by unpin_all at each epoch boundary -- so the driver can warn when a
        # configured persist cache served nothing. manifest_entries is load-time (how
        # many entries a prior run left us) and does NOT reset.
        self.hits = 0
        self.misses = 0
        self.revive_mismatch = 0  # persisted entry whose stored shape/dtype no longer matches
        self.revive_missing = 0  # persisted entry whose .npy was unreadable/gone
        self.manifest_entries = 0
        # Cross-run persistence: keep slot files past close, write a manifest of completed
        # entries, and revive them on reopen. Requires a dir to keep the files in. The dir
        # path is the dataset+pipeline identity (the user buries a version in it); we only
        # auto-check shape/dtype on revive (a mismatch is a miss, not an error).
        self._persistent = persist
        # When the on-disk cache is *stale* (its chunk_transform fingerprint or the log format
        # differs from this run's), the default is to fail fast -- a stale cache is almost never
        # what the user intended. Setting this opts into deleting the stale files and rebuilding.
        self._reset_stale_cache = reset_stale_cache
        if persist and self._dir is None:
            raise ValueError("persist=True requires cache_dir (a backing_dir) to keep files in")
        # key -> on-disk filename for completed entries known to survive a run.
        self._persisted: dict[tuple[str, int], str] = {}
        # Keys already written to the on-disk log this pool's lifetime (loaded entries + entries
        # appended on completion). Gates the append so re-completing a chunk across epochs/runs
        # never duplicates a line -- the log is self-deduplicating and bounded to O(#chunks).
        self._recorded: set[tuple[str, int]] = set()
        # The append-only manifest handle (persist mode), held open for the pool's lifetime so a
        # completion is one write()+flush() -- no per-chunk open(). None in heap/spill mode.
        self._log: TextIO | None = None
        # Fingerprint of the chunk_transform pipeline (only chunk_transforms are baked into
        # cached chunks; batch_transforms run post-cache). A run whose fingerprint differs
        # from the manifest's discards the cache (changed transforms -> stale). batch
        # transforms and the store identity are out of scope (the cache_dir path is the
        # dataset identity -- see the class docstring).
        self._pipeline_fp = ""
        if persist:
            self._pipeline_fp = hashlib.sha256(
                "\n".join(_transform_token(t) for t in self._chunk_transforms).encode()
            ).hexdigest()
            if _has_weak_token(self._chunk_transforms):
                logger.warning(
                    "persist: cloudpickle not installed and a chunk_transform has no "
                    "cache_key -> cache invalidation on transform changes is best-effort "
                    "(source only; closure/global changes may not invalidate). Install "
                    "`insitubatch[cache]` or set a `cache_key` attribute for a stronger guarantee."
                )
            self._load_log()
            self._open_log()
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
        src, out = self._by_path[array], self._out_by_path[array]
        # The cache slot is the OUTPUT geometry (what a reshaping transform produces and what
        # gather reads); the budget is charged for it. Assembly tiles are SOURCE-shaped, so a
        # reshaping path also needs a transient source-shaped scratch buffer (not cached).
        nbytes = int(np.prod(out.slot_shape(chunk_index), dtype=np.int64)) * out.dtype.itemsize
        with self._cv:
            if key in self._slots:
                self._pin(key)  # already resident (in-flight or ready hit) -> incref, reuse
                self._slots[key].claimed = True
                self._cv.notify_all()  # a ready hit may now satisfy a waiter
                return True
            if not self._make_room(nbytes):
                return False
            self.misses += 1  # a fresh slot allocated to fetch -> a cache miss
            scratch = (
                np.empty(src.slot_shape(chunk_index), dtype=src.dtype)
                if self._reshapes[array]
                else None
            )
            self._slots[key] = _Slot(
                data=self._alloc(array, chunk_index, out.slot_shape(chunk_index), out.dtype),
                remaining=src.n_inner_chunks(chunk_index),
                nbytes=nbytes,
                claimed=True,
                scratch=scratch,
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

        A cross-epoch (or, with ``persist``, cross-*run*) cache hit the driver can skip
        fetching -- but it must still be referenced so it stays resident through the
        consumer's use (released at last use like an admitted chunk), else it could be
        evicted before the waiter gathers it and, since the driver fetches each chunk
        once, deadlock. One lock so the check and the incref cannot race an eviction in
        between. A persisted-on-disk chunk is revived here on first touch (see
        :meth:`_revive`), so a cross-run hit costs no fetch.
        """
        with self._cv:
            key = (array, chunk_index)
            slot = self._slots.get(key)
            if not (slot is not None and slot.ready and slot.error is None):
                if not self._revive(key):
                    return False
                slot = self._slots[key]
            self.hits += 1  # resident (cross-epoch) or revived (cross-run) -> no fetch
            self._pin(key)
            slot.claimed = True  # publish the claim so a waiter may now proceed
            self._cv.notify_all()
            return True

    def _revive(self, key: tuple[str, int]) -> bool:  # call under the lock
        """Bring a persisted on-disk chunk back as a ready slot (a cross-run hit).

        Returns ``True`` iff the slot is now resident + ready. Validates the stored
        ``.npy`` shape/dtype against the current geometry; a mismatch (or an unreadable
        file) is a **miss** -- the entry is dropped from the registry and the stale file
        is overwritten when the chunk is next fetched. Charges the slot to the budget,
        evicting unpinned-LRU for room; if none can be freed it stays a miss (the driver
        re-fetches -- correct, just uncached this once).
        """
        if not self._persistent or key in self._slots:
            return False
        fname = self._persisted.get(key)
        if fname is None:
            return False
        array, chunk_index = key
        geom = self._out_by_path.get(array)  # the persisted .npy holds the post-transform chunk
        assert self._dir is not None
        try:
            data = np.lib.format.open_memmap(self._dir / fname, mode="r")
        except (OSError, ValueError) as exc:
            self.revive_missing += 1
            logger.debug("cache: persisted %s unreadable (%s); refetching", key, exc)
            self._persisted.pop(key, None)
            return False
        if geom is None or data.shape != geom.slot_shape(chunk_index) or data.dtype != geom.dtype:
            self.revive_mismatch += 1
            logger.debug(
                "cache: persisted %s shape/dtype %s/%s != current %s/%s; refetching",
                key,
                data.shape,
                data.dtype,
                None if geom is None else geom.slot_shape(chunk_index),
                None if geom is None else geom.dtype,
            )
            del data  # drop the mmap ref; structural fingerprint mismatch -> a miss
            self._persisted.pop(key, None)
            return False
        nbytes = int(data.nbytes)
        if not self._make_room(nbytes):
            del data
            return False
        self._slots[key] = _Slot(data=data, remaining=0, nbytes=nbytes, ready=True)
        self._bytes += nbytes
        self.max_resident = max(self.max_resident, len(self._positions()))
        return True

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
            # Per-epoch observability resets here (the epoch boundary); manifest_entries
            # is load-time and persists.
            self.hits = self.misses = 0
            self.revive_mismatch = self.revive_missing = 0
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
        # In persist mode a *ready* eviction is a cache demotion, not a deletion: keep the
        # .npy on disk so a later epoch/run can revive it. It was already recorded in the log at
        # completion (see scatter -> _record_completed), so eviction touches only the backing.
        # A not-ready partial is garbage either way -> unlink it.
        keep = self._persistent and slot.ready
        self._free(slot, keep_file=keep)

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
        # Tiles assemble at the SOURCE shape: into scratch on a reshaping path, else straight
        # into the slot (which is then both buffer and cache). Disjoint, fixed-shape, lock-free.
        buffer = slot.data if slot.scratch is None else slot.scratch
        buffer[dst] = tile[src]  # disjoint, fixed-shape: lock-free (rule 1)

        with self._cv:
            slot.remaining -= 1
            last = slot.remaining == 0
        if not last:
            return

        # Sole owner now: assemble-stage transforms on the whole (source-shaped) outer chunk,
        # then land the (possibly reshaped) result in the output-sized slot.
        prepped = self._apply_transforms(array, chunk_index, buffer)
        with self._cv:
            slot.data = self._persist(slot.data, prepped)
            slot.scratch = None  # assembly done -- drop the transient source-shaped buffer
            slot.ready = True
            # Record the completed entry *now* (not at eviction/close) so a crash still leaves a
            # usable cache. Appending here, under the lock, serializes writes across decode
            # threads and orders them after the slot's data is durable in its .npy.
            self._record_completed(key, slot)
            self._cv.notify_all()

    def _persist(self, current: np.ndarray, prepped: np.ndarray) -> np.ndarray:
        """Land the prepped (post-transform) array in the slot's (output-sized) backing.

        No transform (``prepped is current``) is a no-op -- the scatter already wrote the
        slot. Otherwise heap just holds the new array; mmap writes it back into the slot's
        file so the cached chunk stays on NVMe. The slot is sized at the transform's *output*
        geometry (see :func:`output_geometry`), so a reshaping transform lands here exactly
        like a shape-preserving one -- ``prepped.shape`` matches the slot by construction. A
        mismatch means ``__call__`` disagreed with its declared ``output_inner``: a bug, raised.
        """
        if prepped is current or self._dir is None:
            return prepped
        if prepped.shape != current.shape:
            raise ValueError(
                f"chunk_transform produced shape {prepped.shape} but the cache slot is sized "
                f"{current.shape} from the declared output geometry; a reshaping transform's "
                "output_inner must agree with what __call__ returns."
            )
        current[:] = prepped  # write the transformed result into the memmap (casts to slot dtype)
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
            geom = self._geom[var]  # source: drives the read math (offset, path, chunking)
            out_geom = self._out_geom[var]  # post-transform: shape/dtype the consumer sees
            sample = anchor + geom.offset  # this view reads array[anchor + offset]
            read_cid = sample // spc
            within = sample % spc
            out = np.empty((n, *out_geom.inner_shape), dtype=out_geom.dtype)
            for cid in np.unique(read_cid):
                mask = read_cid == cid  # rows that read this chunk -> one coalesced index
                out[mask] = self._slots[(geom.path, int(cid))].data[within[mask]]
            arrays[var] = out
        offsets = {var: self._geom[var].offset for var in variables}
        return Batch(arrays=arrays, sample_indices=anchor, offsets=offsets)

    def _free(self, slot: _Slot, *, keep_file: bool) -> None:
        """Release a slot's backing: a no-op for heap, close (and maybe unlink) for mmap.

        ``keep_file`` leaves the ``.npy`` on disk (a persisted cache entry); otherwise the
        file is unlinked (heap/spill teardown or a discarded partial).
        """
        mmap = getattr(slot.data, "_mmap", None)
        if mmap is not None:
            fname = getattr(slot.data, "filename", None)
            mmap.close()
            if fname and not keep_file:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(fname)

    @staticmethod
    def _is_bare_filename(fname: object) -> bool:
        """True iff ``fname`` is a bare basename safe to resolve inside the cache dir.

        An absolute path, a ``..`` component, or any separator would escape via
        ``self._dir / fname`` in :meth:`_revive`. ``Path('..').name`` is ``'..'`` and a
        Windows-style separator is an ordinary char to posix ``basename``, so check the
        separators explicitly rather than trusting ``Path.name``.
        """
        return (
            isinstance(fname, str)
            and fname not in ("", ".", "..")
            and "/" not in fname
            and "\\" not in fname
        )

    def _record_completed(self, key: tuple[str, int], slot: _Slot) -> None:  # under the lock
        """Register a freshly completed slot's ``.npy`` as a surviving cache entry and append
        it to the log -- exactly once per key.

        Recording at *completion* (not at eviction/close) is what makes the cache crash-safe: a
        killed process still leaves every finished chunk in the log. Idempotent per key -- a
        re-completion in a later epoch (after eviction + refetch) refreshes ``_persisted`` but
        the ``_recorded`` gate skips the duplicate log line, so the log stays bounded.
        """
        if not self._persistent:
            return
        fname = getattr(slot.data, "filename", None)
        if fname is None:
            return  # heap backing -- nothing on disk to record
        fname = Path(fname).name
        self._persisted[key] = fname
        if self._log is not None and key not in self._recorded:
            array, chunk_index = key
            self._append_entry(array, chunk_index, fname)
            self._recorded.add(key)

    def _append_entry(self, array: str, chunk_index: int, fname: str) -> None:  # under the lock
        """Append one completed-entry line to the open log and flush it to the page cache.

        ``flush`` (not ``fsync``) makes the entry durable against *process death* -- the target
        failure mode (spot preemption / OOM / SIGTERM); the kernel flushes the page cache. Power
        loss (which would need ``fsync`` per chunk) is out of scope.
        """
        assert self._log is not None
        self._log.write(json.dumps({"array": array, "chunk_index": chunk_index, "file": fname}))
        self._log.write("\n")
        self._log.flush()

    def _open_log(self) -> None:
        """Open the append-only manifest for the pool's lifetime; write the header on a cold
        start (or after a stale-cache reset removed the file). A warm reopen appends after the
        existing entries -- the ``_recorded`` gate (populated by :meth:`_load_log`) keeps those
        from being re-appended."""
        assert self._dir is not None
        path = self._dir / self._MANIFEST_NAME
        fresh = not path.exists()
        self._log = path.open("a")
        if fresh:
            header = {"format_version": self._MANIFEST_FORMAT, "pipeline_hash": self._pipeline_fp}
            self._log.write(json.dumps(header))
            self._log.write("\n")
            self._log.flush()

    def _load_log(self) -> None:
        """Populate the persisted-entry registry from a prior run's append-only log, if any.

        No log is a cold start (silent -- the expected first run). A log whose header ``format``
        or ``pipeline_hash`` differs from this run is a **stale** cache: by default that *raises*
        (a stale cache is almost never what the user intended), or -- with ``reset_stale_cache``
        -- it deletes the listed files + the log and rebuilds (see :meth:`_reset_stale`).

        Corruption always raises, regardless of the flag: an unreadable header, a malformed
        *interior* entry, or a ``file`` that is not a bare filename (an absolute or ``..`` path
        would let :meth:`_revive` ``open_memmap`` escape the cache dir -- path-traversal
        tampering). A torn *final* line (a crash mid-append) is expected and dropped silently.
        """
        assert self._dir is not None
        path = self._dir / self._MANIFEST_NAME
        if not path.exists():
            return  # cold start: no prior cache (the expected first run)
        lines = path.read_text().splitlines()
        if not lines:
            return  # header not yet flushed (a crash before the first write) -- treat as cold
        try:
            header = json.loads(lines[0])
            fmt, fp = header["format_version"], header["pipeline_hash"]
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError(
                f"persist: cache log {path} has an unreadable header ({exc}); this is corruption "
                f"or tampering, not a stale cache -- delete {self._dir} to reset."
            ) from exc
        entry_lines = lines[1:]
        if fmt != self._MANIFEST_FORMAT or fp != self._pipeline_fp:
            why = "log format" if fmt != self._MANIFEST_FORMAT else "chunk_transform fingerprint"
            self._reset_stale(path, entry_lines, why)
            return
        for i, line in enumerate(entry_lines):
            try:
                rec = json.loads(line)
                array, chunk_index, fname = rec["array"], int(rec["chunk_index"]), rec["file"]
            except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
                if i == len(entry_lines) - 1:
                    break  # torn tail: the crash landed mid-append on the last line -- drop it
                raise ValueError(
                    f"persist: cache log {path} has a malformed entry on line {i + 2} ({exc}); "
                    f"this is corruption or tampering, not a stale cache -- delete {self._dir} "
                    "to reset."
                ) from exc
            if not self._is_bare_filename(fname):
                raise ValueError(
                    f"persist: cache log {path} entry file {fname!r} is not a bare filename "
                    f"(possible path-traversal tampering); delete {self._dir} to reset."
                )
            key = (array, chunk_index)
            self._persisted[key] = fname
            self._recorded.add(key)
        self.manifest_entries = len(self._persisted)

    def _reset_stale(self, path: Path, entry_lines: list[str], why: str) -> None:
        """A stale cache: raise by default, or (``reset_stale_cache``) GC + rebuild.

        The GC deletes exactly the ``.npy`` files this stale log named -- each re-checked as a
        bare filename before ``unlink`` (a tampered path is never removed) -- then the log
        itself. Precise: only files we recorded writing; crash-orphans are already in the log.
        """
        assert self._dir is not None
        if not self._reset_stale_cache:
            raise ValueError(
                f"persist: cache at {self._dir} is stale ({why} changed since it was written). "
                "This is not corruption -- pass reset_stale_cache=True to delete and rebuild it, "
                f"or remove {self._dir} yourself."
            )
        for line in entry_lines:
            try:
                fname = json.loads(line)["file"]
            except (json.JSONDecodeError, TypeError, KeyError):
                continue  # a garbage/torn line in a log we're discarding anyway -- skip
            if self._is_bare_filename(fname):
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(self._dir / fname)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        logger.info("persist: stale cache at %s (%s changed) reset; rebuilding.", self._dir, why)

    def close(self) -> None:
        """Free every remaining slot. Persist keeps ready cache files (each already recorded in
        the log at completion, so there is nothing to rewrite -- just flush + close the handle);
        heap/spill mmap files are unlinked."""
        with self._cv:
            if self._log is not None:
                self._log.flush()
                self._log.close()
                self._log = None
            for k in list(self._slots):
                slot = self._slots.pop(k)
                self._free(slot, keep_file=self._persistent and slot.ready)
            self._bytes = 0
            self._pinned.clear()
