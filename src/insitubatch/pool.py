"""ChunkPool: the residency tier *and* the cache -- one machinery.

The scheduler fetches at *stored chunk* (tile) granularity and hands each decoded
tile to the pool, which **adopts it by reference**: the tile *is* the resident
buffer, so the fill path has no memcpy and placement is deferred to ``gather``. A
slot is therefore a grid of stored chunks, not one assembled array. The pool owns
them across their whole life: admit (charge the tiles it will hold, evicting
unpinned-LRU for room) -> deliver tiles (from the decode pool) -> mark ready (when
the last tile lands) -> gather batches (one coalesced write per stored chunk, on
the consumer thread) -> unpin (when a shuffle-block is fully drained).

A slot that must publish a *whole* array -- only a ``chunk_transform``, which must
be handed a real array rather than a dict of tiles -- assembles at completion and
republishes as a **one-tile** slot, so ``gather`` still sees one representation.

Buffer and cache differ only in budget + backing, so they are the same code:

* a **byte budget** with **pin/unpin + LRU** eviction. A chunk is pinned while the
  current epoch needs it; unpinned chunks stay resident (retained) until budget
  pressure evicts them in LRU order. A small budget is read-once (unpinned chunks
  evicted promptly); a large budget retains drained chunks so a still-resident
  prepped chunk is a hit next epoch -- decode-once reuse.
* a **backing**: heap (the adopted tile itself) or an mmap'd ``.npy`` on NVMe
  (``open_memmap``), which stores a chunk's tiles **tile-major** -- each contiguous,
  written straight into its slice, with the slot keeping the view. mmap keeps the
  working set as reclaimable page cache rather than anon heap.

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
Delivery runs on many decode-pool threads at once -- including two tiles of the
*same* outer chunk concurrently (inner-grid parallelism). We never lock the data
write; we lock only the Python-level bookkeeping. Two rules make this correct
under both the GIL build *and* free-threaded 3.13t (where the GIL is no longer an
incidental barrier):

1. **Deliveries do not collide.** Each tile is its own key in ``slot.tiles``
   (its ``inner_coord``), and on the mmap tier its own disjoint, fixed-shape slice
   of the backing. No two writers touch the same bytes, so the write races nothing.
2. **Readiness is published *through the lock*, after the write.** Each delivery
   lands its tile, *then* takes the lock to decrement the completion counter.
   The consumer observes readiness under the same lock. That lock release ->
   acquire pair is the happens-before edge guaranteeing the consumer sees every
   completed write -- we do not lean on the GIL for it.

So the GIL build is just the serialized (slower) case; free threading is upside.
"""

from __future__ import annotations

import codecs
import contextlib
import hashlib
import inspect
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import cast

import numpy as np

from .buffers import BatchBuffers, BufferStats, HostAllocator
from .types import ArrayGeometry, Batch, ChunkRead, DecodedChunk

try:  # optional: stronger transform fingerprint (closures + globals). `--extra cache`.
    import cloudpickle
except ImportError:  # pragma: no cover - exercised by the no-cloudpickle fallback path
    cloudpickle = None

try:  # POSIX advisory locking, which is what arbitrates writers of a shared cache dir.
    import fcntl
except ImportError:  # pragma: no cover - Windows; warned about at pool construction
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

ChunkTransform = Callable[[DecodedChunk], DecodedChunk]


#: Suffix of the private file a slot is written to before it is renamed into place
#: (see :meth:`ChunkPool._alloc`). A crash between the two leaves one behind; the next
#: writer to hold the cache lock sweeps them (:meth:`ChunkPool._sweep_tmp`).
_TMP_SUFFIX = ".insitu-tmp"

#: The advisory-lock file arbitrating writers of one cache dir (see :meth:`ChunkPool._lock`).
_LOCK_NAME = ".insitu.lock"

#: Filesystems on which the cache's arbitration does not hold. A **denylist**, not an
#: allowlist: an unrecognized local filesystem stays quiet, and only the types we know
#: break ``flock``/``rename`` semantics (or are ruinously slow for an mmap tier) warn.
#: NFS in particular: ``flock`` may be emulated per-client (so two hosts both "hold" it),
#: and a silently-dropped lock is exactly the case this arbitration exists to prevent.
_NETWORK_FILESYSTEMS = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "fuse.sshfs",
        "fuse.s3fs",
        "fuse.gcsfuse",
        "9p",
        "lustre",
        "ceph",
        "glusterfs",
    }
)


def _read_holder(fd: int) -> str:
    """The lockfile's self-reported holder, rendered for an error message.

    A **hint**: the ``flock`` is authoritative and this record can lie -- it names the last
    process to take the lock for writing, while the lock may currently be held by readers.
    Never let a parse failure mask the contention error we are on our way to raising.
    """
    try:
        rec = json.loads(os.pread(fd, 4096, 0).decode() or "{}")
        return f"PID {rec['pid']} on host {rec['host']}, since {rec['since']}"
    except Exception:  # noqa: BLE001 - any unreadable record degrades to "unknown"
        return "holder unknown -- the lockfile carries no readable record"


def _filesystem_type(path: str | Path) -> str | None:
    """The filesystem type backing ``path``, or ``None`` where we cannot tell.

    Linux only, read from ``/proc/self/mountinfo`` by longest-prefix match on the
    *resolved* path -- so a bind mount reports the type of what it is bound to, and a
    cache dir under a deeper mount is attributed to that mount rather than to ``/``.
    Returns ``None`` on macOS and Windows (no ``/proc``): we do not know, and a guess
    would be worse than the honest absence, since the caller only ever *warns* on it.
    """
    try:
        entries = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None
    target = os.path.realpath(path)
    best: tuple[int, str] | None = None
    for line in entries:
        # mountinfo: <id> <parent> <maj:min> <root> <mount point> <opts> [<tag>...] - <fstype> ...
        left, sep, right = line.partition(" - ")
        fields, rest = left.split(), right.split()
        if not sep or len(fields) < 5 or not rest:
            continue
        try:  # mount points escape spaces/tabs as octal (\040); decode before comparing
            mount = codecs.decode(fields[4], "unicode_escape")
        except UnicodeDecodeError:  # pragma: no cover - a mount point we cannot read
            continue
        if target != mount and not target.startswith(mount.rstrip("/") + "/"):
            continue
        # Longest prefix wins; on a tie the later line does, since a mount stacked on an
        # existing mount point shadows the one beneath it.
        if best is None or len(mount) >= best[0]:
            best = (len(mount), rest[0])
    return None if best is None else best[1]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def output_geometry(geom: ArrayGeometry, transforms: Sequence[ChunkTransform]) -> ArrayGeometry:
    """The geometry a chunk has *after* the chunk_transform pipeline.

    A reshaping transform (regrid / dtype recast) declares ``output_inner(geom) ->
    (inner_shape, dtype)``; the pipeline folds them so each transform sees the geometry
    produced by the ones before it. Only the *inner* dims and dtype may change -- the
    sample axis is spliced straight back from the source at its physical position, so a
    transform can neither move nor reshape it (the sample-geometry invariant). A transform
    without ``output_inner`` is identity; with none reshaping, the source geometry returns
    unchanged. Output ``chunks`` are set to the inner shape (the cache slot is one assembled
    buffer, never inner-tiled), keeping the geometry self-consistent."""

    def rebuild(inner_shape: tuple[int, ...], dt: np.dtype) -> ArrayGeometry:
        # Reinsert the (whole, single-chunk) sample dim at its physical axis so the
        # output geometry stays in physical order with the same sample_axis.
        ax, n, spc = geom.sample_axis, geom.n_samples, geom.sample_chunk_size
        shape = (*inner_shape[:ax], n, *inner_shape[ax:])
        chunks = (*inner_shape[:ax], spc, *inner_shape[ax:])
        return replace(geom, shape=shape, chunks=chunks, dtype=dt)

    inner, dt = geom.inner_shape, geom.dtype
    for t in transforms:
        declare = getattr(t, "output_inner", None)
        if declare is None:
            continue
        inner, dt = declare(rebuild(inner, dt))
        inner, dt = tuple(int(s) for s in inner), np.dtype(dt)
    return rebuild(inner, dt)


def slot_charge_bytes(
    src: ArrayGeometry, out: ArrayGeometry, chunk_index: int = 0, *, assembles: bool
) -> int:
    """Peak bytes one cache slot occupies. **The single rule for sizing and charging.**

    A slot holds the array's stored tiles, kept **whole** -- a chunk grid that does not
    divide the array evenly still stores full-size edge chunks (721 rows chunked at 180
    occupy 900), and we do not clip them. So residency is ``n_tiles * tile_shape``, which
    can exceed the assembled logical chunk.

    When the pool assembles (a ``chunk_transform`` is configured) the slot holds the source
    tiles for its whole fill and only collapses to the assembled output at completion, so
    the charge is the larger of the two. A transform that *shrinks* the data makes the
    tiles the binding term; one that grows it makes the output the binding term.

    :func:`InSituDataset` sizes its automatic budget with this too. They must not drift:
    sizing from the output shape while charging the tiles under-provisions the budget and
    the pool starves mid-epoch, which is a hang-shaped failure rather than a slow one.
    """
    tiles = (
        src.n_inner_chunks(chunk_index)
        * int(np.prod(src.tile_shape(), dtype=np.int64))
        * src.dtype.itemsize
    )
    if not assembles:
        return tiles
    assembled = int(np.prod(out.slot_shape(chunk_index), dtype=np.int64)) * out.dtype.itemsize
    return max(tiles, assembled)


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


class SlotState(Enum):
    """Where a slot is in its one and only lifecycle.

    ``FILLING -> ASSEMBLED -> READY``, with ``FAILED`` terminal. Exactly one function
    advances it (:meth:`ChunkPool._advance`), so "is this gatherable" and "is this safe
    to take away" are read off one field instead of inferred from agreeing flags.

    ``ASSEMBLED`` is a *real* state, not a formality: the chunk transform and the
    persist write-back run **outside** the lock between the last tile landing and the
    slot being published, so another thread can observe the slot with every tile
    delivered and the data not yet final. A predicate derived only from a tile counter
    would call that window evictable; this one does not.
    """

    FILLING = "filling"
    ASSEMBLED = "assembled"
    READY = "ready"
    FAILED = "failed"


@dataclass(slots=True)
class _Slot:
    """Everything the pool holds for one outer chunk, and the bookkeeping that says when
    it is complete and when it is safe to take away.

    A slot is a unit of **residency**: one array's chunk along the sample axis, held as
    the stored chunks it is made of. **The buffer unit is the stored chunk** -- ``tiles``
    maps each inner coordinate to the decoded tile itself (whatever a single ``store.get``
    returns), adopted by reference, so the tile *is* the resident buffer and the fill path
    has no memcpy. ``gather`` places each tile straight into its sub-rectangle of the
    batch.

    Admission, pinning, the byte budget, eviction and ``wait_ready`` all work at slot
    granularity, so a chunk is never half-resident even though it arrives one tile at a
    time.

    A chunk that needs a whole array collapses to a **one-tile** slot rather than a second
    representation: ``output_geometry`` already sets post-transform ``chunks`` to the full
    inner shape, so a transformed chunk is by definition a 1x1 grid, and ``tiles`` holds
    ``{(0, ...): transformed}``. Same for the mmap tier. So there is exactly one thing
    ``gather`` can find, and no "assembled or tiled?" dispatch anywhere.

    ``scratch`` is the transient *source*-shaped assembly buffer used only while a
    transform runs (it needs one contiguous array); it is dropped at publication.

    Tiles are stored **whole, never clipped** -- one buffer unit, one shape. Clipping edge
    tiles would create two kinds of tile and a ragged shape, which defeats a fixed-shape
    arena later. The padding is therefore real residency, and is charged as such at admit.

    Two counters, because quiescence and completeness are different questions and one
    number cannot answer both:

    * ``writers`` -- tile tasks **currently running**. Incremented when a task enters
      :meth:`ChunkPool.tile_write` and decremented exactly once on every exit path,
      cancellation included. It counts *started* tasks, not planned ones: a pass that is
      cancelled leaves some tiles never started, and counting those would keep the slot
      permanently non-quiescent and its budget unreclaimable. ``writers == 0`` means
      **no thread can still write here**, which is the only safe basis for eviction.
    * ``pending`` -- tiles not yet *delivered*. Reaches 0 only when the chunk is
      genuinely complete. A slot that quiesces with ``pending > 0`` is an abandoned
      partial (its fetch was cancelled), never a cache entry.
    """

    tiles: dict[tuple[int, ...], np.ndarray]  # inner_coord -> stored tile (the residency)
    backing: np.ndarray | None  # the whole tile-major .npy memmap, when persisting
    array: str  # zarr path this slot belongs to
    chunk_index: int  # its outer (sample-axis) chunk
    writers: int  # outstanding tile TASKS -> quiescence
    pending: int  # tiles not yet delivered -> completeness
    nbytes: int  # slot size, charged to the budget (fixed at admit)
    state: SlotState = SlotState.FILLING
    error: BaseException | None = None
    scratch: np.ndarray | None = None  # source-shaped assembly buffer (reshaping path only)

    @property
    def quiescent(self) -> bool:
        """No tile task can still write into this slot."""
        return self.writers == 0


@dataclass(slots=True)
class _TileWrite:
    """One tile task's write scope. See :meth:`ChunkPool.tile_write`.

    Holds the release obligation so no caller has to remember it. ``released`` makes
    the release idempotent: ``deliver`` releases as soon as the copy lands (so the slot
    can publish without waiting for the coroutine to unwind), and the scope's
    ``finally`` then finds nothing to do.
    """

    pool: ChunkPool
    array: str
    chunk_index: int
    inner_coord: tuple[int, ...]
    released: bool = False

    def deliver(self, tile: np.ndarray) -> None:
        """Land this tile in the slot, then release the write."""
        self.pool._deliver(self.array, self.chunk_index, self.inner_coord, tile)
        self.release()

    def fail(self, error: BaseException) -> None:
        """Poison the whole outer chunk, then release the write."""
        self.pool.fail(self.array, self.chunk_index, error)
        self.release()

    def release(self) -> None:
        """Drop this task's claim on the slot. Idempotent; never raises."""
        if self.released:
            return
        self.released = True
        slot = self.pool._slots.get((self.array, self.chunk_index))
        if slot is None:  # already dropped (poisoned + unreferenced) -- nothing to release
            return
        with self.pool._cv:
            slot.writers -= 1
        self.pool._advance(self.array, self.chunk_index)


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

    **One writer per ``backing_dir``** (#42). Whenever a backing dir is set -- with or
    without ``persist``, since ``_alloc`` writes the same filenames either way -- the pool
    takes an advisory lock on it for its lifetime and a second writer fails fast
    (:meth:`_lock`). ``readonly_cache=True`` takes that lock *shared* instead: many such
    openers coexist with each other, none with a writer, none of them writes anything, and
    a miss **raises** (:meth:`_readonly_miss`) rather than fetching -- the flag is an
    assertion that this cache is complete for what the run reads. Slot files are replaced,
    never truncated in place (:meth:`_alloc`), so a reader holding a mapping keeps reading
    real data even while a writer re-admits the same chunk.
    """

    _MANIFEST_NAME = "insitu_cache.jsonl"
    _MANIFEST_FORMAT = 3
    """Bumped to 3: a persisted chunk is now stored **tile-major**.

    A ``.npy`` holds ``(n_tiles, *tile_shape)`` -- each stored tile contiguous, in
    :meth:`ArrayGeometry.inner_index` order -- rather than one assembled
    ``slot_shape`` array. File count per chunk is unchanged. A version-2 cache is
    structurally unreadable under the new layout, so the header check rejects the whole
    log and refetches rather than misreading it as data."""

    def __init__(
        self,
        geometries: dict[str, ArrayGeometry],
        *,
        chunk_transforms: Sequence[ChunkTransform] = (),
        backing_dir: str | Path | None = None,
        budget_bytes: int | None = None,
        persist: bool = False,
        readonly_cache: bool = False,
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
        # Does a slot hold the STORED tiles, or one whole array? Only a chunk_transform
        # forces the latter: user code must be handed a real array, never a dict of tiles.
        # The mmap tier does NOT force it -- a persisted chunk stores its tiles tile-major
        # (see `_tile_grid`), so the cache round-trips tiles rather than assembling them.
        self._whole_chunks = bool(self._chunk_transforms)
        self._reshapes = {
            p: (
                out.inner_shape != self._by_path[p].inner_shape
                or out.dtype != self._by_path[p].dtype
            )
            for p, out in self._out_by_path.items()
        }
        # Cross-process arbitration (#42). Validate before creating anything: a rejected
        # configuration should leave no directory and no lockfile behind.
        if persist and backing_dir is None:
            raise ValueError("persist=True requires cache_dir (a backing_dir) to keep files in")
        if readonly_cache and backing_dir is None:
            raise ValueError("readonly_cache=True requires cache_dir -- there is no cache to read")
        if readonly_cache and reset_stale_cache:
            raise ValueError(
                "readonly_cache=True and reset_stale_cache=True contradict each other: a "
                "read-only opener may not delete a cache another process may be reading. "
                "Reset it from the run that writes it."
            )
        self._readonly = readonly_cache
        self._lock_fd: int | None = None
        # backing: heap (np.empty) or mmap'd .npy under backing_dir (point at NVMe).
        # A heap slot adopts the tile; the mmap tier writes it into its tile-major slice
        # and keeps the view. mmap keeps the working set as reclaimable page cache rather
        # than anon heap. Default heap: writing
        # into mmap is NVMe write traffic even when never reused, so reach for it to
        # spill a working set past RAM or for cross-epoch reuse, not for plain streaming.
        self._dir = Path(backing_dir) if backing_dir is not None else None
        if self._dir is not None:
            if readonly_cache and not self._dir.is_dir():
                raise ValueError(
                    f"readonly_cache=True but {self._dir} does not exist. A read-only opener "
                    "reads a cache another run warmed; it never creates one."
                )
            self._dir.mkdir(parents=True, exist_ok=True)
            self._warn_environment()
            # The lock keys on cache_dir being set, NOT on persist: `_alloc` writes
            # `{array}__{cid}.npy` whenever a backing dir is set, so two processes sharing a
            # spill dir with persist=False collide on identical filenames just as surely.
            self._lock()
            self._sweep_tmp()
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
        # readonly_cache reads the cross-run cache without writing it, so it needs the same
        # revive machinery persist does -- `persist` only additionally *writes*.
        self._persistent = persist or readonly_cache
        # When the on-disk cache is *stale* (its chunk_transform fingerprint or the log format
        # differs from this run's), the default is to fail fast -- a stale cache is almost never
        # what the user intended. Setting this opts into deleting the stale files and rebuilding.
        self._reset_stale_cache = reset_stale_cache
        # key -> on-disk filename for completed entries known to survive a run.
        self._persisted: dict[tuple[str, int], str] = {}
        # Keys already written to the on-disk log this pool's lifetime (loaded entries + entries
        # appended on completion). Gates the append so re-completing a chunk across epochs/runs
        # never duplicates a line -- the log is self-deduplicating and bounded to O(#chunks).
        self._recorded: set[tuple[str, int]] = set()
        # The append-only manifest fd (persist mode), held open for the pool's lifetime so a
        # completion is one os.write() -- no per-chunk open(). None in heap/spill mode, and in
        # readonly_cache mode, which reads the log and never appends to it.
        self._log_fd: int | None = None
        # Fingerprint of the chunk_transform pipeline (only chunk_transforms are baked into
        # cached chunks; batch_transforms run post-cache). A run whose fingerprint differs
        # from the manifest's discards the cache (changed transforms -> stale). batch
        # transforms and the store identity are out of scope (the cache_dir path is the
        # dataset identity -- see the class docstring).
        self._pipeline_fp = ""
        if self._persistent:
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
            try:
                self._load_log()
                if not readonly_cache:
                    self._open_log()
            except BaseException:
                # A stale cache raises here, and the pool never becomes an object anyone can
                # close -- so release the lock ourselves. `__del__` cannot: `close()` reads
                # attributes this half-built pool does not have yet, and its own error
                # suppression would swallow that, leaving the dir locked until the process
                # exits. The user is expected to fix the configuration and construct again.
                self._release_lock()
                raise
        # Batch *output* buffers, distinct from the chunk slots below: gather lends one per
        # variable per batch and reclaims it once the consumer's view is unreferenced. It
        # carries its own lock, and needs it -- one ChunkPool is shared by every active
        # iteration, so `zip(ds.train, ds.val)` or two DataLoaders put two producer threads
        # through `gather` at once. Do not remove that lock on the assumption that gather is
        # single-threaded: two producers deciding the same buffer is free is a silent
        # wrong-data bug. See BatchBuffers' own docstring.
        self._buffers = BatchBuffers()
        self._budget = budget_bytes  # None => unbounded (never self-evicts)
        self._bytes = 0
        # OrderedDict in recency order (LRU front -> MRU back). Eviction targets only
        # a *ready, unreferenced* slot: a not-ready slot is an in-flight fetch, and a
        # refcounted slot is held by a live block (windows let one chunk be read by
        # several blocks at once, so pins are reference-counted, not a boolean).
        self._slots: OrderedDict[tuple[str, int], _Slot] = OrderedDict()
        # key -> {owner: refcount}. Owner-scoped because one pool serves several
        # concurrent iterations (`zip(ds.train, ds.val)`, buffers.py:238-247): a single
        # flat map lets one iteration's epoch prologue release another's live pins, and a
        # single `claimed` bool lets one iteration's claim satisfy another's wait_ready.
        self._pinned: dict[tuple[str, int], dict[int, int]] = {}
        self._owner_seq = 0  # monotonic; owners are opaque tokens minted by new_owner()
        # Owners live from mint to release_owner, NOT from their first pin: an iteration
        # that starves before it can pin anything is exactly the case the starvation
        # diagnostic has to name, and counting pin-holders would miss it.
        self._owners: set[int] = set()
        self._cv = threading.Condition(threading.Lock())
        self._error: BaseException | None = None  # global poison (driver death)
        self.max_resident = 0  # peak distinct outer chunk positions held at once
        # Peak of the running charge, not a re-sum: `_bytes` is already maintained and
        # we are already under `_cv` at the one site that grows it, so this is free.
        self.max_resident_bytes = 0  # peak bytes charged to the budget at once
        # Keys currently blocked in wait_ready. A blocked consumer is one that cannot
        # unpin, which is what lets the scheduler prove an admission starvation is
        # terminal rather than merely slow (see Scheduler._admit).
        self._waiting: dict[tuple[str, int], int] = {}  # key -> blocked waiter count

    # -- cross-process arbitration (#42) -------------------------------------

    def _warn_environment(self) -> None:
        """Warn once, at construction, about the configurations we cannot arbitrate.

        Both messages have to say plainly that this is where two writers can still
        corrupt each other -- it is exactly where the lock that would prevent it cannot
        be taken. A denylist, so an unrecognized local filesystem stays quiet.
        """
        assert self._dir is not None
        fstype = _filesystem_type(self._dir)
        if fstype in _NETWORK_FILESYSTEMS:
            logger.warning(
                "cache_dir %s is on a %s filesystem. The cache is an mmap tier: it wants "
                "local NVMe, and over a network filesystem it is both slow and "
                "**unarbitrated** -- flock may be emulated per client, so two processes on "
                "different hosts can each believe they hold the write lock and silently "
                "corrupt each other's chunks. This is the one configuration where that is "
                "still possible. Point cache_dir at local disk.",
                self._dir,
                fstype,
            )
        if fcntl is None:
            logger.warning(
                "no POSIX advisory locking on this platform (%s), so insitubatch cannot "
                "arbitrate writers of cache_dir %s. Two processes sharing it will silently "
                "corrupt each other's chunks -- give each its own cache_dir. This is the one "
                "configuration where that is still possible.",
                sys.platform,
                self._dir,
            )

    def _lock(self) -> None:
        """Take the cache dir's advisory lock, held for the pool's lifetime.

        ``LOCK_EX`` to write, ``LOCK_SH`` for ``readonly_cache``: one writer at a time, any
        number of concurrent readers, never both. **Non-blocking** -- contention is a
        configuration fact the user has to resolve, not a queue to join, so it raises with
        the diagnosis rather than parking a training job indefinitely.

        The lock cannot go stale. ``flock`` is held by the kernel on behalf of the open
        file description, so it is released when the process dies -- ``SIGKILL``, OOM and
        spot preemption included. There is therefore no cleanup procedure, and deleting the
        lockfile is actively harmful: it releases nothing and makes the next two processes
        lock different inodes, which is the corruption this exists to prevent.
        """
        assert self._dir is not None
        path = self._dir / _LOCK_NAME
        if fcntl is None:  # pragma: no cover - Windows; warned about in _warn_environment
            return
        flags = (os.O_RDONLY if self._readonly else os.O_RDWR) | os.O_CREAT
        fd = os.open(path, flags, 0o644)
        op = (fcntl.LOCK_SH if self._readonly else fcntl.LOCK_EX) | fcntl.LOCK_NB
        try:
            fcntl.flock(fd, op)
        except OSError as exc:
            holder = _read_holder(fd)
            os.close(fd)
            raise RuntimeError(self._contention_message(path, holder)) from exc
        self._lock_fd = fd
        if not self._readonly:
            # Who holds it, for the *next* process's error message. A hint only -- the
            # flock is authoritative and this record can lie (it names the last writer,
            # while the lock may currently be held by readers).
            os.ftruncate(fd, 0)
            os.write(
                fd,
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "since": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                ).encode()
                + b"\n",
            )

    def _contention_message(self, path: Path, holder: str) -> str:
        """What to tell the user when the cache lock is already held.

        Two different situations, so two different leads: a writer blocked by anyone, and
        a ``readonly_cache`` opener blocked by a live writer. Both end with the same
        diagnostic instructions, because both are answered by the same commands.
        """
        how = (
            f"To see who holds it:  fuser -v {path}\n"
            f"                      lsof {path}\n"
            "\n"
            "Do NOT delete the lockfile. The lock is held by the kernel, not by the file, so\n"
            "deleting it releases nothing -- it only makes the next two processes lock "
            "different\ninodes, which is the corruption this check exists to prevent. A hard "
            "kill (SIGKILL,\nOOM, spot preemption) cannot leave a stale lock: the kernel "
            "releases it when the\nprocess dies. If you see this message, that process is "
            "alive."
        )
        if self._readonly:
            return (
                f"insitubatch: cache_dir '{self._dir}' is being written right now "
                f"({holder}), so it\ncannot be opened with readonly_cache=True. That flag "
                "asserts the cache is complete\nfor what this run reads, and a cache still "
                "being warmed is not. Wait for the writing\nrun to finish, or point this run "
                f"at a cache_dir of its own.\n\n{how}"
            )
        return (
            f"insitubatch: cache_dir '{self._dir}' is already open for writing by another\n"
            f"process ({holder}).\n"
            "\n"
            "Is that expected?\n"
            "  - If you meant to run two jobs against one cache: only one may write. Start "
            "the\n    others with readonly_cache=True, once this one has finished warming "
            "it.\n"
            "  - If you did not: two writers silently corrupt each other's chunks, which is "
            "why\n    this is an error rather than a warning. Point them at separate "
            f"cache_dirs.\n\n{how}"
        )

    def _release_lock(self) -> None:
        """Drop the cache dir's lock, if we hold one. Idempotent.

        Closing the fd is what releases the ``flock`` -- the kernel holds it against the
        open file description. The lockfile itself **stays**: it is the inode every process
        locks, and unlinking it is precisely what would let the next two lock different
        inodes and corrupt each other.
        """
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def _sweep_tmp(self) -> None:
        """Delete temp files a crashed writer left behind.

        ``_alloc`` writes ``<name>.<pid>.insitu-tmp`` and renames it into place; a process
        killed between the two leaves one. Sweeping is safe **only** under the exclusive
        lock -- that is what proves no other writer has one in flight -- so a read-only
        opener never sweeps, and neither does a platform where no lock could be taken.
        """
        assert self._dir is not None
        if self._lock_fd is None or self._readonly:
            return
        for stale in self._dir.glob(f"*{_TMP_SUFFIX}"):
            with contextlib.suppress(OSError):
                stale.unlink()

    def _readonly_miss(self, array: str, chunk_index: int) -> RuntimeError:
        """The error a ``readonly_cache`` miss raises -- the contract, not a slow path.

        Two causes worth separating: the entry is not in the cache at all (the usual one,
        and a configuration mismatch with the run that warmed it), or it is there and could
        not be made resident (budget).
        """
        if self._persisted.get((array, chunk_index)) is not None:
            return RuntimeError(
                f"readonly_cache: chunk {chunk_index} of {array!r} is in the cache at "
                f"{self._dir} but could not be made resident. Either its .npy is unreadable "
                "or no longer matches the current geometry (the debug log names which), or "
                "the byte budget is too small to hold the working set and nothing resident "
                "is evictable -- raise cache_budget_bytes."
            )
        return RuntimeError(
            f"readonly_cache: chunk {chunk_index} of {array!r} is not in the cache at "
            f"{self._dir}, and a read-only opener may not fetch it. readonly_cache=True "
            "asserts this cache is complete for what the run reads; it is not. The usual "
            "cause is that the run that warmed it used a different split, sample_range or "
            "transform set. Warm this configuration once with persist=True, then re-run."
        )

    # -- ownership ----------------------------------------------------------

    def new_owner(self) -> int:
        """Mint an opaque token identifying one iteration's references.

        One iteration = one owner: its scheduler's admission pins and its producer's
        block pins are the same owner, so ``unpin_all(owner)`` at the next epoch's
        prologue releases exactly that iteration's state and leaves a concurrent
        iteration's untouched. Deliberately not user-visible -- ``_iterate`` mints one
        and threads it through; nothing in the public API names an owner.
        """
        with self._cv:
            self._owner_seq += 1
            self._owners.add(self._owner_seq)
            return self._owner_seq

    def _refs(self, key: tuple[str, int]) -> int:  # call under the lock
        """Total outstanding references to ``key`` across every owner."""
        return sum(self._pinned.get(key, {}).values())

    def _evictable(self, key: tuple[str, int], slot: _Slot) -> bool:  # call under the lock
        """May the pool take this buffer away?

        The one predicate. Three independent conditions, each with a distinct reason:
        nobody may still be *writing* (``quiescent``), nobody is still *reading*
        (no references), and the contents are a finished cache entry (``READY``).
        A ``FAILED`` slot is never evicted here because it is dropped as soon as it
        quiesces (see :meth:`_advance`); an ``ASSEMBLED`` one is mid-transform.
        """
        return slot.quiescent and self._refs(key) == 0 and slot.state is SlotState.READY

    # -- observability ------------------------------------------------------

    def _positions(self) -> set[int]:
        """Distinct outer chunk indices currently resident (call under the lock)."""
        return {cid for _array, cid in self._slots}

    @property
    def resident_chunks(self) -> int:
        with self._cv:
            return len(self._positions())

    @property
    def assembles(self) -> bool:
        """True if publishing a chunk does real work (assembly / transform / write-back).

        The scheduler needs this to decide *where* delivery runs. On the plain tiled path a
        delivery is a dict write and a counter, so it belongs inline on the loop. When a
        slot must publish a whole array, ``_advance`` also runs the assembly memcpy, the
        user ``chunk_transform`` and the mmap write-back -- none of which may sit on an
        event loop we share with the rest of the process.
        """
        return self._whole_chunks

    @property
    def resident_bytes(self) -> int:
        with self._cv:
            return self._bytes

    @property
    def budget_bytes(self) -> int | None:
        """The residency ceiling, or ``None`` for an unbounded pool."""
        return self._budget

    @property
    def active_owners(self) -> int:
        """How many iterations are live on this pool right now.

        Live means minted and not yet released -- *not* "currently holds a pin". An
        iteration that starves before it can pin anything is precisely the case the
        starvation diagnostic exists to name, and counting pin-holders would miss it.

        More than one means several iterations share this pool (``zip(ds.train,
        ds.val)``, two DataLoaders) and each needs its own working set resident at the
        same time -- the most common reason a budget auto-sized for one cannot admit.
        Read-only snapshot.
        """
        with self._cv:
            return len(self._owners)

    def blocked_waiters(self) -> list[tuple[str, int]]:
        """``(path, chunk_index)`` keys some thread is currently blocked on in
        :meth:`wait_ready`.

        A blocked consumer is a consumer that cannot reach its next
        :meth:`unpin_keys`, so it can never free budget. The scheduler pairs this
        with "no tile in flight" to tell a terminal starvation from a slow consumer
        -- see :meth:`Scheduler._admit`. Read-only snapshot.
        """
        with self._cv:
            return list(self._waiting)

    # -- admission / pinning / eviction -------------------------------------

    def try_admit(self, array: str, chunk_index: int, owner: int) -> bool:
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
        # Charge what the slot will actually HOLD, which is the stored tiles.
        #
        # A stored chunk decodes at full ``chunk_shape`` and we keep it whole (never
        # clipped -- see _Slot), so residency is ``n_tiles * chunk_shape``, which EXCEEDS
        # the assembled ``slot_shape`` wherever the grid does not divide the extent:
        # measured 1.248x on ERA5 721x1440 @ 180x360, and 1.997x on a short final outer
        # chunk. Charging ``slot_shape`` here -- as the assembled design did -- would let a
        # pool told "2048 MiB" resident 2560 MiB while reporting 2048. Grids that divide
        # exactly (720/45, 2048/256) show no difference at all, which is exactly why this
        # is easy to get wrong and needs a ragged-grid test.
        #
        # A path that needs a whole array republishes as a single tile at completion, so
        # its charge is `slot_shape` and this reduces to the same number.
        n_tiles, tile_shape = self._tile_grid(array, chunk_index)
        dtype = out.dtype if self._whole_chunks else src.dtype
        nbytes = self._charge(array, chunk_index)
        with self._cv:
            if key in self._slots:
                # Already resident (in-flight, or a ready cross-epoch hit) -> incref and
                # reuse. A FAILED slot is NOT reusable: it quiesces and is dropped, so a
                # later epoch refetches instead of re-raising a stale error forever.
                self._pin(key, owner)  # the pin IS this owner's claim (see wait_ready)
                self._cv.notify_all()  # a ready hit may now satisfy a waiter
                return True
            if self._readonly:
                # A read-only opener never allocates: the cache is its whole supply. A miss
                # is the `readonly_cache` contract turning out to be false, so say so --
                # raising is what makes it a contract rather than a silent slow path.
                if not self._revive(key):
                    raise self._readonly_miss(array, chunk_index)
                self.hits += 1  # revived from disk -> no fetch (as in pin_if_ready)
                self._pin(key, owner)
                self._cv.notify_all()
                return True
            if not self._make_room(nbytes):
                return False
            self.misses += 1  # a fresh slot allocated to fetch -> a cache miss
            scratch = (
                np.empty(src.slot_shape(chunk_index), dtype=src.dtype)
                if self._reshapes[array]
                else None
            )
            # Persisting without a transform: back the slot with one tile-major .npy now,
            # so a delivered tile is written straight to its slice and the slot's tile is a
            # view of the file. That is the write we already owed -- not an extra copy.
            backing = (
                self._alloc(array, chunk_index, (n_tiles, *tile_shape), dtype)
                if self._dir is not None and not self._whole_chunks
                else None
            )
            self._slots[key] = _Slot(
                tiles={},  # filled by reference as tiles land; no buffer allocated here
                backing=backing,
                array=array,
                chunk_index=chunk_index,
                writers=0,  # no task has started yet; counted on entry to tile_write
                pending=src.n_inner_chunks(chunk_index),
                nbytes=nbytes,
                scratch=scratch,
            )
            self._bytes += nbytes
            self._pin(key, owner)
            self.max_resident = max(self.max_resident, len(self._positions()))
            self.max_resident_bytes = max(self.max_resident_bytes, self._bytes)
            return True

    def is_ready(self, array: str, chunk_index: int) -> bool:
        """True if the chunk is resident, fully assembled, and not failed (a hit)."""
        with self._cv:
            slot = self._slots.get((array, chunk_index))
            return slot is not None and slot.state is SlotState.READY

    def pin_if_ready(self, array: str, chunk_index: int, owner: int) -> bool:
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
            if not (slot is not None and slot.state is SlotState.READY) and not self._revive(key):
                return False
            self.hits += 1  # resident (cross-epoch) or revived (cross-run) -> no fetch
            self._pin(key, owner)  # the pin IS this owner's claim; publish it
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
        n_tiles, tile_shape = self._tile_grid(array, chunk_index)
        want_shape = (n_tiles, *tile_shape)
        want_dtype = geom.dtype if geom is not None else None
        if geom is None or data.shape != want_shape or data.dtype != want_dtype:
            self.revive_mismatch += 1
            logger.debug(
                "cache: persisted %s shape/dtype %s/%s != current %s/%s; refetching",
                key,
                data.shape,
                data.dtype,
                None if geom is None else want_shape,
                want_dtype,
            )
            del data  # drop the mmap ref; structural fingerprint mismatch -> a miss
            self._persisted.pop(key, None)
            return False
        nbytes = int(data.nbytes)
        if not self._make_room(nbytes):
            del data
            return False
        # A revived chunk enters the SAME lifecycle at a later state rather than being a
        # second way to be gatherable: no tiles were ever outstanding, nothing is pending.
        # Tiles come back as zero-copy VIEWS of the one mapping, in the tile-major order
        # `inner_index` defined when it was written -- so a revived chunk is tiled exactly
        # like a freshly fetched one. That is what keeps `gather` on a single path: there
        # is no "revived is assembled, fresh is tiled" split to dispatch on.
        src_geom = self._by_path[array]
        coords = (
            [(0,) * len(self._out_by_path[array].inner_chunks)]
            if self._whole_chunks
            else sorted(src_geom.inner_coords(), key=src_geom.inner_index)
        )
        self._slots[key] = _Slot(
            tiles={c: data[i] for i, c in enumerate(coords)},
            backing=data,
            array=array,
            chunk_index=chunk_index,
            writers=0,
            pending=0,
            nbytes=nbytes,
            state=SlotState.READY,
        )
        self._bytes += nbytes
        self.max_resident = max(self.max_resident, len(self._positions()))
        self.max_resident_bytes = max(self.max_resident_bytes, self._bytes)
        return True

    def pin_keys(self, keys: set[tuple[str, int]], owner: int) -> None:
        """Reference (incref) a set of ``(path, chunk_index)`` slots for a live block.

        Windows make one chunk readable by several concurrent blocks, so pins are
        reference-counted: each block that needs a slot increfs it on entry and
        decrefs it on drain (:meth:`unpin_keys`). A slot with refcount > 0 is never
        evicted. Pinning a not-yet-allocated key is fine -- the count is recorded and
        the slot, once admitted, inherits it.
        """
        with self._cv:
            for key in keys:
                self._pin(key, owner)
            # Now that a reference IS this owner's claim, pinning can be the event that
            # satisfies a consumer parked in `wait_ready` on an already-READY chunk.
            # Under the old shared `claimed` flag it never could, so this notify was
            # not needed; without it that consumer sleeps until some other wake-up.
            self._cv.notify_all()

    def unpin_keys(self, keys: set[tuple[str, int]], owner: int) -> None:
        """Release (decref) a block's ``(path, chunk_index)`` references.

        A slot dropping to refcount 0 becomes LRU-evictable (retained for cross-epoch
        reuse until budget pressure drops it), not dropped now. Wakes any admit parked
        on a full budget.
        """
        with self._cv:
            for key in keys:
                self._unpin_one(key, owner)
            self._cv.notify_all()

    def buffer_stats(self) -> BufferStats:
        """One consistent reading of the batch-output pool, for the per-epoch summary."""
        return self._buffers.stats()

    def set_host_allocator(self, allocator: HostAllocator) -> None:
        """Point the batch-output pool at a different host allocator.

        How the torch adapter installs page-locked buffers without the core importing a
        framework. A method rather than letting callers reach ``pool._buffers`` directly, so
        the buffer pool stays this class's private business and there is one place to look for
        who can change it.
        """
        self._buffers.set_allocator(allocator)

    def release_owner(self, owner: int) -> None:
        """Release everything one iteration holds: its pins, and its abandoned partials.

        Called by that iteration's own teardown, so the state dies with the thing that
        created it. A pin is per-pass working state, not cache membership -- READY
        chunks stay resident (unpinned) for cross-epoch reuse.

        This must happen: an abandoned pass (early ``break``) otherwise leaves its
        read-ahead and un-drained block referenced forever, shrinking every later
        epoch's budget until admission can free no room and the driver deadlocks.

        Scoped to ``owner``, because one pool serves several concurrent iterations.
        Releasing globally is #34 -- it strips a live iteration's pins and its in-use
        chunks become eviction candidates mid-gather. For the same reason a partial is
        dropped only when it is quiescent *and* unreferenced: a slot still being written
        by a live task, or held by another owner, is not ours to reclaim.
        """
        with self._cv:
            for key in [k for k, owners in self._pinned.items() if owner in owners]:
                self._pinned[key].pop(owner, None)
                if not self._pinned[key]:
                    self._pinned.pop(key, None)
            for key in [
                k
                for k, slot in self._slots.items()
                if slot.state is not SlotState.READY and slot.quiescent and self._refs(k) == 0
            ]:
                self._drop(key)  # an abandoned partial can never be a valid cache entry
            self._owners.discard(owner)  # this iteration is done; it no longer counts
            self._cv.notify_all()  # freed budget may unpark an admission

    def reset_epoch_counters(self) -> None:
        """Zero the per-epoch observability counters at a pass boundary.

        Split out of the old ``unpin_all``: releasing references and resetting counters
        are unrelated jobs on different clocks (one belongs to an iteration's teardown,
        the other to a pass's start), and bundling them is why the reset used to force a
        global unpin. ``manifest_entries`` is load-time state and deliberately survives.
        """
        with self._cv:
            self.hits = self.misses = 0
            self.max_resident = self.max_resident_bytes = 0  # peaks are per-pass too
            self.revive_mismatch = self.revive_missing = 0
            self._buffers.reset_counters()  # batch outputs too -- same epoch boundary

    def _pin(self, key: tuple[str, int], owner: int) -> None:  # call under the lock
        owners = self._pinned.setdefault(key, {})
        owners[owner] = owners.get(owner, 0) + 1
        if key in self._slots:
            self._slots.move_to_end(key)  # MRU (the slot may not be allocated yet)

    def _unpin_one(self, key: tuple[str, int], owner: int) -> None:  # call under the lock
        owners = self._pinned.get(key)
        if owners is None:
            return
        n = owners.get(owner, 0)
        if n <= 1:
            owners.pop(owner, None)
        else:
            owners[owner] = n - 1
        if not owners:
            self._pinned.pop(key, None)

    def _make_room(self, nbytes: int) -> bool:  # call under the lock
        if self._budget is None:
            return True
        while self._bytes + nbytes > self._budget:
            # Evict the LRU slot that is ready *and* unreferenced; a not-ready slot is
            # an in-flight fetch and a refcounted slot is held by a live block.
            victim = next((k for k, sl in self._slots.items() if self._evictable(k, sl)), None)
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
        # completion (see _advance -> _record_completed), so eviction touches only the backing.
        # A not-ready partial is garbage either way -> unlink it.
        keep = self._persistent and slot.state is SlotState.READY
        self._free(slot, keep_file=keep)

    def _tile_grid(self, array: str, chunk_index: int) -> tuple[int, tuple[int, ...]]:
        """``(n_tiles, tile_shape)`` for one slot -- the single rule for its residency.

        Without a transform a slot holds the array's **stored** tiles, sample-first and
        whole (padding included). With one it holds a single tile: ``output_geometry`` sets
        post-transform ``chunks`` to the full inner shape, so a transformed chunk is a 1x1
        grid by construction. Both the budget charge and the persisted ``.npy`` layout are
        derived from this, so they cannot drift apart.
        """
        if self._whole_chunks:
            return 1, self._out_by_path[array].slot_shape(chunk_index)
        src = self._by_path[array]
        return src.n_inner_chunks(chunk_index), src.tile_shape()

    def _charge(self, array: str, chunk_index: int) -> int:
        """Bytes to charge one slot against the budget: its **peak** residency.

        For a tiled slot that is just its tiles. An assembling slot is charged the larger
        of the two shapes it takes on, because it holds the **source tiles** for its whole
        fill and only collapses to the assembled output at completion -- and on a ragged
        grid the tiles are the bigger of the two (1.248x on ERA5 721x1440 at 180x360).
        Charging the output alone, as the assembled design did, under-reports the entire
        fill window.

        The brief overlap at completion (tiles + assembled + the transform's own result) is
        not charged, for the same reason the in-flight decode transient is not: it is
        bounded by `max_inflight` and belongs to the fetch, not to residency.
        """
        return slot_charge_bytes(
            self._by_path[array],
            self._out_by_path[array],
            chunk_index,
            assembles=self._whole_chunks,
        )

    def _alloc(
        self, array: str, chunk_index: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
        """Back one slot: heap, or a fresh ``.npy`` under the cache dir.

        The mmap tier **never truncates an existing file in place**. ``open_memmap(mode=
        "w+")`` opens with ``O_TRUNC``, which zeroes the pages of every mapping already
        held on that inode -- including another process's, reading a revived cache entry
        and getting right-shape, right-dtype, wrong numbers (#42). So write a private temp
        file and ``replace()`` the directory entry: POSIX keeps the old inode alive for
        anyone holding it, so their mapping stays intact while new openers see the new
        file. The rename is atomic within a directory, so nobody sees a half-written entry
        either.
        """
        if self._dir is None:
            return np.empty(shape, dtype=dtype)
        path = self._dir / f"{_safe(array)}__{chunk_index}.npy"
        tmp = self._dir / f"{path.name}.{os.getpid()}{_TMP_SUFFIX}"
        backing = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=shape)
        tmp.replace(path)
        # The mapping *is* `path` now -- renaming does not disturb it -- but numpy recorded
        # the name it was opened under. Correct it: `_record_completed` names the cache entry
        # from here and `_free` unlinks through it, and both would otherwise chase a temp
        # path that no longer exists.
        cast("np.memmap", backing).filename = os.path.abspath(path)
        return backing

    @contextlib.contextmanager
    def tile_write(
        self, array: str, chunk_index: int, inner_coord: tuple[int, ...]
    ) -> Iterator[_TileWrite]:
        """Scope one tile task's write, guaranteeing the slot hears about its end.

        The pool owns the guarantee; the caller only has to *be inside the scope*::

            with pool.tile_write(array, cid, coord) as w:
                tile = await fetch_decode(...)     # cancellation here is covered
                w.deliver(tile)                    # or w.fail(exc)

        A bare decrement at the end of the task would not do: a tile task can end
        without ever reaching the pool -- a fetch/decode error takes an early return,
        and a **cancellation** (an early ``break`` closes the scheduler with
        ``cancel_futures=True``) can unwind at any ``await``. ``writers`` is what
        eviction reads, so a missed decrement is a slot that is never safe to take away
        and a budget that never recovers. ``__exit__`` is the only construct that
        cannot be forgotten by a future early return.

        ``deliver`` / ``fail`` / scope exit all funnel into one release, and the
        release is idempotent -- so a delivered tile does not double-decrement.
        """
        key = (array, chunk_index)
        with self._cv:
            slot = self._slots.get(key)
            if slot is not None:
                slot.writers += 1  # this task is now running: the slot is not quiescent
        writer = _TileWrite(self, array, chunk_index, inner_coord)
        try:
            yield writer
        finally:
            writer.release()

    def deliver_tile(
        self, array: str, chunk_index: int, inner_coord: tuple[int, ...], tile: np.ndarray
    ) -> None:
        """Synchronous convenience: scope one tile write and deliver it in one call.

        Sugar over :meth:`tile_write`, not a second lever -- it opens the same scope, so
        the writer count still moves in exactly one place. Use it when there is nothing
        to await between reserving the write and having the tile. The scheduler cannot:
        it awaits the fetch inside the scope, so it uses :meth:`tile_write` directly.
        """
        with self.tile_write(array, chunk_index, inner_coord) as writer:
            writer.deliver(tile)

    def _deliver(
        self, array: str, chunk_index: int, inner_coord: tuple[int, ...], tile: np.ndarray
    ) -> None:
        """Adopt one decoded tile into its slot (called by :class:`_TileWrite`).

        **No copy.** The decoded tile becomes the resident buffer by reference -- this is
        the memcpy the chunked-slot design deletes. Placement is deferred to ``gather``,
        which knows the batch rectangle it is filling; ``tile_placement`` therefore moves
        from the write path to the read path rather than disappearing.

        The dict write happens *before* the lock (rule 1) -- it is this task's own key, and
        no other writer touches it -- and the counter moves *under* it (rule 2).
        Publication is :meth:`_advance`'s business, not this method's.
        """
        key = (array, chunk_index)
        slot = self._slots[key]  # allocated by the scheduler before any delivery
        if slot.backing is None:
            slot.tiles[inner_coord] = tile  # heap: adopt by reference, no copy
        else:
            # Persisting: write the tile into its tile-major slice and keep the VIEW. The
            # copy is the file write the mmap tier already owed -- not an extra one -- and
            # the slot then reads from the same mapping a later run will revive.
            idx = self._by_path[array].inner_index(inner_coord)
            slot.backing[idx] = tile
            slot.tiles[inner_coord] = slot.backing[idx]
        with self._cv:
            slot.pending -= 1

    def _advance(self, array: str, chunk_index: int) -> None:
        """The **only** function that moves a slot's state. Called once per tile release.

        Every transition is decided here, from the slot's own counters, so there is
        exactly one lever:

        * still has writers -> nothing to decide yet.
        * quiesced + FAILED -> **drop it**. A poisoned slot must not survive as a
          resident entry: ``try_admit`` would take the resident branch next epoch and
          return ``True`` without refetching, and ``wait_ready`` would re-raise the
          stale error forever (#33's second half).
        * quiesced + everything delivered -> ASSEMBLED, then run the chunk transform and
          the persist write-back **outside** the lock (no other thread can touch the
          slot once it is quiescent), then publish READY.
        * quiesced + tiles still pending -> an abandoned partial (cancelled fetch). Leave
          it FILLING; ``unpin_all`` drops it once unreferenced.
        """
        key = (array, chunk_index)
        with self._cv:
            slot = self._slots.get(key)
            if slot is None or not slot.quiescent:
                return
            if slot.state is SlotState.FAILED:
                # Deliberately NOT dropped here: a consumer still has to observe the
                # error, and this slot may be the only thing holding it. It is not
                # evictable (that needs READY) and it does not survive the pass --
                # `release_owner` drops it at teardown, so the next epoch refetches
                # instead of re-raising a stale error forever (#33).
                self._cv.notify_all()
                return
            if slot.pending > 0 or slot.state is not SlotState.FILLING:
                # Either tiles are still to come (quiescent only because no task happens
                # to be running right now, or because the pass was cancelled and left an
                # abandoned partial), or the slot is already published.
                return
            slot.state = SlotState.ASSEMBLED
            needs_whole = self._whole_chunks

        # Sole owner now (quiescent, and ASSEMBLED excludes it from eviction).
        #
        # The common path stops here: the tiles ARE the chunk, nothing to assemble.
        # Assembly happens only for the two consumers that genuinely need one contiguous
        # array -- a chunk_transform (user code, which must not be handed a dict of tiles)
        # and the mmap tier (one file per chunk). Both then publish a **one-tile** slot, so
        # `gather` still sees exactly one representation.
        if needs_whole:
            whole = self._assemble(array, chunk_index, slot)
            prepped = self._apply_transforms(array, chunk_index, whole)
            with self._cv:
                slot.tiles = {
                    (0,) * len(self._out_by_path[array].inner_chunks): self._persist(slot, prepped)
                }
        with self._cv:
            slot.scratch = None  # assembly done -- drop the transient source-shaped buffer
            slot.state = SlotState.READY
            # Record the completed entry *now* (not at eviction/close) so a crash still leaves a
            # usable cache. Appending here, under the lock, serializes writes across decode
            # threads and orders them after the slot's data is durable in its .npy.
            self._record_completed(key, slot)
            self._cv.notify_all()

    def _assemble(self, array: str, chunk_index: int, slot: _Slot) -> np.ndarray:
        """Stitch a slot's stored tiles into one contiguous source-shaped array.

        Only for the two consumers that need a whole array (see :meth:`_advance`). This is
        the memcpy the chunked slot removed from the fill path -- reintroduced *once*, at
        completion, for the paths that cannot avoid it, instead of on every tile delivery.
        """
        geom = self._by_path[array]
        buffer = slot.scratch
        if buffer is None:
            buffer = np.empty(geom.slot_shape(chunk_index), dtype=geom.dtype)
        for coord, tile in slot.tiles.items():
            proj = geom.tile_placement(chunk_index, coord)
            buffer[proj.out_selection] = tile[proj.chunk_selection]
        return buffer

    def _persist(self, slot: _Slot, prepped: np.ndarray) -> np.ndarray:
        """Land the assembled (post-transform) chunk in the slot's backing.

        Heap just holds the array. The mmap tier copies it into the slot's ``.npy`` so the
        cached chunk stays on NVMe, and returns the memmap -- which then becomes the slot's
        single tile. The backing is sized at the transform's *output* geometry (see
        :func:`output_geometry`), so a reshaping transform lands here exactly like a
        shape-preserving one. A shape mismatch means ``__call__`` disagreed with its
        declared ``output_inner``: a bug, raised.
        """
        if self._dir is None:
            return prepped
        out = self._out_by_path[slot.array]
        # A transformed chunk is a one-tile grid, so its file is (1, *slot_shape) -- the
        # same tile-major layout the untransformed path writes, with n_tiles == 1.
        backing = self._alloc(slot.array, slot.chunk_index, (1, *prepped.shape), out.dtype)
        slot.backing = backing
        backing = backing[0]
        if prepped.shape != backing.shape:
            raise ValueError(
                f"chunk_transform produced shape {prepped.shape} but the cache slot is sized "
                f"{backing.shape} from the declared output geometry; a reshaping transform's "
                "output_inner must agree with what __call__ returns."
            )
        backing[:] = prepped  # write into the memmap (casts to slot dtype)
        return backing

    def fail(self, array: str, chunk_index: int, error: BaseException) -> None:
        """Poison one chunk so a waiting consumer re-raises instead of hanging.

        Fail-fast: a fetch/decode error on any tile poisons its outer chunk; the
        consumer's ``wait_ready`` surfaces it on the main thread.

        It records the error and **nothing else**. The old version also set
        ``ready = True`` -- because "stop waiting" and "safe to evict" shared one flag,
        the only way to wake a waiter was to declare the slot a finished cache entry,
        while sibling tile tasks were still writing into it (#33). ``wait_ready`` keys
        off ``error`` independently of state, so waiters still wake immediately; the
        slot is disposed of by :meth:`_advance` once it quiesces.
        """
        with self._cv:
            slot = self._slots.get((array, chunk_index))
            if slot is None:
                return
            slot.error = error
            slot.state = SlotState.FAILED
            self._cv.notify_all()
        self._advance(array, chunk_index)  # may already be quiescent (all tiles returned)

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

    def wait_ready(self, array: str, chunk_index: int, owner: int) -> None:
        """Block until the chunk is READY *for this owner* (or raise).

        Requiring **this owner's own reference** (rather than a shared ``claimed``
        bool) closes a cross-epoch race: a chunk still resident-and-READY from the
        prior epoch would otherwise be gathered before the driver references it,
        letting the consumer's last-use release land before the driver's pin -- a lost
        release that leaks a reference and, worse, lets the driver evict a chunk
        mid-gather. Owner-scoped because a single bool is satisfied by *another*
        iteration's claim (#35), which is the same race with a concurrent producer
        instead of a previous epoch.

        Wakes on: READY and referenced by ``owner``; the chunk failed (:meth:`fail`,
        which no longer has to fake readiness to get here); or the pool was poisoned
        (:meth:`set_error`, covering a driver death before this chunk was allocated).
        A key that is simply absent means the driver has not admitted it yet -- keep
        waiting, as before.
        """
        key = (array, chunk_index)
        with self._cv:
            self._waiting[key] = self._waiting.get(key, 0) + 1
            try:
                self._cv.wait_for(
                    lambda: (
                        self._error is not None
                        or (
                            key in self._slots
                            and (
                                self._slots[key].error is not None
                                or (
                                    self._slots[key].state is SlotState.READY
                                    and self._pinned.get(key, {}).get(owner, 0) > 0
                                )
                            )
                        )
                    )
                )
            finally:
                if self._waiting[key] > 1:
                    self._waiting[key] -= 1
                else:
                    del self._waiting[key]
            if self._error is not None:
                raise self._error
            error = self._slots[key].error
        if error is not None:
            raise error

    def gather(self, rows: np.ndarray, variables: list[str], sample_chunk_size: int) -> Batch:
        """Assemble one batch from ``[chunk_id, within]`` *anchor* draw rows.

        Each row is one sample anchor ``t = chunk_id*ref_spc + within`` in the reference
        (manifest) grid; each variable reads its array at ``t + offset`` (offset 0 is the
        plain non-windowed case). Output is in anchor-row order: row ``i`` of every variable
        is the same anchor, ``sample_indices[i] == t_i``. Per variable the reads are grouped
        by the variable's *own* (offset-shifted) chunk -- computed with that variable's
        chunk size, so variables may chunk the sample axis differently -- one coalesced
        fancy-index per chunk, never a Python per-sample loop. The caller must have waited
        every referenced ``(path, offset-shifted chunk)`` ready.
        """
        anchor = rows[:, 0].astype(np.int64) * sample_chunk_size + rows[:, 1].astype(np.int64)
        n = anchor.shape[0]

        arrays: dict[str, np.ndarray] = {}
        for var in variables:
            geom = self._geom[var]  # source: drives the read math (offset, path, chunking)
            out_geom = self._out_geom[var]  # post-transform: shape/dtype the consumer sees
            sample = anchor + geom.offset  # this view reads array[anchor + offset]
            spc = geom.sample_chunk_size  # the variable's own grid (may differ from ref)
            read_cid = sample // spc
            within = sample % spc
            out = self._buffers.take(n, out_geom.inner_shape, out_geom.dtype)
            for cid in np.unique(read_cid):
                mask = read_cid == cid  # rows that read this chunk -> one coalesced index
                slot = self._slots[(geom.path, int(cid))]
                rows = within[mask]
                # One coalesced write per stored tile. A slot that holds a whole array
                # (transformed, or revived) is simply a 1-tile grid, so this is the same
                # single fancy-index the assembled path used to do -- no special case.
                # A tiled slot's tiles sit on the SOURCE stored-chunk grid; a whole-chunk
                # slot has one tile on the output grid (`output_geometry` sets post-transform
                # chunks to the full inner shape, making it a 1x1 grid).
                tile_geom = out_geom if self._whole_chunks else geom
                for coord, tile in slot.tiles.items():
                    proj = tile_geom.tile_placement(int(cid), coord)
                    # `tile_placement` always builds plain slice tuples; zarr types the
                    # fields as the wider SelectorTuple, so narrow them back for indexing.
                    dst = cast("tuple[slice, ...]", proj.out_selection)
                    src = cast("tuple[slice, ...]", proj.chunk_selection)
                    out[(mask, *dst[1:])] = tile[(rows, *src[1:])]  # type: ignore[arg-type]
            arrays[var] = out
        offsets = {var: self._geom[var].offset for var in variables}
        return Batch(arrays=arrays, sample_indices=anchor, offsets=offsets)

    def _free(self, slot: _Slot, *, keep_file: bool) -> None:
        """Release a slot's backing: a no-op for heap, close (and maybe unlink) for mmap.

        ``keep_file`` leaves the ``.npy`` on disk (a persisted cache entry); otherwise the
        file is unlinked (heap/spill teardown or a discarded partial).
        """
        mmap = getattr(slot.backing, "_mmap", None)
        if mmap is not None:
            fname = getattr(slot.backing, "filename", None)
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
        fname = getattr(slot.backing, "filename", None)
        if fname is None:
            return  # heap backing -- nothing on disk to record
        fname = Path(fname).name
        self._persisted[key] = fname
        if self._log_fd is not None and key not in self._recorded:
            array, chunk_index = key
            self._append_entry(array, chunk_index, fname)
            self._recorded.add(key)

    def _append_entry(self, array: str, chunk_index: int, fname: str) -> None:  # under the lock
        """Append one completed-entry line to the open log."""
        self._write_line({"array": array, "chunk_index": chunk_index, "file": fname})

    def _write_line(self, record: dict[str, object]) -> None:
        """Append one JSON record as a **single** ``os.write`` on an ``O_APPEND`` fd.

        POSIX makes an append of less than ``PIPE_BUF`` atomic -- the seek to the end and the
        write are one operation -- and entries are ~100 bytes, so a line can never interleave
        with another writer's. The lock already gives us a single writer; this makes the
        *format* robust independently of it, which is what keeps a log written on an
        unarbitrated platform readable.

        No ``flush``/``fsync``: the write lands in the page cache directly, which is durable
        against *process death* -- the target failure mode (spot preemption / OOM / SIGTERM).
        Power loss (which would need ``fsync`` per chunk) is out of scope. A short write would
        leave a torn line and cannot be retried (a retry would land after another writer's
        line), so it raises.
        """
        assert self._log_fd is not None
        data = (json.dumps(record) + "\n").encode()
        written = os.write(self._log_fd, data)
        if written != len(data):  # pragma: no cover - only reachable on a full disk
            raise OSError(
                f"persist: short write to the cache log ({written} of {len(data)} bytes) -- "
                "the disk is probably full. The log now ends in a torn line, which the next "
                "run drops; the entries before it are intact."
            )

    def _open_log(self) -> None:
        """Open the append-only manifest for the pool's lifetime; write the header on a cold
        start (or after a stale-cache reset removed the file). A warm reopen appends after the
        existing entries -- the ``_recorded`` gate (populated by :meth:`_load_log`) keeps those
        from being re-appended. ``readonly_cache`` never gets here: it reads the log and
        never opens it for writing."""
        assert self._dir is not None
        path = self._dir / self._MANIFEST_NAME
        fresh = not path.exists()
        self._log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        if fresh:
            self._write_line(
                {"format_version": self._MANIFEST_FORMAT, "pipeline_hash": self._pipeline_fp}
            )

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
        """Free every remaining slot; release the log handle and the cache lock. Persist keeps
        ready cache files
        (each already recorded in the log at completion, so there is nothing to rewrite -- just
        flush + close the handle); heap/spill mmap files are unlinked. Idempotent."""
        with self._cv:
            if self._log_fd is not None:
                os.close(self._log_fd)
                self._log_fd = None
            self._release_lock()
            for k in list(self._slots):
                slot = self._slots.pop(k)
                self._free(slot, keep_file=self._persistent and slot.state is SlotState.READY)
            self._bytes = 0
            self._pinned.clear()
            self._buffers.clear()  # batch outputs too; a big-payload pool holds GBs of them

    def __del__(self) -> None:
        # Safety net for a pool dropped without an explicit close() -- release the open log
        # handle (persist mode holds one for its lifetime) and the mmap backings. Best-effort:
        # __del__ runs at GC / interpreter shutdown where exceptions are unraisable. In normal
        # use InSituDataset.close()/__del__ closes the pool; this covers a bare ChunkPool.
        with contextlib.suppress(Exception):
            self.close()
