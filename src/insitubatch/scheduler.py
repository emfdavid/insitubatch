"""Scheduler: the fetch driver.

Orchestration runs on **zarr's** process-global event loop, streaming *stored-chunk*
(tile) reads under a single ``max_inflight`` budget, decoding each tile off the loop
(``ChunkTransform.decode_chunk``, numcodecs C, GIL released) on a process-wide decode
pool, and handing it to a :class:`ChunkPool` that **adopts it by reference**.

The point is that the two things you tune are independent: **read concurrency** is
dialed by ``max_inflight`` alone, while **residency / shuffle span** is governed by
the pool's byte budget -- no nested inner/outer concurrency caps, no double
quantization of the fetch (reading one outer chunk per ``getitem`` and letting zarr
stitch the inner grid under a second cap is what couples them). See
[docs/architecture.md] for the full pipeline.

Two bounded resources, deliberately distinct:

* **in-flight** (``max_inflight``, an ``asyncio.Semaphore``) -- tiles in flight; a
  slot is held from fetch-start to delivery, spanning fetch + decode + deliver. It is
  also what bounds the decode pool's queue, which is otherwise unbounded.
* **residency** (the pool's byte budget) -- admission (``pool.try_admit``) evicts
  ready-and-unreferenced LRU to make room and *references* (refcounted pin) the chunk,
  so it stays resident from in-flight fetch through to the consumer's release; when the
  budget is full of referenced slots the loop awaits a consumer release. So the number
  of outstanding fetch tasks is bounded by the resident window, not by the epoch length.

Per chunk the scheduler first asks the pool whether it already holds it ready
(``pin_if_ready``): a still-resident prepped chunk is a hit and costs no fetch
(cross-epoch reuse, since the pool persists across epochs). Misses are admitted and
their tiles fetched. The consumer waits on slot readiness, gathers, and releases each
chunk at its *last* use (:meth:`unpin_block`) -- windowed reads let one chunk feed
several blocks, so it is released only when no later block needs it. Errors propagate
two ways -- a per-tile
fetch/decode failure poisons just that chunk (``pool.fail``); a driver failure
poisons the whole pool (``pool.set_error``) so any waiter re-raises instead of
hanging.

Budget floor: a batch may draw from any chunk in its shuffle-block, so the whole
block must be co-resident to gather -- and since batches are cut over the whole epoch
order rather than per block, the batch that straddles a boundary needs *two* blocks
co-resident. The producer already sizes the budget to two (the current block plus one
read-ahead, so block-boundary IO overlaps the current block's compute), so the
straddling batch costs nothing beyond that floor -- but the floor is now load-bearing
for correctness, not only for overlap. Never three: a block is released as soon as a
batch has consumed its last row.

That "never three" holds only while a batch fits inside a single block -- a wider
batch spans ``ceil(batch_size / block rows)`` blocks and pins every one of them until
it has gathered. :class:`~insitubatch.source.InSituDataset` guarantees the fit by
widening ``block_chunks``; finely chunked archives (one sample per chunk) are where it
would otherwise bite. Should a working set exceed its budget anyway, admission does not
hang: :meth:`Scheduler._starvation` proves the stall terminal and raises.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr.api.asynchronous as za
from zarr.abc.store import Store
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.chunk_utils import ChunkTransform
from zarr.core.sync import _get_loop

from .plan import build_stored_chunk_reads
from .pool import ChunkPool
from .types import ArrayGeometry, StoredChunkRead

STARVATION_POLL_S = 0.25

logger = logging.getLogger(__name__)

_DECODE_POOL: ThreadPoolExecutor | None = None
_DECODE_POOL_WORKERS = 0
_DECODE_POOL_LOCK = threading.Lock()


def decode_pool(workers: int | None = None) -> ThreadPoolExecutor:
    """The process-wide decode pool. Created once, never shut down.

    Process-lived on purpose. A per-scheduler pool was rebuilt on every pass (churn), and
    -- once orchestration moved onto zarr's loop -- shutting one down was actively unsafe:
    the pool had to be installed as the loop's *default executor* for ``decode_threads`` to
    mean anything, so closing it left zarr's own ``run_in_executor(None, ...)`` raising
    ``cannot schedule new futures after shutdown`` for the rest of the process.

    We no longer touch the loop's default executor at all (see :meth:`Scheduler._one`), and
    this pool outlives every scheduler, so neither failure is reachable.

    ``workers`` sizes it on **first** use only; later callers get the existing pool and are
    warned if they asked for a different size. That is the right shape rather than a
    limitation: how many threads can usefully decode at once is a property of the
    *machine* -- core count, memory bandwidth -- not of the dataset being read. Two
    datasets in one process do not want two pools competing for the same cores, and the
    second one's number is not more correct than the first's.
    """
    global _DECODE_POOL, _DECODE_POOL_WORKERS
    with _DECODE_POOL_LOCK:
        if _DECODE_POOL is None:
            _DECODE_POOL_WORKERS = workers or min(32, (os.cpu_count() or 4) + 4)
            _DECODE_POOL = ThreadPoolExecutor(
                max_workers=_DECODE_POOL_WORKERS, thread_name_prefix="insitu-dec"
            )
        elif workers and workers != _DECODE_POOL_WORKERS:
            logger.warning(
                "decode_threads=%d ignored: the decode pool is process-wide and was already "
                "built with %d threads. Thread count is a property of the machine, not of a "
                "dataset, so the first value wins. Set decode_threads on the first dataset "
                "created in this process.",
                workers,
                _DECODE_POOL_WORKERS,
            )
        return _DECODE_POOL


def reset_decode_pool() -> None:
    """Drop the process-wide decode pool so the next use rebuilds it. **Tests only.**

    Not a user-facing knob: production code never wants this (the whole point is that the
    pool outlives schedulers). Tests need it to size the pool per case, to run a
    free-threaded arm, and to avoid leaking one case's pool into the next.
    """
    global _DECODE_POOL, _DECODE_POOL_WORKERS
    with _DECODE_POOL_LOCK:
        pool, _DECODE_POOL, _DECODE_POOL_WORKERS = _DECODE_POOL, None, 0
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)


"""How often a parked admission re-checks for a *provably* terminal starvation.

Detection latency only: the test is structural (nothing in flight, and a consumer
blocked in ``wait_ready``), never "we have waited a while", so a slow consumer is
never mistaken for a deadlock however long it takes."""


@dataclass(slots=True)
class SchedulerConfig:
    max_inflight: int = 32
    """Tiles in flight at once -- the single concurrency dial. Memory in flight
    ~= max_inflight * stored_chunk_nbytes (+ transform scratch). Residency is
    bounded separately by the pool's byte budget (admission evicts unpinned-LRU)."""

    decode_threads: int = 0
    """Size of the decode pool (the GIL-releasing codec decode runs here). ``0`` = auto =
    ``min(32, cpu+4)``.

    The pool is **process-wide** (see :func:`decode_pool`), so this sizes it once, on the
    first dataset built in the process; a later, different value is ignored with a warning.
    Thread count is a property of the machine rather than of the data, so one number per
    process is the honest shape."""

    on_bad_chunk: str = "raise"
    """What to do when a stored chunk fails to fetch/decode (truncated/corrupt --
    common in GRIB-under-zarr archives like HRRR). ``"raise"`` (default) fails fast;
    ``"nan"`` fills that tile with NaN (float dtypes) or the fill value, so the chunk
    assembles with a hole instead of poisoning the epoch -- the caller then handles
    NaN with a ``chunk_transform`` (interpolate / drop). Bad reads are recorded in
    ``Scheduler.bad_chunks``."""


@dataclass(slots=True)
class _ArrayCtx:
    """Per-variable handles for the stored-chunk fetch+decode path.

    Cached once per array: the store + key encoder address a stored chunk, the
    codec transform + spec decode its bytes (this reconstructs exactly what
    ``arr.getitem`` would stitch, for single-inner and spatially-chunked arrays).

    ``codec`` is a **synchronous** :class:`~zarr.core.chunk_utils.ChunkTransform`, not the
    array's async ``codec_pipeline`` -- see :func:`_sync_transform` for why that matters.
    """

    path: str
    store: Store
    encode: Callable[[tuple[int, ...]], str]
    codec: ChunkTransform
    spec: ArraySpec
    chunk_shape: tuple[int, ...]
    fill_value: object
    dtype: np.dtype
    sample_axis: int  # physical axis to move to the front on decode (0 = no-op)


def _sync_transform(meta: Any) -> ChunkTransform:
    """A **synchronous** full-chain decoder for one array's codecs.

    ``ChunkTransform.decode_chunk`` is pure compute with no IO, so we can call it inside
    an executor task we dispatch ourselves -- which is what lets us keep our own decode
    pool without claiming the loop's single default-executor slot. The async
    ``codec_pipeline.decode`` cannot: it dispatches to the loop default on our behalf.

    Format-agnostic, and v2 is not optional -- **WeatherBench2 ARCO is zarr-v2**, whose
    pipeline is a single ``V2Codec(filters, compressor)`` rather than v3's
    ``(BytesCodec, ZstdCodec, ...)``. A v3-only path would silently miss our main weather
    benchmark. Verified byte-identical to the async pipeline on v3 zstd / uncompressed /
    gzip and on v2.
    """
    codecs = getattr(meta, "codecs", None)
    if codecs is None:  # zarr-v2: the compressor hangs off V2Codec, not a codec chain
        from zarr.codecs._v2 import V2Codec

        codecs = (V2Codec(filters=meta.filters, compressor=meta.compressor),)
    return ChunkTransform(codecs=tuple(codecs))


def _bad_fill(ctx: _ArrayCtx) -> object:
    """Value to fill a bad/truncated tile with under ``on_bad_chunk='nan'``: NaN for
    float arrays, else the array's fill value (0 if it has none)."""
    if np.issubdtype(ctx.dtype, np.floating):
        return np.nan
    return ctx.fill_value if ctx.fill_value is not None else 0


class Scheduler:
    """Owns one event loop + a decode pool; streams tiles into a caller-owned pool.

    The :class:`ChunkPool` is passed in (dataset-owned, so it persists across epochs
    as the cache). :meth:`start` streams the stored chunks of an ordered chunk list;
    the consumer reads assembled chunks via :attr:`pool` and releases drained ones
    via :meth:`unpin`. Per chunk the scheduler skips fetch if the pool already holds
    it (a cross-epoch hit); misses are admitted against the pool's byte budget,
    awaiting an unpin when the working set fills it.
    """

    def __init__(
        self,
        store: Store,
        geometries: dict[str, ArrayGeometry],
        pool: ChunkPool,
        config: SchedulerConfig | None = None,
        owner: int | None = None,
    ) -> None:
        self._store = store
        self._geometries = geometries
        self._config = config or SchedulerConfig()
        if self._config.on_bad_chunk not in ("raise", "nan"):
            raise ValueError(
                f"on_bad_chunk must be 'raise' or 'nan', got {self._config.on_bad_chunk!r}"
            )
        self.pool = pool  # caller-owned: persists across epochs (the cache)
        # One iteration = one owner: this scheduler's admission pins and its producer's
        # block pins share a token, so the next epoch's prologue releases exactly this
        # iteration's references and leaves a concurrent iteration's alone. Minted here
        # when the caller does not supply one, so a standalone Scheduler still works.
        self._owner = pool.new_owner() if owner is None else owner
        self.bad_chunks: list[StoredChunkRead] = []  # tiles NaN-filled this run (observability)
        self._proto = default_buffer_prototype()
        self._arrays: dict[str, _ArrayCtx] = {}

        # in-flight observability (loop-thread only -> no lock needed)
        self.inflight_peak = 0
        self._inflight_now = 0

        self._decode_pool = decode_pool(self._config.decode_threads)
        # ONE loop for the whole process: zarr's. An async fsspec backend binds its aiohttp
        # session to whichever loop first awaits it, and because we read a store opened
        # through zarr that loop is zarr's -- so being on it is what deletes the bridge
        # (gcsfs and obstore now take the same inline `await`). obstore is loop-agnostic
        # and does not care. We are a guest on it: see `close` and `_shutdown`.
        self._loop = _get_loop()
        self._inflight: asyncio.Semaphore | None = None
        self._capacity: asyncio.Event | None = None  # set on unpin -> wakes a parked admit
        self._open_lock: asyncio.Lock | None = None
        self._ready = threading.Event()
        # OUR tasks, and only ours. `asyncio.all_tasks(loop)` is the *whole loop's* task
        # set: correct while the loop is private, destructive the moment it is shared --
        # it cancels unrelated zarr-sync work mid-flight. Mutated only on the loop thread
        # (create_task and the done-callback both run there), so it needs no lock.
        self._tasks: set[asyncio.Task] = set()
        self._loop.call_soon_threadsafe(self._setup)
        self._ready.wait(timeout=10)

    # -- loop lifecycle -----------------------------------------------------

    def _setup(self) -> None:
        """Create our asyncio primitives **on** the loop. Runs there, once, at construction.

        Deliberately does NOT call ``set_default_executor``: a borrowed loop has exactly one
        default-executor slot, and claiming it retunes zarr's own concurrency process-wide
        (then breaks it when we shut our pool down). We pass :attr:`_decode_pool` explicitly
        instead -- which is possible only because decode is now a *synchronous* call we
        dispatch ourselves (:meth:`_one`), rather than an async pipeline that dispatches to
        the loop default on our behalf.
        """
        self._inflight = asyncio.Semaphore(self._config.max_inflight)
        self._capacity = asyncio.Event()
        self._open_lock = asyncio.Lock()
        self._ready.set()

    def close(self) -> None:
        """Cancel this scheduler's in-flight driver. Nothing else.

        Graceful: a consumer may close mid-epoch (early ``break``) while ``_drive`` is
        still streaming, so we cancel our outstanding tasks and let them unwind.

        **A scheduler owns no shared resource, so it tears down no shared resource.** The
        loop is zarr's and lives for the process; the decode pool is process-wide and
        outlives every scheduler; the chunk pool is caller-owned and persists across epochs
        as the cache. Closing any of them here would break unrelated work elsewhere in the
        process -- which is not hypothetical: stopping the loop hangs every later
        ``zarr.core.sync.sync()`` call, and shutting the decode pool down makes the next
        scheduler raise ``cannot schedule new futures after shutdown``. Both were observed.
        """
        with contextlib.suppress(Exception):  # loop may already be down
            fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            fut.result(timeout=5)  # resolves while the loop is still running
        # NB: we do NOT shut the decode pool down. It is process-wide and outlives every
        # scheduler (see `decode_pool`); shutting it here would make the *next* scheduler
        # raise "cannot schedule new futures after shutdown" -- which is precisely the
        # shared-resource teardown bug this design exists to avoid, and the suite catches
        # it immediately. Tests that need a fresh pool call `reset_decode_pool()`.
        # The *chunk* pool is likewise caller-owned (it persists across epochs as the
        # cache) -- the dataset closes it, not us.

    async def _shutdown(self) -> None:
        """Cancel + drain **our** in-flight tasks. Never the loop's other work.

        Do NOT stop the loop here: stopping inside the awaited coroutine would race the
        delivery of this future's result back to :meth:`close`, which then blocks until
        its timeout.

        This deliberately does not use ``asyncio.all_tasks(self._loop)``. That is the
        whole loop's task set, and on a shared loop it cancels unrelated zarr-sync work
        -- observed as a ``CancelledError`` raised inside an innocent ``zarr`` read in a
        different thread. We cancel exactly what we created (:attr:`_tasks`).
        """
        current = asyncio.current_task()
        mine = [t for t in self._tasks if t is not current and not t.done()]
        for task in mine:
            task.cancel()
        await asyncio.gather(*mine, return_exceptions=True)

    def __enter__(self) -> Scheduler:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public, synchronous surface ---------------------------------------

    @property
    def owner(self) -> int:
        """This scheduler's reference token, for the consumer's ``pool.wait_ready``.

        One iteration = one owner: whoever drains this scheduler must wait under the
        same token the scheduler admits under, or its wait can never be satisfied.
        ``InSituDataset._iterate`` mints the token and hands it to both sides; a
        caller driving a ``Scheduler`` directly reads it here.
        """
        return self._owner

    def start(self, chunk_ids: Sequence[int] | np.ndarray, ref_spc: int) -> Future:
        """Begin streaming the stored chunks of ``chunk_ids`` (priority order).

        ``chunk_ids`` are in the reference (manifest) grid; ``ref_spc`` is that grid's
        sample-chunk size, used to map anchor chunks onto each variable's own chunks.
        Returns the driver future; a failure there poisons the pool so consumers
        re-raise. The consumer drives demand independently via :attr:`pool`.
        """
        reads = build_stored_chunk_reads(chunk_ids, self._geometries, ref_spc)
        fut = asyncio.run_coroutine_threadsafe(self._drive(reads), self._loop)
        fut.add_done_callback(self._on_drive_done)
        return fut

    def unpin_block(self, keys: set[tuple[str, int]]) -> None:
        """Release references on a set of drained ``(path, chunk_index)`` slots
        (thread-safe): the slots that hit refcount 0 become LRU-evictable; wake any
        admit parked on a full budget so it can evict them and proceed."""
        self.pool.unpin_keys(keys, self._owner)
        if self._capacity is not None:
            self._loop.call_soon_threadsafe(self._capacity.set)

    def _on_drive_done(self, fut: Future) -> None:
        # Cancellation is normal: close() cancels a still-finishing drive at epoch
        # end. Only a genuine driver exception poisons the pool (which now persists
        # across epochs, so a spurious poison would break the next epoch).
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc is not None:
            self.pool.set_error(exc)

    # -- async internals ----------------------------------------------------

    async def _drive(self, reads: list[StoredChunkRead]) -> None:
        await self._ensure_arrays()
        # Per (path, chunk): decided[k] = True if it was a cache hit (skip its tiles),
        # False if a miss we admitted (fetch its tiles). reads are chunk-major so a
        # (path, chunk) is first-seen on its first tile. Residency is held by the
        # consumer's per-block pins, not here -- admission only allocates the slot, and
        # a not-ready (in-flight) slot is eviction-protected until its fetch completes.
        decided: dict[tuple[str, int], bool] = {}
        tasks: list[asyncio.Task] = []
        me = asyncio.current_task()
        if me is not None:
            self._tasks.add(me)  # so _shutdown can cancel the driver itself
        try:
            for read in reads:
                key = (read.array, read.chunk_index)
                hit = decided.get(key)
                if hit is None:
                    hit = self.pool.pin_if_ready(read.array, read.chunk_index, self._owner)
                    if not hit:
                        await self._admit(read.array, read.chunk_index)  # may await an unpin
                    decided[key] = hit
                if hit:
                    continue  # cross-epoch hit: prepped chunk already resident, no fetch
                task = asyncio.create_task(self._one(read))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
                tasks.append(task)
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise
        finally:
            if me is not None:
                self._tasks.discard(me)

    async def _admit(self, array: str, chunk_index: int) -> None:
        """Admit a miss chunk against the byte budget, awaiting an unpin if full.

        The clear-then-recheck guards the lost wakeup: if an unpin lands between a
        failed admit and the clear, the recheck catches it; otherwise we wait for
        the next unpin. (Admissions are serialized on the loop, so no admit races
        another.)

        The wait is bounded only so we can re-test for a terminal starvation
        (:meth:`_starvation`); a timeout by itself means nothing and just loops.
        """
        assert self._capacity is not None
        while not self.pool.try_admit(array, chunk_index, self._owner):
            self._capacity.clear()
            if self.pool.try_admit(array, chunk_index, self._owner):
                return
            try:
                await asyncio.wait_for(self._capacity.wait(), STARVATION_POLL_S)
            except TimeoutError:
                starved = self._starvation(array, chunk_index)
                if starved is not None:
                    # `from None`: the poll timeout is how we got here, not why -- the
                    # cause is the budget, and chaining it would put a red herring at
                    # the top of the traceback the consumer re-raises.
                    raise starved from None

    def _starvation(self, array: str, chunk_index: int) -> RuntimeError | None:
        """The error for a *provably* unbreakable admission stall, else ``None``.

        Three facts together make the stall terminal, and none of them is a clock:

        * ``try_admit`` just failed -- the budget is full of in-flight or referenced
          slots, so no eviction can make room.
        * nothing is in flight -- no delivery is pending, so no slot can become ready
          and satisfy a waiter on its own.
        * a consumer is blocked in ``wait_ready`` -- and a blocked consumer never
          reaches its next ``unpin_block``, so the one thing that could free budget
          will not happen.

        The consumer is waiting on us and we are waiting on the consumer. Raising
        poisons the pool (see :meth:`_on_drive_done`), so the waiter re-raises this
        instead of hanging -- which is the whole point: the pre-fix failure mode was
        a silent wedge that reads exactly like slow storage.

        A merely slow consumer registers no waiter (it is computing, not blocked) or
        has work in flight, so it never trips this.
        """
        if self._inflight_now:
            return None
        waiting = self.pool.blocked_waiters()
        if not waiting:
            return None
        budget = self.pool.budget_bytes
        # A budget sized for ONE iteration cannot serve several: each holds its own
        # references, so the requirement multiplies. Name it rather than leave the caller
        # to rediscover it -- it is the likeliest cause once more than one owner is live.
        owners = self.pool.active_owners
        concurrent = (
            f" NOTE: {owners} iterations are sharing this pool (e.g. `zip(ds.train, "
            f"ds.val)`, or two DataLoaders); each needs its own working set resident at "
            f"once, so the budget must cover all {owners}."
            if owners > 1
            else ""
        )
        return RuntimeError(
            f"residency budget exhausted: cannot admit chunk {chunk_index} of {array!r}. "
            f"The pool holds {self.pool.resident_chunks} chunk(s) "
            f"({self.pool.resident_bytes} of {budget} bytes), every one of them pinned or "
            f"in flight; no tile is in flight; and the consumer is blocked waiting on "
            f"{sorted(waiting)[:4]}. Nothing can free a slot, so this would hang. The "
            f"working set is larger than the budget it was sized for -- raise "
            f"cache_budget_bytes, or lower batch_size / block_chunks.{concurrent}"
        )

    async def _ensure_arrays(self) -> None:
        if self._arrays:
            return
        assert self._open_lock is not None
        async with self._open_lock:
            if self._arrays:
                return
            store = self._store
            # Open each distinct array once, keyed by its zarr path: several windowed
            # views (same path, different offset) share one open + one decode path.
            for geom in self._geometries.values():
                if geom.path in self._arrays:
                    continue
                aa = await za.open_array(store=store, path=geom.path, mode="r")
                # Format-agnostic: zarr-v2 metadata exposes `dtype`/`encode_chunk_key`
                # where v3 has `data_type`/`chunk_key_encoding.encode_chunk_key` -- so the
                # engine reads public v2 stores (WeatherBench2 ARCO) as well as v3.
                meta = aa.metadata
                dtype = getattr(meta, "data_type", None) or meta.dtype
                spec = ArraySpec(
                    shape=meta.chunks,
                    dtype=dtype,
                    fill_value=meta.fill_value,
                    config=aa.config,
                    prototype=self._proto,
                )
                self._arrays[geom.path] = _ArrayCtx(
                    path=aa.store_path.path,
                    store=aa.store_path.store,
                    encode=meta.encode_chunk_key,
                    codec=_sync_transform(meta),
                    spec=spec,
                    chunk_shape=tuple(aa.metadata.chunks),
                    fill_value=aa.metadata.fill_value,
                    dtype=geom.dtype,
                    sample_axis=geom.sample_axis,
                )

    async def _one(self, read: StoredChunkRead) -> None:
        """Fetch + decode + deliver one stored tile, holding one in-flight slot.

        The in-flight slot is held across all three stages, so ``max_inflight`` is
        total concurrency. Decode and the delivery memcpy run on the decode pool (GIL
        released); the loop only awaits. A *fetch/decode* failure is a bad/truncated
        chunk -> the ``on_bad_chunk`` policy decides (poison, or NaN-fill and carry
        on). A failure *during delivery* is a genuine bug and always poisons.

        The whole body runs inside ``pool.tile_write``, which owns the obligation to
        tell the slot this task ended. That matters on the paths that never reach the
        pool: the ``return`` below, and cancellation at either ``await`` (an early
        ``break`` closes the scheduler with ``cancel_futures=True``). Without it a
        slot keeps a writer forever and can never be evicted.
        """
        assert self._inflight is not None
        ctx = self._arrays[read.array]
        async with self._inflight:
            self._inflight_now += 1
            self.inflight_peak = max(self.inflight_peak, self._inflight_now)
            try:
                with self.pool.tile_write(read.array, read.chunk_index, read.inner_coord) as w:
                    try:
                        tile = await self._fetch_decode(read, ctx)
                    except Exception as exc:  # noqa: BLE001 - bad/truncated stored chunk
                        if self._config.on_bad_chunk != "nan":
                            w.fail(exc)
                            return
                        self.bad_chunks.append(read)
                        tile = np.full(ctx.chunk_shape, _bad_fill(ctx), dtype=ctx.dtype)
                    try:
                        # Delivery used to be a second executor hop, because it was a memcpy
                        # into the assembled slot. On the tiled path it is now a dict
                        # assignment and a counter, so the hop is deleted rather than fused
                        # -- and the decoded tile stops living in this frame across it,
                        # which is what made tile residency scale with `max_inflight` (an IO
                        # dial) instead of with the pool doing the work.
                        #
                        # It stays a hop when the pool assembles, because delivering the
                        # LAST tile then also runs the assembly memcpy, the user
                        # `chunk_transform` and the mmap write-back (`ChunkPool._advance`).
                        # This loop is zarr's, shared with the whole process: running user
                        # code on it would stall every other zarr caller.
                        if self.pool.assembles:
                            await self._loop.run_in_executor(self._decode_pool, w.deliver, tile)
                        else:
                            w.deliver(tile)
                    except Exception as exc:  # noqa: BLE001 - a delivery failure is a real bug
                        w.fail(exc)
            finally:
                self._inflight_now -= 1

    async def _fetch_decode(self, read: StoredChunkRead, ctx: _ArrayCtx) -> np.ndarray:
        # Seam 1: logical (chunk_index, *inner_coord) -> physical chunk coord. The read is
        # addressed sample-first; reinsert the sample-axis index at its physical position
        # before encoding the store key (identity when sample_axis == 0).
        ax = ctx.sample_axis
        phys = read.inner_coord[:ax] + (read.chunk_index,) + read.inner_coord[ax:]
        key = ctx.path + "/" + ctx.encode(phys)
        # One path for every backend. We are on zarr's loop, which is the loop an async
        # fsspec session is bound to, so gcsfs and obstore are both a plain inline await.
        buf = await ctx.store.get(key, prototype=self._proto)
        if buf is None:  # absent chunk == all fill_value (zarr's getitem semantics)
            tile = np.full(ctx.chunk_shape, ctx.fill_value, dtype=ctx.dtype)
        else:
            # Decode on OUR pool, named explicitly -- never `None`, which would mean the
            # loop's default executor and would make us a guest that retunes its host.
            decoded = await self._loop.run_in_executor(
                self._decode_pool, ctx.codec.decode_chunk, buf, ctx.spec
            )
            tile = decoded.as_numpy_array()
        # Seam 2: the decoded tile is in physical order; move the sample axis to the front
        # so it matches the sample-first grid the pool and gather address (no-op when ax == 0).
        return np.moveaxis(tile, ax, 0) if ax else tile
