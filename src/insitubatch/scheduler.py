"""Scheduler: the fetch driver.

One asyncio event loop streams *stored-chunk* (tile) reads under a single
``max_inflight`` budget, decodes each tile off the loop (numcodecs C, GIL
released), and scatters it into its outer chunk's slot in a :class:`ChunkPool`.

The point is that the two things you tune are independent: **read concurrency** is
dialed by ``max_inflight`` alone, while **residency / shuffle span** is governed by
the pool's byte budget -- no nested inner/outer concurrency caps, no double
quantization of the fetch (reading one outer chunk per ``getitem`` and letting zarr
stitch the inner grid under a second cap is what couples them). See
[docs/architecture.md] for the full pipeline.

Two bounded resources, deliberately distinct:

* **in-flight** (``max_inflight``, an ``asyncio.Semaphore``) -- tiles in flight; a
  slot is held from fetch-start to scatter-done, spanning fetch + decode + scatter.
* **residency** (the pool's byte budget) -- admission (``pool.try_admit``) evicts
  unpinned-LRU to make room and *pins* the chunk; when the working set fills the
  budget (everything resident is pinned) the loop awaits a consumer ``unpin``. So
  the number of outstanding fetch tasks is bounded by the resident window, not by
  the epoch length.

Per chunk the scheduler first asks the pool whether it already holds it
(``pin_if_ready``): a still-resident prepped chunk is a hit and costs no fetch
(cross-epoch reuse, since the pool persists across epochs). Misses are admitted and
their tiles fetched. The consumer waits on slot readiness, gathers, and
:meth:`unpin`\\s drained chunks. Errors propagate two ways -- a per-tile
fetch/decode failure poisons just that chunk (``pool.fail``); a driver failure
poisons the whole pool (``pool.set_error``) so any waiter re-raises instead of
hanging.

Budget floor: a batch may draw from any chunk in its shuffle-block, so the whole
block must be co-resident to gather -- the budget must hold at least one block (the
producer sizes it to two: the current block plus one read-ahead block, so
block-boundary IO overlaps the current block's compute).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import zarr.api.asynchronous as za
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype

from .plan import build_stored_chunk_reads
from .pool import ChunkPool
from .store import StoreLike, as_store
from .types import ArrayGeometry, StoredChunkRead


@dataclass(slots=True)
class SchedulerConfig:
    max_inflight: int = 32
    """Tiles in flight at once -- the single concurrency dial. Memory in flight
    ~= max_inflight * stored_chunk_nbytes (+ transform scratch). Residency is
    bounded separately by the pool's byte budget (admission evicts unpinned-LRU)."""

    decode_threads: int = 0
    """Size of the decode/scatter pool (GIL-releasing codec decode + the disjoint
    scatter memcpy run here). ``0`` = auto = ``min(32, cpu+4)``."""

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
    codec pipeline + spec decode its bytes (this reconstructs exactly what
    ``arr.getitem`` would stitch, for single-inner and spatially-chunked arrays).
    """

    path: str
    store: object
    encode: Callable[[tuple[int, ...]], str]
    codec: object
    spec: ArraySpec
    chunk_shape: tuple[int, ...]
    fill_value: object
    dtype: np.dtype


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
        store: StoreLike,
        geometries: dict[str, ArrayGeometry],
        pool: ChunkPool,
        config: SchedulerConfig | None = None,
        **store_kwargs: object,
    ) -> None:
        self._store = store
        self._store_kwargs = store_kwargs
        self._geometries = geometries
        self._config = config or SchedulerConfig()
        if self._config.on_bad_chunk not in ("raise", "nan"):
            raise ValueError(
                f"on_bad_chunk must be 'raise' or 'nan', got {self._config.on_bad_chunk!r}"
            )
        self.pool = pool  # caller-owned: persists across epochs (the cache)
        self.bad_chunks: list[StoredChunkRead] = []  # tiles NaN-filled this run (observability)
        self._proto = default_buffer_prototype()
        self._arrays: dict[str, _ArrayCtx] = {}

        # in-flight observability (loop-thread only -> no lock needed)
        self.inflight_peak = 0
        self._inflight_now = 0

        workers = self._config.decode_threads or min(32, (os.cpu_count() or 4) + 4)
        self._decode_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="insitu-dec")
        self._loop = asyncio.new_event_loop()
        self._inflight: asyncio.Semaphore | None = None
        self._capacity: asyncio.Event | None = None  # set on unpin -> wakes a parked admit
        self._open_lock: asyncio.Lock | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="insitu-sched")
        self._thread.start()
        self._ready.wait()

    # -- loop lifecycle -----------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.set_default_executor(self._decode_pool)  # decode + scatter -> our pool
        self._inflight = asyncio.Semaphore(self._config.max_inflight)
        self._capacity = asyncio.Event()
        self._open_lock = asyncio.Lock()
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def close(self) -> None:
        """Cancel any in-flight driver, then stop the loop and decode pool.

        Graceful: a consumer may close mid-epoch (early ``break``) while ``_drive``
        is still streaming. We cancel outstanding tasks and let them unwind *before*
        stopping the loop, so no coroutine is orphaned (which would surface as
        ``GeneratorExit`` / "never awaited" warnings on GC).
        """
        with contextlib.suppress(Exception):  # loop may already be down
            fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            fut.result(timeout=5)  # resolves while the loop is still running
        self._loop.call_soon_threadsafe(self._loop.stop)  # ...then stop it
        self._thread.join(timeout=5)
        if not self._thread.is_alive():  # loop has exited run_forever -> safe to close
            self._loop.close()  # release the self-pipe now; don't leave it for __del__
        self._decode_pool.shutdown(wait=False, cancel_futures=True)
        # NB: the pool is caller-owned (persists across epochs as the cache) -- the
        # dataset closes it, not us.

    async def _shutdown(self) -> None:
        # Cancel + drain in-flight tasks, but do NOT stop the loop here: stopping
        # inside the awaited coroutine would race the delivery of this future's
        # result back to close(), which then blocks until its timeout.
        tasks = [t for t in asyncio.all_tasks(self._loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def __enter__(self) -> Scheduler:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public, synchronous surface ---------------------------------------

    def start(self, chunk_ids: Sequence[int] | np.ndarray) -> Future:
        """Begin streaming the stored chunks of ``chunk_ids`` (priority order).

        Returns the driver future; a failure there poisons the pool so consumers
        re-raise. The consumer drives demand independently via :attr:`pool`.
        """
        reads = build_stored_chunk_reads(chunk_ids, self._geometries)
        fut = asyncio.run_coroutine_threadsafe(self._drive(reads), self._loop)
        fut.add_done_callback(self._on_drive_done)
        return fut

    def unpin(self, chunk_ids: set[int]) -> None:
        """Release drained outer chunks (thread-safe): now LRU-evictable, and wake
        any admit parked on a full budget so it can evict them and proceed."""
        self.pool.unpin(chunk_ids)
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
        # Per (var, chunk): decided[k] = True if it was a cache hit (skip its tiles),
        # False if a miss we admitted (fetch its tiles). reads are chunk-major so a
        # (var, chunk) is first-seen on its first tile.
        decided: dict[tuple[str, int], bool] = {}
        tasks: list[asyncio.Task] = []
        try:
            for read in reads:
                key = (read.array, read.chunk_index)
                hit = decided.get(key)
                if hit is None:
                    hit = self.pool.pin_if_ready(read.array, read.chunk_index)
                    if not hit:
                        await self._admit(read.array, read.chunk_index)  # may await an unpin
                    decided[key] = hit
                if hit:
                    continue  # cross-epoch hit: prepped chunk already resident, no fetch
                tasks.append(asyncio.create_task(self._one(read)))
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise

    async def _admit(self, array: str, chunk_index: int) -> None:
        """Admit a miss chunk against the byte budget, awaiting an unpin if full.

        The clear-then-recheck guards the lost wakeup: if an unpin lands between a
        failed admit and the clear, the recheck catches it; otherwise we wait for
        the next unpin. (Admissions are serialized on the loop, so no admit races
        another.)
        """
        assert self._capacity is not None
        while not self.pool.try_admit(array, chunk_index):
            self._capacity.clear()
            if self.pool.try_admit(array, chunk_index):
                return
            await self._capacity.wait()

    async def _ensure_arrays(self) -> None:
        if self._arrays:
            return
        assert self._open_lock is not None
        async with self._open_lock:
            if self._arrays:
                return
            store = as_store(self._store, **self._store_kwargs)  # type: ignore[arg-type]
            for name, geom in self._geometries.items():
                aa = await za.open_array(store=store, path=name, mode="r")
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
                self._arrays[name] = _ArrayCtx(
                    path=aa.store_path.path,
                    store=aa.store_path.store,
                    encode=meta.encode_chunk_key,
                    codec=aa.codec_pipeline,
                    spec=spec,
                    chunk_shape=tuple(aa.metadata.chunks),
                    fill_value=aa.metadata.fill_value,
                    dtype=geom.dtype,
                )

    async def _one(self, read: StoredChunkRead) -> None:
        """Fetch + decode + scatter one stored tile, holding one in-flight slot.

        The in-flight slot is held across all three stages, so ``max_inflight`` is
        total concurrency. Decode and the scatter memcpy run on the decode pool (GIL
        released); the loop only awaits. A *fetch/decode* failure is a bad/truncated
        chunk -> the ``on_bad_chunk`` policy decides (poison, or NaN-fill and carry
        on). A failure *during scatter* is a genuine bug and always poisons.
        """
        assert self._inflight is not None
        ctx = self._arrays[read.array]
        async with self._inflight:
            self._inflight_now += 1
            self.inflight_peak = max(self.inflight_peak, self._inflight_now)
            try:
                try:
                    tile = await self._fetch_decode(read, ctx)
                except Exception as exc:  # noqa: BLE001 - bad/truncated stored chunk
                    if self._config.on_bad_chunk != "nan":
                        self.pool.fail(read.array, read.chunk_index, exc)
                        return
                    self.bad_chunks.append(read)
                    tile = np.full(ctx.chunk_shape, _bad_fill(ctx), dtype=ctx.dtype)
                try:
                    await self._loop.run_in_executor(
                        None,
                        self.pool.scatter,
                        read.array,
                        read.chunk_index,
                        read.inner_coord,
                        tile,
                    )
                except Exception as exc:  # noqa: BLE001 - a scatter failure is a real bug
                    self.pool.fail(read.array, read.chunk_index, exc)
            finally:
                self._inflight_now -= 1

    async def _fetch_decode(self, read: StoredChunkRead, ctx: _ArrayCtx) -> np.ndarray:
        key = ctx.path + "/" + ctx.encode(read.coords)
        buf = await ctx.store.get(key, prototype=self._proto)  # type: ignore[attr-defined]
        if buf is None:  # absent chunk == all fill_value (zarr's getitem semantics)
            return np.full(ctx.chunk_shape, ctx.fill_value, dtype=ctx.dtype)
        [decoded] = list(await ctx.codec.decode([(buf, ctx.spec)]))  # type: ignore[attr-defined]
        return decoded.as_numpy_array()
