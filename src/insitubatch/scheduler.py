"""Scheduler: the decoupled fetch driver (DESIGN.md, M1.6).

One asyncio event loop streams *stored-chunk* (tile) reads under a single
``max_inflight`` budget, decodes each tile off the loop (numcodecs C, GIL
released), and scatters it into its outer chunk's slot in a :class:`ChunkPool`.
This is the inversion that breaks v1's coupling: read concurrency is dialed by
``max_inflight`` alone, while residency/shuffle span is dialed by the pool's
``resident_cap`` -- separately, with no nested concurrency caps and no inner/outer
double-quantization.

Two bounded resources, deliberately distinct:

* **``max_inflight``** (an ``asyncio.Semaphore``) -- tiles in flight; a slot is
  held from fetch-start to scatter-done, spanning fetch + decode + scatter.
* **``resident_cap``** (an ``asyncio.Semaphore`` over *outer positions*) -- how
  many outer chunks may be resident in the pool at once. The admission loop
  acquires one unit the first time it sees an outer chunk and allocates that
  chunk's slots; the consumer releases it (cross-thread, via the loop) when it
  evicts the drained chunk. Because the loop *blocks* on this acquire once the cap
  is full, the number of outstanding fetch tasks is bounded by the resident
  window -- not by the epoch length.

Hard invariant: ``resident_cap >= block_chunks``. A batch may draw from any chunk
in its shuffle-block, so the whole block must be co-resident to gather; a smaller
cap would deadlock waiting on a chunk that can never be admitted. The producer
sets ``resident_cap = block_chunks + read_ahead`` (the read-ahead margin is what
keeps the next block's IO overlapping the current block's compute).

The hand-off to the consumer is the pool: the consumer thread waits on slot
readiness, gathers, and calls :meth:`evict`. Errors propagate two ways -- a
per-tile fetch/decode failure poisons just that chunk (``pool.fail``); a driver
failure poisons the whole pool (``pool.set_error``) so any waiter re-raises
instead of hanging.
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
from .pool import ChunkPool, ChunkTransform
from .store import store_from_url
from .types import ArrayGeometry, StoredChunkRead


@dataclass(slots=True)
class SchedulerConfig:
    max_inflight: int = 32
    """Tiles in flight at once -- the single concurrency dial. Memory in flight
    ~= max_inflight * stored_chunk_nbytes (+ transform scratch)."""

    resident_cap: int = 0
    """Outer chunks resident in the pool. ``0`` = unbounded (no admission
    backpressure); the producer sets ``block_chunks + read_ahead``. Must be
    ``>= block_chunks`` whenever batches draw across a shuffle-block."""

    decode_threads: int = 0
    """Size of the decode/scatter pool (GIL-releasing codec decode + the disjoint
    scatter memcpy run here). ``0`` = auto = ``min(32, cpu+4)``."""


@dataclass(slots=True)
class _ArrayCtx:
    """Per-variable handles for the stored-chunk fetch+decode path (spike-validated)."""

    path: str
    store: object
    encode: Callable[[tuple[int, ...]], str]
    codec: object
    spec: ArraySpec
    chunk_shape: tuple[int, ...]
    fill_value: object
    dtype: np.dtype


class Scheduler:
    """Owns one event loop, a decode pool, and a :class:`ChunkPool`; streams tiles.

    Synchronous, thread-friendly surface (the loop is hidden): :meth:`start` kicks
    off the fetch stream for an ordered list of outer chunks; the consumer reads
    assembled chunks via :attr:`pool` and frees residency via :meth:`evict`.
    """

    def __init__(
        self,
        store_url: str,
        geometries: dict[str, ArrayGeometry],
        config: SchedulerConfig | None = None,
        *,
        chunk_transforms: Sequence[ChunkTransform] = (),
        **store_kwargs: object,
    ) -> None:
        self._url = store_url
        self._store_kwargs = store_kwargs
        self._geometries = geometries
        self._config = config or SchedulerConfig()
        self.pool = ChunkPool(geometries, chunk_transforms=chunk_transforms)
        self._proto = default_buffer_prototype()
        self._arrays: dict[str, _ArrayCtx] = {}

        # in-flight observability (loop-thread only -> no lock needed)
        self.inflight_peak = 0
        self._inflight_now = 0

        workers = self._config.decode_threads or min(32, (os.cpu_count() or 4) + 4)
        self._decode_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="insitu-dec")
        self._loop = asyncio.new_event_loop()
        self._inflight: asyncio.Semaphore | None = None
        self._residency: asyncio.Semaphore | None = None
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
        cap = self._config.resident_cap
        self._residency = asyncio.Semaphore(cap) if cap > 0 else None
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
        self.pool.close()  # free any slots left resident (mmap files) on early exit

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

    def evict(self, chunk_ids: set[int]) -> int:
        """Evict drained outer chunks and free their residency (thread-safe).

        The single eviction entry point so pool residency and the admission
        semaphore stay in lock-step. Returns outer positions dropped.
        """
        n = self.pool.evict(chunk_ids)
        if n and self._residency is not None:
            self._loop.call_soon_threadsafe(self._release_residency, n)
        return n

    def _release_residency(self, n: int) -> None:
        assert self._residency is not None
        for _ in range(n):
            self._residency.release()

    def _on_drive_done(self, fut: Future) -> None:
        try:
            fut.result()
        except Exception as exc:  # noqa: BLE001 - poison the pool; waiters re-raise
            self.pool.set_error(exc)

    # -- async internals ----------------------------------------------------

    async def _drive(self, reads: list[StoredChunkRead]) -> None:
        await self._ensure_arrays()
        admitted: set[int] = set()
        tasks: list[asyncio.Task] = []
        try:
            for read in reads:  # chunk-major: a new chunk_index is first-seen here
                if read.chunk_index not in admitted:
                    if self._residency is not None:
                        await self._residency.acquire()  # backpressure on resident chunks
                    admitted.add(read.chunk_index)
                    for var in self._geometries:
                        self.pool.allocate(var, read.chunk_index)
                tasks.append(asyncio.create_task(self._one(read)))
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise

    async def _ensure_arrays(self) -> None:
        if self._arrays:
            return
        assert self._open_lock is not None
        async with self._open_lock:
            if self._arrays:
                return
            store = store_from_url(self._url, **self._store_kwargs)  # type: ignore[arg-type]
            for name, geom in self._geometries.items():
                aa = await za.open_array(store=store, path=name, mode="r")
                spec = ArraySpec(
                    shape=aa.metadata.chunks,
                    dtype=aa.metadata.data_type,
                    fill_value=aa.metadata.fill_value,
                    config=aa.config,
                    prototype=self._proto,
                )
                self._arrays[name] = _ArrayCtx(
                    path=aa.store_path.path,
                    store=aa.store_path.store,
                    encode=aa.metadata.chunk_key_encoding.encode_chunk_key,
                    codec=aa.codec_pipeline,
                    spec=spec,
                    chunk_shape=tuple(aa.metadata.chunks),
                    fill_value=aa.metadata.fill_value,
                    dtype=geom.dtype,
                )

    async def _one(self, read: StoredChunkRead) -> None:
        """Fetch + decode + scatter one stored tile, holding one in-flight slot.

        The slot spans all three stages (the budget is total concurrency, DESIGN).
        Decode and the scatter memcpy run on the decode pool (GIL released); the
        loop only awaits. A failure poisons just this chunk so the consumer's
        ``wait_ready`` re-raises without stalling the rest.
        """
        assert self._inflight is not None
        ctx = self._arrays[read.array]
        async with self._inflight:
            self._inflight_now += 1
            self.inflight_peak = max(self.inflight_peak, self._inflight_now)
            try:
                key = ctx.path + "/" + ctx.encode(read.coords)
                buf = await ctx.store.get(key, prototype=self._proto)  # type: ignore[attr-defined]
                if buf is None:  # absent chunk == all fill_value (zarr's getitem semantics)
                    tile = np.full(ctx.chunk_shape, ctx.fill_value, dtype=ctx.dtype)
                else:
                    [decoded] = list(await ctx.codec.decode([(buf, ctx.spec)]))  # type: ignore[attr-defined]
                    tile = decoded.as_numpy_array()
                await self._loop.run_in_executor(
                    None, self.pool.scatter, read.array, read.chunk_index, read.inner_coord, tile
                )
            except Exception as exc:  # noqa: BLE001 - poison this chunk, surface to consumer
                self.pool.fail(read.array, read.chunk_index, exc)
            finally:
                self._inflight_now -= 1
