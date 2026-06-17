"""Async IO driver: the obstore win.

This is where insitubatch *stands on* solved cloud IO rather than reinventing
it. A single dedicated asyncio event loop (one OS thread) issues many concurrent
chunk reads against a zarr v3 store -- ideally the obstore-backed store, whose
``get_ranges_async`` coalesces concurrent range requests in a single coroutine
and saturates the NIC without spawning Python threads per request.

Design rules (DESIGN.md, "the inversion"):
  * Parallelism lives *here*, in the event loop's concurrency, NOT in
    torch.DataLoader worker processes. The torch surface runs with
    ``num_workers=0``.
  * A bounded in-flight window (semaphore of ``max_inflight`` chunks) caps memory
    at O(in-flight chunks), independent of batch size.
  * Decode releases the GIL (numcodecs C codecs do) so decode overlaps IO. The
    Python hot path stays O(reads); never decode per-sample in Python.

Decode (the CPU step) runs on the loop's executor -- a bounded, reader-owned
``ThreadPoolExecutor`` (``IOConfig.decode_threads``) -- because zarr v3 offloads
codec decode via ``to_thread``. So IO concurrency lives on the loop while decode
parallelizes across cores in one managed pool, without blocking the fan-out.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import zarr.api.asynchronous as za

from .cache import ChunkCache
from .plan import ReadPlan
from .store import store_from_url
from .types import ArrayGeometry, ChunkRead, DecodedChunk


@dataclass(slots=True)
class IOConfig:
    max_inflight: int = 16
    """Upper bound on chunks in flight. Memory ~= max_inflight * chunk_nbytes."""

    decode_threads: int = 0
    """Size of the loop's executor for GIL-releasing decode (zarr offloads codec
    decode via ``to_thread``, so this is where decompression parallelizes across
    cores). ``0`` = auto = ``min(32, cpu+4)`` (Python's default-executor sizing)."""


class AsyncChunkReader:
    """Owns one asyncio event loop on a background thread and fans out reads.

    The public API is deliberately *synchronous and iterator-shaped* so the rest
    of the engine (buffer, torch surface) needs no async knowledge -- the loop is
    an implementation detail hidden behind a thread-safe queue.
    """

    def __init__(
        self,
        store_url: str,
        geometries: dict[str, ArrayGeometry],
        config: IOConfig | None = None,
        *,
        chunk_transforms: Sequence[Callable[[DecodedChunk], DecodedChunk]] = (),
        cache: ChunkCache | None = None,
        **store_kwargs: object,
    ) -> None:
        self._url = store_url
        self._store_kwargs = store_kwargs
        self._geometries = geometries
        self._config = config or IOConfig()
        self._chunk_transforms = tuple(chunk_transforms)
        self._cache = cache
        self._arrays: dict[str, za.AsyncArray] = {}  # opened lazily on the loop
        self._loop = asyncio.new_event_loop()
        self._sem: asyncio.Semaphore | None = None  # created on the loop bootstrap
        self._open_lock: asyncio.Lock | None = None
        # One bounded, named decode pool, owned by this reader and shut down in
        # close(). It backs the loop's run_in_executor / to_thread, which is where
        # zarr's codec decode runs -- so decode parallelizes here without blocking
        # the loop's IO fan-out.
        workers = self._config.decode_threads or min(32, (os.cpu_count() or 4) + 4)
        self._decode_pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="insitu-decode"
        )
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="insitu-io")
        self._thread.start()
        self._ready.wait()  # don't return until the loop + primitives exist

    # -- loop lifecycle -----------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.set_default_executor(self._decode_pool)  # decode -> our bounded pool
        self._sem = asyncio.Semaphore(self._config.max_inflight)
        self._open_lock = asyncio.Lock()
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._decode_pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> AsyncChunkReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public, synchronous surface ---------------------------------------

    _SENTINEL = object()

    def read_plan(self, plan: ReadPlan) -> Iterator[DecodedChunk]:
        """Fetch every read in ``plan`` concurrently; yield decoded chunks ASAP.

        Yields in completion order (not plan order) so a slow chunk never stalls
        the others -- the buffer downstream is responsible for reordering/gather.

        The bridge uses a thread-safe ``queue.Queue`` and a done-callback so the
        sentinel is *always* delivered -- even if the driver raises -- and any
        exception is re-raised on the caller's thread rather than deadlocking it.
        """
        out_q: queue.Queue = queue.Queue()

        def _on_done(fut: object) -> None:
            try:
                fut.result()  # type: ignore[attr-defined]  # surface driver errors
            except Exception as exc:  # noqa: BLE001 - forwarded to the consumer
                out_q.put(exc)
            finally:
                out_q.put(self._SENTINEL)

        fut = asyncio.run_coroutine_threadsafe(self._drive(plan, out_q), self._loop)
        fut.add_done_callback(_on_done)

        while True:
            item = out_q.get()
            if item is self._SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    # -- async internals ----------------------------------------------------

    async def _drive(self, plan: ReadPlan, out_q: queue.Queue) -> None:
        assert self._sem is not None  # guaranteed by _ready.wait() in __init__
        sem = self._sem  # local binding so the closure sees a non-Optional type
        await self._ensure_arrays()

        async def one(read: ChunkRead) -> None:
            async with sem:  # bound in-flight chunks -> bounded memory
                decoded = await self._fetch_and_decode(read)
            out_q.put(decoded)  # stdlib queue: thread-safe, non-blocking

        await asyncio.gather(*(one(r) for r in plan.reads))

    async def _ensure_arrays(self) -> None:
        """Open one AsyncArray per variable, once, sharing the store."""
        if self._arrays:
            return
        assert self._open_lock is not None
        async with self._open_lock:
            if self._arrays:  # double-checked: another coroutine may have won
                return
            store = store_from_url(self._url, **self._store_kwargs)  # type: ignore[arg-type]
            for name in self._geometries:
                self._arrays[name] = await za.open_array(store=store, path=name, mode="r")

    async def _fetch_and_decode(self, read: ChunkRead) -> DecodedChunk:
        """Fetch + decode one chunk via the zarr v3 async codec pipeline.

        The selection is exactly one chunk along the sample axis, full on the
        inner dims (the v1 sample-geometry contract). zarr fans the underlying
        byte-range reads out through obstore and runs the decode pipeline; for
        single-chunk inner dims this touches exactly one stored chunk.

        Decode runs via zarr's codec pipeline, which offloads the GIL-releasing
        decompression to the loop's executor (our bounded ``insitu-decode`` pool),
        so it parallelizes across cores rather than serializing on the loop thread.
        """
        if self._cache is not None:
            hit = self._cache.get(read.array, read.chunk_index)
            if hit is not None:  # prepped chunk: skips fetch + decode + transforms
                return hit

        geom = self._geometries[read.array]
        arr = self._arrays[read.array]
        samples = geom.samples_in_chunk(read.chunk_index)
        selection = (slice(samples.start, samples.stop), *(slice(None) for _ in geom.inner_shape))
        block = await arr.getitem(selection)
        chunk = DecodedChunk(read=read, data=np.asarray(block), sample_offset=samples.start)
        for transform in self._chunk_transforms:  # vectorized numpy -> GIL released
            chunk = transform(chunk)

        if self._cache is not None:
            self._cache.put(read.array, read.chunk_index, chunk)
        return chunk
