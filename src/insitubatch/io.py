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

Status: SKELETON. The async store wiring is stubbed at the marked TODOs so the
module imports and the control flow is reviewable without a live store.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .plan import ReadPlan
from .types import ArrayGeometry, ChunkRead, DecodedChunk


@dataclass(slots=True)
class IOConfig:
    max_inflight: int = 16
    """Upper bound on chunks in flight. Memory ~= max_inflight * chunk_nbytes."""

    decode_threads: int = 4
    """Worker threads for GIL-releasing decode, overlapped with IO."""


class AsyncChunkReader:
    """Owns one asyncio event loop on a background thread and fans out reads.

    The public API is deliberately *synchronous and iterator-shaped* so the rest
    of the engine (buffer, torch surface) needs no async knowledge -- the loop is
    an implementation detail hidden behind a thread-safe queue.
    """

    def __init__(
        self,
        store: object,  # zarr AsyncArray / obstore store; typed loosely until wired
        geometries: dict[str, ArrayGeometry],
        config: IOConfig | None = None,
    ) -> None:
        self._store = store
        self._geometries = geometries
        self._config = config or IOConfig()
        self._loop = asyncio.new_event_loop()
        self._sem: asyncio.Semaphore | None = None  # created on the loop bootstrap
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="insitu-io")
        self._thread.start()
        self._ready.wait()  # don't return until the loop + semaphore exist

    # -- loop lifecycle -----------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._sem = asyncio.Semaphore(self._config.max_inflight)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

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

        async def one(read: ChunkRead) -> None:
            async with self._sem:  # bound in-flight chunks -> bounded memory
                decoded = await self._fetch_and_decode(read)
            out_q.put(decoded)  # stdlib queue: thread-safe, non-blocking

        await asyncio.gather(*(one(r) for r in plan.reads))

    async def _fetch_and_decode(self, read: ChunkRead) -> DecodedChunk:
        """Fetch one chunk's bytes and decode to ndarray.

        TODO(io): wire to the real store. With zarr v3 async this is roughly::

            arr = self._store[read.array]              # AsyncArray
            block = await arr.getitem((slice(c0, c1), ...))

        or, for the obstore-direct path, issue ``get_ranges_async`` for the
        chunk's byte range(s) and hand the compressed buffer to a decode running
        in a thread (numcodecs releases the GIL) via
        ``loop.run_in_executor(decode_pool, decode, buf)``.
        """
        geom = self._geometries[read.array]
        samples = geom.samples_in_chunk(read.chunk_index)
        # Placeholder payload so the pipeline is exercisable end-to-end in tests.
        data = np.zeros((len(samples), *geom.inner_shape), dtype=geom.dtype)
        return DecodedChunk(read=read, data=data, sample_offset=samples.start)
