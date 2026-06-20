"""Torch handoff surface.

Ties the pieces together and exposes them to PyTorch *without* using the classic
DataLoader worker model. Parallelism lives in :class:`AsyncChunkReader`'s event
loop, so the recommended configuration is::

    loader = DataLoader(InSituDataset(...), batch_size=None, num_workers=0)

``batch_size=None`` because the dataset already yields assembled batches;
``num_workers=0`` because forking workers would re-introduce exactly the
redundant-read / nested-parallelism problems we set out to avoid.

torch (and torchdata.nodes) are optional imports so the core engine stays
framework-agnostic and importable on a box without torch installed.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

from .buffer import BufferConfig, ShuffleBlockBuffer
from .cache import ChunkCache
from .io import AsyncChunkReader, IOConfig
from .plan import build_read_plan
from .shuffle import block_shuffled_order, sequential_order
from .split import SplitManifest
from .store import open_geometries
from .types import ArrayGeometry, Batch, DecodedChunk, SplitName

# Optional torch surface. TYPE_CHECKING gives mypy a consistent view (the real
# IterableDataset) regardless of whether torch is installed in the checking env;
# the runtime branch handles torch-less installs.
if TYPE_CHECKING:
    from torch.utils.data import IterableDataset

    _HAS_TORCH = True
else:
    try:
        from torch.utils.data import IterableDataset

        _HAS_TORCH = True
    except ImportError:  # pragma: no cover - torch-less installs
        IterableDataset = object
        _HAS_TORCH = False


def _partition_blocks(order: np.ndarray, block_chunks: int) -> list[tuple[int, int, np.ndarray]]:
    """Split a draw ``order`` into shuffle-blocks: ``(row_start, row_stop, chunk_ids)``.

    ``block_shuffled_order`` shuffles samples *within* a block of ``block_chunks``
    chunks and concatenates blocks in chunk-permutation order, so every block is a
    contiguous row range over disjoint chunks. We recover the blocks from the chunks'
    first-appearance order (vectorized: O(chunks) of Python, not O(samples)), which
    is robust to short final chunks where fixed-stride slicing would misalign.
    """
    if not len(order):
        return []
    cids = order[:, 0].astype(np.int64)
    _, first_pos = np.unique(cids, return_index=True)
    appearance = cids[np.sort(first_pos)]  # chunk ids in order of first appearance
    block_of = np.full(int(cids.max()) + 1, -1, dtype=np.int64)
    block_of[appearance] = np.arange(len(appearance)) // block_chunks
    block_per_row = block_of[cids]
    starts = [0, *(np.flatnonzero(np.diff(block_per_row) != 0) + 1).tolist(), len(order)]
    return [
        (starts[k], starts[k + 1], appearance[k * block_chunks : (k + 1) * block_chunks])
        for k in range(len(starts) - 1)
    ]


class InSituDataset(IterableDataset):
    """An IterableDataset that streams shuffled batches from a Zarr archive.

    One epoch = permute the split's chunks -> walk shuffle-blocks -> for each
    block, async-fetch its chunks, fill the buffer, emit coalesced batches.
    """

    def __init__(
        self,
        store_url: str,
        manifest: SplitManifest,
        geometries: dict[str, ArrayGeometry] | None = None,
        split: SplitName = SplitName.TRAIN,
        *,
        batch_size: int = 32,
        block_chunks: int = 16,
        max_inflight: int | None = None,
        seed: int = 0,
        to_tensor: bool = True,
        shuffle: bool = True,
        prefetch_depth: int = 2,
        cache: ChunkCache | None = None,
        chunk_transforms: Sequence[Callable[[DecodedChunk], DecodedChunk]] = (),
        batch_transforms: Sequence[Callable[[Batch], Batch]] = (),
        **store_kwargs: Any,
    ) -> None:
        self.store_url = store_url
        self.store_kwargs = store_kwargs
        self.geometries = (
            geometries if geometries is not None else open_geometries(store_url, **store_kwargs)
        )
        self.manifest = manifest
        self.split = split
        self.variables = list(self.geometries)

        # v1 invariant: variables must share the sample axis (length + chunk size).
        # The draw order and gather use a single chunk size for all variables.
        spcs = {g.sample_chunk_size for g in self.geometries.values()}
        lengths = {g.n_samples for g in self.geometries.values()}
        if len(spcs) > 1 or len(lengths) > 1:
            raise ValueError(
                "All variables must share the same sample-axis length and chunk "
                f"size (v1 invariant); got sample_chunk_size={sorted(spcs)}, "
                f"n_samples={sorted(lengths)}."
            )
        # Read concurrency follows the block size unless overridden: a block fetches
        # its block_chunks chunks, so max_inflight defaults to block_chunks (raising
        # block_chunks alone then actually raises concurrency). Decoupling read
        # concurrency from the shuffle-block size is the V2 fetch scheduler.
        self.io_config = IOConfig(max_inflight=max_inflight or block_chunks)
        self.buffer_config = BufferConfig(block_chunks=block_chunks, batch_size=batch_size)
        self.seed = seed
        self.to_tensor = to_tensor and _HAS_TORCH
        self.shuffle = shuffle
        self.prefetch_depth = max(int(prefetch_depth), 1)
        # A caller-owned cache (MemoryCache / DiskCache) persists across epochs and
        # runs; None disables caching. See insitubatch.cache.
        self.cache = cache
        self.chunk_transforms = tuple(chunk_transforms)
        self.batch_transforms = tuple(batch_transforms)
        self._epoch = 0
        self.buffer_peak = 0  # peak resident chunks in the last epoch (observability)

    def set_epoch(self, epoch: int) -> None:
        """Call from the training loop so each epoch reshuffles deterministically."""
        self._epoch = epoch

    _SENTINEL = object()

    def __iter__(self) -> Iterator[Batch | dict]:
        """Drain assembled batches from a background producer (prefetch).

        A producer thread walks the draw order, assembles batches (async fan-out
        + gather + batch transforms) and pushes them onto a bounded queue; this
        consumer pops them. The queue (depth ``prefetch_depth``) provides the
        backpressure and the inter-batch overlap: while the caller works on batch
        N, the producer is already building N+1..N+depth.

        The producer walks shuffle-blocks and reads one block ahead, so the
        block-boundary IO overlaps the per-batch compute of the current block
        instead of stalling. (At literally zero compute the loader is
        IO-throughput-bound and the boundary is only smoothed, not removed.)
        A decoupled fetch scheduler that keeps many reads continuously in flight
        is V2 (see DESIGN.md).
        """
        geom = self.geometries[self.variables[0]]
        chunk_ids = np.asarray(self.manifest.chunks[self.split.value], dtype=np.int64)
        spc = geom.sample_chunk_size
        order = (
            block_shuffled_order(
                chunk_ids,
                spc,
                geom.n_samples,
                block_chunks=self.buffer_config.block_chunks,
                seed=self.seed,
                epoch=self._epoch,
            )
            if self.shuffle
            else sequential_order(chunk_ids, spc, geom.n_samples)
        )

        out_q: queue.Queue = queue.Queue(maxsize=self.prefetch_depth)
        stop = threading.Event()
        buf = ShuffleBlockBuffer(self.buffer_config, seed=self.seed)

        def produce(reader: AsyncChunkReader) -> None:
            bs = self.buffer_config.batch_size
            blocks = _partition_blocks(order, self.buffer_config.block_chunks)

            def read_block(chunk_ids: np.ndarray) -> list[DecodedChunk]:
                # Fetch + decode a whole block's chunks (deduped). Runs on the read-ahead
                # thread so it overlaps with the consumer draining the previous block.
                plan = build_read_plan(sorted({int(c) * spc for c in chunk_ids}), self.geometries)
                return list(reader.read_plan(plan))

            try:
                # One-block read-ahead: while we emit block b, the chunks for block b+1
                # are already being fetched, so the block-boundary IO is hidden instead
                # of stalling the consumer. Blocks use disjoint chunks, so the buffer
                # holds one block plus the in-flight block's decoded list (still
                # O(block_chunks)). Per-batch chunk-granularity look-ahead is V2 (the
                # decoupled fetch scheduler).
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="insitu-readahead") as ex:
                    inflight: Future[list[DecodedChunk]] | None = (
                        ex.submit(read_block, blocks[0][2]) if blocks else None
                    )
                    for b, (rstart, rstop, _chunk_ids) in enumerate(blocks):
                        assert inflight is not None
                        for decoded in inflight.result():
                            buf.add(decoded)
                        inflight = (
                            ex.submit(read_block, blocks[b + 1][2]) if b + 1 < len(blocks) else None
                        )
                        for start in range(rstart, rstop, bs):
                            if stop.is_set():
                                return
                            rows = order[start : min(start + bs, rstop)]
                            batch = buf.gather_batch(rows, self.variables, spc)
                            for transform in self.batch_transforms:
                                batch = transform(batch)
                            out_q.put(batch)  # blocks when full -> backpressure
                        # Block b is fully drained and shares no chunk with later blocks.
                        buf.evict_drained(set())
            except Exception as exc:  # noqa: BLE001 - forwarded to the consumer
                out_q.put(exc)
            finally:
                out_q.put(self._SENTINEL)

        with AsyncChunkReader(
            self.store_url,
            self.geometries,
            self.io_config,
            chunk_transforms=self.chunk_transforms,
            cache=self.cache,
            **self.store_kwargs,
        ) as reader:
            producer = threading.Thread(
                target=produce, args=(reader,), name="insitu-prefetch", daemon=True
            )
            producer.start()
            try:
                while True:
                    item = out_q.get()
                    if item is self._SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield self._maybe_tensor(item)
            finally:
                # Signal stop, then drain so a producer parked on a full queue can
                # proceed and exit before the reader (context manager) is closed.
                stop.set()
                while producer.is_alive():
                    with contextlib.suppress(queue.Empty):
                        out_q.get(timeout=0.05)
                producer.join(timeout=10)
                self.buffer_peak = buf.max_resident  # peak residency this epoch

    def _maybe_tensor(self, batch: Batch) -> Batch | dict:
        if not self.to_tensor:
            return batch
        import torch

        return {k: torch.from_numpy(v) for k, v in batch.arrays.items()}
