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
        max_inflight: int = 16,
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
        self.io_config = IOConfig(max_inflight=max_inflight)
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

        Prefetch granularity is per-batch in v1; chunk-granularity look-ahead
        (reads for N+2 starting before N+1 is assembled) is a later refinement.
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
            # Last window index at which each chunk id is drawn, so a chunk is
            # evicted only once no later batch needs it -> each chunk is read and
            # decoded once per epoch (not re-read per batch). Vectorized: O(chunks)
            # of Python, the scatter-max runs in C.
            if len(order):
                windows = np.arange(len(order)) // bs
                last_use = np.zeros(int(order[:, 0].max()) + 1, dtype=np.int64)
                np.maximum.at(last_use, order[:, 0], windows)
            else:
                last_use = np.zeros(0, dtype=np.int64)
            try:
                for w, start in enumerate(range(0, len(order), bs)):
                    if stop.is_set():
                        break
                    rows = order[start : start + bs]
                    needed = {(v, int(c)) for v in self.variables for c in np.unique(rows[:, 0])}
                    missing = [int(c) * spc for (v, c) in needed if (v, c) not in buf._chunks]
                    if missing:
                        plan = build_read_plan(sorted(set(missing)), self.geometries)
                        for decoded in reader.read_plan(plan):
                            buf.add(decoded)
                    batch = buf.gather_batch(rows, self.variables, spc)
                    for transform in self.batch_transforms:
                        batch = transform(batch)
                    out_q.put(batch)  # blocks when full -> backpressure
                    # Evict only chunks not needed in any later batch.
                    keep = {key for key in buf._chunks if last_use[key[1]] > w}
                    buf.evict_drained(keep)
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
