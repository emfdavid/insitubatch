"""Torch handoff surface.

Ties the pieces together and exposes them to PyTorch *without* using the classic
DataLoader worker model. Parallelism lives in :class:`Scheduler`'s event loop, so
the recommended configuration is::

    loader = DataLoader(InSituDataset(...), batch_size=None, num_workers=0)

``batch_size=None`` because the dataset already yields assembled batches;
``num_workers=0`` because forking workers would re-introduce exactly the
redundant-read / nested-parallelism problems we set out to avoid.

The engine is the decoupled fetch scheduler (DESIGN.md, M1.6): one event loop
streams stored-chunk reads under a single ``max_inflight`` budget and scatters
decoded tiles into a :class:`ChunkPool`; this producer walks the shuffle order,
waits on each block's assembled chunks, gathers coalesced batches, and evicts the
block to free residency. Read concurrency (``max_inflight``) and shuffle span /
residency (``block_chunks``) are independent dials.

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

from .scheduler import Scheduler, SchedulerConfig
from .shuffle import block_shuffled_order, sequential_order
from .split import SplitManifest
from .store import open_geometries
from .types import ArrayGeometry, Batch, DecodedChunk, SplitName

# Default for the single concurrency dial when the caller does not pin it. Unlike
# v1 (where read concurrency followed block_chunks), max_inflight is independent
# of the shuffle window -- it is sized to saturate the network, not the buffer.
DEFAULT_MAX_INFLIGHT = 32

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
    block, stream-fetch its stored chunks into the pool, gather coalesced batches,
    evict the block.
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

        self.batch_size = batch_size
        self.block_chunks = block_chunks
        self.seed = seed
        self.to_tensor = to_tensor and _HAS_TORCH
        self.shuffle = shuffle
        self.prefetch_depth = max(int(prefetch_depth), 1)
        self.chunk_transforms = tuple(chunk_transforms)
        self.batch_transforms = tuple(batch_transforms)
        self._epoch = 0
        self.resident_peak = 0  # peak resident outer chunks last epoch (observability)

        # One concurrency dial (max_inflight, network) decoupled from the shuffle
        # window (block_chunks). resident_cap = 2*block_chunks holds the current
        # block plus one read-ahead block, so block-boundary IO overlaps compute
        # (matching the v1 baseline residency in DESIGN's exp_c table). The invariant
        # resident_cap >= block_chunks is required: a batch may draw across a whole
        # block, so the block must be co-resident to gather. Exposed so the probe can
        # tune decode_threads at iteration time.
        self.scheduler_config = SchedulerConfig(
            max_inflight=max_inflight or DEFAULT_MAX_INFLIGHT,
            resident_cap=2 * block_chunks,
        )

    def set_epoch(self, epoch: int) -> None:
        """Call from the training loop so each epoch reshuffles deterministically."""
        self._epoch = epoch

    _SENTINEL = object()

    def _draw_order(self) -> np.ndarray:
        geom = self.geometries[self.variables[0]]
        chunk_ids = np.asarray(self.manifest.chunks[self.split.value], dtype=np.int64)
        spc = geom.sample_chunk_size
        if self.shuffle:
            return block_shuffled_order(
                chunk_ids,
                spc,
                geom.n_samples,
                block_chunks=self.block_chunks,
                seed=self.seed,
                epoch=self._epoch,
            )
        return sequential_order(chunk_ids, spc, geom.n_samples)

    def __iter__(self) -> Iterator[Batch | dict]:
        """Drain assembled batches from a background producer (prefetch).

        A producer thread starts the scheduler over the epoch's chunks (in draw
        order), then for each shuffle-block waits the block assembled, gathers its
        batches, and evicts it; this consumer pops from a bounded queue (depth
        ``prefetch_depth``) that provides backpressure and inter-batch overlap.

        The scheduler keeps ``max_inflight`` tiles continuously in flight and (via
        ``resident_cap``) fetches one block ahead, so block-boundary IO overlaps
        the current block's per-batch compute instead of stalling.
        """
        geom = self.geometries[self.variables[0]]
        spc = geom.sample_chunk_size
        order = self._draw_order()
        blocks = _partition_blocks(order, self.block_chunks)
        ordered_chunks = [int(c) for _rstart, _rstop, cids in blocks for c in cids]

        out_q: queue.Queue = queue.Queue(maxsize=self.prefetch_depth)
        stop = threading.Event()

        def produce(sched: Scheduler) -> None:
            bs = self.batch_size
            try:
                sched.start(ordered_chunks)
                for rstart, rstop, block_cids in blocks:
                    block = [int(c) for c in block_cids]
                    # A block's batches draw across all its chunks, so wait the whole
                    # block assembled before gathering (each wait is cheap once ready).
                    for cid in block:
                        for var in self.variables:
                            sched.pool.wait_ready(var, cid)
                    for start in range(rstart, rstop, bs):
                        if stop.is_set():
                            return
                        rows = order[start : min(start + bs, rstop)]
                        batch = sched.pool.gather(rows, self.variables, spc)
                        for transform in self.batch_transforms:
                            batch = transform(batch)
                        out_q.put(batch)  # blocks when full -> backpressure
                    sched.evict(set(block))  # frees residency for the next read-ahead
            except Exception as exc:  # noqa: BLE001 - forwarded to the consumer
                out_q.put(exc)
            finally:
                out_q.put(self._SENTINEL)

        with Scheduler(
            self.store_url,
            self.geometries,
            self.scheduler_config,
            chunk_transforms=self.chunk_transforms,
            **self.store_kwargs,
        ) as sched:
            producer = threading.Thread(
                target=produce, args=(sched,), name="insitu-prefetch", daemon=True
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
                # proceed and exit before the scheduler (context manager) is closed.
                stop.set()
                while producer.is_alive():
                    with contextlib.suppress(queue.Empty):
                        out_q.get(timeout=0.05)
                producer.join(timeout=10)
                self.resident_peak = sched.pool.max_resident  # peak residency this epoch

    def _maybe_tensor(self, batch: Batch) -> Batch | dict:
        if not self.to_tensor:
            return batch
        import torch

        return {k: torch.from_numpy(v) for k, v in batch.arrays.items()}
