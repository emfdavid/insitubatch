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

from collections.abc import Iterator

import numpy as np

from .buffer import BufferConfig, ShuffleBlockBuffer
from .io import AsyncChunkReader, IOConfig
from .plan import build_read_plan
from .shuffle import block_shuffled_order
from .split import SplitManifest
from .store import open_geometries
from .types import ArrayGeometry, Batch, SplitName

try:  # optional torch surface
    from torch.utils.data import IterableDataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised on torch-less installs
    IterableDataset = object  # type: ignore[assignment,misc]
    _HAS_TORCH = False


class InSituDataset(IterableDataset):  # type: ignore[misc]
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
        **store_kwargs: object,
    ) -> None:
        self.store_url = store_url
        self.store_kwargs = store_kwargs
        self.geometries = geometries if geometries is not None else open_geometries(store_url)
        self.manifest = manifest
        self.split = split
        self.variables = list(self.geometries)
        self.io_config = IOConfig(max_inflight=max_inflight)
        self.buffer_config = BufferConfig(block_chunks=block_chunks, batch_size=batch_size)
        self.seed = seed
        self.to_tensor = to_tensor and _HAS_TORCH
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call from the training loop so each epoch reshuffles deterministically."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[Batch | dict]:
        geom = self.geometries[self.variables[0]]
        chunk_ids = np.asarray(self.manifest.chunks[self.split.value], dtype=np.int64)
        spc = geom.sample_chunk_size

        order = block_shuffled_order(
            chunk_ids,
            spc,
            block_chunks=self.buffer_config.block_chunks,
            seed=self.seed,
            epoch=self._epoch,
        )

        with AsyncChunkReader(
            self.store_url, self.geometries, self.io_config, **self.store_kwargs
        ) as reader:
            buf = ShuffleBlockBuffer(self.buffer_config, seed=self.seed)
            bs = self.buffer_config.batch_size

            # Walk the emitted draw order in batch-sized windows. For each window
            # we ensure its chunks are resident (async fan-out), then gather.
            for start in range(0, len(order), bs):
                rows = order[start : start + bs]
                needed = {(v, int(c)) for v in self.variables for c in np.unique(rows[:, 0])}
                missing_samples = [
                    int(c) * spc for (v, c) in needed if (v, c) not in buf._chunks
                ]
                if missing_samples:
                    plan = build_read_plan(sorted(set(missing_samples)), self.geometries)
                    for decoded in reader.read_plan(plan):
                        buf.add(decoded)

                batch = buf.gather_batch(rows, self.variables, spc)
                yield self._maybe_tensor(batch)
                buf.evict_drained(needed)

    def _maybe_tensor(self, batch: Batch) -> Batch | dict:
        if not self.to_tensor:
            return batch
        import torch

        return {k: torch.from_numpy(v) for k, v in batch.arrays.items()}
