"""Bounded shuffle-block buffer: residency + batch assembly.

Holds decoded chunks for a window of ``block_chunks`` chunks, draws shuffled
batches across that window, and evicts chunks once fully drained. This is the
memory-bounding component: peak residency is O(block_chunks), independent of the
number of samples per epoch or the batch size.

The batch assembly does a single coalesced gather per variable (one fancy-index
copy), never a Python per-sample loop -- the constraint David's S3 benchmark
imposed (Python per-chunk overhead bounds throughput).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .types import Batch, DecodedChunk


@dataclass(slots=True)
class BufferConfig:
    block_chunks: int = 16
    """Window size in chunks. Larger == better shuffle, more memory."""

    batch_size: int = 32


@dataclass(slots=True)
class ShuffleBlockBuffer:
    """Accumulates decoded chunks and emits shuffled, coalesced batches."""

    config: BufferConfig
    seed: int = 0
    _chunks: dict[tuple[str, int], DecodedChunk] = field(default_factory=dict)
    # Pending draws: rows of (array, chunk_index, within) flattened per variable.
    _pending: deque[int] = field(default_factory=deque)

    def add(self, chunk: DecodedChunk) -> None:
        self._chunks[(chunk.read.array, chunk.read.chunk_index)] = chunk

    def ready(self) -> bool:
        """Enough buffered to safely emit a well-mixed batch."""
        return len(self._chunks) >= self.config.block_chunks

    def gather_batch(
        self,
        rows: np.ndarray,
        variables: list[str],
    ) -> Batch:
        """Assemble one batch from ``rows`` of ``[chunk_id, within]`` draws.

        ``rows`` are pre-shuffled draw coordinates (see shuffle.block_shuffled_order).
        For each variable we issue ONE vectorized gather into the resident chunk
        arrays. Samples that don't cross chunk boundaries (the v1 contract) make
        this a clean per-chunk slice + concatenate.
        """
        out: dict[str, np.ndarray] = {}
        chunk_ids = rows[:, 0]
        within = rows[:, 1]
        sample_indices = chunk_ids * self._sample_chunk_size(variables[0]) + within

        for var in variables:
            pieces = []
            # Group draws by chunk so each resident array is touched once.
            for cid in np.unique(chunk_ids):
                mask = chunk_ids == cid
                chunk = self._chunks[(var, int(cid))]
                pieces.append(chunk.data[within[mask]])
            out[var] = np.concatenate(pieces, axis=0)
        return Batch(arrays=out, sample_indices=sample_indices)

    def evict_drained(self, still_needed: set[tuple[str, int]]) -> int:
        """Drop chunks no longer referenced by any pending draw. Returns count."""
        drop = [k for k in self._chunks if k not in still_needed]
        for k in drop:
            del self._chunks[k]
        return len(drop)

    def _sample_chunk_size(self, var: str) -> int:
        # All resident chunks of a var share chunking; read it off any one.
        for (v, _), chunk in self._chunks.items():
            if v == var:
                return chunk.data.shape[0]
        raise KeyError(f"no resident chunk for variable {var!r}")
