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
        sample_chunk_size: int,
    ) -> Batch:
        """Assemble one batch from ``rows`` of ``[chunk_id, within]`` draws.

        ``rows`` are pre-shuffled draw coordinates (see shuffle.block_shuffled_order).
        Draws are grouped by chunk so each resident array is touched once (one
        coalesced fancy-index per chunk); ``data`` and ``sample_indices`` are
        emitted in the *same* grouped order so row ``i`` of every variable and
        ``sample_indices[i]`` refer to the same sample. Intra-batch order is thus
        grouped-by-chunk -- irrelevant for training, and the cross-batch shuffle
        is preserved.

        ``sample_chunk_size`` is the array's true chunk length (from geometry),
        used to recover global sample indices -- NOT inferred from a resident
        chunk, which may be a short final chunk.
        """
        chunk_ids = rows[:, 0]
        within = rows[:, 1]
        uniq = np.unique(chunk_ids)

        out: dict[str, list[np.ndarray]] = {v: [] for v in variables}
        idx_pieces: list[np.ndarray] = []
        for cid in uniq:
            w = within[chunk_ids == cid]
            idx_pieces.append(cid * sample_chunk_size + w)
            for var in variables:
                out[var].append(self._chunks[(var, int(cid))].data[w])

        arrays = {var: np.concatenate(pieces, axis=0) for var, pieces in out.items()}
        return Batch(arrays=arrays, sample_indices=np.concatenate(idx_pieces))

    def evict_drained(self, still_needed: set[tuple[str, int]]) -> int:
        """Drop chunks no longer referenced by any pending draw. Returns count."""
        drop = [k for k in self._chunks if k not in still_needed]
        for k in drop:
            del self._chunks[k]
        return len(drop)
