"""insitubatch -- train in place on n-dimensional cloud tensors.

The loader-orchestration layer that sits on top of *already-solved* async cloud
IO (obstore / zarr v3 / icechunk): turns an existing Zarr archive into a
shuffled, split-aware, GPU-saturating PyTorch source with no reshard and a
Python hot path that scales with chunks, not samples.

See DESIGN.md for the full rationale.
"""

from __future__ import annotations

from .buffer import BufferConfig, ShuffleBlockBuffer
from .io import AsyncChunkReader, IOConfig
from .plan import ReadPlan, build_read_plan, dedup_ratio
from .shuffle import block_shuffled_order, chunk_permutation, shuffle_quality
from .split import SplitManifest, split_by_chunk
from .types import ArrayGeometry, Batch, ChunkRead, DecodedChunk, SplitName

__version__ = "0.1.0"

__all__ = [
    "ArrayGeometry",
    "AsyncChunkReader",
    "Batch",
    "BufferConfig",
    "ChunkRead",
    "DecodedChunk",
    "IOConfig",
    "ReadPlan",
    "ShuffleBlockBuffer",
    "SplitManifest",
    "SplitName",
    "block_shuffled_order",
    "build_read_plan",
    "chunk_permutation",
    "dedup_ratio",
    "shuffle_quality",
    "split_by_chunk",
]
