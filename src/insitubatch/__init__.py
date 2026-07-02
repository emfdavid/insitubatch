"""insitubatch -- train in place on n-dimensional cloud tensors.

The loader-orchestration layer that sits on top of *already-solved* async cloud
IO (obstore / zarr v3 / icechunk): turns an existing Zarr archive into a
shuffled, split-aware, GPU-saturating PyTorch source with no reshard and a
Python hot path that scales with chunks, not samples.

See DESIGN.md for the full rationale.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .plan import build_stored_chunk_reads
from .pool import ChunkPool
from .scheduler import Scheduler, SchedulerConfig
from .shuffle import (
    block_shuffled_order,
    chunk_permutation,
    sequential_order,
    shuffle_quality,
)
from .split import SplitManifest, split_by_chunk, valid_anchor_range
from .store import (
    StoreLike,
    arraylake_store,
    as_store,
    ensure_local_dir,
    fsspec_store,
    open_geometries,
    store_from_url,
)
from .transforms import (
    BatchTransform,
    ChunkTransform,
    StandardScaler,
)
from .types import ArrayGeometry, Batch, ChunkRead, DecodedChunk, SplitName, StoredChunkRead

# Single source of truth is pyproject.toml -> installed dist metadata; never a
# second hardcoded string to drift. Fallback covers running from a source tree
# with no install (rare in this uv-managed repo, but keeps import non-fatal).
try:
    __version__ = version("insitubatch")
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "ArrayGeometry",
    "Batch",
    "BatchTransform",
    "ChunkPool",
    "ChunkRead",
    "ChunkTransform",
    "DecodedChunk",
    "Scheduler",
    "SchedulerConfig",
    "SplitManifest",
    "SplitName",
    "StandardScaler",
    "StoreLike",
    "StoredChunkRead",
    "arraylake_store",
    "as_store",
    "block_shuffled_order",
    "build_stored_chunk_reads",
    "chunk_permutation",
    "ensure_local_dir",
    "fsspec_store",
    "open_geometries",
    "sequential_order",
    "shuffle_quality",
    "split_by_chunk",
    "store_from_url",
    "valid_anchor_range",
]
