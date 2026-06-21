"""Preprocessing transforms — three stages, placed by cost.

See docs/architecture.md ("Transforms"). This module implements the two CPU
stages; ``device_transform`` lives in the framework adapters (M2/M3).

- **chunk_transform**: per-chunk, runs in the IO loop before shuffle/gather,
  amortized over every sample drawn from the chunk, and *cacheable* (deterministic,
  position-independent). Home for scaling, unit conversion, chunk-local regrid.
  Must be **vectorized numpy** so it releases the GIL and overlaps IO.
- **batch_transform**: per-batch, after gather. For cross-variable derived fields,
  per-sample random augmentation, channel stacking. Not cached.

Rule of thumb: per-variable + per-chunk + deterministic -> chunk stage;
cross-variable or per-sample-random -> batch stage; cross-chunk -> not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .types import Batch, DecodedChunk


@runtime_checkable
class ChunkTransform(Protocol):
    """Per-chunk transform applied before shuffle/gather (cacheable)."""

    def __call__(self, chunk: DecodedChunk) -> DecodedChunk: ...


@runtime_checkable
class BatchTransform(Protocol):
    """Per-batch transform applied after gather (not cached)."""

    def __call__(self, batch: Batch) -> Batch: ...


@dataclass(slots=True)
class StandardScaler:
    """Global per-variable standardization with PRE-FIT, FIXED statistics.

    ``mean``/``std`` are keyed by variable and shaped to broadcast over a chunk's
    ``(n_samples, *inner)`` array WITHOUT the sample axis: a surface variable uses
    shape ``(1, 1)``; per-level stats use ``(level, 1, 1)``. The same stats are
    applied to every chunk of that variable -- never recomputed per chunk.

    Pre-fit the stats however you like and pass them in. The recommended path is to
    fit over the loader with ``sklearn``'s incremental ``StandardScaler.partial_fit``
    (which also warms the cache) and scale at the *batch* stage -- see
    ``examples/fit_scaler.py``; this class is the *chunk*-stage applier for when you
    want the normalization cached with the decoded chunk.
    """

    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    eps: float = 1e-8

    def __call__(self, chunk: DecodedChunk) -> DecodedChunk:
        m = self.mean[chunk.read.array]
        s = self.std[chunk.read.array]
        chunk.data = (chunk.data - m) / (s + self.eps)
        return chunk

    def save(self, path: str | Path) -> None:
        flat = {f"{k}.mean": v for k, v in self.mean.items()}
        flat.update({f"{k}.std": v for k, v in self.std.items()})
        np.savez(path, **flat)  # type: ignore[arg-type]  # np stub collides **kwds w/ allow_pickle

    @classmethod
    def load(cls, path: str | Path) -> StandardScaler:
        d = np.load(path)
        mean = {k[:-5]: d[k] for k in d.files if k.endswith(".mean")}
        std = {k[:-4]: d[k] for k in d.files if k.endswith(".std")}
        return cls(mean=mean, std=std)
