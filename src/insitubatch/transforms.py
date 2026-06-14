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
cross-variable or per-sample-random -> batch stage; cross-chunk -> not v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .io import AsyncChunkReader
from .plan import build_read_plan
from .split import SplitManifest
from .types import ArrayGeometry, Batch, DecodedChunk, SplitName


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
    applied to every chunk of that variable -- never recomputed per chunk. Fit
    once with :func:`fit_standard_scaler`.
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


def _reduce_axes(chunk_ndim: int, keep_inner: tuple[int, ...]) -> tuple[int, ...]:
    """Chunk axes to reduce over: the sample axis (0) + every inner axis not kept.

    ``keep_inner`` indexes the *inner* dims (0-based within ``inner_shape``); the
    corresponding chunk axis is ``1 + i``.
    """
    keep = {1 + i for i in keep_inner}
    return tuple(ax for ax in range(chunk_ndim) if ax not in keep)


def fit_standard_scaler(
    url: str,
    manifest: SplitManifest,
    geometries: dict[str, ArrayGeometry],
    *,
    split: SplitName = SplitName.TRAIN,
    keep_axes: tuple[int, ...] = (),
    **store_kwargs: object,
) -> StandardScaler:
    """Fit a :class:`StandardScaler` over ``split`` -- with our own infra.

    One streaming, bounded-memory pass per variable through the async reader:
    accumulate sum / sumsq / count, reducing over the sample axis (+ spatial),
    keeping the inner dims named in ``keep_axes`` (e.g. ``(0,)`` keeps a leading
    level axis -> per-level stats). ``()`` gives one scalar mean/std per variable.

    Note
    ----
    Uses float64 sum-of-squares for simplicity; for production prefer Welford or a
    shifted mean to avoid catastrophic cancellation on large means.
    """
    sums: dict[str, np.ndarray] = {}
    sqs: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    for var, geom in geometries.items():
        idx = manifest.sample_indices(split, geom)
        plan = build_read_plan(idx, {var: geom})
        with AsyncChunkReader(url, {var: geom}, **store_kwargs) as reader:  # type: ignore[arg-type]
            for chunk in reader.read_plan(plan):
                x = chunk.data.astype(np.float64)
                axes = _reduce_axes(x.ndim, keep_axes)
                sums[var] = sums.get(var, 0.0) + x.sum(axis=axes, keepdims=True)
                sqs[var] = sqs.get(var, 0.0) + (x * x).sum(axis=axes, keepdims=True)
                counts[var] = counts.get(var, 0) + int(np.prod([x.shape[a] for a in axes]))

    mean: dict[str, np.ndarray] = {}
    std: dict[str, np.ndarray] = {}
    for var in geometries:
        n = counts[var]
        m = sums[var] / n
        var_ = np.maximum(sqs[var] / n - m * m, 0.0)
        # Drop the sample axis (size 1) -> stats broadcast over (n_samples, *inner).
        mean[var] = np.squeeze(m, axis=0)
        std[var] = np.squeeze(np.sqrt(var_), axis=0)
    return StandardScaler(mean=mean, std=std)
