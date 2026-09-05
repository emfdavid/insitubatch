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

**Scope is declared at the call site**, with :func:`applies` -- never by a name test inside
the transform's body. "Subtract 273.15" is a fact about Kelvin; "``2m_temperature`` is in
Kelvin" is a fact about *your store*, and only the call site knows it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .types import ArrayGeometry, Batch, DecodedChunk


@runtime_checkable
class ChunkTransform(Protocol):
    """Per-chunk transform applied before shuffle/gather (cacheable)."""

    def __call__(self, chunk: DecodedChunk, /) -> DecodedChunk: ...


@runtime_checkable
class ReshapingChunkTransform(ChunkTransform, Protocol):
    """A chunk_transform that changes a chunk's *inner* geometry (shape and/or dtype).

    The canonical case is regrid; a dtype recast (e.g. decode ``f2`` storage, cache ``f4``)
    qualifies too. Declaring the output lets the engine size the cache slot and the gather
    buffer at the post-transform geometry, so a reshaping transform is a first-class,
    cacheable chunk stage rather than a heap-only special case.

    ``output_inner`` returns only the **inner** geometry (everything past the sample axis):
    a chunk_transform must never cross or reshape axis 0 (the sample-geometry invariant), so
    it has no business naming axis 0, and the engine splices the source's sample axis back on.
    Shape/dtype-preserving transforms (e.g. :class:`StandardScaler`) simply do **not** define
    ``output_inner`` -- the engine treats their output geometry as identical to the source.
    """

    def output_inner(self, geom: ArrayGeometry) -> tuple[tuple[int, ...], np.dtype]:
        """``(inner_shape, dtype)`` of this transform's output, given its input ``geom``.

        Composes across a pipeline: the engine feeds each transform the geometry produced
        by the ones before it. The sample axis (``geom.shape[0]``) is unchanged."""
        ...


@runtime_checkable
class BatchTransform(Protocol):
    """Per-batch transform applied after gather (not cached)."""

    def __call__(self, batch: Batch, /) -> Batch: ...


# -- scope: which arrays a chunk_transform applies to ------------------------


@dataclass(frozen=True, slots=True)
class _Scoped:
    """One chunk_transform, restricted to a set of array paths. Built by :func:`applies`.

    The engine reads ``arrays`` directly: it skips the call for an array outside the scope,
    and -- the part a name test in ``__call__`` cannot provide -- it excludes the transform
    from those arrays' *declared output geometry* and from their *cache fingerprint*.
    """

    arrays: frozenset[str]
    transform: ChunkTransform

    def __call__(self, chunk: DecodedChunk) -> DecodedChunk:
        return self.transform(chunk)


@dataclass(frozen=True, slots=True)
class _ScopedReshaping(_Scoped):
    """A scoped :class:`ReshapingChunkTransform`.

    ``output_inner`` exists as a *separate class* rather than a delegating method on
    :class:`_Scoped`, because the engine detects a reshaping transform by
    ``hasattr(t, "output_inner")``: a wrapper that always defined it would make every
    shape-preserving transform claim to reshape.
    """

    def output_inner(self, geom: ArrayGeometry) -> tuple[tuple[int, ...], np.dtype]:
        return self.transform.output_inner(geom)  # type: ignore[attr-defined]


def applies(arrays: Sequence[str], transform: ChunkTransform) -> ChunkTransform:
    """Restrict a ``chunk_transform`` to the named arrays; every other array passes through.

    ``arrays`` are zarr array **paths** (what ``open_geometries`` keys on, and what
    ``chunk.read.array`` carries) -- not dict labels, since two labels may alias one array
    (``t2m_now`` / ``t2m_next``). Unknown names raise at dataset construction::

        InSituDataset(..., chunk_transforms=[
            applies(["2m_temperature"], kelvin_to_celsius),
            Coarsen(2),                    # bare = every array
        ])

    Declaring scope here rather than with an ``if chunk.read.array == ...`` inside the
    transform is not a style preference: an in-body gate is invisible to the engine, which
    folds a reshaping transform's ``output_inner`` into *every* array's declared geometry.
    The unaffected arrays are then gathered as truncated prefixes of themselves and can
    never revive from cache -- with no exception raised. Scope also enters the cache
    fingerprint per array, so editing one variable's transform leaves the rest valid.

    Scope is matched against the zarr array **path**. Two labels that alias one array
    (``t2m_now`` / ``t2m_next``) therefore share one chunk pipeline; per-label work belongs in
    a ``batch_transform``.

    A transform may be *parameterized* by variable (a fitted scaler's statistics
    legitimately are) but must not decide whether it runs. If it can validate a declared
    scope -- :class:`StandardScaler` checks the names against its own stats dict -- it
    defines ``validate_scope(arrays)``, which this calls at construction rather than leaving
    it to fail in a decode thread.
    """
    if isinstance(arrays, str):
        raise TypeError(
            f"applies() takes a sequence of array names, not the bare string {arrays!r} -- "
            f'a string would scope to its characters. Write applies(["{arrays}"], ...).'
        )
    scope = frozenset(arrays)
    if not scope:
        raise ValueError(
            "applies() was given an empty scope, so the transform would never run. Drop it "
            "from chunk_transforms instead, or name the arrays it applies to."
        )
    check = getattr(transform, "validate_scope", None)
    if check is not None:
        check(scope)
    if hasattr(transform, "output_inner"):
        return _ScopedReshaping(scope, transform)
    return _Scoped(scope, transform)


def transform_scope(transform: ChunkTransform) -> frozenset[str] | None:
    """The array paths ``transform`` is restricted to, or ``None`` for "every array"."""
    return transform.arrays if isinstance(transform, _Scoped) else None


def unwrap_transform(transform: ChunkTransform) -> ChunkTransform:
    """The user's own callable, with any :func:`applies` wrapper removed.

    The cache fingerprint hashes *this*, never the wrapper: scope already selects which
    transforms enter an array's hash, so folding it into the token as well would invalidate
    the variable that **stayed** in scope every time another one left it."""
    while isinstance(transform, _Scoped):
        transform = transform.transform
    return transform


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

    **Give a re-fitted scaler a new ``cache_key`` if you persist the cache without
    cloudpickle.** The fallback fingerprint hashes this class's source plus its ``repr``, and
    numpy summarizes an array over 1000 elements in ``repr`` -- so per-gridpoint statistics
    (say ``(721, 1440)``) repr identically whatever their values, and a re-fit reopens a
    persisted cache as a *hit*, serving chunks normalized with the old numbers. Any of three
    closes the hole: install ``insitubatch[cache]`` (cloudpickle hashes the values), pass
    ``cache_key="<stats version>"`` and bump it on every fit, or keep stats small enough to
    repr in full.
    """

    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    eps: float = 1e-8
    cache_key: str | None = None
    """The identity this scaler declares to the cache fingerprint, and the only one that is
    exact: the fallback hashes ``repr``, which numpy summarizes for large statistics. ``None``
    (the default) leaves identity to the fingerprint's own resolution."""

    def __call__(self, chunk: DecodedChunk) -> DecodedChunk:
        m = self.mean[chunk.read.array]
        s = self.std[chunk.read.array]
        chunk.data = (chunk.data - m) / (s + self.eps)
        return chunk

    def validate_scope(self, arrays: frozenset[str]) -> None:
        """Reject a scope this scaler has no statistics for (called by :func:`applies`).

        ``applies(["u10"], scaler_fitted_on_t2m)`` would otherwise ``KeyError`` inside a
        decode thread, one chunk into training. The stats dict may be *wider* than the
        scope -- one fitted scaler shared by several scoped uses is normal."""
        missing = sorted(arrays - (self.mean.keys() | self.std.keys()))
        if missing:
            raise ValueError(
                f"StandardScaler has no mean/std for {missing}; it was fitted for "
                f"{sorted(self.mean.keys() & self.std.keys())}. Fit those variables, or "
                "scope this scaler to the ones it knows."
            )

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
