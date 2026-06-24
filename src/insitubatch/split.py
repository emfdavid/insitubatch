"""Chunk-aligned train/val/test splits.

Splits are done *ahead of time* and at *chunk granularity* along the sample
axis. Two reasons (DESIGN.md, "splits"):

  1. Leakage: splitting mid-chunk would scatter temporally adjacent, highly
     autocorrelated samples across train and val. Chunk-aligned boundaries keep
     a contiguous block of time in a single split.
  2. Zero-copy: a split that respects chunk boundaries means every read serves
     exactly one split, so the engine never decodes a chunk and throws half of
     it away.

The manifest is a plain, serializable record of which chunk indices belong to
which split, so a run is reproducible and shareable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .types import ArrayGeometry, SplitName


def valid_anchor_range(offsets: Iterable[int], n_samples: int) -> tuple[int, int]:
    """Half-open ``[lo, hi)`` of anchor sample-indices whose *every* windowed read
    ``anchor + offset`` stays in ``[0, n_samples)`` -- the anchors a windowed dataset
    may draw, with array-edge anchors dropped.

    Offsets ``{-1, 0, 1}`` over ``T`` samples -> anchors ``[1, T-1)``. Range is the
    *only* validity the engine enforces; whether the user's offset choices define a
    meaningful (non-leaky) task is theirs to decide (DESIGN, M-W). Empty/too-wide
    windows return an empty range ``(lo, lo)``.
    """
    offs = list(offsets)
    if not offs:
        return (0, n_samples)
    lo = max(0, -min(offs))
    hi = n_samples - max(0, max(offs))
    return (lo, max(lo, hi))


@dataclass(slots=True)
class SplitManifest:
    """Which sample-axis chunk indices belong to each split."""

    n_chunks: int
    sample_chunk_size: int
    n_samples: int
    chunks: dict[str, list[int]]  # SplitName.value -> sorted chunk indices
    seed: int

    def sample_indices(self, split: SplitName, geom: ArrayGeometry) -> np.ndarray:
        """Expand a split's chunks into the global sample indices they contain."""
        out: list[int] = []
        for c in self.chunks[split.value]:
            out.extend(geom.samples_in_chunk(c))
        return np.asarray(out, dtype=np.int64)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> SplitManifest:
        return cls(**json.loads(Path(path).read_text()))


def split_by_chunk(
    geom: ArrayGeometry,
    *,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    contiguous: bool = True,
    sample_range: tuple[int, int] | None = None,
) -> SplitManifest:
    """Partition a variable's sample-axis chunks into train/val/test.

    Parameters
    ----------
    fractions:
        (train, val, test) fractions of *chunks* (not samples). Must sum to ~1.
    contiguous:
        If True (default), assign contiguous blocks of chunks to each split --
        the safest choice for time series, where a randomly interleaved split
        still risks leakage through autocorrelation across chunk boundaries. If
        False, chunks are shuffled before partitioning (acceptable when samples
        are exchangeable, e.g. independent scenes).
    sample_range:
        Optional half-open ``(start, stop)`` window of sample (outer-axis) indices
        to restrict the split to *before* partitioning -- e.g. train on one date
        range of a long archive. **The selection is chunk-aligned and contiguous:**
        every chunk that *overlaps* ``[start, stop)`` is kept whole, so a window
        starting or ending mid-chunk pulls in that partial edge chunk (splits are
        chunk-granular -- you subset whole chunks, never individual samples). Use it
        for a single contiguous window; it is **not** a tool for scattered/boolean
        selections (those would drag in straddling chunks and silently add samples).
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {fractions} -> {sum(fractions)}")

    if sample_range is None:
        order = np.arange(geom.n_chunks)
    else:
        start, stop = sample_range
        if not 0 <= start < stop <= geom.n_samples:
            raise ValueError(
                f"sample_range must satisfy 0 <= start < stop <= {geom.n_samples}, "
                f"got {sample_range}"
            )
        spc = geom.sample_chunk_size
        order = np.arange(start // spc, -(-stop // spc))  # chunks overlapping [start, stop)

    if not contiguous:
        order = np.random.default_rng(seed).permutation(order)

    n_sel = len(order)
    n_train = int(round(fractions[0] * n_sel))
    n_val = int(round(fractions[1] * n_sel))
    train = sorted(order[:n_train].tolist())
    val = sorted(order[n_train : n_train + n_val].tolist())
    test = sorted(order[n_train + n_val :].tolist())

    return SplitManifest(
        n_chunks=geom.n_chunks,  # the array's chunk count; the splits may be a subset
        sample_chunk_size=geom.sample_chunk_size,
        n_samples=geom.n_samples,
        chunks={
            SplitName.TRAIN.value: train,
            SplitName.VAL.value: val,
            SplitName.TEST.value: test,
        },
        seed=seed,
    )
