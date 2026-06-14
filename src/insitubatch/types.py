"""Core data types shared across the insitubatch engine.

The central design choice (see DESIGN.md): the unit of work is neither the
*sample* nor the *chunk* in isolation, but a **read plan** that maps the samples
required for a step to a *deduplicated* set of chunk reads. This lets the same
scheduler serve the whole spectrum from fat chunks (many samples per chunk,
shared-cache wins) to the degenerate GRIB-per-timestep case (one sample per
chunk, async fan-out is everything).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class SplitName(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class ArrayGeometry:
    """The minimal geometry the engine needs about one zarr array.

    We only model the *sample axis* (the outer dimension, axis 0 by convention:
    time for ERA5/HRRR) explicitly, because that is the axis we split, shuffle,
    and batch along. The trailing dims are carried opaquely as ``inner_shape``
    and are kept contiguous to preserve partial zero-copy.
    """

    name: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype

    @property
    def n_samples(self) -> int:
        """Length of the sample (outer) axis."""
        return self.shape[0]

    @property
    def sample_chunk_size(self) -> int:
        """How many samples live in one chunk along the sample axis."""
        return self.chunks[0]

    @property
    def inner_shape(self) -> tuple[int, ...]:
        """Shape of a single sample (everything past the sample axis)."""
        return self.shape[1:]

    @property
    def n_chunks(self) -> int:
        """Number of chunks along the sample axis."""
        return -(-self.n_samples // self.sample_chunk_size)  # ceil div

    def chunk_of(self, sample_index: int) -> int:
        """Which sample-axis chunk a given sample index falls in."""
        return sample_index // self.sample_chunk_size

    def samples_in_chunk(self, chunk_index: int) -> range:
        """The half-open range of global sample indices in ``chunk_index``."""
        start = chunk_index * self.sample_chunk_size
        stop = min(start + self.sample_chunk_size, self.n_samples)
        return range(start, stop)


@dataclass(frozen=True, slots=True)
class ChunkRead:
    """A single chunk to fetch, addressed along the sample axis.

    ``array`` names which zarr array (variable) this read belongs to; a training
    sample that concatenates several variables produces one ``ChunkRead`` per
    variable that must be co-scheduled.
    """

    array: str
    chunk_index: int


@dataclass(slots=True)
class DecodedChunk:
    """A decoded, in-memory chunk, keyed by its read.

    ``data`` has shape ``(n_samples_in_chunk, *inner_shape)``. The buffer holds a
    bounded number of these; memory overhead is O(in-flight chunks), independent
    of batch size.
    """

    read: ChunkRead
    data: np.ndarray
    sample_offset: int  # global sample index of data[0]


@dataclass(slots=True)
class Batch:
    """A model-ready batch.

    ``arrays`` maps variable name -> stacked array of shape ``(batch, *inner)``.
    ``sample_indices`` records provenance (which global samples, in order) for
    determinism checks and resumption.
    """

    arrays: dict[str, np.ndarray]
    sample_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
