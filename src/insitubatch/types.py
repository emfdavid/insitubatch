"""Core data types shared across the insitubatch engine.

The central design choice (see DESIGN.md): the unit of work is neither the
*sample* nor the *chunk* in isolation, but a **read plan** that maps the samples
required for a step to a *deduplicated* set of chunk reads. This lets the same
scheduler serve the whole spectrum from fat chunks (many samples per chunk,
shared-cache wins) to the degenerate GRIB-per-timestep case (one sample per
chunk, async fan-out is everything).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np


class SplitName(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class ArrayGeometry:
    """The minimal geometry the engine needs about one zarr array.

    We only model the *sample axis* explicitly, because that is the axis we split,
    shuffle, and batch along; the remaining dims are carried opaquely as
    ``inner_shape`` and kept contiguous to preserve partial zero-copy. ``shape`` and
    ``chunks`` are in **physical** (zarr) axis order -- they mirror the array's own
    metadata -- and ``sample_axis`` names which physical axis is the sample axis
    (``0`` by convention: time for ERA5/HRRR; e.g. ``2`` for the Z of an OME-NGFF
    ``(T,C,Z,Y,X)`` microscopy stack sampled slice-by-slice). The engine works in a
    *logical* view where the sample axis leads and the inner axes follow in physical
    order; the one physical<->logical permutation is confined to the scheduler
    (:meth:`physical_chunk_coord` for read addressing; a ``moveaxis`` on the decoded
    tile). Everything downstream -- planning, pooling, gather -- is sample-first.

    ``offset`` makes a variable a *windowed view*: it reads ``array[anchor + offset]``
    along the sample axis around a shared anchor. Two geometries with the same ``path``
    and different ``offset`` (e.g. ``g`` and ``g.shift(1)``) are two views of one array --
    they decode once and share slots. Offset 0 is not special; everything is relative to
    the anchor.
    """

    path: str  # the array's zarr path within the store, e.g. "t2m" or "surface/hourly/t2m"
    shape: tuple[int, ...]  # physical (zarr) axis order
    chunks: tuple[int, ...]  # physical (zarr) axis order
    dtype: np.dtype
    offset: int = 0  # sample-axis read shift: this view reads array[anchor + offset]
    sample_axis: int = 0  # which *physical* axis is the sample (outer) axis

    def __post_init__(self) -> None:
        if not 0 <= self.sample_axis < len(self.shape):
            raise ValueError(
                f"sample_axis {self.sample_axis} out of range for {len(self.shape)}-D "
                f"array {self.path!r} with shape {self.shape}"
            )

    def shift(self, k: int) -> ArrayGeometry:
        """A view of the same array read ``k`` samples later (composes: ``shift(1).shift(1)``
        is ``offset += 2``). Declare a forecast target as ``g.shift(horizon)``."""
        return replace(self, offset=self.offset + k)

    def _drop_sample(self, dims: tuple[int, ...]) -> tuple[int, ...]:
        """``dims`` (a physical shape/chunks tuple) with the sample axis removed --
        the inner axes in physical order, which is the engine's logical inner order."""
        return dims[: self.sample_axis] + dims[self.sample_axis + 1 :]

    def physical_chunk_coord(
        self, chunk_index: int, inner_coord: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Full physical zarr stored-chunk coordinate for a logical read.

        Reinsert the sample-axis chunk index ``chunk_index`` at ``sample_axis`` among the
        inner-axis chunk coords (which are in physical inner order). With ``sample_axis == 0``
        this is exactly ``(chunk_index, *inner_coord)`` -- the identity the old code assumed."""
        return inner_coord[: self.sample_axis] + (chunk_index,) + inner_coord[self.sample_axis :]

    @property
    def n_samples(self) -> int:
        """Length of the sample (outer) axis."""
        return self.shape[self.sample_axis]

    @property
    def sample_chunk_size(self) -> int:
        """How many samples live in one chunk along the sample axis."""
        return self.chunks[self.sample_axis]

    @property
    def inner_shape(self) -> tuple[int, ...]:
        """Shape of a single sample (every axis but the sample axis, physical order)."""
        return self._drop_sample(self.shape)

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

    # -- stored-chunk geometry ----------------------------------------------
    # An outer (sample-axis) chunk is exactly one stored chunk along axis 0, but
    # the inner dims may be gridded (the ARCO/ERA5 norm). The fetch scheduler
    # works at *stored chunk* granularity -- (chunk_index, *inner_coord) -- so one
    # in-flight budget spans inner and outer reads (no nested concurrency caps).
    # It therefore needs the inner grid and, per tile, where it lands in the chunk.

    @property
    def inner_chunks(self) -> tuple[int, ...]:
        """Stored-chunk shape on the inner (non-sample) axes (physical order)."""
        return self._drop_sample(self.chunks)

    def inner_grid(self) -> tuple[range, ...]:
        """Per-inner-axis range of stored-chunk indices (ceil div of shape/chunk)."""
        return tuple(
            range(-(-s // c)) for s, c in zip(self.inner_shape, self.inner_chunks, strict=True)
        )

    def inner_coords(self) -> Iterator[tuple[int, ...]]:
        """Every inner stored-chunk coordinate, row-major over the inner grid."""
        return itertools.product(*self.inner_grid())

    def n_inner_chunks(self, chunk_index: int) -> int:
        """How many stored tiles compose one outer chunk (the inner-grid size).

        Independent of ``chunk_index`` (a short final *outer* chunk is still one
        axis-0 stored chunk), but kept index-keyed so the pool's completion count
        reads naturally and the API survives a future per-axis sample chunking.
        """
        n = 1
        for r in self.inner_grid():
            n *= len(r)
        return n

    def slot_shape(self, chunk_index: int) -> tuple[int, ...]:
        """Shape of the assembled outer chunk: ``(n_samples_in_chunk, *inner_shape)``.

        Axis 0 uses the *actual* sample count so the final short chunk is sized
        exactly (no over-allocation, no out-of-range scatter).
        """
        return (len(self.samples_in_chunk(chunk_index)), *self.inner_shape)

    def tile_placement(
        self, chunk_index: int, inner_coord: tuple[int, ...]
    ) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        """``(dst, src)`` slices for scattering one decoded tile into its slot.

        ``dst`` indexes the outer-chunk slot; ``src`` clips the full chunk-shaped
        decoded tile to the (possibly partial) edge region -- both axis 0 (short
        final outer chunk) and the inner edges. After the copy the tile is free.
        """
        n0 = len(self.samples_in_chunk(chunk_index))
        dst = [slice(0, n0)]
        for i, c, s in zip(inner_coord, self.inner_chunks, self.inner_shape, strict=True):
            start = i * c
            dst.append(slice(start, min(start + c, s)))
        src = tuple(slice(0, sl.stop - sl.start) for sl in dst)
        return tuple(dst), src


@dataclass(frozen=True, slots=True)
class ChunkRead:
    """A single chunk to fetch, addressed along the sample axis.

    ``array`` names which zarr array (variable) this read belongs to; a training
    sample that concatenates several variables produces one ``ChunkRead`` per
    variable that must be co-scheduled.
    """

    array: str
    chunk_index: int


@dataclass(frozen=True, slots=True)
class StoredChunkRead:
    """One *stored* chunk to fetch: a single tile of the chunk grid.

    Reading a whole outer chunk per ``getitem`` lets zarr stitch the inner grid
    under a *second* concurrency cap; fetching at stored-chunk granularity instead
    -- ``(chunk_index, *inner_coord)`` -- lets a single ``max_inflight`` budget
    span inner *and* outer reads, with no nested caps. ``chunk_index`` is the
    sample-axis (outer) stored-chunk index; ``inner_coord`` is the stored-chunk
    index on each inner axis (empty tuple when the inner dims are single-chunk --
    the degenerate GRIB-per-timestep case).

    Frozen + hashable so a plan can dedup tiles and key the in-flight set.
    """

    array: str
    chunk_index: int
    inner_coord: tuple[int, ...]

    @property
    def coords(self) -> tuple[int, ...]:
        """Full zarr stored-chunk coordinate (axis 0 is the sample axis)."""
        return (self.chunk_index, *self.inner_coord)


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

    ``arrays`` maps variable *label* -> stacked array of shape ``(batch, *inner)``.
    ``sample_indices`` is the per-row *anchor* sample index ``t`` (provenance for
    determinism / resumption). ``offsets`` maps each label to its sample-axis read
    offset, so label ``v`` of row ``i`` was read from global sample ``sample_indices[i]
    + offsets[v]``. A plain (non-windowed) batch has every offset 0; a forecast batch
    pairs e.g. an input at offset 0 with a target at offset ``horizon``.

    The batch stays a flat ``{label: array}`` dict -- there is no lead/role axis in the
    engine. Use :meth:`stack` to assemble a multi-step window into one array and
    :meth:`read_indices` for a label's true provenance.
    """

    arrays: dict[str, np.ndarray]
    sample_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    offsets: dict[str, int] = field(default_factory=dict)  # label -> sample-axis read offset

    def read_indices(self, label: str) -> np.ndarray:
        """Global sample index each row of ``label`` was read from: ``anchor + offset``.

        Provenance for a windowed view (e.g. to confirm a target leads its input by the
        intended horizon). Defaults the offset to 0 for a label without one recorded.
        """
        return self.sample_indices + self.offsets.get(label, 0)

    def stack(self, labels: Sequence[str], axis: int = 1) -> np.ndarray:
        """Stack several labels into one array along a new ``axis`` (default 1).

        The obvious way to build a multi-step input window from a set of time-shifted
        views, e.g. ``batch.stack(["t_m2", "t_m1", "t_0"])`` -> ``(batch, 3, *inner)``.
        Order follows ``labels``; row ``i`` of every label shares anchor
        ``sample_indices[i]``, so the stacked steps stay aligned. The caller chooses the
        labels and their order -- the engine does not impose a window layout.
        """
        return np.stack([self.arrays[label] for label in labels], axis=axis)
