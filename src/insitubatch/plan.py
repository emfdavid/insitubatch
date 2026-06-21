"""Read planning: samples -> deduplicated chunk reads.

This is the crux abstraction. Given the samples required for a window of the
epoch and the array geometries, produce the *minimal* set of chunk reads plus a
gather map describing where each sample lives once those chunks are decoded.

Why this matters (DESIGN.md, "the spectrum"):
  - Fat chunks: many samples share one chunk -> dedup collapses N samples to 1
    read; the shared decoded chunk is gathered N times. This is the shared-cache
    win that the classic per-worker DataLoader cannot get.
  - GRIB-per-timestep: one sample per chunk -> no dedup possible, but the plan
    still drives a single wide async fan-out (B samples == B concurrent reads),
    which is exactly where obstore earns its keep.

The Python hot path here is O(reads), never O(samples) once gathered, which is
the constraint David's S3 benchmark imposed (Python per-chunk overhead bounds
throughput; never loop per-sample in Python).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .types import ArrayGeometry, ChunkRead, StoredChunkRead


@dataclass(slots=True)
class Gather:
    """Where one sample lives within a decoded chunk.

    ``read_index`` indexes into ``ReadPlan.reads``; ``within`` is the offset of
    the sample inside that chunk's decoded array along the sample axis.
    """

    read_index: int
    within: int


@dataclass(slots=True)
class ReadPlan:
    """A deduplicated batch of chunk reads plus the gather map back to samples.

    One ``ReadPlan`` typically covers enough samples to (a) saturate the async
    fan-out and (b) fill the shuffle-block buffer. ``reads`` is what the IO
    driver fetches; ``gathers[v]`` reconstructs the requested samples for
    variable ``v`` from the decoded chunks.
    """

    reads: list[ChunkRead]
    gathers: dict[str, list[Gather]]
    sample_indices: np.ndarray  # global sample indices, in requested order

    @property
    def n_reads(self) -> int:
        return len(self.reads)


def build_read_plan(
    sample_indices: Sequence[int] | np.ndarray,
    geometries: dict[str, ArrayGeometry],
) -> ReadPlan:
    """Build a deduplicated read plan for ``sample_indices`` across all variables.

    All variables are assumed aligned on the sample axis (same length, possibly
    different chunking) -- the common case for co-registered NWP variables. A
    sample at global index ``s`` requires chunk ``geom.chunk_of(s)`` from *each*
    variable; identical chunks requested by multiple samples are read once.

    Parameters
    ----------
    sample_indices:
        Global sample-axis indices to fetch, in the order they should appear.
    geometries:
        Variable name -> :class:`ArrayGeometry`.

    Returns
    -------
    ReadPlan
    """
    idx = np.asarray(sample_indices, dtype=np.int64)
    reads: list[ChunkRead] = []
    read_lookup: dict[ChunkRead, int] = {}
    gathers: dict[str, list[Gather]] = {name: [] for name in geometries}

    for name, geom in geometries.items():
        # Vectorized chunk assignment for this variable across all samples.
        chunk_ids = idx // geom.sample_chunk_size
        within = idx - chunk_ids * geom.sample_chunk_size
        for c, w in zip(chunk_ids.tolist(), within.tolist(), strict=True):
            read = ChunkRead(array=name, chunk_index=int(c))
            ri = read_lookup.get(read)
            if ri is None:
                ri = len(reads)
                read_lookup[read] = ri
                reads.append(read)
            gathers[name].append(Gather(read_index=ri, within=int(w)))

    return ReadPlan(reads=reads, gathers=gathers, sample_indices=idx)


def build_stored_chunk_reads(
    chunk_ids: Sequence[int] | np.ndarray,
    geometries: dict[str, ArrayGeometry],
) -> list[StoredChunkRead]:
    """Expand outer chunk ids into deduped stored-chunk reads, in priority order.

    Where :func:`build_read_plan` plans *outer-chunk* reads plus a sample gather
    map (for the streaming reader), this plans *stored-chunk* (tile) reads with no
    gather map: the scheduler scatters tiles into per-outer-chunk slots in a
    :class:`~insitubatch.pool.ChunkPool`, and batches are gathered straight from
    those assembled slots by ``(chunk_id, within)`` draw rows -- the same
    coordinates the shuffle order already produces. So the result is just *what to
    fetch, in what order*; the scheduler keeps ``max_inflight`` tiles in flight
    across the list.

    ``chunk_ids`` are outer (sample-axis) chunk indices in *draw/priority* order
    (e.g. the next shuffle-block's chunks first), so the soonest-needed tiles go
    first. Each outer chunk expands to its inner grid; every variable contributes
    its own grid (variables may chunk the inner dims differently). Order is
    ``chunk -> variable -> inner`` so a whole outer chunk's tiles are scheduled
    together (it can be assembled and drained promptly). Dedup is
    belt-and-suspenders -- a draw order visits each outer chunk once -- but makes
    the function safe to call with repeated ids.
    """
    reads: list[StoredChunkRead] = []
    seen: set[StoredChunkRead] = set()
    for cid in chunk_ids:
        for name, geom in geometries.items():
            for inner in geom.inner_coords():
                read = StoredChunkRead(array=name, chunk_index=int(cid), inner_coord=inner)
                if read not in seen:
                    seen.add(read)
                    reads.append(read)
    return reads


def dedup_ratio(plan: ReadPlan) -> float:
    """Samples-per-read averaged over variables.

    1.0 == degenerate (GRIB-per-timestep, no sharing); higher == fatter chunks
    with more cache reuse. A quick lever for understanding which regime a dataset
    + batch size lands in.
    """
    n_samples = len(plan.sample_indices) * max(len(plan.gathers), 1)
    return n_samples / plan.n_reads if plan.n_reads else 0.0
