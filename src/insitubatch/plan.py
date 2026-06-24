"""Stored-chunk read planning for the scheduler.

Expand the outer (sample-axis) chunks a draw order needs into the deduplicated set
of *stored chunks* (tiles) the scheduler fetches. Python touches O(reads), never
O(samples) -- the constraint David's S3 benchmark imposed (per-chunk overhead
bounds throughput; never loop per-sample in Python). The scheduler scatters each
decoded tile into its outer-chunk slot in the :class:`~insitubatch.pool.ChunkPool`;
batches gather straight from those slots by ``(chunk_id, within)`` draw rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import ArrayGeometry, StoredChunkRead


def build_stored_chunk_reads(
    chunk_ids: Sequence[int] | np.ndarray,
    geometries: dict[str, ArrayGeometry],
) -> list[StoredChunkRead]:
    """Expand outer chunk ids into deduped stored-chunk reads, in priority order.

    There is no gather map: the scheduler scatters tiles into per-outer-chunk slots
    in a :class:`~insitubatch.pool.ChunkPool`, and batches are gathered straight
    from those assembled slots by ``(chunk_id, within)`` draw rows -- the same
    coordinates the shuffle order already produces. So the result is just *what to
    fetch, in what order*; the scheduler keeps ``max_inflight`` tiles in flight
    across the list.

    ``chunk_ids`` are outer (sample-axis) chunk indices in *draw/priority* order
    (e.g. the next shuffle-block's chunks first), so the soonest-needed tiles go
    first. Each outer chunk expands to its inner grid; every variable contributes
    its own grid (variables may chunk the inner dims differently). Order is
    ``chunk -> variable -> inner`` so a whole outer chunk's tiles are scheduled
    together (it can be assembled and drained promptly). Reads are keyed by the
    array *path* (not the dict label), so several windowed views of one array
    (same path, different ``offset``) collapse to a single fetch -- decode-once.
    Dedup also makes the function safe to call with repeated ids.

    ``chunk_ids`` are *anchor* chunks. A windowed variable reads ``array[anchor +
    offset]``, so its anchor chunk's samples map to one or two offset-shifted *read*
    chunks; each is expanded here (clamped to the array, since edge anchors that would
    read off the array are dropped from the draw upstream). With every ``offset == 0``
    this is exactly ``anchor chunk -> itself``.
    """
    reads: list[StoredChunkRead] = []
    seen: set[StoredChunkRead] = set()
    for cid in chunk_ids:
        for geom in geometries.values():
            for read_cid in _read_chunks(geom, int(cid)):
                for inner in geom.inner_coords():
                    read = StoredChunkRead(geom.path, read_cid, inner)
                    if read not in seen:
                        seen.add(read)
                        reads.append(read)
    return reads


def _read_chunks(geom: ArrayGeometry, anchor_chunk: int) -> range:
    """Offset-shifted read chunks an anchor chunk needs for ``geom`` (1-2, clamped).

    The anchor samples ``[start, stop)`` of ``anchor_chunk`` read array samples
    ``[start+offset, stop-1+offset]``; clamp to ``[0, n_samples)`` (edge anchors are
    dropped from the draw) and return the half-open range of chunks they span.
    """
    spc = geom.sample_chunk_size
    anchors = geom.samples_in_chunk(anchor_chunk)
    lo = max(0, anchors.start + geom.offset)
    hi = min(geom.n_samples - 1, anchors.stop - 1 + geom.offset)
    if lo > hi:  # the whole anchor chunk reads off the array (edge); nothing to fetch
        return range(0)
    return range(lo // spc, hi // spc + 1)
