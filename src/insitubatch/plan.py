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
    ref_spc: int,
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

    ``chunk_ids`` are *anchor* chunks in the **reference** grid (``ref_spc`` = the
    manifest's sample-chunk size, which defines the shuffle/split anchor grid). A windowed
    variable reads ``array[anchor + offset]``, and a variable that chunks the sample axis
    differently from the reference maps those anchor samples onto *its own* chunks -- so one
    anchor chunk expands to the (offset-shifted) chunks each variable needs. With every
    ``offset == 0`` and a uniform chunk size this is exactly ``anchor chunk -> itself``.
    """
    reads: list[StoredChunkRead] = []
    seen: set[StoredChunkRead] = set()
    for cid in chunk_ids:
        for geom in geometries.values():
            for read_cid in _read_chunks(geom, int(cid), ref_spc):
                for inner in geom.inner_coords():
                    read = StoredChunkRead(geom.path, read_cid, inner)
                    if read not in seen:
                        seen.add(read)
                        reads.append(read)
    return reads


def _read_chunks(geom: ArrayGeometry, anchor_chunk: int, ref_spc: int) -> range:
    """Offset-shifted read chunks an anchor chunk needs for ``geom`` (clamped).

    ``anchor_chunk`` is in the reference grid: its anchor samples are ``[k*ref_spc,
    (k+1)*ref_spc)`` (clamped to the array). Those read array samples ``[start+offset,
    stop-1+offset]``, mapped onto ``geom``'s *own* chunk grid (``geom.sample_chunk_size``).
    A variable coarser than the reference collapses several anchor chunks onto one of its
    chunks; a finer one spans several. Edge anchors that read off the array are dropped from
    the draw upstream, so an empty span here means nothing to fetch.
    """
    start = anchor_chunk * ref_spc
    stop = min(start + ref_spc, geom.n_samples)
    lo = max(0, start + geom.offset)
    hi = min(geom.n_samples - 1, stop - 1 + geom.offset)
    if lo > hi:  # the whole anchor chunk reads off the array (edge); nothing to fetch
        return range(0)
    spc = geom.sample_chunk_size
    return range(lo // spc, hi // spc + 1)
