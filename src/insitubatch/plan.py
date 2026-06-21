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
