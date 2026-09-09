"""Approximate-global shuffle for chunk-aligned data.

True global shuffle is incompatible with chunk-aligned, low-copy reads: it would
demand a random chunk per sample. The compromise (DESIGN.md, "shuffle"), adapted
from MosaicML Streaming's shuffle-block algorithms (py1e / py1br), is two-level:

  1. **Chunk permutation** -- shuffle the *order chunks are scheduled* each epoch.
  2. **Shuffle-block buffer** -- hold samples from a window of B chunks and draw
     batches across the whole window, so samples from different chunks interleave.

Setting the block span B >= ~10x the samples-per-chunk yields shuffle quality
close to global, at memory cost O(B chunks). B is the single quality<->memory
knob. This module owns the *index math*; buffer.py owns the residency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def chunk_permutation(chunk_ids: np.ndarray, *, seed: int, epoch: int) -> np.ndarray:
    """Deterministically permute chunk ids for one epoch.

    Determinism is keyed on (seed, epoch) only -- not on world size or worker
    count -- so a run is reproducible and resumable across hardware (the
    "canonical" property from MosaicML).
    """
    rng = np.random.default_rng((seed, epoch))
    return rng.permutation(chunk_ids)


def _chunk_rows(chunk_id: int, samples_per_chunk: int, n_samples: int) -> np.ndarray:
    """``[chunk_id, within]`` rows for one chunk, honouring a short final chunk.

    The last chunk on the sample axis holds ``n_samples - chunk_id*spc`` samples,
    which is < ``spc`` when ``n_samples`` is not a multiple of ``spc``. Emitting
    ``within`` only up to the real length avoids out-of-range sample indices.
    """
    clen = min(samples_per_chunk, n_samples - chunk_id * samples_per_chunk)
    return np.stack([np.full(clen, chunk_id), np.arange(clen)], axis=1)


# The shape every draw order carries: (N, 2) of [chunk_id, within]. Callers index column
# 0, so an empty order has to keep the second axis or it breaks them differently.
_EMPTY_ORDER = np.empty((0, 2), dtype=np.int64)


@dataclass(eq=False, slots=True)
class DrawOrder:
    """One epoch's ``[chunk_id, within]`` draw rows, and the shuffle-blocks holding them.

    The blocks are *laid down* here, by the builder that creates them, and travel with the
    rows. They are not inferred from the rows afterwards, because in general they cannot
    be: once a windowed view's edge anchors are dropped, a two-chunk block that lost a
    chunk is indistinguishable from a one-chunk block, and any reconstruction has to guess
    (#68). Dropping rows shrinks the block that held them; it never re-cuts the blocks.

    ``bounds`` holds ``n_blocks + 1`` row offsets, so block ``i`` is
    ``rows[bounds[i] : bounds[i + 1]]``. Blocks are contiguous, gapless, and cover every
    row.
    """

    rows: np.ndarray
    bounds: np.ndarray

    def __len__(self) -> int:
        return int(len(self.rows))

    @property
    def n_blocks(self) -> int:
        return max(int(len(self.bounds)) - 1, 0)

    def block(self, index: int) -> tuple[int, int]:
        """Half-open row range ``[start, stop)`` of block ``index``."""
        return int(self.bounds[index]), int(self.bounds[index + 1])

    def keep(self, mask: np.ndarray) -> DrawOrder:
        """Drop the rows ``mask`` excludes, narrowing the blocks that held them.

        A block emptied outright is dropped -- it names no chunks and does no work -- but
        the blocks around it keep their identity, which is the whole point of carrying the
        bounds rather than re-deriving them from what survived.
        """
        kept = np.flatnonzero(mask)
        # searchsorted maps an old row offset to the number of surviving rows before it,
        # which is exactly its offset in the filtered array.
        bounds = np.searchsorted(kept, self.bounds)
        nonempty = np.concatenate([[True], np.diff(bounds) > 0])
        return DrawOrder(self.rows[kept], bounds[nonempty])


def _from_blocks(blocks: list[np.ndarray]) -> DrawOrder:
    """Assemble one draw order from per-block row arrays, recording where each begins."""
    if not blocks:
        return DrawOrder(_EMPTY_ORDER.copy(), np.zeros(1, dtype=np.int64))
    bounds = np.concatenate([[0], np.cumsum([len(b) for b in blocks])]).astype(np.int64)
    return DrawOrder(np.concatenate(blocks, axis=0), bounds)


def block_shuffled_order(
    chunk_ids: np.ndarray,
    samples_per_chunk: int,
    n_samples: int,
    *,
    block_chunks: int,
    seed: int,
    epoch: int,
) -> DrawOrder:
    """Produce a shuffle-block-ordered list of ``[chunk_id, within]`` draws.

    Chunks are permuted per epoch; within each window of ``block_chunks`` chunks
    all samples are shuffled together. ``n_samples`` is the global sample-axis
    length, used to size a short final chunk correctly.

    Returns a :class:`DrawOrder`: ``rows`` of shape ``(N, 2)`` covering every sample in
    ``chunk_ids``, and the ``bounds`` of the blocks this function just laid down. The
    caller needs those bounds and cannot recover them from ``rows`` -- see
    :class:`DrawOrder` -- so they are returned rather than left to be inferred.
    """
    if not len(chunk_ids):
        return _from_blocks([])  # an empty split yields nothing; see sequential_order
    perm = chunk_permutation(chunk_ids, seed=seed, epoch=epoch)
    rng = np.random.default_rng((seed, epoch, 7919))

    blocks: list[np.ndarray] = []
    for start in range(0, len(perm), block_chunks):
        block = perm[start : start + block_chunks]
        pairs = np.concatenate(
            [_chunk_rows(int(cid), samples_per_chunk, n_samples) for cid in block], axis=0
        )
        rng.shuffle(pairs)  # in-place, along axis 0
        blocks.append(pairs)
    return _from_blocks(blocks)


def sequential_order(
    chunk_ids: np.ndarray,
    samples_per_chunk: int,
    n_samples: int,
    *,
    block_chunks: int,
) -> DrawOrder:
    """In-order ``[chunk_id, within]`` draws (no permutation, no shuffle).

    Used when ``shuffle=False`` (eval / inference / reconstruction): chunks in the
    given order, samples in order within each. Honours a short final chunk.

    ``block_chunks`` changes no row -- nothing is shuffled -- but an unshuffled pass still
    streams and releases in blocks, so it is grouped the same way. One block definition
    across both orders is what lets the rest of the engine stay ignorant of which it walks.

    No chunks is an empty order of the same shape, not an error: a split can hold nothing
    because it was asked to (``fractions=(1.0, 0.0, 0.0)``) or because a small store
    rounded it to zero, and iterating it should yield nothing the way an empty list does.
    """
    if not len(chunk_ids):
        return _from_blocks([])
    return _from_blocks(
        [
            np.concatenate(
                [
                    _chunk_rows(int(cid), samples_per_chunk, n_samples)
                    for cid in chunk_ids[start : start + block_chunks]
                ],
                axis=0,
            )
            for start in range(0, len(chunk_ids), block_chunks)
        ]
    )


def shuffle_quality(order: np.ndarray, samples_per_chunk: int) -> float:
    """A 0..1 score for how well an emitted order mixes the source.

    Heuristic: the mean absolute *source-rank* gap between consecutive emitted
    samples, normalised by the gap a perfect global shuffle would give. 1.0 ~=
    global; values near 0 mean adjacent samples still come out near each other
    (poor mixing). Cheap to compute, good enough to tune ``block_chunks``.
    """
    source_rank = order[:, 0] * samples_per_chunk + order[:, 1]
    gaps = np.abs(np.diff(source_rank.astype(np.int64)))
    n = len(source_rank)
    # Expected mean gap of a uniform random permutation of 0..n-1 is ~n/3.
    expected = n / 3.0
    return float(min(gaps.mean() / expected, 1.0)) if n > 1 and expected else 0.0
