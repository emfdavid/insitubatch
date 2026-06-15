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


def block_shuffled_order(
    chunk_ids: np.ndarray,
    samples_per_chunk: int,
    n_samples: int,
    *,
    block_chunks: int,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Produce a shuffle-block-ordered list of ``[chunk_id, within]`` draws.

    Chunks are permuted per epoch; within each window of ``block_chunks`` chunks
    all samples are shuffled together. ``n_samples`` is the global sample-axis
    length, used to size a short final chunk correctly. Returns an array of shape
    ``(N, 2)`` where ``N`` is the number of samples covered by ``chunk_ids``.
    """
    perm = chunk_permutation(chunk_ids, seed=seed, epoch=epoch)
    rng = np.random.default_rng((seed, epoch, 7919))

    rows: list[np.ndarray] = []
    for start in range(0, len(perm), block_chunks):
        block = perm[start : start + block_chunks]
        pairs = np.concatenate(
            [_chunk_rows(int(cid), samples_per_chunk, n_samples) for cid in block], axis=0
        )
        rng.shuffle(pairs)  # in-place, along axis 0
        rows.append(pairs)
    return np.concatenate(rows, axis=0)


def sequential_order(
    chunk_ids: np.ndarray,
    samples_per_chunk: int,
    n_samples: int,
) -> np.ndarray:
    """In-order ``[chunk_id, within]`` draws (no permutation, no shuffle).

    Used when ``shuffle=False`` (eval / inference / reconstruction): chunks in the
    given order, samples in order within each. Honours a short final chunk.
    """
    return np.concatenate(
        [_chunk_rows(int(cid), samples_per_chunk, n_samples) for cid in chunk_ids], axis=0
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
