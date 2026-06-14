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


def block_shuffled_order(
    chunk_ids: np.ndarray,
    samples_per_chunk: int,
    *,
    block_chunks: int,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Produce a globally-ordered list of (chunk_id, within) draws.

    Emulates the shuffle-block draw order the live buffer will realise, useful
    for the quality harness and for deterministic single-process iteration.
    Returns an array of shape ``(n_samples, 2)`` of ``[chunk_id, within]`` rows.
    """
    perm = chunk_permutation(chunk_ids, seed=seed, epoch=epoch)
    rng = np.random.default_rng((seed, epoch, 7919))

    rows: list[np.ndarray] = []
    for start in range(0, len(perm), block_chunks):
        block = perm[start : start + block_chunks]
        # Materialise every (chunk, within) pair in this block, then shuffle.
        cc = np.repeat(block, samples_per_chunk)
        ww = np.tile(np.arange(samples_per_chunk), len(block))
        pairs = np.stack([cc, ww], axis=1)
        rng.shuffle(pairs)  # in-place, along axis 0
        rows.append(pairs)
    return np.concatenate(rows, axis=0)


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
