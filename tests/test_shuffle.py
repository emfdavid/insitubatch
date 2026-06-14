"""Shuffle: determinism, coverage, and the quality<->block-size relationship."""

from __future__ import annotations

import numpy as np

from insitubatch import block_shuffled_order, chunk_permutation, shuffle_quality


def test_permutation_is_deterministic_per_epoch() -> None:
    ids = np.arange(50)
    a = chunk_permutation(ids, seed=1, epoch=0)
    b = chunk_permutation(ids, seed=1, epoch=0)
    c = chunk_permutation(ids, seed=1, epoch=1)
    assert np.array_equal(a, b)  # same (seed, epoch) -> identical
    assert not np.array_equal(a, c)  # next epoch reshuffles


def test_order_covers_every_sample_exactly_once() -> None:
    ids = np.arange(20)
    spc = 8
    order = block_shuffled_order(ids, spc, block_chunks=4, seed=0, epoch=0)
    assert order.shape == (20 * spc, 2)
    ranks = order[:, 0] * spc + order[:, 1]
    assert np.array_equal(np.sort(ranks), np.arange(20 * spc))


def test_bigger_blocks_improve_shuffle_quality() -> None:
    ids = np.arange(64)
    spc = 16
    q_small = shuffle_quality(block_shuffled_order(ids, spc, block_chunks=1, seed=0, epoch=0), spc)
    q_large = shuffle_quality(block_shuffled_order(ids, spc, block_chunks=32, seed=0, epoch=0), spc)
    assert q_large > q_small  # wider block -> closer to global mixing
