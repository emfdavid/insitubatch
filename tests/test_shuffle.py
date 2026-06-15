"""Shuffle: determinism, coverage (incl. partial final chunk), quality, in-order."""

from __future__ import annotations

import numpy as np

from insitubatch import (
    block_shuffled_order,
    chunk_permutation,
    sequential_order,
    shuffle_quality,
)


def test_permutation_is_deterministic_per_epoch() -> None:
    ids = np.arange(50)
    a = chunk_permutation(ids, seed=1, epoch=0)
    b = chunk_permutation(ids, seed=1, epoch=0)
    c = chunk_permutation(ids, seed=1, epoch=1)
    assert np.array_equal(a, b)  # same (seed, epoch) -> identical
    assert not np.array_equal(a, c)  # next epoch reshuffles


def test_order_covers_every_sample_exactly_once() -> None:
    ids = np.arange(20)
    spc, n = 8, 20 * 8
    order = block_shuffled_order(ids, spc, n, block_chunks=4, seed=0, epoch=0)
    assert order.shape == (n, 2)
    ranks = order[:, 0] * spc + order[:, 1]
    assert np.array_equal(np.sort(ranks), np.arange(n))


def test_order_handles_partial_final_chunk() -> None:
    # 5 chunks, spc=8, but only 37 samples -> last chunk holds 5, not 8.
    ids = np.arange(5)
    spc, n = 8, 37
    order = block_shuffled_order(ids, spc, n, block_chunks=2, seed=0, epoch=0)
    assert order.shape == (n, 2)
    ranks = order[:, 0] * spc + order[:, 1]
    assert np.array_equal(np.sort(ranks), np.arange(n))  # no out-of-range indices
    last = order[order[:, 0] == 4]  # final chunk contributes exactly 5 samples
    assert sorted(last[:, 1].tolist()) == [0, 1, 2, 3, 4]


def test_sequential_order_is_in_order_and_complete() -> None:
    ids = np.arange(5)
    spc, n = 8, 37
    order = sequential_order(ids, spc, n)
    ranks = order[:, 0] * spc + order[:, 1]
    assert ranks.tolist() == list(range(n))  # strictly in order, no shuffle


def test_bigger_blocks_improve_shuffle_quality() -> None:
    ids = np.arange(64)
    spc, n = 16, 64 * 16
    q_small = shuffle_quality(
        block_shuffled_order(ids, spc, n, block_chunks=1, seed=0, epoch=0), spc
    )
    q_large = shuffle_quality(
        block_shuffled_order(ids, spc, n, block_chunks=32, seed=0, epoch=0), spc
    )
    assert q_large > q_small  # wider block -> closer to global mixing
