"""Chunk-aligned splits: no leakage, full coverage, round-trips to disk."""

from __future__ import annotations

import numpy as np
import pytest

from insitubatch import ArrayGeometry, SplitName, split_by_chunk
from insitubatch.split import SplitManifest


def _geom() -> ArrayGeometry:
    return ArrayGeometry("t2m", shape=(1000, 4), chunks=(10, 4), dtype=np.dtype("f4"))


def test_splits_are_disjoint_and_cover_all_chunks() -> None:
    geom = _geom()  # 100 chunks
    m = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))
    train = set(m.chunks[SplitName.TRAIN.value])
    val = set(m.chunks[SplitName.VAL.value])
    test = set(m.chunks[SplitName.TEST.value])
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(range(100))
    assert len(train) == 80 and len(val) == 10 and len(test) == 10


def test_contiguous_blocks_prevent_interleave() -> None:
    geom = _geom()
    m = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1), contiguous=True)
    # contiguous train should be exactly chunks 0..79
    assert m.chunks[SplitName.TRAIN.value] == list(range(80))


def test_sample_indices_expand_within_chunk() -> None:
    geom = _geom()
    m = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))
    idx = m.sample_indices(SplitName.VAL, geom)
    # 10 val chunks * 10 samples each
    assert len(idx) == 100
    assert idx.min() >= 0 and idx.max() < geom.n_samples


def test_sample_range_restricts_to_overlapping_chunks() -> None:
    geom = _geom()  # 100 chunks of 10 samples (n_samples=1000)
    # window [250, 500) is chunk-aligned -> chunks 25..49, split 80/10/10 of those 25
    m = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1), sample_range=(250, 500))
    kept = (
        set(m.chunks[SplitName.TRAIN.value])
        | set(m.chunks[SplitName.VAL.value])
        | set(m.chunks[SplitName.TEST.value])
    )
    assert kept == set(range(25, 50))  # only the windowed chunks
    assert m.chunks[SplitName.TRAIN.value] == list(range(25, 45))  # contiguous, 80% of 25
    assert m.n_chunks == 100  # array metadata unchanged


def test_sample_range_snaps_outward_to_chunk_bounds() -> None:
    geom = _geom()  # chunk size 10
    # a window starting/ending mid-chunk pulls in the partial edge chunks
    m = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0), sample_range=(13, 27))
    assert m.chunks[SplitName.TRAIN.value] == [1, 2]  # chunks covering 13..27 -> 1,2


def test_sample_range_validates_bounds() -> None:
    geom = _geom()
    for bad in [(-1, 10), (10, 10), (10, 5), (0, 1001)]:
        with pytest.raises(ValueError, match="sample_range"):
            split_by_chunk(geom, sample_range=bad)


def test_manifest_json_roundtrip(tmp_path) -> None:
    geom = _geom()
    m = split_by_chunk(geom)
    p = tmp_path / "manifest.json"
    m.to_json(p)
    back = SplitManifest.from_json(p)
    assert back.chunks == m.chunks
    assert back.n_chunks == m.n_chunks
