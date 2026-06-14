"""Chunk-aligned splits: no leakage, full coverage, round-trips to disk."""

from __future__ import annotations

import numpy as np

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


def test_manifest_json_roundtrip(tmp_path) -> None:
    geom = _geom()
    m = split_by_chunk(geom)
    p = tmp_path / "manifest.json"
    m.to_json(p)
    back = SplitManifest.from_json(p)
    assert back.chunks == m.chunks
    assert back.n_chunks == m.n_chunks
