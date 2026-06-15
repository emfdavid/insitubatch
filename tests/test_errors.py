"""Error propagation: a failing chunk transform must raise, not deadlock.

Guards the two exception bridges — AsyncChunkReader.read_plan's done-callback and
the prefetch producer thread — both of which forward exceptions to the consumer.
If either regressed into a hang, these tests would hang (a real bug to surface).
"""

from __future__ import annotations

import pytest

from insitubatch import open_geometries, split_by_chunk
from insitubatch.io import AsyncChunkReader
from insitubatch.plan import build_read_plan
from insitubatch.source import InSituDataset
from insitubatch.types import DecodedChunk


def _boom(chunk: DecodedChunk) -> DecodedChunk:
    raise ValueError("boom in transform")


def test_error_propagates_through_reader(write_zarr) -> None:
    url, _ = write_zarr(n=40, spc=8)
    geom = open_geometries(url)["t2m"]
    plan = build_read_plan([0], {"t2m": geom})
    with (
        AsyncChunkReader(url, {"t2m": geom}, chunk_transforms=[_boom]) as reader,
        pytest.raises(ValueError, match="boom"),
    ):
        list(reader.read_plan(plan))


def test_error_propagates_through_prefetch(write_zarr) -> None:
    url, _ = write_zarr(n=40, spc=8)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(url, manifest, to_tensor=False, chunk_transforms=[_boom])
    ds.set_epoch(0)
    with pytest.raises(ValueError, match="boom"):
        list(ds)
