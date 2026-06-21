"""End-to-end: real async zarr read through obstore LocalStore.

Writes a zarr with known values to a temp ``file://`` store and checks that
InSituDataset returns (a) the correct values at the right sample indices and
(b) exactly the train split's samples, once each.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from insitubatch import (
    SplitName,
    ensure_local_dir,
    open_geometries,
    split_by_chunk,
    store_from_url,
)
from insitubatch.io import AsyncChunkReader, IOConfig
from insitubatch.plan import build_read_plan
from insitubatch.source import InSituDataset


def _write_known_zarr(tmp_path, *, n=80, spc=8, inner=(4, 4)) -> tuple[str, np.ndarray]:
    url = f"file://{tmp_path}/d.zarr"
    ensure_local_dir(url)
    store = store_from_url(url, read_only=False)
    group = zarr.open_group(store=store, mode="w")
    arr = group.create_array("t2m", shape=(n, *inner), chunks=(spc, *inner), dtype="f4")
    src = np.arange(n * int(np.prod(inner)), dtype="f4").reshape(n, *inner)
    arr[:] = src
    return url, src


def test_async_reader_returns_correct_chunk(tmp_path) -> None:
    url, src = _write_known_zarr(tmp_path)
    geom = open_geometries(url)["t2m"]
    plan = build_read_plan([0, 8, 30], {"t2m": geom})  # chunks 0, 1, 3
    with AsyncChunkReader(url, {"t2m": geom}, IOConfig(max_inflight=4)) as reader:
        decoded = list(reader.read_plan(plan))
    assert len(decoded) == 3
    by_chunk = {d.read.chunk_index: d for d in decoded}
    for c in (0, 1, 3):
        np.testing.assert_array_equal(by_chunk[c].data, src[c * 8 : c * 8 + 8])


def test_reader_close_closes_the_loop(tmp_path) -> None:
    """close() must close the loop, not leave it for GC -- an unclosed loop's
    __del__ re-closes it during finalization and raises on the gone self-pipe fd
    (a noisy unraisable first seen under free-threaded 3.13t)."""
    url, _ = _write_known_zarr(tmp_path)
    geom = open_geometries(url)["t2m"]
    reader = AsyncChunkReader(url, {"t2m": geom})
    reader.close()
    assert reader._loop.is_closed()


def test_insitu_dataset_values_and_coverage(tmp_path) -> None:
    url, src = _write_known_zarr(tmp_path)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))

    ds = InSituDataset(
        url, manifest, split=SplitName.TRAIN, batch_size=10, block_chunks=4, to_tensor=False
    )
    ds.set_epoch(0)

    seen: list[np.ndarray] = []
    for batch in ds:
        idx = batch.sample_indices
        # values must match the source at exactly those indices
        np.testing.assert_array_equal(batch.arrays["t2m"], src[idx])
        seen.append(idx)

    all_seen = np.concatenate(seen)
    train = manifest.sample_indices(SplitName.TRAIN, geom)
    assert sorted(all_seen.tolist()) == sorted(train.tolist())  # full coverage, no dupes


def test_geometry_introspection_matches_metadata(tmp_path) -> None:
    url, _ = _write_known_zarr(tmp_path, n=50, spc=5, inner=(3, 3))
    geom = open_geometries(url)["t2m"]
    assert geom.shape == (50, 3, 3)
    assert geom.sample_chunk_size == 5
    assert geom.n_chunks == 10


def test_open_geometries_rejects_subgroup(tmp_path) -> None:
    # A name resolving to a subgroup (not an array) is bad input: must raise a
    # clear TypeError that survives `python -O`, not a bare/strippable assert.
    url = f"file://{tmp_path}/d.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=store_from_url(url, read_only=False), mode="w")
    group.create_array("t2m", shape=(4, 2), chunks=(2, 2), dtype="f4")
    group.create_group("nested")

    with pytest.raises(TypeError, match="not an array"):
        open_geometries(url, variables=["nested"])
