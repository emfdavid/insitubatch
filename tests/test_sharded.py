"""Sharded zarr-v3 arrays, and what "one stored chunk" means (#56).

On a sharded array zarr reports two shapes and only one of them is the storage unit:
``metadata.chunks`` is the *inner* chunk (read granularity inside a shard), while
``chunk_grid.chunk_shape`` is the shard -- what a chunk key addresses and what ``store.get``
returns. Planning reads or building an ``ArraySpec`` from the former asks for a key holding a
shard and then decodes it as one inner chunk, which fails the shard index's CRC.

WeatherBench2 and ARCO are not sharded, which is why every other test in this suite passes
either way. These fixtures make the two shapes differ on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from insitubatch import InSituDataset, obstore_store, open_geometries, split_by_chunk


def test_geometry_reports_the_shard_not_the_inner_chunk(write_sharded_zarr):
    """The geometry drives the read plan, so it must describe the stored object."""
    url, _ = write_sharded_zarr(n=48, shard=(16, 8, 8), chunks=(16, 4, 4), inner=(8, 8))
    store = obstore_store(url)
    geom = open_geometries(store, variables=["t2m"])["t2m"]

    arr = zarr.open_group(store=store, mode="r")["t2m"]
    assert arr.chunks == (16, 4, 4) and arr.shards == (16, 8, 8), "fixture must be sharded"
    assert geom.chunks == (16, 8, 8), "geometry must describe the shard, not the inner chunk"
    assert geom.sample_chunk_size == 16


def test_sharded_batches_match_the_source(write_sharded_zarr):
    """The bug's visible form: iterating raised a CRC error from the shard index. Assert the
    values, not merely that it does not raise -- a wrong stored-chunk shape could also scatter
    the right bytes into the wrong places."""
    url, srcs = write_sharded_zarr(n=48, shard=(16, 8, 8), chunks=(16, 4, 4), inner=(8, 8))
    store = obstore_store(url)
    geoms = open_geometries(store, variables=["t2m"])
    manifest = split_by_chunk(geoms["t2m"], fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(store, manifest, geometries=geoms, batch_size=4, shuffle=False)

    seen = 0
    for batch in ds.all:
        idx = batch.sample_indices
        np.testing.assert_array_equal(batch.arrays["t2m"], srcs["t2m"][idx])
        seen += len(idx)
    ds.close()
    assert seen == 48


def test_sharded_with_tiled_inner_dims(write_sharded_zarr):
    """The dynamical.org shape: a shard that covers only part of the field, so the engine
    assembles several shards per sample-chunk. Inner tiling and sharding at once is where a
    wrong stored-chunk shape stops being a decode error and becomes silently misplaced data.
    """
    url, srcs = write_sharded_zarr(
        n=32,
        shard=(8, 4, 4),
        chunks=(8, 2, 2),
        inner=(8, 8),  # 2x2 shards per sample-chunk
    )
    store = obstore_store(url)
    geoms = open_geometries(store, variables=["t2m"])
    assert geoms["t2m"].inner_shape == (8, 8), "inner geometry is the whole field"
    manifest = split_by_chunk(geoms["t2m"], fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(store, manifest, geometries=geoms, batch_size=4, shuffle=False)
    for batch in ds.all:
        np.testing.assert_array_equal(batch.arrays["t2m"], srcs["t2m"][batch.sample_indices])
    ds.close()


def test_unsharded_arrays_are_unchanged(write_zarr):
    """The fix must be a no-op where the two shapes coincide -- which is every store we
    benchmark against."""
    url, srcs = write_zarr(n=32, spc=8, inner=(4, 4))
    store = obstore_store(url)
    geom = open_geometries(store, variables=["t2m"])["t2m"]
    assert geom.chunks == (8, 4, 4)
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(store, manifest, geometries={"t2m": geom}, batch_size=4, shuffle=False)
    for batch in ds.all:
        np.testing.assert_array_equal(batch.arrays["t2m"], srcs["t2m"][batch.sample_indices])
    ds.close()


@pytest.mark.remote
def test_real_sharded_icechunk_store_matches_zarr():
    """A live Icechunk + sharded store, because that is where this defect lived and no local
    fixture would have found it. NOAA GFS analysis on anonymous S3 (dynamical.org)."""
    pytest.importorskip("icechunk")
    from insitubatch import icechunk_store

    store = icechunk_store(
        "s3://dynamical-noaa-gfs/noaa-gfs-analysis/v0.1.0.icechunk",
        anonymous=True,
        region="us-west-2",
    )
    geoms = open_geometries(store, variables=["temperature_2m"])
    assert geoms["temperature_2m"].chunks == (1440, 400, 400)  # the shard

    manifest = split_by_chunk(
        geoms["temperature_2m"], fractions=(1.0, 0.0, 0.0), sample_range=(1440, 1448)
    )
    ds = InSituDataset(
        store, manifest, geometries=geoms, batch_size=2, block_chunks=1, shuffle=False
    )
    batch = next(iter(ds.all))
    ref = zarr.open_group(store=store, mode="r")["temperature_2m"][
        batch.sample_indices[0] : batch.sample_indices[-1] + 1
    ]
    np.testing.assert_allclose(batch.arrays["temperature_2m"], ref)
    ds.close()


# -- byte budget vs the residency floor --------------------------------------


def _floor(store, geoms, manifest):
    """The floor the engine computes for itself, read off an auto-sized dataset."""
    ds = InSituDataset(store, manifest, geometries=geoms, batch_size=4, shuffle=False)
    floor = ds.cache_budget_bytes
    ds.close()
    return floor


def test_a_budget_below_the_residency_floor_raises(write_zarr):
    """An explicit budget the run cannot satisfy was silently promoted to the floor.

    That silence is survivable when a chunk is 64 KB. It is not on a sharded analysis
    archive whose stored chunk is hundreds of MB, where the number you passed is the
    difference between running and being OOM-killed -- and where being quietly given ten
    times what you asked for is indistinguishable, until the kernel intervenes, from the
    loader honouring it. The floor is computable from geometry alone, before any IO.
    """
    url, _ = write_zarr(n=64, spc=16, inner=(32, 32))
    store = obstore_store(url)
    geoms = open_geometries(store, variables=["t2m"])
    manifest = split_by_chunk(geoms["t2m"], fractions=(1.0, 0.0, 0.0))
    floor = _floor(store, geoms, manifest)

    with pytest.raises(ValueError) as exc:
        InSituDataset(store, manifest, geometries=geoms, batch_size=4, cache_budget_bytes=floor - 1)
    msg = str(exc.value)
    assert str(floor) in msg, "name the floor the run actually needs"
    assert "cache_budget_bytes" in msg, "and the knob that sets it"


def test_exactly_the_floor_is_accepted(write_zarr):
    """The boundary is inclusive: the floor is what the run needs, not one byte more."""
    url, _ = write_zarr(n=64, spc=16, inner=(32, 32))
    store = obstore_store(url)
    geoms = open_geometries(store, variables=["t2m"])
    manifest = split_by_chunk(geoms["t2m"], fractions=(1.0, 0.0, 0.0))
    floor = _floor(store, geoms, manifest)

    ds = InSituDataset(
        store,
        manifest,
        geometries=geoms,
        batch_size=4,
        shuffle=False,
        cache_budget_bytes=floor,
    )
    assert ds.cache_budget_bytes == floor
    assert sum(len(b.sample_indices) for b in ds.all) == 64
    ds.close()


def test_no_budget_still_sizes_itself(write_zarr):
    """Passing nothing is not passing zero: the automatic floor is unchanged."""
    url, _ = write_zarr(n=64, spc=16, inner=(32, 32))
    store = obstore_store(url)
    geoms = open_geometries(store, variables=["t2m"])
    manifest = split_by_chunk(geoms["t2m"], fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(store, manifest, geometries=geoms, batch_size=4, shuffle=False)
    assert ds.cache_budget_bytes > 0
    ds.close()


def test_an_over_budget_run_fails_rather_than_hanging(write_zarr, run_by):
    """The failure this replaces is residency starvation, which is a hang. Assert the raise
    lands under a deadline so a regression cannot wedge the suite."""
    url, _ = write_zarr(n=64, spc=16, inner=(32, 32))
    store = obstore_store(url)
    geoms = open_geometries(store, variables=["t2m"])
    manifest = split_by_chunk(geoms["t2m"], fractions=(1.0, 0.0, 0.0))

    def build():
        with pytest.raises(ValueError):
            InSituDataset(store, manifest, geometries=geoms, batch_size=4, cache_budget_bytes=1)
        return "raised"

    assert run_by(30, build) == "raised"


# -- icechunk_store ----------------------------------------------------------


def test_icechunk_store_round_trips_a_local_repo(tmp_path):
    """The local scheme, so the constructor is covered without a bucket. An Icechunk repo is
    a zarr Store like any other -- that is the whole 'one contract, any backend' claim."""
    icechunk = pytest.importorskip("icechunk")
    from insitubatch import icechunk_store

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(repo_dir)))
    session = repo.writable_session("main")
    group = zarr.open_group(store=session.store, mode="w")
    src = np.arange(32 * 4 * 4, dtype="f4").reshape(32, 4, 4)
    group.create_array("t2m", shape=src.shape, chunks=(8, 4, 4), dtype="f4")[:] = src
    session.commit("write t2m")

    store = icechunk_store(f"file://{repo_dir}")
    geoms = open_geometries(store, variables=["t2m"])
    manifest = split_by_chunk(geoms["t2m"], fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(store, manifest, geometries=geoms, batch_size=8, shuffle=False)
    for batch in ds.all:
        np.testing.assert_array_equal(batch.arrays["t2m"], src[batch.sample_indices])
    ds.close()


def test_icechunk_store_rejects_an_unknown_scheme():
    """An unusable URL should say which schemes work and point at the Arraylake route,
    rather than failing inside icechunk with something about storage backends."""
    pytest.importorskip("icechunk")
    from insitubatch import icechunk_store

    with pytest.raises(ValueError, match="arraylake_store"):
        icechunk_store("https://example.com/repo.icechunk")
