"""Icechunk support: pass a session Store directly instead of a URL.

An Icechunk store has no URL that round-trips to it (it is bound to a
repository snapshot/branch), so the engine must accept a prebuilt zarr Store.
The hot path already speaks only the zarr-v3 Store interface, so these tests
assert the end-to-end stream is byte-identical to the obstore/URL path and that
the store survives being driven from the scheduler's separate event-loop thread.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from insitubatch import (
    obstore_store,
    open_geometries,
    split_by_chunk,
)
from insitubatch.source import InSituDataset

icechunk = pytest.importorskip("icechunk")


@pytest.fixture
def icechunk_store():
    """Factory: write an in-memory Icechunk repo, return (readonly_store, {var: src})."""

    def _write(*, n=80, spc=8, inner=(2, 2), variables=("t2m",), seed=0):
        repo = icechunk.Repository.create(icechunk.in_memory_storage())
        session = repo.writable_session("main")
        group = zarr.open_group(store=session.store, mode="a")
        rng = np.random.default_rng(seed)
        srcs: dict[str, np.ndarray] = {}
        for var in variables:
            arr = group.create_array(var, shape=(n, *inner), chunks=(spc, *inner), dtype="f4")
            data = rng.standard_normal((n, *inner)).astype("f4")
            arr[:] = data
            srcs[var] = data
        session.commit("write fixture")
        return repo.readonly_session("main").store, srcs

    return _write


def test_open_geometries_from_store(icechunk_store) -> None:
    store, srcs = icechunk_store(n=40, spc=8, inner=(2, 2))
    geoms = open_geometries(store)
    assert set(geoms) == {"t2m"}
    geom = geoms["t2m"]
    assert geom.shape == (40, 2, 2)
    assert geom.chunks == (8, 2, 2)
    assert geom.dtype == np.dtype("f4")


def test_dataset_streams_from_icechunk_store(icechunk_store) -> None:
    # End-to-end: the dataset introspects the store, the scheduler drives it from
    # its own thread/loop, and the reassembled stream equals the source array.
    store, srcs = icechunk_store(n=40, spc=8, inner=(2, 2))
    src = srcs["t2m"]
    geom = open_geometries(store)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))

    ds = InSituDataset(
        store,
        manifest,
        shuffle=False,
        batch_size=5,
        block_chunks=2,
    )
    ds.set_epoch(0)
    idx = np.concatenate([b.sample_indices for b in ds.train])
    assert idx.tolist() == list(range(40))  # strictly in order

    ds.set_epoch(0)
    out = np.concatenate([b.arrays["t2m"] for b in ds.train], axis=0)
    np.testing.assert_array_equal(out, src)


def test_icechunk_matches_url_path(icechunk_store, write_zarr) -> None:
    # Same bytes via two stores: an Icechunk store and an obstore file:// URL.
    store, srcs = icechunk_store(n=40, spc=8, seed=7)
    src = srcs["t2m"]

    # Mirror the identical data into a file:// zarr so both paths see the same array.
    url, _ = write_zarr(n=40, spc=8, seed=7)

    geom = open_geometries(store)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))

    def collect(spec):
        ds = InSituDataset(spec, manifest, shuffle=False, batch_size=5, block_chunks=2)
        ds.set_epoch(0)
        return np.concatenate([b.arrays["t2m"] for b in ds.train], axis=0)

    np.testing.assert_array_equal(collect(store), src)
    np.testing.assert_array_equal(collect(obstore_store(url)), src)
