"""Shared test fixtures."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import zarr

from insitubatch import ensure_local_dir, obstore_store


@pytest.fixture
def run_by():
    """Factory: ``run_by(seconds, fn)`` -> ``fn()``'s result, failing the test on timeout.

    For the cases whose regression is a *hang* rather than a wrong answer -- residency
    starvation, error propagation through the prefetch producer. A plain call would block
    the whole suite; running under a deadline turns the wedge into a failure. The worker is
    daemonic and deliberately abandoned on timeout: the test has already failed, and a
    deadlocked scheduler cannot be joined.
    """

    def _run_by(seconds: float, fn: Callable[[], Any]) -> Any:
        box: list[tuple[str, Any]] = []

        def go() -> None:
            try:
                box.append(("ok", fn()))
            except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
                box.append(("raised", exc))

        thread = threading.Thread(target=go, daemon=True, name="insitu-deadline")
        thread.start()
        thread.join(seconds)
        if thread.is_alive():
            pytest.fail(f"deadlocked: no result within {seconds}s")
        kind, value = box[0]
        if kind == "raised":
            raise value
        return value

    return _run_by


def pytest_addoption(parser):
    """``--remote`` opts into the tests that read somebody else's live bucket.

    Off by default: the suite is otherwise fully offline and deterministic, and a public
    store going away must not turn into a red build for a contributor who changed nothing.
    These tests exist because a local fixture cannot reproduce a backend-specific defect --
    the sharded-decode bug (#56) lived in an Icechunk store and no synthetic fixture would
    have found it.
    """
    parser.addoption(
        "--remote",
        action="store_true",
        default=False,
        help="run tests that read live public cloud stores (network, slow)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--remote"):
        return
    skip = pytest.mark.skip(reason="needs --remote (reads a live public bucket)")
    for item in items:
        if "remote" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def write_zarr(tmp_path):
    """Factory: write a zarr group of random f4 variables, return (url, {var: src})."""

    def _write(*, n=80, spc=8, inner=(2, 2), variables=("t2m",), seed=0):
        url = f"file://{tmp_path}/d.zarr"
        ensure_local_dir(url)
        group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
        rng = np.random.default_rng(seed)
        srcs: dict[str, np.ndarray] = {}
        for var in variables:
            arr = group.create_array(var, shape=(n, *inner), chunks=(spc, *inner), dtype="f4")
            data = rng.standard_normal((n, *inner)).astype("f4")
            arr[:] = data
            srcs[var] = data
        return url, srcs

    return _write


@pytest.fixture
def write_sharded_zarr(tmp_path):
    """Factory: a **sharded** zarr-v3 group -- the layout every dynamical.org store uses.

    ``shard`` is the stored object (what a chunk key addresses); ``chunks`` is the read
    granularity inside it. The two differ here on purpose: that difference is the whole of
    #56, and a fixture where they coincide proves nothing.
    """

    def _write(
        *, n=48, shard=(16, 8, 8), chunks=(16, 4, 4), inner=(8, 8), variables=("t2m",), seed=0
    ):
        url = f"file://{tmp_path}/sharded.zarr"
        ensure_local_dir(url)
        group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
        rng = np.random.default_rng(seed)
        srcs: dict[str, np.ndarray] = {}
        for var in variables:
            arr = group.create_array(
                var, shape=(n, *inner), shards=shard, chunks=chunks, dtype="f4"
            )
            data = rng.standard_normal((n, *inner)).astype("f4")
            arr[:] = data
            srcs[var] = data
        return url, srcs

    return _write


@pytest.fixture
def write_cf_zarr(tmp_path):
    """Factory: a CF/xarray-shaped group -- data variables plus coordinate arrays and a 0-D
    ``spatial_ref`` grid-mapping scalar, which is what every rioxarray-written store carries.
    """

    def _write(*, n=32, spc=8, lat=4, lon=5, variables=("t2m", "u10"), v3=True, seed=0):
        url = f"file://{tmp_path}/cf.zarr"
        ensure_local_dir(url)
        group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
        rng = np.random.default_rng(seed)
        srcs: dict[str, np.ndarray] = {}
        dims = ("time", "latitude", "longitude")

        def _dim(arr, names):
            # v3 carries dimension_names in metadata; v2 carries xarray's _ARRAY_DIMENSIONS.
            if not v3:
                arr.attrs["_ARRAY_DIMENSIONS"] = list(names)

        for name, size in (("time", n), ("latitude", lat), ("longitude", lon)):
            c = group.create_array(
                name,
                shape=(size,),
                chunks=(size,),
                dtype="f8",
                dimension_names=(name,) if v3 else None,
            )
            c[:] = np.arange(size, dtype="f8")
            _dim(c, (name,))

        crs = group.create_array("spatial_ref", shape=(), chunks=(), dtype="i4")
        crs.attrs["grid_mapping_name"] = "latitude_longitude"
        crs.attrs["crs_wkt"] = 'GEOGCS["WGS 84"]'

        for var in variables:
            arr = group.create_array(
                var,
                shape=(n, lat, lon),
                chunks=(spc, lat, lon),
                dtype="f4",
                dimension_names=dims if v3 else None,
            )
            data = rng.standard_normal((n, lat, lon)).astype("f4")
            arr[:] = data
            _dim(arr, dims)
            srcs[var] = data
        return url, srcs

    return _write
