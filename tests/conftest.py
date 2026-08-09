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
