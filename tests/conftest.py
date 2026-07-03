"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from insitubatch import ensure_local_dir, obstore_store


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
