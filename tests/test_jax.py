"""JAX handoff: frameworks.to_jax (DLPack). Skips if jax is absent.

JAX has no dataloader base class -- you iterate the dataset directly and convert
each numpy ``Batch`` with :func:`insitubatch.frameworks.to_jax`.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from insitubatch import SplitName, open_geometries, split_by_chunk  # noqa: E402
from insitubatch.frameworks import to_jax  # noqa: E402
from insitubatch.source import InSituDataset  # noqa: E402


def test_to_jax_roundtrip(write_zarr) -> None:
    url, srcs = write_zarr(n=40, spc=8)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        url, manifest, split=SplitName.TRAIN, shuffle=False, batch_size=8, block_chunks=2
    )
    ds.set_epoch(0)

    seen = []
    for batch in ds:
        arrays = to_jax(batch)
        assert isinstance(arrays["t2m"], jax.Array)
        assert arrays["t2m"].shape[1:] == (2, 2)
        seen.append(np.asarray(arrays["t2m"]))

    np.testing.assert_array_equal(np.concatenate(seen, axis=0), srcs["t2m"])
