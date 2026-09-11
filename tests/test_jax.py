"""JAX handoff: frameworks.to_jax (DLPack). Skips if jax is absent.

JAX has no dataloader base class -- you iterate the dataset directly and convert
each numpy ``Batch`` with :func:`insitubatch.frameworks.to_jax`.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from insitubatch import obstore_store, open_geometries, split_by_chunk  # noqa: E402
from insitubatch.frameworks import to_jax  # noqa: E402
from insitubatch.source import InSituDataset  # noqa: E402


def test_to_jax_roundtrip(write_zarr) -> None:
    url, srcs = write_zarr(n=40, spc=8)
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(obstore_store(url), manifest, shuffle=False, batch_size=8, block_chunks=2)
    ds.set_epoch(0)

    seen = []
    for batch in ds.train:
        arrays = to_jax(batch)
        assert isinstance(arrays["t2m"], jax.Array)
        assert arrays["t2m"].shape[1:] == (2, 2)
        seen.append(np.asarray(arrays["t2m"]))

    np.testing.assert_array_equal(np.concatenate(seen, axis=0), srcs["t2m"])


def test_to_jax_lands_on_the_default_device(write_zarr) -> None:
    """A batch must arrive where every other jax array-creation path puts one.

    ``jnp.from_dlpack`` imports a host buffer and so commits the result to ``cpu:0``. Since
    ``jax.jit`` will not move a committed array, returning that unchanged meant a GPU run
    trained on the CPU -- correct values, full speed of the wrong device, nothing raised. The
    assertion is against ``jax.devices()[0]`` rather than a literal, so it is meaningful on a
    CUDA box and still guards the contract on a CPU-only one.
    """
    url, srcs = write_zarr(n=40, spc=8)
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(obstore_store(url), manifest, shuffle=False, batch_size=8, block_chunks=2)
    ds.set_epoch(0)

    default = jax.devices()[0]
    for batch in ds.train:
        for name, arr in to_jax(batch).items():
            assert list(arr.devices()) == [default], f"{name} on {list(arr.devices())}"


def test_to_jax_honours_an_explicit_device(write_zarr) -> None:
    """``device=`` overrides the default, and the values survive the transfer."""
    url, srcs = write_zarr(n=16, spc=8)
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(obstore_store(url), manifest, shuffle=False, batch_size=8, block_chunks=2)
    ds.set_epoch(0)

    target = jax.devices("cpu")[0]
    batch = next(iter(ds.train))
    arrays = to_jax(batch, device=target)
    assert list(arrays["t2m"].devices()) == [target]
    np.testing.assert_array_equal(np.asarray(arrays["t2m"]), batch.arrays["t2m"])
