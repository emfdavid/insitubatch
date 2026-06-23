"""TensorFlow handoff: frameworks.to_tf + as_tf_dataset. Skips if tensorflow is absent.

TF has no base class to inherit -- :func:`insitubatch.frameworks.as_tf_dataset` adapts
the stream via ``tf.data.Dataset.from_generator``; :func:`to_tf` is the per-batch
(zero-copy DLPack) path.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from insitubatch import SplitName, open_geometries, split_by_chunk  # noqa: E402
from insitubatch.frameworks import as_tf_dataset, to_tf  # noqa: E402
from insitubatch.source import InSituDataset  # noqa: E402


def _ds(write_zarr):
    url, srcs = write_zarr(n=40, spc=8)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        url, manifest, split=SplitName.TRAIN, shuffle=False, batch_size=8, block_chunks=2
    )
    ds.set_epoch(0)
    return ds, srcs


def test_to_tf_roundtrip(write_zarr) -> None:
    ds, srcs = _ds(write_zarr)
    seen = []
    for batch in ds:
        tensors = to_tf(batch)
        assert isinstance(tensors["t2m"], tf.Tensor)
        seen.append(tensors["t2m"].numpy())
    np.testing.assert_array_equal(np.concatenate(seen, axis=0), srcs["t2m"])


def test_as_tf_dataset_roundtrip(write_zarr) -> None:
    ds, srcs = _ds(write_zarr)
    tfds = as_tf_dataset(ds, prefetch=2)
    seen = [b["t2m"].numpy() for b in tfds]
    np.testing.assert_array_equal(np.concatenate(seen, axis=0), srcs["t2m"])
