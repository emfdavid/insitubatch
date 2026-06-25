"""Torch handoff: frameworks.to_torch / as_torch + DataLoader round-trip.

Skips if torch is absent (it's an optional adapter dep).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from insitubatch import open_geometries, split_by_chunk  # noqa: E402
from insitubatch.frameworks import as_torch, to_torch  # noqa: E402
from insitubatch.source import InSituDataset  # noqa: E402


def _ds(write_zarr):
    url, srcs = write_zarr(n=40, spc=8)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(url, manifest, shuffle=False, batch_size=8, block_chunks=2)
    ds.set_epoch(0)
    return ds, srcs


def test_to_torch_yields_tensors_zero_copy(write_zarr) -> None:
    ds, _ = _ds(write_zarr)
    batch = next(iter(ds.train))
    tensors = to_torch(batch)
    assert isinstance(tensors, dict)
    assert isinstance(tensors["t2m"], torch.Tensor)
    assert tensors["t2m"].shape[1:] == (2, 2)
    # DLPack is zero-copy on CPU: mutating the numpy batch shows through the tensor.
    batch.arrays["t2m"][0, 0, 0] = 123.0
    assert float(tensors["t2m"][0, 0, 0]) == 123.0


def test_as_torch_dataloader_roundtrip(write_zarr) -> None:
    ds, srcs = _ds(write_zarr)
    src = srcs["t2m"]
    # batch_size=None: the dataset already yields assembled batches; num_workers=0:
    # parallelism is in insitubatch's event loop, not in worker processes.
    loader = DataLoader(as_torch(ds.train), batch_size=None, num_workers=0)
    recon = torch.cat([b["t2m"] for b in loader], dim=0).numpy()
    np.testing.assert_array_equal(recon, src)  # shuffle=False + all chunks -> src in order
