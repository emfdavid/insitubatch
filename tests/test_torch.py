"""Torch handoff: to_tensor + DataLoader round-trip. Skips if torch is absent."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from insitubatch import SplitName, open_geometries, split_by_chunk  # noqa: E402
from insitubatch.source import InSituDataset  # noqa: E402


def test_to_tensor_yields_torch_tensors(write_zarr) -> None:
    url, _ = write_zarr(n=40, spc=8)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        to_tensor=True,
        shuffle=False,
        batch_size=8,
        block_chunks=2,
    )
    ds.set_epoch(0)
    batch = next(iter(ds))
    assert isinstance(batch, dict)
    assert isinstance(batch["t2m"], torch.Tensor)
    assert batch["t2m"].shape[1:] == (2, 2)


def test_dataloader_roundtrip_reconstructs(write_zarr) -> None:
    url, srcs = write_zarr(n=40, spc=8)
    src = srcs["t2m"]
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        to_tensor=True,
        shuffle=False,
        batch_size=8,
        block_chunks=2,
    )
    ds.set_epoch(0)

    # batch_size=None: the dataset already yields assembled batches; num_workers=0:
    # parallelism is in insitubatch's event loop, not in worker processes.
    loader = DataLoader(ds, batch_size=None, num_workers=0)
    recon = torch.cat([b["t2m"] for b in loader], dim=0).numpy()
    np.testing.assert_array_equal(recon, src)  # shuffle=False + all chunks -> src in order
