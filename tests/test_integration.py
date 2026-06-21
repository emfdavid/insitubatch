"""Combined: chunk transform + prefetch + shuffle=False, end to end.

Concatenates the batch arrays, inverts the (deterministic) StandardScaler, and
asserts the original input is reconstructed exactly — across two epochs (so
determinism holds epoch to epoch) and with a non-divisible sample count so the
partial final chunk is exercised through the scatter/gather path.
"""

from __future__ import annotations

import numpy as np

from insitubatch import (
    SplitName,
    StandardScaler,
    open_geometries,
    split_by_chunk,
)
from insitubatch.source import InSituDataset


def test_transforms_prefetch_reconstruct(write_zarr) -> None:
    url, srcs = write_zarr(n=50, spc=8, inner=(2, 2))  # 7 chunks; last holds 50-48=2
    src = srcs["t2m"]
    geoms = open_geometries(url)
    manifest = split_by_chunk(geoms["t2m"], fractions=(1.0, 0.0, 0.0))
    mean, std = src.astype("f8").mean(), src.astype("f8").std()  # global stats, pre-fit
    scaler = StandardScaler(mean={"t2m": np.float64(mean)}, std={"t2m": np.float64(std)})

    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        shuffle=False,
        batch_size=6,
        block_chunks=3,
        prefetch_depth=2,
        to_tensor=False,
        chunk_transforms=[scaler],
    )

    for _epoch in range(2):
        ds.set_epoch(_epoch)
        normalized = np.concatenate([b.arrays["t2m"] for b in ds], axis=0)
        reconstructed = normalized * (std + 1e-8) + mean  # invert the scaler
        np.testing.assert_allclose(reconstructed, src, rtol=1e-4, atol=1e-4)
