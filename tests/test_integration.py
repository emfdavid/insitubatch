"""Combined: transforms + cache (memory & disk) + prefetch + shuffle=False.

Concatenates the batch arrays, inverts the (deterministic) StandardScaler, and
asserts the original input is reconstructed exactly — across two epochs (the
second served from the cache) and with a non-divisible sample count so the
partial final chunk is exercised end-to-end. Parametrized over the in-memory and
mmap'd-NVMe caches so the disk/mmap path gets value-level coverage too.
"""

from __future__ import annotations

import numpy as np
import pytest

from insitubatch import (
    DiskCache,
    MemoryCache,
    SplitName,
    fit_standard_scaler,
    open_geometries,
    split_by_chunk,
)
from insitubatch.source import InSituDataset


@pytest.mark.parametrize("kind", ["memory", "disk"])
def test_transforms_cache_prefetch_reconstruct(write_zarr, tmp_path, kind) -> None:
    url, srcs = write_zarr(n=50, spc=8, inner=(2, 2))  # 7 chunks; last holds 50-48=2
    src = srcs["t2m"]
    geoms = open_geometries(url)
    manifest = split_by_chunk(geoms["t2m"], fractions=(1.0, 0.0, 0.0))
    scaler = fit_standard_scaler(url, manifest, geoms)
    mean, std = np.squeeze(scaler.mean["t2m"]), np.squeeze(scaler.std["t2m"])

    cache = MemoryCache(10_000_000) if kind == "memory" else DiskCache(tmp_path / "c", 10_000_000)
    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        shuffle=False,
        batch_size=6,
        block_chunks=3,
        prefetch_depth=2,
        cache=cache,
        to_tensor=False,
        chunk_transforms=[scaler],
    )

    for _epoch in range(2):
        ds.set_epoch(_epoch)
        normalized = np.concatenate([b.arrays["t2m"] for b in ds], axis=0)
        reconstructed = normalized * (std + 1e-8) + mean  # invert the scaler
        np.testing.assert_allclose(reconstructed, src, rtol=1e-4, atol=1e-4)

    assert cache.hits > 0  # the cache was exercised (epoch 2)
