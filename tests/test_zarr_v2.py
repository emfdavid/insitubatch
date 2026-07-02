"""Reading zarr **v2** stores (e.g. the public WeatherBench2 ARCO archive) the same as v3.

The V2-decode path opens each array's metadata to build the decode spec + chunk-key
encoder. zarr-v2 metadata (`ArrayV2Metadata`) exposes `dtype` / `encode_chunk_key` where
v3 has `data_type` / `chunk_key_encoding` -- so the engine must read them
format-agnostically. Regression for `'ArrayV2Metadata' object has no attribute
'data_type'` on WeatherBench2.

Parametrized over **format × inner chunking** with compression on, so all three
metadata-sensitive pieces are exercised for v2:
  * dtype (`data_type` vs `dtype`) -- a wrong dtype mis-decodes;
  * chunk-key encoding -- the 2×2 inner grid yields non-zero inner coords, so the key
    shape (`v2 '0.1.1'` vs `v3 'c/0/1/1'`) must be right or the wrong tile is fetched;
  * codec pipeline -- v2 filters+compressor must decode.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from insitubatch import (
    ensure_local_dir,
    obstore_store,
    open_geometries,
    split_by_chunk,
)
from insitubatch.source import InSituDataset


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("inner_chunks", [(4, 4), (2, 2)], ids=["single-inner", "2x2-grid"])
def test_iterates_store_v2_and_v3(
    tmp_path, zarr_format: int, inner_chunks: tuple[int, int]
) -> None:
    url = f"file://{tmp_path}/d.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(
        store=obstore_store(url, read_only=False), mode="w", zarr_format=zarr_format
    )
    arr = group.create_array(
        "t2m",
        shape=(40, 4, 4),
        chunks=(8, *inner_chunks),
        dtype="f4",
        compressors="auto",
    )
    src = np.arange(40 * 16, dtype="f4").reshape(40, 4, 4)
    arr[:] = src

    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        obstore_store(url),
        manifest,
        shuffle=False,
        batch_size=8,
        block_chunks=2,
    )
    ds.set_epoch(0)

    # Every chunk (and every inner tile) decoded + scattered into the right place.
    recon = np.concatenate([b.arrays["t2m"] for b in ds.train], axis=0)
    np.testing.assert_array_equal(recon, src)
