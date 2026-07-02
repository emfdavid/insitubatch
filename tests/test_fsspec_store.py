"""fsspec-backed store path (insitubatch.fsspec_store).

Exercises the fsspec store path without needing a real cloud bucket: fsspec's
local ``file://`` backend, which zarr auto-wraps as async. gcsfs/Rapid ride the
same ``FsspecStore.from_url`` path, so a green local round-trip proves the
wiring; the GCS-specific validation lives on the bench box, not in unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

# fsspec is not a core dep; it arrives via a backend extra
# (insitubatch[gcsfs] -> gcsfs -> fsspec). Skip where absent.
pytest.importorskip("fsspec")

from insitubatch import (  # noqa: E402
    ensure_local_dir,
    fsspec_store,
    open_geometries,
    split_by_chunk,
)
from insitubatch.source import InSituDataset  # noqa: E402


def test_fsspec_store_round_trips_an_epoch(write_zarr) -> None:
    url, srcs = write_zarr(n=40, spc=8)
    store = fsspec_store(url)  # read-only zarr FsspecStore over the local fs

    geom = open_geometries(store)["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(store, manifest, shuffle=False, batch_size=5, block_chunks=2)
    ds.set_epoch(0)

    got = np.concatenate([b.arrays["t2m"] for b in ds.train])
    # shuffle=False -> samples arrive in order, so they equal the written source.
    np.testing.assert_array_equal(got, srcs["t2m"])


def test_fsspec_store_writes_local_zarr(tmp_path) -> None:
    # Regression: a local fsspec write must create the nested chunk dirs zarr emits.
    # LocalFileSystem won't (unlike obstore's LocalStore), so fsspec_store defaults
    # auto_mkdir for file:// -- without it this 404s on the first chunk write.
    url = f"file://{tmp_path}/w.zarr"
    ensure_local_dir(url)
    src = np.arange(8 * 3 * 3, dtype="f4").reshape(8, 3, 3)

    group = zarr.open_group(store=fsspec_store(url, read_only=False), mode="w")
    arr = group.create_array(
        "t2m", shape=src.shape, chunks=(4, 3, 3), dtype="f4", dimension_names=("time", "y", "x")
    )
    arr[:] = src

    back = zarr.open_array(store=fsspec_store(url), path="t2m", mode="r")
    np.testing.assert_array_equal(np.asarray(back[:]), src)
