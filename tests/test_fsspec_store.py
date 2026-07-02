"""fsspec-backed store path (insitubatch.fsspec_store).

Exercises the fsspec store path without needing a real cloud bucket: fsspec's
local ``file://`` backend, which zarr auto-wraps as async. gcsfs/Rapid ride the
same ``FsspecStore.from_url`` path, so a green local round-trip proves the
wiring; the GCS-specific validation lives on the bench box, not in unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

# fsspec is not a core dep; it arrives via a backend extra
# (insitubatch[gcsfs] -> gcsfs -> fsspec). Skip where absent.
pytest.importorskip("fsspec")

from insitubatch import fsspec_store, open_geometries, split_by_chunk  # noqa: E402
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
