"""Error propagation: a failing chunk transform must raise, not deadlock.

Guards the prefetch producer thread's exception bridge, which forwards exceptions
to the consumer. If it regressed into a hang, this test would hang (a real bug to
surface).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest

from insitubatch import obstore_store, open_geometries, split_by_chunk
from insitubatch.source import InSituDataset
from insitubatch.types import DecodedChunk


def _boom(chunk: DecodedChunk) -> DecodedChunk:
    raise ValueError("boom in transform")


def test_error_propagates_through_prefetch(write_zarr) -> None:
    url, _ = write_zarr(n=40, spc=8)
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(obstore_store(url), manifest, chunk_transforms=[_boom])
    ds.set_epoch(0)
    with pytest.raises(ValueError, match="boom"):
        list(ds.train)


def test_bad_chunk_raises_by_default_then_nan_fills(write_zarr) -> None:
    """A corrupt/truncated stored chunk fails fast by default; on_bad_chunk='nan'
    fills it with NaN and carries on (the rest of the data intact)."""
    url, srcs = write_zarr(n=40, spc=8, inner=(2, 2))  # 5 chunks of 8
    src = srcs["t2m"]
    geom = open_geometries(obstore_store(url))["t2m"]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))

    # Corrupt chunk 1's stored bytes so decode fails (not a valid compressed stream).
    chunk = Path(urlparse(url).path) / "t2m" / "c" / "1" / "0" / "0"
    assert chunk.exists(), f"chunk file not found: {chunk}"
    chunk.write_bytes(b"\x00\x01\x02\x03")

    ds = InSituDataset(obstore_store(url), manifest, shuffle=False, batch_size=8)
    ds.set_epoch(0)
    with pytest.raises(Exception):  # noqa: B017 - decode raises a codec-specific type
        list(ds.train)

    ds = InSituDataset(
        obstore_store(url), manifest, shuffle=False, batch_size=8, on_bad_chunk="nan"
    )
    ds.set_epoch(0)
    batches = list(ds.train)
    idx = np.concatenate([b.sample_indices for b in batches])
    out = np.concatenate([b.arrays["t2m"] for b in batches])[np.argsort(idx)]

    assert np.isnan(out[8:16]).all()  # chunk 1 -> NaN
    good = np.r_[0:8, 16:40]
    np.testing.assert_array_equal(out[good], src[good])  # everything else intact
    # ds.bad_chunks records which stored reads were corrupt (chunk 1's tile here).
    assert ds.bad_chunks and {r.chunk_index for r in ds.bad_chunks} == {1}
