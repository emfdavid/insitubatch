"""Spike: validate the V2 fetch scheduler's core mechanic against zarr 3.2 internals.

V2 (DESIGN.md, M1.6) reads at STORED-CHUNK granularity (outer x inner) under ONE
flat concurrency budget, then scatters each decoded inner tile into its outer
chunk's array. This is the risky zarr-internals dependency, so this spike proves
that fetch + decode + scatter reconstructs *exactly* what ``arr.getitem`` stitches
-- for both single-inner and spatially-chunked arrays, including partial edge
chunks. Validation/throwaway, not shipped in the wheel.

    uv run python bench/spike_v2_decode.py
"""

from __future__ import annotations

import asyncio
import itertools
import math
import tempfile

import numpy as np
import zarr
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype

from insitubatch.store import ensure_local_dir, store_from_url


async def fetch_decode_scatter(url: str, var: str, max_inflight: int) -> np.ndarray:
    """Reconstruct the whole array by fetching every stored chunk under one flat
    ``max_inflight`` budget, decoding it, and scattering it into place."""
    aa = zarr.open_array(store=store_from_url(url), path=var, mode="r")._async_array
    proto = default_buffer_prototype()
    spec = ArraySpec(
        shape=aa.metadata.chunks,
        dtype=aa.metadata.data_type,
        fill_value=aa.metadata.fill_value,
        config=aa.config,
        prototype=proto,
    )
    shape, chunks = tuple(aa.metadata.shape), tuple(aa.metadata.chunks)
    grid = [range(math.ceil(s / c)) for s, c in zip(shape, chunks, strict=True)]
    out = np.empty(shape, dtype=np.float32)
    sem = asyncio.Semaphore(max_inflight)  # ONE budget over all stored chunks (inner + outer)

    async def one(coords: tuple[int, ...]) -> None:
        async with sem:
            key = aa.store_path.path + "/" + aa.metadata.chunk_key_encoding.encode_chunk_key(coords)
            buf = await aa.store_path.store.get(key, prototype=proto)
            [tile] = list(await aa.codec_pipeline.decode([(buf, spec)]))
        # Scatter into place; the decoded tile is full chunk-shaped, so clip the
        # (possibly partial) edge region. After this copy the tile is free.
        dst = tuple(
            slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(coords, chunks, shape, strict=True)
        )
        src = tuple(slice(0, sl.stop - sl.start) for sl in dst)
        out[dst] = tile.as_numpy_array()[src]

    await asyncio.gather(*(one(c) for c in itertools.product(*grid)))
    return out


def _check(url: str, var: str, ref: np.ndarray) -> None:
    got = asyncio.run(fetch_decode_scatter(url, var, max_inflight=8))
    ok = got.shape == ref.shape and np.allclose(got, ref)
    print(f"  {var:14} {ref.shape}  chunk-scatter == getitem: {ok}")
    assert ok


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="spike-v2-")
    url = f"file://{tmp}/s.zarr"
    ensure_local_dir(url)
    g = zarr.open_group(store=store_from_url(url, read_only=False), mode="w")
    rng = np.random.default_rng(0)
    # partial edge chunks on every axis, both single-inner and spatially chunked
    shape = (5, 9, 7)
    data = rng.standard_normal(shape).astype("f4")
    for var, chunks in (("single_inner", (2, 9, 7)), ("spatial", (2, 4, 4))):
        a = g.create_array(var, shape=shape, chunks=chunks, dtype="f4")
        a[:] = data
        _check(url, var, np.asarray(a[:]))
    print("spike OK: V2 fetch+decode+scatter validated against zarr", zarr.__version__)


if __name__ == "__main__":
    main()
