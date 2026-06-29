"""The two user transform stages, side by side, on a tiny synthetic store (no network).

insitubatch has two places to hook preprocessing, chosen by *cost* and *what the transform
needs to see* (the full model is in docs/architecture.md, "Transforms"):

- **chunk_transform** ``(DecodedChunk) -> DecodedChunk`` runs per decoded chunk, before
  shuffle/gather, on the decode thread pool. It sees **one variable, one chunk**, so it is
  the home for per-element, sample-order-independent work: unit conversion, scaling, dtype
  cast. Because it is deterministic and runs before the cache boundary, its output is
  **cached** -- amortized over every sample drawn from that chunk, and over later epochs.

- **batch_transform** ``(Batch) -> Batch`` runs per assembled batch, after gather. It sees
  **all variables aligned on the sample axis**, so it is the home for cross-variable derived
  fields, per-sample random augmentation, and collation. It runs *after* the cache, so its
  output is **recomputed every draw** (never cached).

The rule of thumb: per-variable + per-chunk + deterministic -> chunk stage (cacheable);
cross-variable or per-sample-random -> batch stage. This example makes the split concrete:
a Kelvin->Celsius conversion is a chunk_transform (one variable, elementwise), while
windspeed = sqrt(u10^2 + v10^2) *must* be a batch_transform -- a chunk_transform cannot see
``u10`` and ``v10`` at once.

    uv run python -m examples.transforms
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import zarr

from insitubatch import ensure_local_dir, open_geometries, split_by_chunk, store_from_url
from insitubatch.source import InSituDataset
from insitubatch.types import Batch, DecodedChunk

VARIABLES = ("t2m", "u10", "v10")


def build_store(tmp: str, *, n: int = 64, lat: int = 16, lon: int = 32, spc: int = 8) -> str:
    """Write a tiny 3-variable zarr: temperature in Kelvin and the two wind components."""
    url = f"file://{tmp}/synthetic.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=store_from_url(url, read_only=False), mode="w")
    rng = np.random.default_rng(0)
    fields = {
        "t2m": 273.15 + 10.0 * rng.standard_normal((n, lat, lon)).astype("f4"),
        "u10": 5.0 * rng.standard_normal((n, lat, lon)).astype("f4"),
        "v10": 5.0 * rng.standard_normal((n, lat, lon)).astype("f4"),
    }
    for name, data in fields.items():
        arr = group.create_array(
            name,
            shape=(n, lat, lon),
            chunks=(spc, lat, lon),
            dtype="f4",
            dimension_names=("time", "lat", "lon"),
        )
        arr[:] = data
    return url


def kelvin_to_celsius(chunk: DecodedChunk) -> DecodedChunk:
    """A chunk_transform: convert temperature to Celsius in place.

    Gated on the variable name, because a chunk_transform is called once per (variable,
    chunk) and should only touch the field it understands. Leaves u10/v10 untouched.
    """
    if chunk.read.array == "t2m":
        chunk.data = chunk.data - 273.15
    return chunk


def add_windspeed(batch: Batch) -> Batch:
    """A batch_transform: derive windspeed from the two wind components.

    This cannot be a chunk_transform: it needs u10 and v10 together, and the assembled
    batch is the first place all variables are aligned on the sample axis.
    """
    batch.arrays["windspeed"] = np.sqrt(batch.arrays["u10"] ** 2 + batch.arrays["v10"] ** 2)
    return batch


def run_demo(
    *, url: str | None = None, batch_size: int = 16, block_chunks: int = 4, verbose: bool = True
) -> dict:
    """Build the store (if no ``url``), run both transform stages, return a summary."""
    tmp = None
    if url is None:
        tmp = tempfile.mkdtemp(prefix="insitu-transforms-")
        url = build_store(tmp)
    try:
        geoms = open_geometries(url, variables=list(VARIABLES))
        manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))

        ds = InSituDataset(
            url,
            manifest,
            geometries=geoms,
            batch_size=batch_size,
            block_chunks=block_chunks,
            shuffle=False,
            chunk_transforms=[kelvin_to_celsius],
            batch_transforms=[add_windspeed],
        )

        batch = next(iter(ds.all))
        t2m, wind = batch.arrays["t2m"], batch.arrays["windspeed"]
        summary = {
            "variables": sorted(batch.arrays),
            "t2m_mean_c": float(t2m.mean()),
            "windspeed_mean": float(wind.mean()),
            "windspeed_nonneg": bool((wind >= 0.0).all()),
            "samples": int(t2m.shape[0]),
            "sample_shape": tuple(t2m.shape[1:]),
        }
        if verbose:
            print(f"batch variables: {summary['variables']}")
            print(f"t2m  (chunk_transform K->C): mean {summary['t2m_mean_c']:+.2f} C")
            print(f"windspeed (batch_transform): mean {summary['windspeed_mean']:.2f} m/s")
            print("both transform stages ran.")
        return summary
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    run_demo(verbose=True)


if __name__ == "__main__":
    main()
