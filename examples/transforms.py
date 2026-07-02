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
cross-variable or per-sample-random -> batch stage. This example makes the split concrete with
three transforms: :func:`kelvin_to_celsius` (a shape-preserving chunk stage) and :class:`Coarsen`
(a *reshaping* chunk stage -- a chunk-local regrid that halves the grid and declares its output
geometry so the cache can size the slot), then ``windspeed = sqrt(u10^2 + v10^2)`` at the batch
stage -- which *must* be a batch_transform, since a chunk_transform cannot see ``u10`` and
``v10`` at once. Validate any of them against a real store with ``insitubatch-check-transform``.

    uv run python -m examples.transforms
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
import zarr

from insitubatch import ensure_local_dir, obstore_store, open_geometries, split_by_chunk
from insitubatch.source import InSituDataset
from insitubatch.types import ArrayGeometry, Batch, DecodedChunk

VARIABLES = ("t2m", "u10", "v10")


def build_store(tmp: str, *, n: int = 64, lat: int = 16, lon: int = 32, spc: int = 8) -> str:
    """Write a tiny 3-variable zarr: temperature in Kelvin and the two wind components."""
    url = f"file://{tmp}/synthetic.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
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


# Temperature fields this K->C transform understands: the synthetic ``t2m`` here and the public
# WeatherBench2 ERA5 name (``2m_temperature``), so the same callable works against real cloud data.
TEMPERATURE_VARS = ("t2m", "2m_temperature")


def kelvin_to_celsius(chunk: DecodedChunk) -> DecodedChunk:
    """A chunk_transform: convert 2 m temperature from Kelvin to Celsius, vectorized.

    Gated on the variable name, because a chunk_transform is called once per (variable,
    chunk) and should only touch the field it understands (leaves u10/v10 untouched). It is
    per-variable, per-chunk, deterministic and elementwise -- the textbook cacheable chunk
    stage -- and pure vectorized numpy, so it releases the GIL on the decode pool.

    Check it against one chunk of your real store before training (geometry, cacheability, and
    an empirical GIL-release verdict)::

        insitubatch-check-transform <URL> --var 2m_temperature \\
            --transform examples/transforms.py:kelvin_to_celsius --skip-signature
    """
    if chunk.read.array in TEMPERATURE_VARS:
        chunk.data = chunk.data - 273.15
    return chunk


@dataclass
class Coarsen:
    """A *reshaping* chunk_transform: block-mean the spatial grid by ``factor`` (a local regrid).

    ``(n, lat, lon) -> (n, lat//factor, lon//factor)`` -- the canonical geometry-changing chunk
    stage. Because the output shape differs from the source, it declares ``output_inner`` so the
    cache can size its slot at the coarsened shape (a reshaping transform that *forgot* to declare
    it is rejected -- try deleting the method and running ``check-transform``). Pure vectorized
    numpy (a strided reshape + mean), so it releases the GIL on the decode pool. Assumes a 2-D
    ``(lat, lon)`` inner grid; a ragged edge (not divisible by ``factor``) is trimmed.

    Check the reshaping path against real data (validates the declared vs actual output shape)::

        insitubatch-check-transform <URL> --var 2m_temperature \\
            --transform examples/transforms.py:Coarsen --skip-signature
    """

    factor: int = 2

    def __call__(self, chunk: DecodedChunk) -> DecodedChunk:
        n, lat, lon = chunk.data.shape
        f = self.factor
        lat2, lon2 = (lat // f) * f, (lon // f) * f  # trim the ragged edge to a multiple of f
        blocks = chunk.data[:, :lat2, :lon2].reshape(n, lat2 // f, f, lon2 // f, f)
        chunk.data = blocks.mean(axis=(2, 4)).astype(chunk.data.dtype)  # (n, lat//f, lon//f)
        return chunk

    def output_inner(self, geom: ArrayGeometry) -> tuple[tuple[int, ...], np.dtype]:
        lat, lon = geom.inner_shape
        return (lat // self.factor, lon // self.factor), geom.dtype


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
        geoms = open_geometries(obstore_store(url), variables=list(VARIABLES))
        manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
        source_inner = geoms["t2m"].inner_shape

        # Two chunk stages: K->C (shape-preserving) then a reshaping Coarsen (halves the grid).
        # The cache slot is sized at the *coarsened* shape via Coarsen.output_inner.
        ds = InSituDataset(
            obstore_store(url),
            manifest,
            geometries=geoms,
            batch_size=batch_size,
            block_chunks=block_chunks,
            shuffle=False,
            chunk_transforms=[kelvin_to_celsius, Coarsen(factor=2)],
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
            "source_inner": tuple(source_inner),
            "sample_shape": tuple(t2m.shape[1:]),  # coarsened by the reshaping chunk_transform
        }
        if verbose:
            src, out = summary["source_inner"], summary["sample_shape"]
            print(f"batch variables: {summary['variables']}")
            print(f"t2m  (chunk_transform K->C): mean {summary['t2m_mean_c']:+.2f} C")
            print(f"grid (chunk_transform Coarsen): {src} -> {out}")
            print(f"windspeed (batch_transform): mean {summary['windspeed_mean']:.2f} m/s")
            print("all three stages ran: K->C + reshaping Coarsen (chunk), windspeed (batch).")
        return summary
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    run_demo(verbose=True)


if __name__ == "__main__":
    main()
