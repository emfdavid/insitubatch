"""Fit a scaler over the loader with scikit-learn ``partial_fit`` — warming the cache.

The recommended normalization pattern (vs caching the *scaled* chunk at the chunk stage):

  1. **Fit pass** — iterate the loader ONCE with *no scaler*. Each chunk is decoded
     and **cached** (raw); meanwhile a sklearn `StandardScaler.partial_fit` accumulates
     per-variable stats incrementally over the batches. The fit *is* the cache warm-up.
  2. **Train** — attach the fitted scaler as a **batch** transform and re-iterate. The
     cache now serves the raw chunks (decode-once), and scaling is a cheap per-batch op.

Why this over a chunk-stage scaler:

  * **Familiar tooling** — fit with `sklearn` (or `dask_ml`) `partial_fit`, on the
    completed batch or per variable, exactly as you would off any iterator.
  * **The cache stays normalization-agnostic** — it holds *raw* chunks, so you can
    change the scaler (or sweep normalizations) without re-decoding.
  * **Composes correctly** — the fit pass runs the dataset's own `chunk_transforms`
    (e.g. a regrid), so the scaler is fit on exactly what training will see.

    uv run python -m examples.fit_scaler                 # synthetic file://
    uv run python -m examples.fit_scaler --wb2           # public WeatherBench2 ERA5
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import zarr

from insitubatch import (
    Batch,
    ensure_local_dir,
    obstore_store,
    open_geometries,
    split_by_chunk,
)
from insitubatch.source import InSituDataset

try:
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - example-only dep
    raise SystemExit("this example needs scikit-learn:  uv sync --extra bench") from exc

WB2 = (
    "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr"
)


def _synthetic(tmp: str, *, n: int = 256, inner: tuple[int, int] = (32, 32), spc: int = 8) -> str:
    """Two variables with deliberately different mean/scale, so standardization shows."""
    url = f"file://{tmp}/era5.zarr"
    ensure_local_dir(url)
    g = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    rng = np.random.default_rng(0)
    for i, var in enumerate(("t2m", "u10")):
        a = g.create_array(var, shape=(n, *inner), chunks=(spc, *inner), dtype="f4")
        a[:] = (rng.standard_normal((n, *inner)) * (3 * i + 2) + (20 * i + 5)).astype("f4")
    return url


def fit_over_loader(ds: InSituDataset) -> tuple[dict[str, StandardScaler], float, int]:
    """One pass over the loader (no scaler): warm the cache + ``partial_fit`` per variable.

    Per-variable *global* stats: every pixel is a sample of one feature
    (``reshape(-1, 1)``). For per-pixel climatology use ``reshape(len(x), -1)`` instead.
    """
    scalers = {v: StandardScaler() for v in ds.variables}
    ds.set_epoch(0)
    n, t = 0, time.perf_counter()
    for batch in ds.train:
        for v in ds.variables:
            scalers[v].partial_fit(batch.arrays[v].reshape(-1, 1))
        n += len(batch.sample_indices)
    return scalers, time.perf_counter() - t, n


def make_batch_scaler(scalers: dict[str, StandardScaler]) -> Callable[[Batch], Batch]:
    """A batch_transform that applies the fitted scalers (post-gather, per batch)."""

    def scale(batch: Batch) -> Batch:
        for v, s in scalers.items():
            x = batch.arrays[v]
            batch.arrays[v] = s.transform(x.reshape(-1, 1)).reshape(x.shape).astype("f4")
        return batch

    return scale


def run_demo(
    *,
    url: str | None = None,
    cache_dir: str | None = None,
    variables: list[str] | None = None,
    verbose: bool = True,
    **store_kwargs: Any,
) -> dict:
    tmp = None
    if url is None:
        tmp = tempfile.mkdtemp(prefix="fit-scaler-")
        url = _synthetic(tmp)
    if url.startswith("gs://"):
        store_kwargs["skip_signature"] = True  # public bucket, anonymous read

    # A real store (WeatherBench2) has many variables on *different* sample axes (coords,
    # pressure levels); InSituDataset requires one shared sample axis, so select a
    # compatible set. None = every variable (fine for the synthetic two-var store).
    geoms = open_geometries(obstore_store(url), variables=variables, **store_kwargs)
    var0 = next(iter(geoms))
    manifest = split_by_chunk(geoms[var0], fractions=(1.0, 0.0, 0.0))

    # A cache big enough to hold the split: the fit pass warms it, training reuses it.
    ds = InSituDataset(
        obstore_store(url),
        manifest,
        geometries=geoms,
        batch_size=16,
        block_chunks=4,
        shuffle=False,
        cache_dir=cache_dir or (f"{tmp}/cache" if tmp else None),
        cache_budget_bytes=4 << 30,
        **store_kwargs,
    )

    # 1) fit over the loader (cold: decode + cache) -----------------------------------
    scalers, fit_s, n = fit_over_loader(ds)

    # verify the streamed partial_fit matches the true global mean/std
    grp = zarr.open_group(store=obstore_store(url, **store_kwargs), mode="r")
    max_err = 0.0
    for v, s in scalers.items():
        arr = grp[v]
        assert isinstance(arr, zarr.Array)
        true = np.asarray(arr[:])
        max_err = max(max_err, abs(s.mean_[0] - true.mean()), abs(s.scale_[0] - true.std()))

    # 2) attach the scaler as a BATCH transform; re-iterate (warm: cache hits) ---------
    ds.batch_transforms = (make_batch_scaler(scalers),)
    ds.set_epoch(1)
    means, stds, t = [], [], time.perf_counter()
    for batch in ds.train:
        for v in ds.variables:
            means.append(float(batch.arrays[v].mean()))
            stds.append(float(batch.arrays[v].std()))
    warm_s = time.perf_counter() - t

    stats: dict[str, tuple[float, float]] = {
        v: (float(s.mean_[0]), float(s.scale_[0])) for v, s in scalers.items()
    }
    summary = {
        "samples": n,
        "fit_s": fit_s,
        "warm_s": warm_s,
        "speedup": (fit_s / warm_s) if warm_s else 0.0,
        "stat_max_err": max_err,
        "scaled_mean": float(np.mean(means)),
        "scaled_std": float(np.mean(stds)),
        "stats": stats,
    }
    if verbose:
        print(f"fit pass (cold, decodes + caches): {n} samples in {fit_s:.3f}s")
        for v, (m, sd) in stats.items():
            print(f"   {v}: mean={m:8.3f}  std={sd:7.3f}")
        print(f"   partial_fit vs true global stats: max |err| = {max_err:.2e}  ✓")
        print(
            f"train pass (cache-warm, scaler at batch stage): {warm_s:.3f}s "
            f"({summary['speedup']:.1f}x faster than the cold fit)"
        )
        print(
            f"   scaled batches: mean ~ {summary['scaled_mean']:+.2e}  "
            f"std ~ {summary['scaled_std']:.3f}  (≈ 0 / 1)"
        )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--wb2", action="store_true", help="use public WeatherBench2 ERA5 (GCS)")
    p.add_argument("--url", default=None, help="zarr URL (default: synthetic file://)")
    p.add_argument("--cache-dir", default=None, help="spill the cache to this dir (NVMe)")
    p.add_argument(
        "--variables",
        default=None,
        help="comma list; must share the sample axis + chunking. --wb2 default: two "
        "surface fields. Synthetic/other URLs default to every variable in the store.",
    )
    a = p.parse_args()
    url = WB2 if a.wb2 else a.url
    if a.variables:
        variables = a.variables.split(",")
    elif a.wb2:
        # WB2 surface fields share the (time, lat, lon) grid + chunking; two with
        # different mean/scale make the per-variable standardization visible.
        variables = ["2m_temperature", "10m_u_component_of_wind"]
    else:
        variables = None
    run_demo(url=url, cache_dir=a.cache_dir, variables=variables)


if __name__ == "__main__":
    main()
