"""Shared data for the advected-field forecast: the store, the one dataset, the eval.

**The task — a direct 24-hour forecast.** Inputs are three fields at time ``t`` --
temperature ``t2m`` and the 10 m wind ``u10`` / ``v10`` -- and the target is ``t2m`` 24 h
later. Nothing is resharded: input and target are two *views* of the same in-place array
read at different sample-axis offsets (``g`` and ``g.shift(horizon)``), and the three
inputs are three arrays gathered at the *same* anchor. That is the whole M-W unlock.

Persistence -- predict ``t2m(t+24h) = t2m(t)`` -- is wrong by exactly the advection the
wind drives over 24 h. A tiny CNN that *sees* the wind can do better, so the models learn
the **tendency** (the change on top of persistence), the pattern weather-ML uses in
practice. On the synthetic store that beats persistence by construction; on the real
WeatherBench2 store the same code runs on real ERA5 (we claim "same pipeline, real data,
no reshard" -- not SOTA skill: t2m persistence at a 24 h multiple is a strong baseline).

``forecast_dataset`` returns one dataset whose labels are always ``t2m, u10, v10, target``
regardless of the store, so the training files are store-agnostic.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import numpy as np
import zarr

from insitubatch import (
    SplitName,
    ensure_local_dir,
    open_geometries,
    split_by_chunk,
    store_from_url,
)
from insitubatch.source import InSituDataset
from insitubatch.types import Batch

# The public WeatherBench2 ERA5 (6-hourly) ARCO store the wb2_* examples use.
WB2_URL = (
    "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr"
)
WB2_VARS = ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind")
WB2_HORIZON = 4  # 4 steps x 6 h = 24 h

SYNTH_VARS = ("t2m", "u10", "v10")
SYNTH_HORIZON = 24  # 24 steps x 1 h = 24 h

LABELS = ("t2m", "u10", "v10")  # canonical input labels (store-independent)


def _advect(field: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """One semi-Lagrangian advection step on a periodic grid (bilinear backtrace).

    ``out[y, x] = field[y - v, x - u]`` -- each cell pulls from where the flow came from.
    Periodic wrap keeps the field in-domain; the displacement is small (sub-cell) so 24
    steps move features a few cells -- enough that persistence is clearly wrong.
    """
    h, w = field.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    sy, sx = (ys - v) % h, (xs - u) % w
    fy, fx = np.floor(sy), np.floor(sx)
    wy, wx = sy - fy, sx - fx  # fractional part, always in [0, 1) -- robust to fp modulo == h
    y0, x0 = fy.astype(int) % h, fx.astype(int) % w
    y1, x1 = (y0 + 1) % h, (x0 + 1) % w
    return (
        field[y0, x0] * (1 - wy) * (1 - wx)
        + field[y0, x1] * (1 - wy) * wx
        + field[y1, x0] * wy * (1 - wx)
        + field[y1, x1] * wy * wx
    )


def make_advection_store(url: str, *, n_steps: int = 768, size: int = 48, seed: int = 0) -> None:
    """Write a synthetic 3-variable advected-field zarr (``t2m``, ``u10``, ``v10``).

    A smooth, time-*constant* deformation wind advects the temperature field; every hour is
    one sample. The process is kept *stationary* (advect, replenish small scales, renormalize
    -- see below) so every timestep has the same statistics and a contiguous split is honest.
    ``u10`` / ``v10`` are stored over time (broadcast) so all three arrays share the sample
    axis the engine batches along. Fields are ~unit-scale (no normalization needed). Chunked
    at 48 steps so there are several chunks to split/shuffle.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    # A smooth, spatially-varying deformation wind (a couple of low modes), ~0.2 cells/step
    # peak so the 24 h displacement is ~5 cells -- visible, within a small CNN's receptive
    # field, and varying enough that the model must *read* the wind, not memorize a shift.
    u = 0.20 * (np.sin(2 * np.pi * yy) * np.cos(2 * np.pi * xx))
    v = 0.20 * (np.cos(2 * np.pi * yy) * np.sin(2 * np.pi * xx))

    def smooth_field() -> np.ndarray:
        """A random unit-scale field of mid-frequency modes (a ~5-cell shift is visible)."""
        f = np.zeros((size, size))
        for _ in range(12):
            kx, ky = rng.integers(2, 6, size=2)
            f += rng.normal() * np.cos(2 * np.pi * (kx * xx + ky * yy) + rng.uniform(0, 2 * np.pi))
        return (f - f.mean()) / f.std()

    # A *stationary* forced-advection process: each step advects the field by the wind, then
    # adds a little fresh structure to replenish what bilinear interpolation dissipates, and
    # renormalizes. Every timestep then has the same statistics (so a contiguous split is
    # honest), and t2m(t+24) is the field advected 24 steps -- predictable from t2m(t) + the
    # wind (the model beats persistence by learning the displacement) up to the small forcing.
    field = smooth_field()
    t2m = np.empty((n_steps, size, size), dtype="f4")
    for t in range(n_steps):
        t2m[t] = field
        field = _advect(field, u, v) + 0.12 * smooth_field()
        field = (field - field.mean()) / field.std()

    wind_u = np.broadcast_to(u.astype("f4"), (n_steps, size, size))
    wind_v = np.broadcast_to(v.astype("f4"), (n_steps, size, size))

    ensure_local_dir(url)
    group = zarr.open_group(store=store_from_url(url, read_only=False), mode="w")
    for name, data in (("t2m", t2m), ("u10", wind_u), ("v10", wind_v)):
        arr = group.create_array(
            name,
            shape=data.shape,
            chunks=(48, size, size),
            dtype="f4",
            dimension_names=("time", "lat", "lon"),
        )
        arr[:] = data


def forecast_dataset(
    url: str,
    *,
    variables: tuple[str, str, str] = SYNTH_VARS,
    horizon: int = SYNTH_HORIZON,
    split: SplitName = SplitName.TRAIN,
    batch_size: int = 32,
    shuffle: bool = True,
    **store_kwargs: Any,
) -> InSituDataset:
    """One windowed dataset: inputs ``{t2m, u10, v10}@t`` + ``target = t2m@(t+horizon)``.

    ``variables`` names the three input arrays in the store (synthetic short names or the
    WeatherBench2 long names); the dataset's labels are always ``t2m, u10, v10, target`` so
    the model code is store-agnostic. The target is the first variable shifted by
    ``horizon`` -- two views of one array, no reshard. ``split_by_chunk`` partitions the
    time axis (contiguous, the time-series default).
    """
    opened = open_geometries(url, variables=list(variables), **store_kwargs)
    geoms = {label: opened[var] for label, var in zip(LABELS, variables, strict=True)}
    geoms["target"] = opened[variables[0]].shift(horizon)
    manifest = split_by_chunk(opened[variables[0]], fractions=(0.8, 0.1, 0.1))
    return InSituDataset(
        url,
        manifest,
        geometries=geoms,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        **store_kwargs,
    )


def inputs_and_targets(batch: Batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a windowed ``Batch`` into ``(x, persistence, target)`` numpy arrays.

    ``x`` is the stacked 3-channel input ``(B, 3, H, W)`` (``Batch.stack`` -- the Phase 3
    helper), ``persistence`` is ``t2m(t)`` ``(B, 1, H, W)`` (the no-skill forecast), and
    ``target`` is ``t2m(t+horizon)`` ``(B, 1, H, W)``. Models predict a tendency added to
    ``persistence``; the framework loops only ever touch these three.
    """
    x = batch.stack(["t2m", "u10", "v10"])  # (B, 3, H, W)
    persistence = batch.arrays["t2m"][:, None]  # (B, 1, H, W)
    target = batch.arrays["target"][:, None]  # (B, 1, H, W)
    return x, persistence, target


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def evaluate(ds: InSituDataset, predict: Callable[[Batch], np.ndarray]) -> tuple[float, float]:
    """24-hour-forecast skill on a held-out split: ``(model_rmse, persistence_rmse)``.

    ``predict`` maps a windowed ``Batch`` to the model's ``t2m(t+horizon)`` forecast
    ``(B, 1, H, W)`` (each framework supplies its own, so this stays framework-neutral).
    Persistence -- predict no change -- is the baseline a useful model must beat.
    """
    ds.set_epoch(0)
    preds, targets, persists = [], [], []
    for batch in ds:
        _x, persistence, target = inputs_and_targets(batch)
        preds.append(predict(batch))
        targets.append(target)
        persists.append(persistence)
    pred, target, persistence = (np.concatenate(a) for a in (preds, targets, persists))
    return rmse(pred, target), rmse(persistence, target)


def cli() -> argparse.Namespace:
    """Shared CLI for the three training files (only the model + loop differ)."""
    p = argparse.ArgumentParser(description="advected-field 24 h forecast")
    p.add_argument("--wb2", action="store_true", help="read the public WeatherBench2 ARCO store")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=768, help="synthetic trajectory length (hours)")
    p.add_argument("--url", default=None, help="synthetic store path (default: a temp file)")
    return p.parse_args()


def build_datasets(args: argparse.Namespace) -> tuple[InSituDataset, InSituDataset]:
    """``(train, val)`` forecast datasets from CLI args. Synthetic by default (written
    fresh -- deterministic, offline); ``--wb2`` reads the public WeatherBench2 store
    (anonymous ``gs://``, real ERA5, 6-hourly so the 24 h horizon is 4 steps)."""
    if args.wb2:
        url, variables, horizon, kw = WB2_URL, WB2_VARS, WB2_HORIZON, {"skip_signature": True}
    else:
        url = args.url or "file:///tmp/insitu_advection.zarr"
        make_advection_store(url, n_steps=args.n_steps)
        variables, horizon, kw = SYNTH_VARS, SYNTH_HORIZON, {}
    train = forecast_dataset(
        url,
        variables=variables,
        horizon=horizon,
        split=SplitName.TRAIN,
        batch_size=args.batch_size,
        shuffle=True,
        **kw,
    )
    val = forecast_dataset(
        url,
        variables=variables,
        horizon=horizon,
        split=SplitName.VAL,
        batch_size=args.batch_size,
        shuffle=False,
        **kw,
    )
    return train, val
