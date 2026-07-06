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
import time
from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import zarr
from zarr.abc.store import Store

from insitubatch import (
    Batch,
    InSituDataset,
    arraylake_store,
    ensure_local_dir,
    obstore_store,
    open_geometries,
    split_by_chunk,
)

# The public WeatherBench2 ERA5 (6-hourly) ARCO store the wb2_* examples use.
WB2_URL = (
    "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr"
)
WB2_VARS = ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind")
WB2_HORIZON = 4  # 4 steps x 6 h = 24 h

# The same data as an Arraylake/Icechunk repo (real ERA5; grouped by resolution).
ARRAYLAKE_REPO = "earthmover-public/weatherbench2"
ARRAYLAKE_GROUP = "era5/6h_240x121"

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


def make_advection_store(
    url: str,
    *,
    n_steps: int = 768,
    size: int = 48,
    sample_chunk: int = 48,
    inner_chunk: int | None = None,
    seed: int = 0,
    compress: bool = True,
    write_batch_mb: int = 256,
    write_concurrency: int = 32,
) -> None:
    """Write a synthetic 3-variable advected-field zarr (``t2m``, ``u10``, ``v10``).

    A smooth, time-*constant* deformation wind advects the temperature field; every hour is
    one sample. The process is kept *stationary* (advect, replenish small scales, renormalize
    -- see below) so every timestep has the same statistics and a contiguous split is honest.
    ``u10`` / ``v10`` are stored over time (broadcast) so all three arrays share the sample
    axis the engine batches along. Fields are ~unit-scale (no normalization needed). Chunked
    at ``sample_chunk`` steps so there are several chunks to split/shuffle.

    ``inner_chunk`` tiles the spatial dims into ``inner_chunk x inner_chunk`` stored chunks
    (default: one chunk covering the whole ``size x size`` field). Tiling makes one sample's
    field fan out into a grid of concurrent reads -- the ARCO/ERA5 norm, and the axis for
    sweeping inner fan-out vs the single-fat-chunk regime.

    Scales to a large cloud store (``url`` a ``gs://`` / ``s3://`` prefix) without
    materializing the whole array: the sequential advection is generated and written a
    **slab of whole chunks at a time**, bounding writer RAM to ``write_batch_mb`` while
    ``write_concurrency`` overlaps the chunk PUTs on zarr's async loop. ``size`` sets the
    spatial resolution; total volume is ``3 * n_steps * size**2 * 4`` bytes uncompressed.
    """
    ic = inner_chunk or size
    if not 1 <= ic <= size:
        raise ValueError(f"inner_chunk {ic} must be in 1..size ({size})")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    # A smooth, spatially-varying deformation wind (a couple of low modes), ~0.2 cells/step
    # peak so the 24 h displacement is ~5 cells -- visible, within a small CNN's receptive
    # field, and varying enough that the model must *read* the wind, not memorize a shift.
    u = 0.20 * (np.sin(2 * np.pi * yy) * np.cos(2 * np.pi * xx))
    v = 0.20 * (np.cos(2 * np.pi * yy) * np.sin(2 * np.pi * xx))
    wind_u, wind_v = u.astype("f4"), v.astype("f4")

    def smooth_field() -> np.ndarray:
        """A random unit-scale field of mid-frequency modes (a ~5-cell shift is visible)."""
        f = np.zeros((size, size))
        for _ in range(12):
            kx, ky = rng.integers(2, 6, size=2)
            f += rng.normal() * np.cos(2 * np.pi * (kx * xx + ky * yy) + rng.uniform(0, 2 * np.pi))
        return (f - f.mean()) / f.std()

    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    shape = (n_steps, size, size)
    chunks = (sample_chunk, ic, ic)
    compressors: Literal["auto"] | None = "auto" if compress else None
    arrays = {
        name: group.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype="f4",
            compressors=compressors,
            dimension_names=("time", "lat", "lon"),
        )
        for name in ("t2m", "u10", "v10")
    }

    # Slab = a whole number of sample-axis chunks fitting in write_batch_mb, so every slab
    # write is chunk-aligned (no read-modify-write) and the writer's RAM stays bounded.
    rows_that_fit = max(1, (write_batch_mb * 1_000_000) // (size * size * 4))
    slab_rows = max(sample_chunk, (rows_that_fit // sample_chunk) * sample_chunk)

    total_mb = 3 * n_steps * size * size * 4 / 1e6
    print(
        f"make_advection_store: {url}\n"
        f"  3 vars x ({n_steps}, {size}, {size})  chunks=({sample_chunk}, {ic}, {ic})  "
        f"~{total_mb:.0f} MB uncompressed  compress={'auto' if compress else 'none'}",
        flush=True,
    )
    t0 = time.perf_counter()

    # A *stationary* forced-advection process: each step advects the field by the wind, then
    # adds a little fresh structure to replenish what bilinear interpolation dissipates, and
    # renormalizes. Every timestep then has the same statistics (so a contiguous split is
    # honest), and t2m(t+24) is the field advected 24 steps -- predictable from t2m(t) + the
    # wind (the model beats persistence by learning the displacement) up to the small forcing.
    field = smooth_field()
    buf = np.empty((slab_rows, size, size), dtype="f4")
    with zarr.config.set({"async.concurrency": write_concurrency}):
        for start in range(0, n_steps, slab_rows):
            stop = min(start + slab_rows, n_steps)
            n = stop - start
            for i in range(n):
                buf[i] = field
                field = _advect(field, u, v) + 0.12 * smooth_field()
                field = (field - field.mean()) / field.std()
            arrays["t2m"][start:stop] = buf[:n]
            arrays["u10"][start:stop] = np.broadcast_to(wind_u, (n, size, size))
            arrays["v10"][start:stop] = np.broadcast_to(wind_v, (n, size, size))
            elapsed = time.perf_counter() - t0
            rate = stop / elapsed if elapsed else 0.0
            print(
                f"  wrote {stop}/{n_steps} steps ({stop / n_steps:.0%})  "
                f"{elapsed:.1f}s  {rate:.0f} steps/s",
                flush=True,
            )
    print(f"  done: {url} in {time.perf_counter() - t0:.1f}s", flush=True)


def _synthetic_ready(
    url: str, n_steps: int, size: int, sample_chunk: int, inner_chunk: int | None
) -> bool:
    """True if a synthetic store already exists at ``url`` with the requested geometry.

    Lets ``--source synthetic --url gs://...`` generate a large cloud store once and reuse
    it across training runs instead of rewriting it every time. A mismatch in shape
    (``--n-steps`` / ``--size``) *or chunking* (``--sample-chunk`` / ``--inner-chunk``), or any
    open failure (absent / partial), returns False, so a changed request regenerates rather
    than silently training on the stale store.
    """
    ic = inner_chunk or size
    try:
        t2m = zarr.open_group(store=obstore_store(url), mode="r")["t2m"]
        return (
            isinstance(t2m, zarr.Array)
            and t2m.shape == (n_steps, size, size)
            and t2m.chunks == (sample_chunk, ic, ic)
        )
    except Exception:
        return False


def forecast_dataset(
    store: Store,
    *,
    variables: tuple[str, str, str] = SYNTH_VARS,
    horizon: int = SYNTH_HORIZON,
    sample_range: tuple[int, int] | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    cache_dir: str | None = None,
    max_inflight: int | None = None,
) -> InSituDataset:
    """One windowed dataset: inputs ``{t2m, u10, v10}@t`` + ``target = t2m@(t+horizon)``.

    ``store`` is a zarr Store -- build it with :func:`~insitubatch.obstore_store` /
    :func:`~insitubatch.fsspec_store` / :func:`~insitubatch.arraylake_store`.
    ``variables`` names the three input arrays in the store (synthetic short names, the
    WeatherBench2 long names, or group-qualified paths); the dataset's labels are always
    ``t2m, u10, v10, target`` so the model code is store-agnostic. The target is the first
    variable shifted by ``horizon`` -- two views of one array, no reshard. ``sample_range``
    restricts the split to a finite time window (subset a multi-decade store).
    ``cache_dir`` spills the decoded-chunk cache to NVMe (fast cross-epoch reuse over a
    high-latency network); ``max_inflight`` throttles read-ahead. Iterate the returned
    dataset's ``.train`` / ``.val`` views.
    """
    opened = open_geometries(store, variables=list(variables))
    geoms = {label: opened[var] for label, var in zip(LABELS, variables, strict=True)}
    geoms["target"] = opened[variables[0]].shift(horizon)
    manifest = split_by_chunk(
        opened[variables[0]], fractions=(0.8, 0.1, 0.1), sample_range=sample_range
    )
    return InSituDataset(
        store,
        manifest,
        geometries=geoms,
        batch_size=batch_size,
        shuffle=shuffle,
        cache_dir=cache_dir,
        max_inflight=max_inflight,
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


def evaluate(view: Iterable[Batch], predict: Callable[[Batch], np.ndarray]) -> tuple[float, float]:
    """24-hour-forecast skill on a held-out split view (e.g. ``ds.val``): ``(model_rmse,
    persistence_rmse)``.

    ``predict`` maps a windowed ``Batch`` to the model's ``t2m(t+horizon)`` forecast
    ``(B, 1, H, W)`` (each framework supplies its own, so this stays framework-neutral).
    Persistence -- predict no change -- is the baseline a useful model must beat. The view
    is deterministic (eval splits don't shuffle), so no epoch is set.
    """
    preds, targets, persists = [], [], []
    for batch in view:
        _x, persistence, target = inputs_and_targets(batch)
        preds.append(predict(batch))
        targets.append(target)
        persists.append(persistence)
    pred, target, persistence = (np.concatenate(a) for a in (preds, targets, persists))
    return rmse(pred, target), rmse(persistence, target)


def _range(s: str) -> tuple[int, int]:
    start, stop = (int(x) for x in s.split(","))
    return (start, stop)


def build_parser() -> argparse.ArgumentParser:
    """The shared example CLI (source, device, dataset geometry). The three training files
    parse it via :func:`cli`; ``train_torch_metrics.py`` extends it with its benchmark flags."""
    p = argparse.ArgumentParser(description="advected-field 24 h forecast")
    p.add_argument(
        "--source",
        choices=("synthetic", "wb2", "arraylake"),
        default="synthetic",
        help="offline synthetic | WeatherBench2 gs:// | Arraylake/Icechunk (real ERA5)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda -- the train loop moves tensors there",
    )
    p.add_argument(
        "--sample-range",
        type=_range,
        default=None,
        metavar="START,STOP",
        help="finite training window on the time axis (subset a multi-decade real store)",
    )
    p.add_argument("--repo", default=ARRAYLAKE_REPO, help="Arraylake repo (--source arraylake)")
    p.add_argument("--group", default=ARRAYLAKE_GROUP, help="resolution group (--source arraylake)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=768, help="synthetic trajectory length (hours)")
    p.add_argument("--size", type=int, default=48, help="synthetic spatial resolution (NxN grid)")
    p.add_argument(
        "--sample-chunk", type=int, default=48, help="synthetic sample-axis chunk length (steps)"
    )
    p.add_argument(
        "--inner-chunk",
        type=int,
        default=None,
        help="synthetic inner (spatial) chunk edge; default one chunk over the whole field",
    )
    p.add_argument(
        "--url",
        default=None,
        help="synthetic store path (file:// temp default; gs://... for a cloud store)",
    )
    p.add_argument(
        "--regenerate",
        action="store_true",
        help="force-rewrite the synthetic store even if one with the same geometry exists",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="spill the decoded-chunk cache here (e.g. NVMe) for cross-epoch reuse",
    )
    p.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="throttle read-ahead depth (lower = lower cold TTFB over a high-latency network)",
    )
    return p


def cli() -> argparse.Namespace:
    """Parse the shared CLI for the three canonical training files."""
    return build_parser().parse_args()


def build_datasets(args: argparse.Namespace) -> InSituDataset:
    """One forecast dataset from CLI args -- iterate ``ds.train`` / ``ds.val``. ``--source``
    picks offline synthetic (written fresh), the public WeatherBench2 ``gs://`` store, or the
    Arraylake/Icechunk repo (real ERA5; pass ``--sample-range`` to subset the archive)."""
    store: Store
    if args.source == "arraylake":
        store = arraylake_store(args.repo)
        a, b, c = WB2_VARS
        variables = (f"{args.group}/{a}", f"{args.group}/{b}", f"{args.group}/{c}")
        horizon = WB2_HORIZON
    elif args.source == "wb2":
        store = obstore_store(WB2_URL, skip_signature=True)  # anonymous read of the public bucket
        variables, horizon = WB2_VARS, WB2_HORIZON
    else:
        url = args.url or "file:///tmp/insitu_advection.zarr"
        if args.regenerate or not _synthetic_ready(
            url, args.n_steps, args.size, args.sample_chunk, args.inner_chunk
        ):
            make_advection_store(
                url,
                n_steps=args.n_steps,
                size=args.size,
                sample_chunk=args.sample_chunk,
                inner_chunk=args.inner_chunk,
            )
        store = obstore_store(url)
        variables, horizon = SYNTH_VARS, SYNTH_HORIZON
    return forecast_dataset(
        store,
        variables=variables,
        horizon=horizon,
        sample_range=args.sample_range,
        batch_size=args.batch_size,
        shuffle=True,
        cache_dir=args.cache_dir,
        max_inflight=args.max_inflight,
    )
