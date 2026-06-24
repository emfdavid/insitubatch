"""Earthmover's ``dataloader-demo`` stack — xbatcher + torch DataLoader — with a
focus on **cold-start latency** and how to cut it.

xbatcher (from the Earthmover / pangeo community) is the domain-standard way to *define*
ndim batches; this runs their published demo as-is — ``xarray`` + ``xbatcher`` for the
batch definition, fed to a torch ``DataLoader`` with ``num_workers`` worker processes
(https://github.com/earth-mover/dataloader-demo) — and uses it to study where the
worker-process model spends startup time (useful whichever loader you ship). The
companion ``examples/wb2_dataloader.py`` runs the *same task* on insitubatch's
single-event-loop engine; the two are complementary engines for the same problem, and
pairing them makes the cold-start trade-off concrete.

**Why this script exists: inference startup.** Training amortizes worker spin-up
over many epochs, so it barely shows. Inference usually does *not* — you make one
pass from a cold loader, and a long-lived server keeping a DataLoader open (with
its pinned workers and file handles) is rare. So time-to-first-batch (TTFB) is a
real cost, and the worker start method is the lever:

- ``spawn`` — a fresh interpreter per worker; re-imports torch/xarray/zarr each
  time. Safe, slowest startup.
- ``forkserver`` — fork each worker from a pristine server (no inherited obstore
  tokio runtime; safe). Cheaper than spawn.
- ``forkserver-preload`` — same, but the server imports the heavy modules **once**
  (``set_forkserver_preload``); forked workers skip the re-import. Lowest safe TTFB.
- ``fork`` — fastest startup, but unsafe with async cloud stores and so **excluded
  from ``--compare``**. fork copies only the calling thread, so a store opened in
  the parent (here ``gcsfs``/aiohttp; equally obstore/tokio) is inherited bound to
  the parent's now-dead event loop — the first worker read raises ``got Future
  attached to a different loop`` (obstore deadlocks instead). ``--mp fork`` still
  lets you reproduce the failure on purpose.

insitubatch's single in-process event loop sidesteps this entirely — no fork, nothing
to relaunch — so it starts fast from cold; this script exists to make the trade-off
visible (and to show how ``forkserver-preload`` already cuts the worker-stack TTFB a
lot) so you can pick what fits your workload.

    uv run python -m examples.wb2_xbatcher --wb2 --compare --max-batches 100  # teaching table
    uv run python -m examples.wb2_xbatcher --wb2 --max-batches 100  # one run, default regime
    uv run python -m examples.wb2_xbatcher  # tiny synthetic data, no network
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from multiprocessing.context import BaseContext
from typing import Any, cast

import numpy as np

# The public WeatherBench2 ERA5 used by the Earthmover demo (zarr v2, consolidated).
WB2_URL = (
    "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr"
)
WB2_VAR = "2m_temperature"

# Preloaded once in the forkserver; forked workers inherit these already-imported.
# Heavy *pure* imports only — never obstore: importing it is fine, but a preload runs
# in the server BEFORE forking, and any obstore *operation* there would start the
# tokio runtime that fork can't inherit.
PRELOAD = ["torch", "xarray", "zarr", "numcodecs", "xbatcher", "numpy", "pandas"]

# The regimes --compare runs. fork is excluded: unsafe with async cloud stores (see above).
COMPARE_MODES = ["spawn", "forkserver", "forkserver-preload"]


def _storage_options(url: str, request_payer: bool) -> dict:
    if url.startswith("gs://"):
        return {"token": "anon"}  # public bucket, no credentials
    if url.startswith("s3://") and request_payer:
        return {"requester_pays": True}
    return {}


def _mp_context(mode: str, num_workers: int) -> str | BaseContext | None:
    """Resolve a DataLoader multiprocessing_context for a named regime."""
    if not num_workers or mode == "fork":
        return None if mode != "fork" else "fork"
    if mode == "spawn":
        return "spawn"
    if mode in ("forkserver", "forkserver-preload"):
        ctx = mp.get_context("forkserver")
        if mode == "forkserver-preload":
            ctx.set_forkserver_preload(PRELOAD)
        return ctx
    raise ValueError(f"unknown mp mode: {mode}")


def _synthetic(tmp: str, *, n: int = 240, lat: int = 32, lon: int = 32) -> tuple[str, str]:
    import xarray as xr

    path = f"{tmp}/era5.zarr"
    data = np.random.default_rng(0).standard_normal((n, lat, lon)).astype("f4")
    ds = xr.Dataset({WB2_VAR: (("time", "lat", "lon"), data)})
    # v2 + consolidated, like the real WeatherBench2 store (and what open_zarr expects).
    ds.to_zarr(path, mode="w", zarr_format=2, consolidated=True)
    return path, WB2_VAR


class _CenterCrop:
    """Top-level (picklable) wrapper: center-crop each xbatcher sample to (h, w) and
    drop the singleton time axis, so a batch collates to (batch, h, w)."""

    def __init__(self, base: Any, h: int, w: int) -> None:  # base: xbatcher MapDataset
        self.base = base
        self.h = h
        self.w = w

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Any:  # torch tensor from the xbatcher MapDataset
        x = self.base[i]  # (1, LAT, LON) tensor from xbatcher MapDataset
        x = x.reshape(x.shape[-2], x.shape[-1])
        lat, lon = x.shape
        top = max((lat - self.h) // 2, 0)
        left = max((lon - self.w) // 2, 0)
        return x[top : top + self.h, left : left + self.w]


def run_xbatcher_demo(
    *,
    url: str | None = None,
    var: str = WB2_VAR,
    subregion: tuple[int, int] = (16, 16),
    batch_size: int = 32,
    num_workers: int = 8,
    mp_mode: str = "auto",
    num_epochs: int = 1,
    max_batches: int = 0,
    shuffle: bool = True,
    request_payer: bool = False,
    verbose: bool = True,
) -> dict:
    import xarray as xr
    import xbatcher
    from torch.utils.data import DataLoader
    from xbatcher.loaders.torch import MapDataset

    if mp_mode == "auto":
        mp_mode = "forkserver-preload" if sys.platform != "darwin" else "spawn"

    tmp = None
    if url is None:
        tmp = tempfile.mkdtemp(prefix="wb2-xb-")
        url, var = _synthetic(tmp)

    # None (not {}) for local paths: zarr rejects unused storage_options.
    da = xr.open_zarr(url, storage_options=_storage_options(url, request_payer) or None)[var]
    h, w = subregion
    input_dims = {da.dims[0]: 1, **{d: int(da.sizes[d]) for d in da.dims[1:]}}
    base = _CenterCrop(MapDataset(xbatcher.BatchGenerator(da, input_dims=input_dims)), h, w)

    waits: list[float] = []
    total = 0
    sample_shape: tuple[int, ...] = ()
    t_start = time.perf_counter()
    for _ in range(num_epochs):
        # Fresh loader each epoch: this measures *cold start*, the inference-relevant
        # cost (the bench/ suite uses persistent workers to measure steady state).
        # base (_CenterCrop) is map-style (len/getitem) -- DataLoader accepts it at runtime;
        # cast(Any) satisfies the stub (which wants a Dataset subclass) without a
        # `type: ignore` that flips to "unused" when torch (its stub) isn't installed.
        loader: DataLoader = DataLoader(
            cast(Any, base),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            persistent_workers=False,
            prefetch_factor=(2 if num_workers else None),
            multiprocessing_context=_mp_context(mp_mode, num_workers),
        )
        it: Iterator = iter(loader)
        t_prev = time.perf_counter()
        for i, batch in enumerate(it):
            now = time.perf_counter()
            waits.append(now - t_prev)
            total += batch.shape[0]
            sample_shape = tuple(batch.shape[1:])
            if verbose:
                print(f"  batch {i}: {tuple(batch.shape)}  {1e3 * (now - t_prev):.1f} ms")
            t_prev = now
            if max_batches and i + 1 >= max_batches:
                break
        del loader, it
    dt = time.perf_counter() - t_start

    summary = {
        "regime": f"xbatcher {mp_mode}",
        "num_workers": num_workers,
        "samples": total,
        "sample_shape": sample_shape,
        "ttfb_ms": 1e3 * waits[0] if waits else 0.0,  # cold-start time-to-first-batch
        "seconds": dt,
        "samples_per_s": total / dt if dt else 0.0,
    }
    if verbose:
        print(f"\nsummary: {summary}")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return summary


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'regime':<28}{'workers':>8}{'ttfb_ms':>10}{'samples/s':>12}{'wall_s':>9}")
    for r in rows:
        print(
            f"{r['regime']:<28}{r['num_workers']:>8}{r['ttfb_ms']:>10.1f}"
            f"{r['samples_per_s']:>12.0f}{r['seconds']:>9.2f}"
        )


def _subregion(s: str) -> tuple[int, int]:
    h, w = (int(x) for x in s.split(","))
    return (h, w)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--url", default=None, help="zarr URL (gs://, s3://, local); default = synthetic"
    )
    p.add_argument(
        "--wb2", action="store_true", help="use the public WeatherBench2 ERA5 (sets --url/--var)"
    )
    p.add_argument("--var", default=WB2_VAR)
    p.add_argument("--subregion", type=_subregion, default=(16, 16), help="center-crop H,W")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--mp",
        dest="mp_mode",
        choices=["auto", "spawn", "forkserver", "forkserver-preload", "fork"],
        default="auto",
        help="worker start regime (default: forkserver-preload on Linux, spawn on macOS)",
    )
    p.add_argument("--compare", action="store_true", help="run all safe regimes and print a table")
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=0, help="cap batches per epoch (0 = all)")
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--request-payer", action="store_true")
    a = p.parse_args()

    url, var = (WB2_URL, WB2_VAR) if a.wb2 else (a.url, a.var)
    common = dict(
        url=url,
        var=var,
        subregion=a.subregion,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        num_epochs=a.num_epochs,
        max_batches=a.max_batches,
        shuffle=not a.no_shuffle,
        request_payer=a.request_payer,
    )

    if a.compare:
        rows = []
        for mode in COMPARE_MODES:  # fork excluded: unsafe with async cloud stores (see --mp fork)
            print(f"\n=== {mode} ===")
            try:
                rows.append(run_xbatcher_demo(mp_mode=mode, verbose=False, **common))
            except Exception as e:  # noqa: BLE001 - report and keep comparing other regimes
                print(f"  {mode} failed: {type(e).__name__}: {e}")
        _print_table(rows)
    else:
        run_xbatcher_demo(mp_mode=a.mp_mode, **common)


if __name__ == "__main__":
    main()
