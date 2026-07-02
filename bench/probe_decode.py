"""Probe: is the read pipeline network-bound or decode-bound, and does V2 decouple?

    uv run python -m bench.probe_decode                              # synthetic file://
    uv run python -m bench.probe_decode --url s3://bucket/era5_c8.zarr --var t2m
    uv run python -m bench.probe_decode --url gs://.../era5.zarr --var 2m_temperature --anon

Measurements over the first ``--max-chunks`` chunks (quick on a 25 GB store):

  1. insitu throughput vs ``decode_threads`` (1,2,4,8,auto) — where decode parallelism
     saturates; flat past N cores => decode isn't the limit.
  1b. **The V2 decoupling headline**: insitu throughput + peak residency vs
     ``max_inflight``, with ``block_chunks`` fixed small. V2 dials network
     concurrency with ``max_inflight`` alone; throughput should rise to the network
     knee and then stay *flat* (not fall, as v1's nested caps did when oversubscribed)
     while residency stays pinned at ``2*block_chunks`` — independent of concurrency.
  2. raw obstore concurrent GET MB/s vs concurrency (1,4,8,16,32) — pure fetch, no
     decode.
  3. (synthetic only) insitu compressed vs uncompressed — the codec's share.

Diagnosis: if raw GET (2) far exceeds insitu (1/1b), we're decode/loop-limited and
``max_inflight`` / the decode pool is the lever; if raw GET also caps near insitu,
it's the network/endpoint (more/bigger parallel streams, in-region S3, the gateway).
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import itertools
import math
import shutil
import statistics
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import numpy as np
import obstore
import zarr

from insitubatch import SplitManifest, SplitName, open_geometries, split_by_chunk
from insitubatch.source import InSituDataset
from insitubatch.store import obstore_store

from ._profile import record_pyspy
from .make_dataset import make_dataset


def _store_kwargs(url: str, anon: bool, request_payer: bool, s3_express: bool) -> dict:
    kw: dict = {}
    if anon or url.startswith("gs://"):
        kw["skip_signature"] = True
    if request_payer:
        kw["request_payer"] = True
    if s3_express:  # S3 Express One Zone directory bucket (--x-s3); not inferred from the name
        kw["s3_express"] = True
    return kw


def _stats(fn: Callable[[], float], repeats: int) -> tuple[float, float, float]:
    """Run ``fn`` ``repeats`` times; return (median, min, max). Cloud reads of a
    small (cold) sample are noisy, so a single number lies -- report the spread."""
    xs = sorted(fn() for _ in range(repeats))
    return statistics.median(xs), xs[0], xs[-1]


def _insitu(
    url: str,
    var: str,
    kw: dict[str, Any],
    *,
    decode_threads: int,
    max_chunks: int,
    block_chunks: int = 2,
    max_inflight: int = 32,
    window: int = 0,
) -> tuple[float, int]:
    """One insitu pass over the first ``max_chunks`` chunks; (MB/s, peak resident chunks).

    ``block_chunks`` sets the shuffle window / residency; ``max_inflight`` is the
    single network-concurrency dial (V2 -- no nested ``read_concurrency`` cap).

    ``window`` adds a forecast view-set: the anchor input plus ``window`` shifted
    targets (``geom.shift(1..window)``) read from the same array. With ``window=0`` it
    is the plain single-variable baseline; ``window>0`` exercises the offset gather,
    per-block read-union, and *refcounted* residency that plain reads do not -- the cost
    most exposed at the GRIB end (one sample/chunk = max chunk-rate = max pin/unpin/lock
    churn) and under free-threading. MB/s counts the anchor-input bytes either way, so
    the windowing machinery's overhead shows as a drop at equal anchor rate.
    """
    store = obstore_store(url, **kw)
    geom = open_geometries(store, variables=[var])[var]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    geoms = {var: geom}
    for k in range(1, window + 1):
        geoms[f"{var}_t{k}"] = geom.shift(k)
    ds = InSituDataset(
        store,
        manifest,
        geometries=geoms,
        batch_size=16,
        block_chunks=block_chunks,
        max_inflight=max_inflight,
        shuffle=False,
    )
    ds.scheduler_config.decode_threads = decode_threads  # read by the scheduler at iteration time
    bps = int(np.prod(geom.inner_shape)) * 4
    limit = max_chunks * geom.sample_chunk_size
    n = 0
    t = time.perf_counter()
    for b in ds.train:
        n += b.arrays[var].shape[0]
        if n >= limit:
            break
    dt = time.perf_counter() - t
    return n * bps / 1e6 / dt, ds.resident_peak


def _insitu_mb(url: str, var: str, kw: dict[str, Any], **opts: int) -> float:
    """Just the MB/s of an :func:`_insitu` pass -- a typed target for ``partial`` in
    the ``show`` sweeps (a ``lambda x=x:`` loop-capture defeats mypy inference)."""
    return _insitu(url, var, kw, **opts)[0]


def _insitu_cache(
    url: str,
    var: str,
    kw: dict[str, Any],
    *,
    max_chunks: int,
    block_chunks: int,
    cache_dir: str,
) -> tuple[float, float]:
    """Two epochs over the first ``max_chunks`` chunks with a budget that holds them
    all; returns (cold MB/s, warm MB/s). The manifest is restricted to exactly those
    chunks so read-ahead can't LRU-evict the early ones before epoch 1 -- which is
    then served entirely from the cache (no S3, no decode). ``cache_dir`` spills the
    slots to NVMe (mmap); pass it to keep heap bounded on fat data.
    """
    store = obstore_store(url, **kw)
    geom = open_geometries(store, variables=[var])[var]
    full = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    train = full.chunks[SplitName.TRAIN.value][:max_chunks]
    manifest = SplitManifest(
        n_chunks=full.n_chunks,
        sample_chunk_size=full.sample_chunk_size,
        n_samples=full.n_samples,
        chunks={"train": train, "val": [], "test": []},
        seed=full.seed,
    )
    outer_nbytes = geom.sample_chunk_size * int(np.prod(geom.inner_shape)) * geom.dtype.itemsize
    budget = (len(train) + 2) * outer_nbytes  # hold every probed chunk + a margin
    ds = InSituDataset(
        store,
        manifest,
        geometries={var: geom},
        batch_size=16,
        block_chunks=block_chunks,
        shuffle=False,
        cache_dir=cache_dir,
        cache_budget_bytes=budget,
    )
    bps = int(np.prod(geom.inner_shape)) * 4

    def epoch_mb(epoch: int) -> float:
        ds.set_epoch(epoch)
        n = 0
        t = time.perf_counter()
        for b in ds.train:
            n += b.arrays[var].shape[0]
        return n * bps / 1e6 / (time.perf_counter() - t)

    cold, warm = epoch_mb(0), epoch_mb(1)
    ds.close()
    return cold, warm


def _raw_get_mb_s(
    url: str, var: str, kw: dict[str, Any], concurrency: int, max_chunks: int
) -> float:
    """Fetch raw (still-encoded) chunk objects via obstore — no decode.

    Enumerates the *real* stored-chunk grid (outer x inner) from the array's
    chunks, so it's correct for spatially-chunked arrays (not just single inner
    chunk). max_chunks bounds the number of OUTER chunks; every inner chunk under
    them is fetched.
    """
    arr = zarr.open_array(store=obstore_store(url, **kw), path=var, mode="r")  # sync; for chunks
    obs = obstore.store.from_url(url, **kw)
    n_outer = min(max_chunks, math.ceil(arr.shape[0] / arr.chunks[0]))
    inner_ranges = [
        range(math.ceil(s / c)) for s, c in zip(arr.shape[1:], arr.chunks[1:], strict=True)
    ]
    keys = [
        f"{var}/c/{oi}/" + "/".join(map(str, inner))
        for oi in range(n_outer)
        for inner in itertools.product(*inner_ranges)
    ]

    def fetch(key: str) -> int:
        return len(bytes(obstore.get(obs, key).bytes()))

    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        total = sum(ex.map(fetch, keys))
    return total / 1e6 / (time.perf_counter() - t)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default=None, help="zarr URL; default = synthetic file://")
    p.add_argument("--var", default="t2m")
    p.add_argument("--max-chunks", type=int, default=64, help="chunks to probe (bounds the cost)")
    p.add_argument("--repeats", type=int, default=3, help="runs per point; report median (min-max)")
    p.add_argument("--decode-threads", default="1,2,4,8,0", help="sec 1 sweep (0=auto)")
    p.add_argument(
        "--block-chunks",
        type=int,
        default=2,
        help="fixed shuffle window / residency for the max_inflight sweep (sec 1b)",
    )
    p.add_argument("--max-inflight", default="8,16,32,64", help="sec 1b sweep (the V2 dial)")
    p.add_argument(
        "--window",
        type=int,
        default=0,
        help="forecast views: anchor input + N shifted targets (0 = plain single var). "
        "Exercises the windowed offset-gather + refcounted residency; pair with a c1 "
        "store (max chunk-rate) and/or PYTHON_GIL=0 to surface any lock serialization.",
    )
    p.add_argument("--concurrency", default="1,4,8,16,32", help="sec 2 raw-GET sweep")
    p.add_argument(
        "--profile",
        nargs="?",
        const="probe-profile.svg",
        default=None,
        help="record a py-spy --native flamegraph of the sec 1b sweep to PATH "
        "(default probe-profile.svg; .json => speedscope). Needs ptrace_scope=0 "
        "or sudo; see bench/_profile.py.",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="run the cross-epoch cache test (sec 1c): cold vs cached epoch, mmap slots here",
    )
    p.add_argument("--no-raw", action="store_true", help="skip sec 2 (raw GET re-reads the data)")
    p.add_argument(
        "--no-decode-sweep",
        action="store_true",
        help="skip sec 1 (the decode_threads sweep) -- a one-time decode-saturation "
        "check; redundant when you only want the sec-1b max_inflight sawtooth",
    )
    p.add_argument(
        "--no-warm",
        action="store_true",
        help="skip the throwaway prewarm burst (warms obstore's TLS pool + the S3 "
        "per-prefix request-rate ramp so the first sweep point isn't cold)",
    )
    p.add_argument("--anon", action="store_true", help="anonymous (public gs:// / s3://)")
    p.add_argument("--request-payer", action="store_true")
    p.add_argument(
        "--s3-express", action="store_true", help="target an S3 Express One Zone directory bucket"
    )
    a = p.parse_args()

    tmp = None
    if a.url is None:
        tmp = tempfile.mkdtemp(prefix="probe-")
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)  # don't leak GBs to /tmp
        a.url = f"file://{tmp}/era5.zarr"
        a.var = "t2m"
        make_dataset(a.url, n_samples=512, inner=(721, 1440), sample_chunk=8, variables=["t2m"])
    kw = _store_kwargs(a.url, a.anon, a.request_payer, a.s3_express)
    print(f"probe {a.url}  var={a.var}  first {a.max_chunks} chunks\n")

    def show(label: str, fn: Callable[[], float]) -> None:
        med, lo, hi = _stats(fn, a.repeats)
        print(f"   {label}: {med:8.1f} MB/s  ({lo:.0f}-{hi:.0f})")

    bc = a.block_chunks

    # Prewarm: a cold S3 prefix is rate-limited and obstore's TLS pool is empty, so
    # the first sweep point otherwise eats the ramp-up and reads at ~1 stream. One
    # throwaway burst at the top fixes it for every section below (and makes 1c's
    # "cold" epoch S3-warm/pool-cold, isolating decode+pool reuse from the S3 ramp).
    if not a.no_warm:
        mi_warm = max(int(x) for x in a.max_inflight.split(","))
        print(f"0) prewarm: {a.max_chunks} chunks @ max_inflight={mi_warm} (discarded) ...")
        try:
            _insitu(
                a.url,
                a.var,
                kw,
                decode_threads=0,
                max_chunks=a.max_chunks,
                block_chunks=bc,
                max_inflight=mi_warm,
                window=a.window,
            )
        except Exception as exc:  # noqa: BLE001 - prewarm is best-effort
            print(f"   prewarm failed: {type(exc).__name__}: {exc}")
        print()

    if a.window:
        print(f"(windowed: anchor input + {a.window} shifted target view(s); MB/s = input bytes)\n")

    if not a.no_decode_sweep:
        print(f"1) insitu MB/s vs decode_threads (median of {a.repeats}, block_chunks={bc}):")
        for dt in (int(x) for x in a.decode_threads.split(",")):
            show(
                f"decode_threads={dt or 'auto':>4}",
                partial(
                    _insitu_mb,
                    a.url,
                    a.var,
                    kw,
                    decode_threads=dt,
                    max_chunks=a.max_chunks,
                    block_chunks=bc,
                    window=a.window,
                ),
            )

    print(f"\n1b) insitu MB/s + peak residency vs max_inflight (block_chunks={bc}, decode auto):")
    print("    V2 wants throughput flat (not falling) past the knee, residency pinned.")
    profile = record_pyspy(a.profile) if a.profile else contextlib.nullcontext()
    with profile:  # profile scope = the V2 acceptance sweep (insitu only, no raw-GET)
        for mi in (int(x) for x in a.max_inflight.split(",")):
            xs = sorted(
                _insitu(
                    a.url,
                    a.var,
                    kw,
                    decode_threads=0,
                    max_chunks=a.max_chunks,
                    block_chunks=bc,
                    max_inflight=mi,
                    window=a.window,
                )
                for _ in range(a.repeats)
            )
            med = statistics.median(x[0] for x in xs)
            resident = xs[0][1]  # deterministic across repeats
            print(
                f"   max_inflight={mi:>4}: {med:8.1f} MB/s  ({xs[0][0]:.0f}-{xs[-1][0]:.0f})"
                f"  resident={resident} chunks"
            )

    if a.cache_dir:
        print(f"\n1c) cross-epoch cache ({a.max_chunks} chunks, budget holds all, mmap):")
        cold, warm = _insitu_cache(
            a.url, a.var, kw, max_chunks=a.max_chunks, block_chunks=bc, cache_dir=a.cache_dir
        )
        print(f"   epoch 0 (cold):   {cold:8.1f} MB/s")
        print(f"   epoch 1 (cached): {warm:8.1f} MB/s   ({warm / cold:.1f}x cold)")

    if not a.no_raw:
        print("\n2) raw obstore concurrent GET MB/s (no decode):")
        for c in (int(x) for x in a.concurrency.split(",")):
            show(f"concurrency={c:>2}", partial(_raw_get_mb_s, a.url, a.var, kw, c, a.max_chunks))

    if tmp:
        none_url = f"file://{tmp}/era5_none.zarr"
        make_dataset(
            none_url,
            n_samples=512,
            inner=(721, 1440),
            sample_chunk=8,
            variables=["t2m"],
            compress=False,
        )
        print("\n3) codec cost (synthetic, compressed vs uncompressed, auto decode_threads):")
        for label, u in (("compressed", a.url), ("uncompressed", none_url)):
            show(
                f"{label:12}",
                partial(_insitu_mb, u, "t2m", {}, decode_threads=0, max_chunks=a.max_chunks),
            )


if __name__ == "__main__":
    main()
