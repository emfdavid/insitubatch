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
import itertools
import math
import shutil
import statistics
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import obstore
import zarr

from insitubatch import SplitName, open_geometries, split_by_chunk
from insitubatch.source import InSituDataset
from insitubatch.store import store_from_url

from .make_dataset import make_dataset


def _store_kwargs(url: str, anon: bool, request_payer: bool) -> dict:
    kw: dict = {}
    if anon or url.startswith("gs://"):
        kw["skip_signature"] = True
    if request_payer:
        kw["request_payer"] = True
    return kw


def _stats(fn: Callable[[], float], repeats: int) -> tuple[float, float, float]:
    """Run ``fn`` ``repeats`` times; return (median, min, max). Cloud reads of a
    small (cold) sample are noisy, so a single number lies -- report the spread."""
    xs = sorted(fn() for _ in range(repeats))
    return statistics.median(xs), xs[0], xs[-1]


def _insitu(
    url, var, kw, *, decode_threads, max_chunks, block_chunks=2, max_inflight=32
) -> tuple[float, int]:
    """One insitu pass over the first ``max_chunks`` chunks; (MB/s, peak resident chunks).

    ``block_chunks`` sets the shuffle window / residency; ``max_inflight`` is the
    single network-concurrency dial (V2 -- no nested ``read_concurrency`` cap).
    """
    geom = open_geometries(url, variables=[var], **kw)[var]
    manifest = split_by_chunk(geom, fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        url,
        manifest,
        geometries={var: geom},
        split=SplitName.TRAIN,
        batch_size=16,
        block_chunks=block_chunks,
        max_inflight=max_inflight,
        to_tensor=False,
        shuffle=False,
        **kw,
    )
    ds.scheduler_config.decode_threads = decode_threads  # read by the scheduler at iteration time
    bps = int(np.prod(geom.inner_shape)) * 4
    limit = max_chunks * geom.sample_chunk_size
    n = 0
    t = time.perf_counter()
    for b in ds:
        n += b.arrays[var].shape[0]
        if n >= limit:
            break
    dt = time.perf_counter() - t
    return n * bps / 1e6 / dt, ds.resident_peak


def _raw_get_mb_s(url, var, kw, concurrency, max_chunks) -> float:
    """Fetch raw (still-encoded) chunk objects via obstore — no decode.

    Enumerates the *real* stored-chunk grid (outer x inner) from the array's
    chunks, so it's correct for spatially-chunked arrays (not just single inner
    chunk). max_chunks bounds the number of OUTER chunks; every inner chunk under
    them is fetched.
    """
    arr = zarr.open_array(store=store_from_url(url, **kw), path=var, mode="r")  # sync; for chunks
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
    p.add_argument("--concurrency", default="1,4,8,16,32", help="sec 2 raw-GET sweep")
    p.add_argument("--no-raw", action="store_true", help="skip sec 2 (raw GET re-reads the data)")
    p.add_argument("--anon", action="store_true", help="anonymous (public gs:// / s3://)")
    p.add_argument("--request-payer", action="store_true")
    a = p.parse_args()

    tmp = None
    if a.url is None:
        tmp = tempfile.mkdtemp(prefix="probe-")
        atexit.register(shutil.rmtree, tmp, ignore_errors=True)  # don't leak GBs to /tmp
        a.url = f"file://{tmp}/era5.zarr"
        a.var = "t2m"
        make_dataset(a.url, n_samples=512, inner=(721, 1440), sample_chunk=8, variables=["t2m"])
    kw = _store_kwargs(a.url, a.anon, a.request_payer)
    print(f"probe {a.url}  var={a.var}  first {a.max_chunks} chunks\n")

    def show(label: str, fn: Callable[[], float]) -> None:
        med, lo, hi = _stats(fn, a.repeats)
        print(f"   {label}: {med:8.1f} MB/s  ({lo:.0f}-{hi:.0f})")

    bc = a.block_chunks
    print(f"1) insitu MB/s vs decode_threads (median of {a.repeats}, block_chunks={bc}):")
    for dt in (int(x) for x in a.decode_threads.split(",")):
        show(
            f"decode_threads={dt or 'auto':>4}",
            lambda dt=dt: _insitu(
                a.url, a.var, kw, decode_threads=dt, max_chunks=a.max_chunks, block_chunks=bc
            )[0],
        )

    print(f"\n1b) insitu MB/s + peak residency vs max_inflight (block_chunks={bc}, decode auto):")
    print("    V2 wants throughput flat (not falling) past the knee, residency pinned.")
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
            )
            for _ in range(a.repeats)
        )
        med = statistics.median(x[0] for x in xs)
        resident = xs[0][1]  # deterministic across repeats
        print(
            f"   max_inflight={mi:>4}: {med:8.1f} MB/s  ({xs[0][0]:.0f}-{xs[-1][0]:.0f})"
            f"  resident={resident} chunks"
        )

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
                lambda u=u: _insitu(u, "t2m", {}, decode_threads=0, max_chunks=a.max_chunks)[0],
            )


if __name__ == "__main__":
    main()
