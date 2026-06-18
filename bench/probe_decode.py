"""Probe: is the read pipeline network-bound or decode-bound?

    uv run python -m bench.probe_decode                              # synthetic file://
    uv run python -m bench.probe_decode --url s3://bucket/era5_c8.zarr --var t2m
    uv run python -m bench.probe_decode --url gs://.../era5.zarr --var 2m_temperature --anon

Three measurements over the first ``--max-chunks`` chunks (so it's quick on a 25 GB
store):

  1. insitu throughput vs ``decode_threads`` (1,2,4,8,auto) — where decode parallelism
     saturates; flat past N cores => decode isn't the limit.
  2. raw obstore concurrent GET MB/s vs concurrency (1,4,8,16,32) — pure fetch, no
     decode.
  3. (synthetic only) insitu compressed vs uncompressed — the codec's share.

Diagnosis: if raw GET (2) far exceeds insitu (1), we're decode/loop-limited and the
decode pool / max_inflight is the lever; if raw GET also caps near insitu, it's the
network/endpoint (more/bigger parallel streams, in-region S3, the gateway endpoint).
"""

from __future__ import annotations

import argparse
import atexit
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

from insitubatch import SplitName, open_geometries, split_by_chunk
from insitubatch.source import InSituDataset

from .make_dataset import make_dataset


def _store_kwargs(url: str, anon: bool, request_payer: bool) -> dict:
    kw: dict = {}
    if anon or url.startswith("gs://"):
        kw["skip_signature"] = True
    if request_payer:
        kw["request_payer"] = True
    return kw


def _stats(fn: Callable[[], float], repeats: int) -> tuple[float, float, float]:
    """Run ``fn`` ``repeats`` times; return (median, min, max) MB/s. Cloud reads of a
    small (cold) sample are noisy, so a single number lies -- report the spread."""
    xs = sorted(fn() for _ in range(repeats))
    return statistics.median(xs), xs[0], xs[-1]


def _insitu_mb_s(
    url, var, kw, decode_threads, max_chunks, block_chunks=8, max_inflight=16
) -> float:
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
    ds.io_config.decode_threads = decode_threads  # read by the reader at iteration time
    bps = int(np.prod(geom.inner_shape)) * 4
    limit = max_chunks * geom.sample_chunk_size
    n = 0
    t = time.perf_counter()
    for b in ds:
        n += b.arrays[var].shape[0]
        if n >= limit:
            break
    dt = time.perf_counter() - t
    return n * bps / 1e6 / dt


def _raw_get_mb_s(url, var, kw, concurrency, max_chunks) -> float:
    """Fetch raw (still-encoded) chunk objects via obstore — no decode."""
    geom = open_geometries(url, variables=[var], **kw)[var]
    obs = obstore.store.from_url(url, **kw)
    nchunks = min(max_chunks, math.ceil(geom.n_samples / geom.sample_chunk_size))
    zeros = "/".join("0" for _ in geom.inner_shape)
    keys = [f"{var}/c/{i}/{zeros}" for i in range(nchunks)]

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
    p.add_argument("--block-chunks", default="8,16,32,64", help="comma list for the 1b sweep")
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

    print(f"1) insitu MB/s vs decode_threads (median of {a.repeats}):")
    for dt in (1, 2, 4, 8, 0):
        show(
            f"decode_threads={dt or 'auto':>4}",
            partial(_insitu_mb_s, a.url, a.var, kw, dt, a.max_chunks),
        )

    print("\n1b) insitu MB/s vs read concurrency (block_chunks=max_inflight, decode auto):")
    for bc in (int(x) for x in a.block_chunks.split(",")):
        show(
            f"block_chunks={bc:>2}",
            partial(
                _insitu_mb_s, a.url, a.var, kw, 0, a.max_chunks, block_chunks=bc, max_inflight=bc
            ),
        )

    print("\n2) raw obstore concurrent GET MB/s (no decode):")
    for c in (1, 4, 8, 16, 32):
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
        print("\n3) codec cost (synthetic, auto vs uncompressed, auto decode_threads):")
        for label, u in (("compressed", a.url), ("uncompressed", none_url)):
            show(f"{label:12}", partial(_insitu_mb_s, u, "t2m", {}, 0, a.max_chunks))


if __name__ == "__main__":
    main()
