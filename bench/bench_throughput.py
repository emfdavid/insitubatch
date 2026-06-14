"""Throughput harness: insitubatch vs a naive synchronous baseline.

Phase 0 runs against a local ``file://`` zarr -- proving correctness and that the
async-driven engine is no slower than naive sequential reads. The dramatic win
(async fan-out hiding latency) shows up in Phase 1 against real S3; the same
harness runs there by passing ``--url s3://...``.

Metrics per run: samples/s, MB/s (decoded), time-to-first-batch, peak RSS.
Results append to ``bench/results/throughput.jsonl`` so paid EC2 numbers are
never re-derived.

    uv run python bench/bench_throughput.py                 # local, both regimes
    uv run python bench/bench_throughput.py --url s3://...   # bring your own data
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(__file__))
from make_dataset import make_dataset  # noqa: E402

from insitubatch import SplitName, open_geometries, split_by_chunk, store_from_url  # noqa: E402
from insitubatch.source import InSituDataset  # noqa: E402

RESULTS = Path(__file__).parent / "results" / "throughput.jsonl"


def peak_rss_mb() -> float:
    """Peak resident set size in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if platform.system() == "Darwin" else rss / 1e3


@dataclass
class Result:
    label: str
    regime: str
    n_samples: int
    seconds: float
    samples_per_s: float
    mb_per_s: float
    ttfb_s: float
    peak_rss_mb: float
    config: dict


def _bytes_per_sample(geom) -> int:
    return int(np.prod(geom.inner_shape)) * geom.dtype.itemsize


def run_insitu(url, manifest, geom, *, batch_size, block_chunks, max_inflight) -> Result:
    ds = InSituDataset(
        url,
        manifest,
        geometries={geom.name: geom},
        split=SplitName.TRAIN,
        batch_size=batch_size,
        block_chunks=block_chunks,
        max_inflight=max_inflight,
        to_tensor=False,
    )
    ds.set_epoch(0)
    bps = _bytes_per_sample(geom)

    t0 = time.perf_counter()
    ttfb = None
    n = 0
    for batch in ds:
        if ttfb is None:
            ttfb = time.perf_counter() - t0
        n += batch.arrays[geom.name].shape[0]
    dt = time.perf_counter() - t0
    return Result(
        "insitubatch", "", n, dt, n / dt, n * bps / 1e6 / dt, ttfb or dt, peak_rss_mb(),
        {"batch_size": batch_size, "block_chunks": block_chunks, "max_inflight": max_inflight},
    )


def run_naive(url, manifest, geom, *, batch_size) -> Result:
    """Floor: read each train chunk once, sequentially, no concurrency."""
    store = store_from_url(url)
    group = zarr.open_group(store=store, mode="r")
    arr = group[geom.name]
    spc = geom.sample_chunk_size
    bps = _bytes_per_sample(geom)

    t0 = time.perf_counter()
    ttfb = None
    n = 0
    for c in manifest.chunks[SplitName.TRAIN.value]:
        s0, s1 = c * spc, min(c * spc + spc, geom.n_samples)
        block = arr[s0:s1]  # synchronous read + decode
        if ttfb is None:
            ttfb = time.perf_counter() - t0
        n += block.shape[0]
    dt = time.perf_counter() - t0
    return Result(
        "naive_sync", "", n, dt, n / dt, n * bps / 1e6 / dt, ttfb or dt, peak_rss_mb(),
        {"batch_size": batch_size},
    )


def log(result: Result) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps({"ts": time.time(), **asdict(result)}) + "\n")


CASES = [
    ("fat", dict(n_samples=2048, inner=(64, 64), sample_chunk=64,
                 batch=32, block=16, inflight=16)),
    ("grib", dict(n_samples=2048, inner=(128, 128), sample_chunk=1,
                  batch=32, block=256, inflight=32)),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=None, help="existing zarr URL; default = temp local datasets")
    a = p.parse_args()

    print(f"{'case':6s} {'engine':12s} {'samples':>8s} {'sec':>7s} "
          f"{'samp/s':>10s} {'MB/s':>8s} {'ttfb_ms':>8s} {'rssMB':>7s}")
    tmp = tempfile.mkdtemp() if a.url is None else None
    for regime, c in CASES:
        url = a.url or f"file://{tmp}/{regime}.zarr"
        if a.url is None:
            make_dataset(url, n_samples=c["n_samples"], inner=c["inner"],
                         sample_chunk=c["sample_chunk"], variables=["t2m"])
        geom = open_geometries(url)["t2m"]
        manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))

        for res in (
            run_naive(url, manifest, geom, batch_size=c["batch"]),
            run_insitu(url, manifest, geom, batch_size=c["batch"],
                       block_chunks=c["block"], max_inflight=c["inflight"]),
        ):
            res.regime = regime
            log(res)
            print(f"{regime:6s} {res.label:12s} {res.n_samples:8d} {res.seconds:7.3f} "
                  f"{res.samples_per_s:10.1f} {res.mb_per_s:8.1f} "
                  f"{res.ttfb_s * 1e3:8.1f} {res.peak_rss_mb:7.0f}")


if __name__ == "__main__":
    main()
