"""Probe: how much of the object store's raw-GET ceiling survives decode + gather?

    uv run python -m bench.probe_ceiling --url gs://bucket/era5_fat_g16.zarr --var t2m
    uv run python -m bench.probe_ceiling --url s3://bucket/era5_fat_g16.zarr --anon

This is `docs/benchmarks.md` story 4 as a standalone, self-checking measurement. It exists
because that story's published ratio is easy to get wrong in two specific ways, and both are
silent -- the run completes and prints a plausible percentage either way.

**1. The two sides are counted in different units.** insitu's MB/s is *decoded* bytes; a raw
GET moves *stored* bytes. Dividing one by the other overstates "% kept" by exactly the
**logical-to-stored** ratio -- which is what this probe measures, via ``HEAD`` on the same
objects the raw sweep fetches, reporting the corrected number alongside the naive one.

That ratio is not the compression ratio. `standard_normal` f32 is close to incompressible
but not actually so (zstd level 0 takes ~7.6% off it), while zarr pads edge chunks to full
size, which puts ~0.8% back on the wire. On the benchmark grids the two nearly cancel and
the ratio is **~1.08x** -- ~6 points of "% kept". Assuming either effect instead of
measuring both gets a plausible, flattering number: a store with no padding and no codec
must report exactly 1.000, and a padded uncompressed one exactly `inner / padded_inner`.

**2. Read-ahead permits can cap insitu below the concurrency being compared.** A pass may
hold ``read_ahead_bound`` chunks ahead of its own consumer, so in-flight tiles are bounded by
``bound x tiles-per-chunk`` however high ``max_inflight`` goes -- while the raw sweep, a flat
thread pool over a key list, has no such bound. Comparing the two at a concurrency insitu
cannot reach measures the bound, not the loader. The insitu sweep therefore reports
``inflight_peak`` beside each point and flags any that never reached its ``max_inflight``.

Both sides are measured in one session, interleaved rather than blocked, so provider drift
hits every arm equally. The raw sweep is the control: it is the same bytes over the same
network from the same box, with decode and gather removed.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings
from typing import Any

import numpy as np
import obstore

from insitubatch import open_geometries, split_by_chunk
from insitubatch.source import InSituDataset

from .backend import build_store
from .probe_decode import _raw_get_mb_s, _store_kwargs, stored_chunk_keys


def stored_bytes(url: str, var: str, kw: dict[str, Any], max_chunks: int) -> int:
    """Bytes actually on the wire for one pass -- ``HEAD``, not ``GET``, so it costs nothing.

    Paired with the logical size this gives the **logical-to-stored** ratio the two sides
    differ by -- not the compression ratio, which is a different number: zarr pads edge
    chunks to full size, so the stored form carries padding the user never sees. Measured
    per store, never assumed; ~8% of the wire is 6 points of "% kept".
    """
    store = build_store("obstore", url, **kw)
    obs = obstore.store.from_url(url, **kw)
    return sum(obstore.head(obs, k)["size"] for k in stored_chunk_keys(store, var, max_chunks))


def insitu_pass(
    url: str, var: str, kw: dict[str, Any], *, block_chunks: int, max_inflight: int
) -> tuple[float, int, int]:
    """One full pass; (decoded MB/s, peak in-flight tiles, peak resident chunks)."""
    store = build_store("obstore", url, **kw)
    geom = open_geometries(store, variables=[var])[var]
    ds = InSituDataset(
        store,
        split_by_chunk(geom, fractions=(1.0, 0.0, 0.0)),
        geometries={var: geom},
        batch_size=32,
        block_chunks=block_chunks,
        max_inflight=max_inflight,
        shuffle=False,
    )
    bps = int(np.prod(geom.inner_shape)) * geom.dtype.itemsize
    n = 0
    t = time.perf_counter()
    for b in ds.train:
        n += b.arrays[var].shape[0]
    dt = time.perf_counter() - t
    return n * bps / 1e6 / dt, ds.inflight_peak, ds.resident_peak


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", required=True, help="zarr URL (gs:// | s3:// | file://)")
    p.add_argument("--var", default="t2m")
    p.add_argument("--max-chunks", type=int, default=256, help="outer chunks per pass")
    p.add_argument("--repeats", type=int, default=3, help="passes per point; median reported")
    p.add_argument("--block-chunks", default="2,4,8", help="insitu sweep (sets the permit bound)")
    p.add_argument("--max-inflight", default="32,64,128,256", help="insitu sweep")
    p.add_argument("--concurrency", default="16,32,64,128", help="raw-GET sweep")
    p.add_argument("--anon", action="store_true", help="anonymous (public gs:// / s3://)")
    p.add_argument("--request-payer", action="store_true")
    p.add_argument("--s3-express", action="store_true")
    a = p.parse_args()

    kw = _store_kwargs(a.url, "obstore", a.anon, a.request_payer, a.s3_express)
    bcs = [int(x) for x in a.block_chunks.split(",")]
    mis = [int(x) for x in a.max_inflight.split(",")]

    store = build_store("obstore", a.url, **kw)
    geom = open_geometries(store, variables=[a.var])[a.var]
    tiles = len(stored_chunk_keys(store, a.var, a.max_chunks)) // min(a.max_chunks, geom.n_chunks)
    n_chunks = min(a.max_chunks, geom.n_chunks)
    logical = n_chunks * geom.sample_chunk_size * int(np.prod(geom.inner_shape))
    logical *= geom.dtype.itemsize
    on_wire = stored_bytes(a.url, a.var, kw, a.max_chunks)
    ratio = logical / on_wire

    print(f"probe {a.url}  var={a.var}")
    print(f"  {n_chunks} outer chunks x {tiles} stored objects each")
    print(f"  {logical / 1e6:.0f} MB logical / {on_wire / 1e6:.0f} MB on the wire = {ratio:.4f}x")

    print("\nprewarm (discarded) ...")
    insitu_pass(a.url, a.var, kw, block_chunks=bcs[0], max_inflight=mis[-1])

    grid = [(bc, mi) for bc in bcs for mi in mis]
    acc: dict[tuple[int, int], list[float]] = {k: [] for k in grid}
    peak: dict[tuple[int, int], tuple[int, int]] = {}
    for _ in range(a.repeats):  # interleaved, so drift hits every arm alike
        for bc, mi in grid:
            mb, ip, rp = insitu_pass(a.url, a.var, kw, block_chunks=bc, max_inflight=mi)
            acc[(bc, mi)].append(mb)
            peak[(bc, mi)] = (ip, rp)

    print(f"\ninsitu decoded MB/s (median of {a.repeats}):")
    print(
        f"  {'block_chunks':>12} {'max_inflight':>12} {'inflight_pk':>11}"
        f" {'resident':>8} {'MB/s':>9}"
    )
    for bc, mi in grid:
        ip, rp = peak[(bc, mi)]
        flag = "  <-- permit-bound" if ip < mi else ""
        med = statistics.median(acc[(bc, mi)])
        print(f"  {bc:>12} {mi:>12} {ip:>11} {rp:>8} {med:>9.1f}{flag}")

    print(f"\nraw obstore GET, the control -- same bytes, no decode (median of {a.repeats}):")
    raws: dict[int, float] = {}
    for c in (int(x) for x in a.concurrency.split(",")):
        raws[c] = statistics.median(
            _raw_get_mb_s(a.url, a.var, kw, c, a.max_chunks, "obstore") for _ in range(a.repeats)
        )
        print(f"  concurrency={c:>4}: {raws[c]:8.1f} MB/s")

    best_in = max(statistics.median(v) for v in acc.values())
    best_raw = max(raws.values())
    if any(ip < mi for (_, mi), (ip, _) in peak.items()):
        warnings.warn(
            "some insitu points never reached their max_inflight -- the permit bound capped "
            "them. Raise --block-chunks before reading the ratio.",
            stacklevel=1,
        )
    print(f"\nbest insitu (decoded) : {best_in:8.1f} MB/s")
    print(f"best raw GET (stored) : {best_raw:8.1f} MB/s")
    print(f"  naive ratio         : {100 * best_in / best_raw:5.1f}%   <-- mixed units, too high")
    print(f"  raw as decoded-equiv: {best_raw * ratio:8.1f} MB/s")
    print(f"  % of ceiling kept   : {100 * best_in / (best_raw * ratio):5.1f}%")


if __name__ == "__main__":
    main()
