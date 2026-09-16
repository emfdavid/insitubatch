"""Probe: is the read pipeline network-bound or decode-bound, and does V2 decouple?

    uv run python -m bench.probe_decode                              # synthetic file://
    uv run python -m bench.probe_decode --url s3://bucket/era5_c8.zarr --var t2m
    uv run python -m bench.probe_decode --url gs://.../era5.zarr --var 2m_temperature --anon

Measurements over the first ``--max-chunks`` chunks (quick on a 25 GB store):

  1. **The V2 decoupling headline**: insitu throughput + peak residency vs
     ``max_inflight``, with ``block_chunks`` fixed small. V2 dials network
     concurrency with ``max_inflight`` alone; throughput should rise to the network
     knee and then stay *flat* (not fall, as v1's nested caps did when oversubscribed)
     while residency stays **pinned** — independent of concurrency. Pinned is the
     claim, not a particular number: a plain single-variable pass sits at
     ``2*block_chunks``, and a windowed one (``--window``) sits higher, because a
     chunk feeds every block whose anchors reach it and is released only at the last.
  2. raw concurrent GET MB/s vs concurrency (1,4,8,16,32) — pure fetch, no decode.
  3. (synthetic only) insitu compressed vs uncompressed — the codec's share.

Diagnosis: if raw GET (2) far exceeds insitu (1), we're decode/loop-limited and
``max_inflight`` / the decode pool is the lever; if raw GET also caps near insitu,
it's the network/endpoint (more/bigger parallel streams, in-region S3, the gateway).

**Comparing (1) against (2) assumes the data barely compresses.** insitu MB/s counts
*decoded* bytes; raw GET counts the *stored* bytes it moved. On the synthetic store
those agree, deliberately — ``make_dataset`` writes ``standard_normal`` f32, which zstd
does not shrink (see ``bench/benchmark_plan.md``). On a real ``--url`` they do not: ERA5
compresses ~2-4x, so raw GET is understated against insitu by that ratio, which biases
the diagnosis *toward* "not decode-limited". Divide insitu's MB/s by the store's
compression ratio before reading the gap.

``decode_threads`` is a **process-wide** setting, not a sweep: the decode pool is built
once, by the first dataset in the process, and a later different value is ignored. Set
``--decode-threads`` and run the probe again to compare -- one value per process is the
only way to measure it honestly.
"""

from __future__ import annotations

import argparse
import asyncio
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

from insitubatch import SplitManifest, SplitName, close_store, open_geometries, split_by_chunk
from insitubatch.scheduler import decode_pool_workers
from insitubatch.source import InSituDataset

from ._profile import record_pyspy
from .backend import build_store
from .make_dataset import make_dataset


def _store_kwargs(
    url: str, backend: str, anon: bool, request_payer: bool, s3_express: bool
) -> dict:
    """Backend-specific store kwargs. ``--anon`` must be explicit: an owned (private)
    GCS bucket reads with ambient credentials, so gs:// is not forced anonymous."""
    kw: dict = {}
    if backend == "fsspec":
        if anon:
            kw["token"] = "anon"  # gcsfs anonymous read of a public bucket
        if request_payer:
            kw["requester_pays"] = True
        return kw
    if anon:
        kw["skip_signature"] = True  # obstore anonymous read
    if request_payer:
        kw["request_payer"] = True
    if s3_express:  # S3 Express One Zone directory bucket (--x-s3); not inferred from the name
        kw["s3_express"] = True
    return kw


def _close_fsspec_session(fs: Any) -> None:
    """Close a gcsfs aiohttp session on the loop it lives on, so gcsfs's finalizer --
    which captures ``self.loop`` (None here) and otherwise closes on the wrong loop at
    GC -- becomes a no-op. Silences the "Task was destroyed / different loop" teardown
    traceback. Best-effort: guarded for non-gcsfs backends and already-closed sessions.
    """
    sess = getattr(fs, "_session", None)
    loop = getattr(sess, "_loop", None)
    if sess is None or loop is None or sess.closed or not loop.is_running():
        return
    with contextlib.suppress(Exception):
        asyncio.run_coroutine_threadsafe(sess.close(), loop).result(timeout=5)
        fs._session = None


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
    backend: str = "obstore",
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

    ``backend`` selects the store (obstore | fsspec) so the max_inflight sweep can be
    run on either -- fsspec routes its reads through zarr's loop (the cross-loop fix),
    so this is where any bridge overhead at high concurrency would surface.
    """
    store = build_store(backend, url, **kw)
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


def _insitu_mb(
    url: str, var: str, kw: dict[str, Any], *, backend: str = "obstore", **opts: int
) -> float:
    """Just the MB/s of an :func:`_insitu` pass -- a typed target for ``partial`` in
    the ``show`` sweeps (a ``lambda x=x:`` loop-capture defeats mypy inference)."""
    return _insitu(url, var, kw, backend=backend, **opts)[0]


def _insitu_cache(
    url: str,
    var: str,
    kw: dict[str, Any],
    *,
    max_chunks: int,
    block_chunks: int,
    cache_dir: str,
    backend: str = "obstore",
) -> tuple[float, float]:
    """Two epochs over the first ``max_chunks`` chunks with a budget that holds them
    all; returns (cold MB/s, warm MB/s). The manifest is restricted to exactly those
    chunks so read-ahead can't LRU-evict the early ones before epoch 1 -- which is
    then served entirely from the cache (no S3, no decode). ``cache_dir`` spills the
    slots to NVMe (mmap); pass it to keep heap bounded on fat data.
    """
    store = build_store(backend, url, **kw)
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


def stored_chunk_keys(store: Any, var: str, max_chunks: int) -> list[str]:
    """Every stored-object key under the first ``max_chunks`` OUTER chunks of ``var``.

    The grid comes from ``ChunkGrid.from_metadata`` and the key from the array's own
    ``encode_chunk_key`` -- the same "one spelling of the shape of one stored object" the
    engine plans reads with. Neither is cosmetic. ``metadata.chunks`` is the *inner* chunk
    on a sharded array while a key addresses the *shard*, so building the grid from
    ``chunks`` enumerates keys that do not exist (every dynamical.org archive is sharded);
    and the ``c/i/j`` layout is zarr-v3's, while v2 writes ``i.j`` (WeatherBench2 ARCO is
    v2). Either way a hand-built key 404s on exactly the stores the engine reads fine.
    """
    from zarr.core.chunk_grids import ChunkGrid

    arr = zarr.open_array(store=store, path=var, mode="r")
    meta = arr.metadata
    stored = tuple(ChunkGrid.from_metadata(meta).chunk_shape)  # the shard, if sharded
    n_outer = min(max_chunks, math.ceil(arr.shape[0] / stored[0]))
    inner_ranges = [range(math.ceil(s / c)) for s, c in zip(arr.shape[1:], stored[1:], strict=True)]
    return [
        f"{var}/" + meta.encode_chunk_key((oi, *inner))
        for oi in range(n_outer)
        for inner in itertools.product(*inner_ranges)
    ]


def _raw_get_mb_s(
    url: str, var: str, kw: dict[str, Any], concurrency: int, max_chunks: int, backend: str
) -> float:
    """Fetch raw (still-encoded) chunk objects — no decode, no engine — as the pure
    transfer-stack floor. ``backend`` picks the fetcher: obstore's Rust GET or a
    concurrent gcsfs ``cat_file``. Comparing the two is the cleanest "what does the
    Python fsspec layer cost" number, isolated from decode/gather/loop.

    Reads the *real* stored-chunk grid via :func:`stored_chunk_keys`, so it is correct for
    spatially-chunked, sharded and zarr-v2 arrays alike. max_chunks bounds the number of
    OUTER chunks; every stored object under them is fetched, concurrency threads at a time.
    """
    meta_store = build_store(backend, url, **kw)
    keys = stored_chunk_keys(meta_store, var, max_chunks)

    if backend == "fsspec":
        import fsspec

        proto, _, rest = url.partition("://")
        root = rest.rstrip("/")
        # A dedicated *sync* fs (own session on fsspec's loop), separate from the async
        # zarr store above -- so the thread pool drives concurrent cat_file without the
        # cross-loop session sharing that trips the engine path.
        raw_fs = fsspec.filesystem(proto, **kw)

        def fetch(key: str) -> int:
            return len(raw_fs.cat_file(f"{root}/{key}"))
    else:
        obs = obstore.store.from_url(url, **kw)

        def fetch(key: str) -> int:
            return len(bytes(obstore.get(obs, key).bytes()))

    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        total = sum(ex.map(fetch, keys))
    mb = total / 1e6 / (time.perf_counter() - t)
    if backend == "fsspec":
        _close_fsspec_session(raw_fs)  # the bare raw-fetch fs (no Store wrapper)
        close_store(meta_store)  # the async grid-metadata store's session
    return mb


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default=None, help="zarr URL; default = synthetic file://")
    p.add_argument("--var", default="t2m")
    p.add_argument(
        "--backend",
        default="obstore",
        choices=["obstore", "fsspec"],
        help="store backend for the fetch-sensitive sections (1, 2): obstore or fsspec/gcsfs",
    )
    p.add_argument("--max-chunks", type=int, default=64, help="chunks to probe (bounds the cost)")
    p.add_argument("--repeats", type=int, default=3, help="runs per point; report median (min-max)")
    p.add_argument(
        "--decode-threads",
        type=int,
        default=0,
        help="size the process-wide decode pool (0 = auto = min(32, cpu+4)). Applied to "
        "the first dataset this process builds, which is the only one that can set it; "
        "to compare thread counts, run the probe once per value.",
    )
    p.add_argument(
        "--block-chunks",
        type=int,
        default=2,
        help="fixed shuffle window / residency for the max_inflight sweep (sec 1)",
    )
    p.add_argument("--max-inflight", default="8,16,32,64", help="sec 1 sweep (the V2 dial)")
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
        help="record a py-spy --native flamegraph of the sec 1 sweep to PATH "
        "(default probe-profile.svg; .json => speedscope). Needs ptrace_scope=0 "
        "or sudo; see bench/_profile.py.",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="run the cross-epoch cache test (sec 1b): cold vs cached epoch, mmap slots here",
    )
    p.add_argument("--no-raw", action="store_true", help="skip sec 2 (raw GET re-reads the data)")
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
    kw = _store_kwargs(a.url, a.backend, a.anon, a.request_payer, a.s3_express)
    print(f"probe {a.url}  var={a.var}  backend={a.backend}  first {a.max_chunks} chunks\n")

    def show(label: str, fn: Callable[[], float]) -> None:
        med, lo, hi = _stats(fn, a.repeats)
        print(f"   {label}: {med:8.1f} MB/s  ({lo:.0f}-{hi:.0f})")

    bc = a.block_chunks

    # Prewarm: a cold S3 prefix is rate-limited and obstore's TLS pool is empty, so
    # the first sweep point otherwise eats the ramp-up and reads at ~1 stream. One
    # throwaway burst at the top fixes it for every section below (and makes 1b's
    # "cold" epoch S3-warm/pool-cold, isolating decode+pool reuse from the S3 ramp).
    if not a.no_warm:
        mi_warm = max(int(x) for x in a.max_inflight.split(","))
        print(f"0) prewarm: {a.max_chunks} chunks @ max_inflight={mi_warm} (discarded) ...")
        try:
            # Also where decode_threads lands: the pool is process-wide and the first
            # dataset built sizes it, so passing it only to the sections below would be
            # ignored -- with a warning, under a table that still looked like a result.
            _insitu(
                a.url,
                a.var,
                kw,
                decode_threads=a.decode_threads,
                max_chunks=a.max_chunks,
                block_chunks=bc,
                max_inflight=mi_warm,
                window=a.window,
                backend=a.backend,
            )
        except Exception as exc:  # noqa: BLE001 - prewarm is best-effort
            print(f"   prewarm failed: {type(exc).__name__}: {exc}")
        print()

    if a.window:
        print(f"(windowed: anchor input + {a.window} shifted target view(s); MB/s = input bytes)\n")

    print(
        f"\n1) insitu MB/s + peak residency vs max_inflight (block_chunks={bc}, "
        f"decode pool={decode_pool_workers() or 'unbuilt'} threads):"
    )
    print("    V2 wants throughput flat (not falling) past the knee, residency pinned.")
    profile = record_pyspy(a.profile) if a.profile else contextlib.nullcontext()
    with profile:  # profile scope = the V2 acceptance sweep (insitu only, no raw-GET)
        for mi in (int(x) for x in a.max_inflight.split(",")):
            xs = sorted(
                _insitu(
                    a.url,
                    a.var,
                    kw,
                    decode_threads=a.decode_threads,
                    max_chunks=a.max_chunks,
                    block_chunks=bc,
                    max_inflight=mi,
                    window=a.window,
                    backend=a.backend,
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
        print(f"\n1b) cross-epoch cache ({a.max_chunks} chunks, budget holds all, mmap):")
        cold, warm = _insitu_cache(
            a.url,
            a.var,
            kw,
            max_chunks=a.max_chunks,
            block_chunks=bc,
            cache_dir=a.cache_dir,
            backend=a.backend,
        )
        print(f"   epoch 0 (cold):   {cold:8.1f} MB/s")
        print(f"   epoch 1 (cached): {warm:8.1f} MB/s   ({warm / cold:.1f}x cold)")

    if not a.no_raw:
        fetcher = "gcsfs cat_file" if a.backend == "fsspec" else "obstore GET"
        print(f"\n2) raw concurrent {fetcher} MB/s (no decode):")
        for c in (int(x) for x in a.concurrency.split(",")):
            show(
                f"concurrency={c:>2}",
                partial(_raw_get_mb_s, a.url, a.var, kw, c, a.max_chunks, a.backend),
            )

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
        print("\n3) codec cost (synthetic, compressed vs uncompressed):")
        for label, u in (("compressed", a.url), ("uncompressed", none_url)):
            show(
                f"{label:12}",
                partial(
                    _insitu_mb,
                    u,
                    "t2m",
                    {},
                    decode_threads=a.decode_threads,
                    max_chunks=a.max_chunks,
                ),
            )


if __name__ == "__main__":
    main()
