"""Probe: peak resident memory by engine, measured correctly (the G5 rebuild).

The suite's per-row ``peak_rss_mb`` is invalid for *comparing* engines: it reads
``/proc/self/status`` of the single suite process, so it's a monotonic high-water
shared by every engine, and it counts only the **main** process -- so the 32
DataLoader worker children of ``workers`` / ``xbatcher`` (their entire memory cost,
the whole point of the comparison) are never seen.

This probe runs **each engine in its own subprocess** (clean per-engine high-water)
and samples **peak RSS over the whole process tree** (main + children) on a timer, so
insitu's bounded working set vs ``num_workers x prefetch`` growth is finally
measurable. Use a read-once config (no ``--cache-dir``) so the comparison is anon
working set, apples to apples.

    uv run python -m bench.probe_memory --url s3://bucket/era5_c1.zarr \
      --engines insitu,workers,xbatcher --num-workers 32 --max-batches 64
"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
import time
from multiprocessing.queues import Queue as MPQueue

import psutil

from .engines import Cfg, run


def _proc_anon_bytes(p: psutil.Process) -> int:
    """RssAnon for one process (Linux /proc; 0 elsewhere or on race)."""
    try:
        with open(f"/proc/{p.pid}/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _sample_tree(root: psutil.Process) -> tuple[int, int, int]:
    """(total RSS bytes, total RssAnon bytes, process count) for root + descendants."""
    procs = [root]
    with contextlib.suppress(psutil.Error):
        procs += root.children(recursive=True)
    rss = anon = 0
    for p in procs:
        try:
            rss += p.memory_info().rss
            anon += _proc_anon_bytes(p)
        except psutil.Error:  # a worker came/went between enumerate and read
            pass
    return rss, anon, len(procs)


def _monitor(pid: int, interval: float) -> tuple[int, int, int]:
    """Poll the process tree until it exits; return peak (RSS, anon-at-peak, nproc)."""
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0, 0, 0
    peak_rss = peak_anon = peak_n = 0
    while True:
        try:
            if not root.is_running() or root.status() == psutil.STATUS_ZOMBIE:
                break
            rss, anon, n = _sample_tree(root)
        except psutil.NoSuchProcess:
            break
        if rss > peak_rss:
            peak_rss, peak_anon = rss, anon
        peak_n = max(peak_n, n)
        time.sleep(interval)
    return peak_rss, peak_anon, peak_n


def _child(cfg: Cfg, store_kwargs: dict, q: MPQueue) -> None:
    """Run one engine to completion in this subprocess; report median MB/s back."""
    try:
        rows = run(cfg, store_kwargs=store_kwargs)
        mbs = sorted(r.mb_per_s for r in rows)
        q.put(("ok", mbs[len(mbs) // 2] if mbs else 0.0))
    except Exception as exc:  # noqa: BLE001 - forwarded to the parent for the table
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", required=True, help="zarr URL (file:// or s3://)")
    p.add_argument("--var", default="t2m")
    p.add_argument("--engines", default="insitu,workers,xbatcher", help="comma list")
    p.add_argument("--storage", default="s3", choices=["file", "s3"])
    p.add_argument("--sample-chunk", type=int, default=1, help="the dataset's chunk0 (for the row)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-chunks", type=int, default=16, help="insitu residency window")
    p.add_argument("--num-workers", type=int, default=32, help="DataLoader workers (B1/B2)")
    p.add_argument("--max-batches", type=int, default=64, help="batches/epoch; reach steady state")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--interval-ms", type=float, default=25.0, help="RSS poll interval")
    p.add_argument("--anon", action="store_true", help="anonymous store (public gs:// / s3://)")
    p.add_argument("--request-payer", action="store_true")
    p.add_argument("--s3-express", action="store_true")
    a = p.parse_args()

    store_kwargs: dict[str, bool] = {}
    if a.anon or a.url.startswith("gs://"):
        store_kwargs["skip_signature"] = True
    if a.request_payer:
        store_kwargs["request_payer"] = True
    if a.s3_express:
        store_kwargs["s3_express"] = True

    # spawn (not fork): obstore's tokio runtime isn't fork-safe, and it matches how the
    # DataLoader engines start their own workers -- so the tree we measure is realistic.
    ctx = mp.get_context("spawn")
    print(f"peak RSS by engine  {a.url}")
    print(f"c{a.sample_chunk}  nw={a.num_workers}  bs={a.batch_size}  mb={a.max_batches}\n")
    print(f"{'engine':9} {'peakRSS_MB':>10} {'anon_MB':>8} {'procs':>5} {'MB/s':>7}")
    for engine in a.engines.split(","):
        cfg = Cfg(
            engine=engine,
            url=a.url,
            storage=a.storage,
            sample_chunk=a.sample_chunk,
            var=a.var,
            batch_size=a.batch_size,
            block_chunks=a.block_chunks,
            num_workers=a.num_workers,
            max_batches=a.max_batches,
            epochs=a.epochs,
        )
        q: MPQueue = ctx.Queue()
        proc = ctx.Process(target=_child, args=(cfg, store_kwargs, q), name=f"mem-{engine}")
        proc.start()
        assert proc.pid is not None  # set by start()
        peak_rss, peak_anon, peak_n = _monitor(proc.pid, a.interval_ms / 1000.0)
        proc.join()
        status, payload = q.get() if not q.empty() else ("err", "no result")
        if status == "err":
            print(f"{engine:9} {'skip':>10}  {payload}")
            continue
        print(
            f"{engine:9} {peak_rss / 1e6:10.0f} {peak_anon / 1e6:8.0f} "
            f"{peak_n:5d} {float(payload):7.1f}"
        )


if __name__ == "__main__":
    main()
