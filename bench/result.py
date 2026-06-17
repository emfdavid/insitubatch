"""Benchmark result row + JSONL logging.

One self-describing row per (engine, config, epoch) so the JSONL can be sliced
into any of the planned graphs (see benchmark_plan.md).
"""

from __future__ import annotations

import json
import platform
import resource
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def peak_rss_mb() -> float:
    """Peak resident set size in MB (ru_maxrss is bytes on macOS, KB on Linux).

    Note: this is a process-wide *monotonic* high-water mark, so in the
    single-process suite it sticks at the largest config seen so far (a MemoryCache
    run inflates every later row). Prefer the per-row ``rss_anon_mb`` below.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if platform.system() == "Darwin" else rss / 1e3


def rss_breakdown_mb() -> tuple[float, float]:
    """Current (anon, file) resident MB from ``/proc/self/status`` (Linux only).

    Splits resident memory into **RssAnon** (heap/stack — true memory pressure) and
    **RssFile** (file-backed mmap — e.g. DiskCache's mmap'd ``.npy``, which is
    reclaimable page cache, not heap). ru_maxrss conflates the two; this is sampled
    per row so DiskCache shows a bounded heap even when total RSS looks large.
    Returns ``(0.0, 0.0)`` off Linux (e.g. macOS has no ``/proc``).
    """
    if platform.system() != "Linux":
        return 0.0, 0.0
    anon = file = 0.0
    with open("/proc/self/status") as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "RssAnon:":
                anon = float(parts[1]) / 1e3  # kB -> MB
            elif parts and parts[0] == "RssFile:":
                file = float(parts[1]) / 1e3
    return anon, file


@dataclass
class Result:
    engine: str  # insitu | naive | memory | workers | xbatcher
    cache: str  # none | memory | disk
    storage: str  # file | s3
    sample_chunk: int
    n_samples: int
    epoch: int
    batch_size: int
    block_chunks: int
    prefetch_depth: int
    num_workers: int
    compute_ms: float
    seconds: float
    samples_per_s: float
    mb_per_s: float
    ttfb_ms: float
    peak_rss_mb: float  # ru_maxrss high-water (monotonic, process-wide)
    rss_anon_mb: float = 0.0  # heap/stack at row time (Linux); the real memory bound
    rss_file_mb: float = 0.0  # file-backed mmap at row time (DiskCache .npy; reclaimable)
    host: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=platform.platform)
    ts: float = field(default_factory=time.time)


def append_jsonl(path: str | Path, result: Result) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
