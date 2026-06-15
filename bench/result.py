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
    """Peak resident set size in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if platform.system() == "Darwin" else rss / 1e3


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
    peak_rss_mb: float
    host: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=platform.platform)
    ts: float = field(default_factory=time.time)


def append_jsonl(path: str | Path, result: Result) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
