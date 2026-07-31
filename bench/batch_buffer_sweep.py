"""Benchmark: what batch-buffer reuse is worth, as a function of payload size.

``pool.gather`` lends a reused buffer per variable instead of allocating one per batch. The
saving is not the allocation -- ``np.empty`` only reserves address space -- but the
**first-touch page faults** on the scatter-write that follows. glibc recycles a freed block on
the heap below its 32 MiB dynamic ``mmap`` threshold and ``mmap``/``munmap``s above it, so
reuse is worth nothing for small batches and a third of assembly time for large ones. This
sweeps batch size across that boundary on a real pipeline, which the isolated measurement in
``probe_batch_buffers.py`` cannot do: reuse saves *producer-thread* time, and whether that
shows up end to end depends on whether IO is hiding it.

Two columns matter, and the second is the sensitive one. ``samples_per_s`` is the headline but
goes quiet whenever the pipeline is IO-bound; ``minor_faults`` measures the mechanism directly
and moves whether or not the time is visible. A run where faults drop sharply and throughput
does not is not a null result -- it means this workload's assembly was already hidden behind
IO, which is worth knowing and is *not* what a wall-clock-only benchmark would report.

**One child process per config, always.** ``ru_minflt`` is cumulative, and glibc's threshold
is itself stateful -- it rises once a large block has been freed -- so configs sharing a
process would contaminate each other in exactly the dimension under measurement.

    # local smoke test: seconds, well under the threshold, proves the plumbing
    uv run python -m bench.batch_buffer_sweep --inner 32,32 --batch-sizes 8,32 \
        --n-samples 512 --repeats 1

    # the real run: 8 MiB -> 128 MiB per batch, crossing the cliff
    uv run python -m bench.batch_buffer_sweep

An A/B against a checkout without the pool needs no change here -- point ``--child-package``
at another worktree and the same sweep measures it (see ``docs/benchmarks.md``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .engines import Cfg, run
from .result import append_jsonl

DEFAULT_OUT = Path(__file__).parent / "results" / "batch_buffers.jsonl"
# 256x256 f4 = 256 KiB per sample, so these span ~8 MiB to ~128 MiB per batch -- two points
# below glibc's 32 MiB threshold and three above, which is where the effect should appear.
DEFAULT_BATCH_SIZES = (32, 64, 128, 256, 512)
DEFAULT_INNER = (256, 256)


def batch_mib(batch_size: int, inner: tuple[int, ...], itemsize: int = 4) -> float:
    """MiB per batch -- the axis that actually decides whether reuse pays."""
    return batch_size * int(np.prod(inner)) * itemsize / 2**20


def _inner(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def _child_cmd(args: argparse.Namespace, batch_size: int, out: str) -> list[str]:
    """One config, one fresh interpreter. ``--child-package`` selects which insitubatch the
    child imports, which is what lets the same sweep measure another worktree."""
    cmd = [sys.executable]
    if args.child_package:
        cmd = ["uv", "run", "--no-project", "--with-editable", args.child_package, "python"]
    return [
        *cmd,
        "-m",
        "bench.batch_buffer_sweep",
        "--child",
        "--url",
        args.url,
        "--batch-sizes",
        str(batch_size),
        "--sample-chunk",
        str(args.sample_chunk),
        "--block-chunks",
        str(args.block_chunks),
        "--prefetch-depth",
        str(args.prefetch_depth),
        "--compute-ms",
        str(args.compute_ms),
        "--epochs",
        str(args.epochs),
        "--child-out",
        out,
    ]


def _run_child(cmd: list[str], cfg: dict[str, Any], repeat: int, out_path: Path) -> dict[str, Any]:
    """Run one child, tag its rows with the sweep config, append them, return the last row."""
    with tempfile.NamedTemporaryFile("r+", suffix=".jsonl") as tmp:
        subprocess.run([*cmd[:-1], tmp.name], check=True)
        tmp.seek(0)
        rows = [json.loads(line) for line in tmp if line.strip()]
    for row in rows:
        row["config"] = cfg
        row["repeat"] = repeat
        with out_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return rows[-1] if rows else {}


def _child(args: argparse.Namespace) -> None:
    """Measure one config in this (fresh) process and write its rows."""
    cfg = Cfg(
        engine="insitu",
        url=args.url,
        storage="file",
        sample_chunk=args.sample_chunk,
        batch_size=args.batch_sizes[0],
        block_chunks=args.block_chunks,
        prefetch_depth=args.prefetch_depth,
        compute_ms=args.compute_ms,
        epochs=args.epochs,
    )
    for result in run(cfg):
        append_jsonl(args.child_out, result)


def main() -> None:
    p = argparse.ArgumentParser(description="batch-buffer reuse sweep (payload-size crossover)")
    p.add_argument("--url", default=None, help="existing store; omitted -> a temp one is built")
    p.add_argument("--batch-sizes", type=_ints, default=list(DEFAULT_BATCH_SIZES))
    p.add_argument("--inner", type=_inner, default=DEFAULT_INNER, help="per-sample shape")
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--sample-chunk", type=int, default=8)
    p.add_argument("--block-chunks", type=int, default=16)
    p.add_argument("--prefetch-depth", type=int, default=2)
    p.add_argument("--compute-ms", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--child-package",
        default=None,
        help="path to another insitubatch checkout for the child to import (A/B vs main)",
    )
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.child:
        _child(args)
        return

    from .make_dataset import make_dataset

    tmpdir: tempfile.TemporaryDirectory[str] | None = None
    if args.url is None:
        tmpdir = tempfile.TemporaryDirectory()
        args.url = f"file://{tmpdir.name}/bb.zarr"
        make_dataset(
            args.url,
            n_samples=args.n_samples,
            inner=args.inner,
            sample_chunk=args.sample_chunk,
            variables=["t2m"],
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, float, dict[str, Any]]] = []
    # Repeat-major so a slow drift in the machine spreads across batch sizes instead of
    # landing entirely on the ones measured last.
    for repeat in range(args.repeats):
        for batch_size in args.batch_sizes:
            mib = batch_mib(batch_size, args.inner)
            cfg = {"inner": list(args.inner), "batch_size": batch_size, "batch_mib": mib}
            row = _run_child(_child_cmd(args, batch_size, ""), cfg, repeat, out_path)
            rows.append((batch_size, mib, row))

    print(f"\n{'batch':>7}{'MiB':>9}{'samples/s':>12}{'minor faults':>14}{'rss anon MB':>13}")
    for batch_size in args.batch_sizes:
        got = [r for bs, _m, r in rows if bs == batch_size and r]
        if not got:
            continue
        mib = batch_mib(batch_size, args.inner)
        sps = float(np.median([r["samples_per_s"] for r in got]))
        flt = float(np.median([r.get("minor_faults", 0) for r in got]))
        rss = float(np.median([r.get("rss_anon_mb", 0.0) for r in got]))
        print(f"{batch_size:>7}{mib:>9.1f}{sps:>12.1f}{flt:>14.0f}{rss:>13.1f}")
    print(f"\n{len(rows)} rows -> {out_path}")
    if tmpdir is not None:
        tmpdir.cleanup()


if __name__ == "__main__":
    main()
