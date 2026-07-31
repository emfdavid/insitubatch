"""Benchmark: what batch-buffer reuse is worth, as a function of payload size.

``pool.gather`` lends a reused buffer per variable instead of allocating one per batch. Above
glibc's 32 MiB dynamic ``mmap`` threshold a fresh batch is ``mmap``ed and ``munmap``ed *every
batch*; below it the freed block is recycled on the heap and reuse buys nothing. This sweeps
batch size across that boundary on a real pipeline, which the isolated measurement in
``probe_batch_buffers.py`` cannot do: reuse saves *producer-thread* time, and whether that
surfaces end to end depends on whether IO is hiding it.

**Read the shape, not one number.** Reuse holds throughput flat across payload size while
fresh allocation falls off a cliff at 32 MiB. A single headline percentage hides that the
small-batch end is a ~2-3% *regression*, which is a real trade-off rather than noise.

``minor_faults`` is here because faulting a fresh page set is part of what a per-batch
``mmap`` costs, but treat it as **environment-dependent**: it matched prediction closely on
one host (678k vs 1039k for a 12-batch epoch of 128 MiB batches, ~328k predicted) and did not
separate at all on another under the same nominal config. If it stays flat while throughput
moves, believe the throughput and check the syscalls directly::

    strace -f -e trace=mmap,munmap -o t.txt <child cmd>
    grep -c 'mmap(NULL, <batch_bytes+4096>, PROT_READ|PROT_WRITE' t.txt

That count scales with batch count without the pool and stays flat with it, which is the
mechanism measured rather than inferred. Ignore same-sized ``PROT_NONE ... MAP_NORESERVE``
mappings -- those are allocator arena reservations and track thread count, not batches.

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


def effective_batch(batch_size: int, block_chunks: int, sample_chunk: int) -> int:
    """Rows a batch actually gets: a batch never crosses a shuffle-block boundary.

    ``source.py`` draws ``order[start : min(start + bs, rstop)]`` within one block, so a
    ``batch_size`` larger than ``block_chunks * sample_chunk`` is silently clipped to it --
    and this sweep's whole axis is batch *bytes*, so an unnoticed clip would collapse several
    configs onto the same point while still labelling them apart.
    """
    return min(batch_size, block_chunks * sample_chunk)


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
    p.add_argument("--url", default=None, help="existing store; omitted -> one is built")
    p.add_argument(
        "--data-dir",
        default=None,
        help="where to build the store (point at NVMe). Reused across runs if the geometry "
        "matches; omitted -> a temp dir, rebuilt every run",
    )
    p.add_argument("--batch-sizes", type=_ints, default=list(DEFAULT_BATCH_SIZES))
    p.add_argument("--inner", type=_inner, default=DEFAULT_INNER, help="per-sample shape")
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--sample-chunk", type=int, default=8)
    # Must cover the largest batch: a batch never crosses a shuffle block, so a small value
    # here silently clips the sweep's biggest configs onto one another.
    p.add_argument("--block-chunks", type=int, default=64)
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
        # Name the store after its geometry so a rerun reuses it rather than rewriting GBs.
        # Reuse is what makes an A/B valid, not merely fast: both arms must read the *same*
        # store, or the comparison also contains whatever differs between two datasets.
        shape = "x".join(str(x) for x in args.inner)
        name = f"bb_n{args.n_samples}_{shape}_c{args.sample_chunk}.zarr"
        if args.data_dir:
            directory = Path(args.data_dir)
            directory.mkdir(parents=True, exist_ok=True)
        else:
            tmpdir = tempfile.TemporaryDirectory()
            directory = Path(tmpdir.name)
        store = directory / name
        args.url = f"file://{store}"
        if store.exists():
            print(f"reusing {args.url}")
        else:
            make_dataset(
                args.url,
                n_samples=args.n_samples,
                inner=args.inner,
                sample_chunk=args.sample_chunk,
                variables=["t2m"],
            )

    block_rows = args.block_chunks * args.sample_chunk
    clipped = [b for b in args.batch_sizes if b > block_rows]
    if clipped:
        raise SystemExit(
            f"--batch-sizes {clipped} exceed the shuffle block ({args.block_chunks} chunks x "
            f"{args.sample_chunk} = {block_rows} rows). A batch never crosses a block, so these "
            f"would all be clipped to {block_rows} rows and land on one point in the sweep. "
            f"Raise --block-chunks to at least {max(args.batch_sizes) // args.sample_chunk}."
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
