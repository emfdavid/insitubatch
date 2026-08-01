"""Drive the advection stall/ceiling benchmark across the axes that back the finding.

    uv run python -m bench.advection_sweep --url-prefix gs://bucket/adv \
        --device cuda --epochs 5 --repeats 5 --sweeps inflight,size,chunk,inner

Each config runs ``examples.advection.train_torch_metrics`` **in its own process** -- so the
unbounded in-memory decoded cache and CUDA allocator start clean every time, and peak host
RSS / peak GPU memory are that config's alone (an in-process loop would let one config's 14 GB
cache inflate the next). The child writes its per-(run, epoch) rows to a temp JSONL; this
runner stamps the config onto each row and appends them to one combined ``--out`` file that
``bench.advection_report`` aggregates.

The sweeps, and the claim each one confirms:

* **inflight** -- throttle read-ahead depth over a fixed store (WB2 by default). Stall must
  *rise* as prefetch is starved and *fall* back as it is restored: validates the stall metric
  is real and produces the IO-bound datapoint the compute-bound runs never hit.
* **size** -- same model, synthetic field 64/128/256. Shows MB/s demand is ~size-invariant
  (bytes and conv compute both scale with pixels), so growing the field can't reach IO-bound.
* **chunk** -- sample-axis fat <-> GRIB (``sample_chunk`` large -> 1). The DESIGN spectrum:
  does the loader stay ahead as chunks shrink toward one-sample-per-read?
* **inner** -- spatial fan-out (``inner_chunk``: one fat chunk -> a tiled grid). The ARCO norm;
  restores concurrency *within* a fat sample chunk.
* **payload** -- bytes per batch (``batch_size`` over one 256^2 store: 8 -> 64 MiB per variable
  buffer, 4x that per step). The axis the batch-buffer work acts on, and the one the ``size``
  sweep cannot reach: reuse only pays above glibc's 32 MiB ``mmap`` threshold, which applies
  per allocation, and pinning's saved transfer time scales with bytes moved. Run it twice, back
  to back, with and without ``--pin`` -- absolute throughput is only comparable within a
  session (see the metric warning in docs/benchmarks.md).

The compute-only ceiling depends only on the field geometry (the conv cost), not on
inflight / chunking / cache, so ``--ceiling`` is run exactly once per distinct ``geom`` and the
report matches every insitu config to its geom's ceiling.

``--child-package`` A/Bs a change to the *library* by putting another checkout's ``src/`` ahead
of the installed one, so both arms share this checkout's example code, interpreter and torch --
only insitubatch differs. **Point it at a checkout that differs solely in the change under
test.** Pointing it at ``main`` does not work and is not what it is for: main predates the
examples' own API (``to_torch(..., device=...)``, ``pin_host_buffers``), so the child fails to
import, and even where it imports, main differs in several changes at once. For the batch-buffer
reuse arm, branch off this checkout and revert only the ``gather`` call site::

    git worktree add -b bench/no-reuse-arm ../ib-noreuse batch-buffer-ring
    # in that worktree, pool.gather: out = np.empty((n, *out_geom.inner_shape), ...)
    uv run python -m bench.advection_sweep --sweeps payload --child-package ../ib-noreuse ...

Every row records ``child_package``, so the two files an A/B produces cannot be confused after
the fact.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from insitubatch import InSituDataset

DEFAULT_OUT = Path(__file__).parent / "results" / "advection_sweep.jsonl"

# n_steps for the synthetic sweep stores: enough chunks to split/shuffle and to make cloud IO
# non-trivial, small enough to generate quickly. Overridable for a heavier run.
SYNTH_STEPS = 4000

# Payload sweep: batch sizes over the 256^2 store -> 8 / 16 / 32 / 64 MiB **per variable
# buffer** (4x that per step across the four variables). Two points below glibc's 32 MiB mmap
# threshold and two at or above it, since that threshold applies per allocation and the pool
# lends one buffer per variable. Capped at 256 because the 0.8 train split of a 4000-step store
# gives only 12 batches there -- fewer still and an epoch is too short to time.
PAYLOAD_BATCH_SIZES = (32, 64, 128, 256)

# Read from the engine rather than restated here, so the batch-clipping guard below cannot go
# stale against it (the examples do not pass block_chunks, so they get this default).
_BLOCK_CHUNKS: int = inspect.signature(InSituDataset).parameters["block_chunks"].default


def _synth(size: int, sample_chunk: int, inner_chunk: int | None) -> dict[str, Any]:
    """A synthetic-store config: its geom (compute identity) is the field size alone.

    ``store`` is the *dataset* identity (which zarr store to read) and ``geom`` the *compute*
    identity (which ceiling to score against). They coincide everywhere except the payload
    sweep, which varies batch size over one store -- and a bigger batch is a different compute
    step, so it needs its own ceiling while reading the same bytes.
    """
    return {
        "source": "synthetic",
        "size": size,
        "sample_chunk": sample_chunk,
        "inner_chunk": inner_chunk,
        "store": f"synth{size}",
        "geom": f"synth{size}",
    }


def buffer_mib(size: int, batch_size: int) -> float:
    """MiB in **one variable's** batch buffer -- what the reuse cliff keys on.

    The pool lends one buffer per variable per batch, so it is this number, not the step total,
    that glibc compares against its 32 MiB ``mmap`` threshold. Matches the per-variable payload
    column in docs/benchmarks.md.
    """
    return batch_size * size * size * 4 / 2**20


def step_mib(size: int, batch_size: int, n_variables: int = 4) -> float:
    """MiB crossing PCIe per step -- what pinning acts on.

    All four f4 variables (t2m/u10/v10/target), which is what one ``to_torch(..., device=...)``
    moves. Four times :func:`buffer_mib`, and the two axes are worth keeping apart: reuse is a
    per-allocation effect, pinning a per-transfer one.
    """
    return n_variables * buffer_mib(size, batch_size)


def _configs(
    sweeps: set[str], *, wb2_range: str, payload_batch_sizes: tuple[int, ...] = PAYLOAD_BATCH_SIZES
) -> Iterator[dict[str, Any]]:
    """Yield the insitu configs for the requested sweeps (ceiling is added per geom later).

    Each config is a dict of the knobs that vary; fixed knobs take the store/CLI defaults.
    ``max_inflight=None`` means the engine default (unthrottled)."""
    if "inflight" in sweeps:
        # Fixed real store (WB2), vary only read-ahead depth. geom "wb2" -> its own ceiling.
        for mi in (1, 2, 4, 8, 16, None):
            yield {
                "sweep": "inflight",
                "source": "wb2",
                "geom": "wb2",
                "sample_range": wb2_range,
                "max_inflight": mi,
            }
    if "size" in sweeps:
        for size in (64, 128, 256):
            yield {"sweep": "size", **_synth(size, 64, None)}
    if "chunk" in sweeps:
        # Sample-axis fat (256) -> GRIB-ish (4), field size fixed so compute (and ceiling) match.
        for spc in (256, 64, 16, 4):
            yield {"sweep": "chunk", **_synth(128, spc, None)}
    if "inner" in sweeps:
        # Spatial fan-out: one fat 128-chunk -> 64 (4 tiles) -> 32 (16 tiles), field size fixed.
        for ic in (128, 64, 32):
            yield {"sweep": "inner", **_synth(128, 64, ic)}
    if "payload" in sweeps:
        # Bytes per step, over ONE store (256^2) so only the batch changes: 8 -> 64 MiB, which
        # straddles glibc's 32 MiB mmap threshold. This is the axis both batch-buffer changes
        # act on -- reuse removes the re-zeroing cliff above the threshold, and pinning's saved
        # transfer time scales with bytes -- and the only one the 0.5-8 MiB size sweep cannot
        # reach. Each batch size gets its own geom: a bigger step is a different compute
        # ceiling, so they must not share one.
        for bs in payload_batch_sizes:
            cfg = _synth(256, 64, None)
            yield {**cfg, "sweep": "payload", "batch_size": bs, "geom": f"{cfg['store']}b{bs}"}


def _command(
    cfg: dict[str, Any],
    *,
    url_prefix: str,
    device: str,
    epochs: int,
    n_steps: int,
    ceiling: bool,
    pin: bool = False,
) -> list[str]:
    """Build the child ``train_torch_metrics`` command for one config (``--metrics-out`` is
    appended per run by :func:`_run_config`)."""
    cmd = [
        sys.executable,
        "-m",
        "examples.advection.train_torch_metrics",
        "--source",
        cfg["source"],
        "--device",
        device,
        "--epochs",
        str(epochs),
    ]
    if cfg["source"] == "synthetic":
        inner = cfg["inner_chunk"] or cfg["size"]
        # Keyed on `store`, not `geom`: the payload sweep gives each batch size its own geom
        # (its own ceiling) while they all read the one store.
        url = f"{url_prefix}_{cfg['store']}_c{cfg['sample_chunk']}_i{inner}.zarr"
        cmd += ["--url", url, "--n-steps", str(n_steps)]
        cmd += ["--size", str(cfg["size"]), "--sample-chunk", str(cfg["sample_chunk"])]
        if cfg["inner_chunk"] is not None:
            cmd += ["--inner-chunk", str(cfg["inner_chunk"])]
    if cfg.get("batch_size") is not None:
        cmd += ["--batch-size", str(cfg["batch_size"])]
    if cfg.get("sample_range"):
        cmd += ["--sample-range", cfg["sample_range"]]
    if cfg.get("max_inflight") is not None:
        cmd += ["--max-inflight", str(cfg["max_inflight"])]
    if ceiling:
        cmd += ["--ceiling"]
    if pin:
        # Unconditional, including on the runs that also collect the ceiling: `--ceiling`
        # *adds* a second fit rather than replacing the insitu one, so gating on it would
        # leave the first repeat of every geom unpinned while the rest were pinned.
        cmd += ["--pin"]
    return cmd


def child_env(child_package: str | None) -> dict[str, str] | None:
    """Environment for the child, optionally importing insitubatch from another checkout.

    Swapping the library by ``PYTHONPATH`` rather than by running the child in its own uv
    environment (which is how ``batch_buffer_sweep.py`` does it) is deliberate: the arms of an
    A/B differ *only* in insitubatch. An ephemeral environment would resolve its own torch, and
    a benchmark whose two arms ran different torch builds would be measuring that instead. Here
    the interpreter, torch, numpy and the example code are all shared; ``src/`` simply precedes
    site-packages, so ``import insitubatch`` finds the other checkout.
    """
    if not child_package:
        return None
    src = Path(child_package).expanduser().resolve() / "src"
    if not (src / "insitubatch" / "__init__.py").exists():
        raise SystemExit(f"--child-package {child_package}: no insitubatch package under {src}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(src), env.get("PYTHONPATH", "")) if p)
    return env


def _check_arm(out: Path, arm: dict[str, Any], p: argparse.ArgumentParser) -> None:
    """Refuse to append one arm's rows onto another's.

    Output is opened in append mode so a sweep can be resumed or extended, which also means two
    arms written to one path silently merge -- and because both number their repeats from zero,
    the report then medians across arms and prints a blend that looks like a clean result. That
    happened once; the cost was a full A/B. An arm is identified by what distinguishes it
    (``--pin``, ``--child-package``), and rows carry it, so the mismatch is detectable.
    """
    if not out.exists() or out.stat().st_size == 0:
        return
    with out.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            existing = {"pin": row.get("pin", False), "child_package": row.get("child_package", "")}
            if existing != arm:
                p.error(
                    f"{out} already holds rows from a different arm ({existing}); this run is "
                    f"{arm}. Two arms in one file merge on repeat index and the report medians "
                    f"across them. Write this arm to its own --out."
                )
            break  # one row settles it; the guard above keeps a file single-armed


def _run_config(
    cmd: list[str],
    cfg: dict[str, Any],
    repeat: int,
    out_fh: Any,
    env: dict[str, str] | None = None,
    arm: dict[str, Any] | None = None,
) -> None:
    """Run one child, tag its rows with the config + repeat, append to the combined file."""
    with tempfile.NamedTemporaryFile("r+", suffix=".jsonl") as tmp:
        subprocess.run([*cmd, "--metrics-out", tmp.name], check=True, env=env)
        tmp.seek(0)
        for line in tmp:
            if not line.strip():
                continue
            row = json.loads(line)
            row["config"] = cfg
            row["repeat"] = repeat
            # Stamped on every row: an A/B lives in two files that are indistinguishable
            # afterwards otherwise, and mixing them up is silent.
            row.update(arm or {})
            out_fh.write(json.dumps(row) + "\n")
    out_fh.flush()


def main() -> None:
    p = argparse.ArgumentParser(description="advection stall/ceiling sweep runner")
    p.add_argument(
        "--sweeps",
        default="inflight,size,chunk",
        help="comma list of: inflight,size,chunk,inner",
    )
    p.add_argument(
        "--url-prefix",
        default="file:///tmp/insitu_adv_sweep",
        help="synthetic store URL prefix (gs://bucket/adv on the box; file:// for local)",
    )
    p.add_argument("--device", default="cuda", help="cpu or cuda")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5, help="repeats per config (variance)")
    p.add_argument(
        "--n-steps", type=int, default=SYNTH_STEPS, help="synthetic trajectory length per store"
    )
    p.add_argument(
        "--wb2-range", default="0,4000", metavar="START,STOP", help="WB2 time window for --inflight"
    )
    p.add_argument(
        "--pin",
        action="store_true",
        help="page-lock batch buffers in the insitu runs (not the ceiling); A/B for pinning",
    )
    p.add_argument(
        "--child-package",
        default=None,
        metavar="PATH",
        help="run the children against another insitubatch checkout by putting its src/ ahead of "
        "the installed one. The A/B arm for changes to the library itself -- identical example "
        "code, interpreter and torch on both sides. Point it at a checkout differing only in the "
        "change under test, NOT at main (whose API predates the examples' calls)",
    )
    p.add_argument(
        "--payload-batch-sizes",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=PAYLOAD_BATCH_SIZES,
        metavar="N,N,...",
        help=f"batch sizes for the payload sweep (default {','.join(map(str, PAYLOAD_BATCH_SIZES))}"
        "); bigger needs a longer --n-steps to keep an epoch worth timing",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    sweeps = {s.strip() for s in args.sweeps.split(",") if s.strip()}
    unknown = sweeps - {"inflight", "size", "chunk", "inner", "payload"}
    if unknown:
        p.error(f"unknown sweeps: {sorted(unknown)}")

    configs = list(
        _configs(sweeps, wb2_range=args.wb2_range, payload_batch_sizes=args.payload_batch_sizes)
    )
    # A 0.8/0.1/0.1 split needs >=3 sample-axis chunks, so the synthetic trajectory must be a
    # few chunks long; catch a too-short --n-steps here rather than as an empty-val crash later.
    for cfg in configs:
        if cfg["source"] == "synthetic" and args.n_steps < 3 * cfg["sample_chunk"]:
            p.error(
                f"--n-steps {args.n_steps} too short for sample_chunk {cfg['sample_chunk']} "
                f"(need >= 3 chunks to split; use --n-steps >= {3 * cfg['sample_chunk']})"
            )
        # A batch never crosses a shuffle block, so an oversized batch_size is silently clipped
        # to the block -- which on the payload sweep would collapse two payload points onto one
        # while still labelling them apart. Fail instead.
        block_rows = _BLOCK_CHUNKS * cfg.get("sample_chunk", 0)
        if cfg.get("batch_size") is not None and cfg["batch_size"] > block_rows:
            p.error(
                f"batch_size {cfg['batch_size']} exceeds the shuffle block "
                f"({_BLOCK_CHUNKS} chunks x {cfg['sample_chunk']} = {block_rows} rows); it "
                f"would be clipped to {block_rows} and land on another config's payload"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arm = {"pin": bool(args.pin), "child_package": args.child_package or ""}
    _check_arm(args.out, arm, p)
    env = child_env(args.child_package)
    # Print which insitubatch the children will actually import, before spending an hour on
    # runs. A PYTHONPATH swap that silently did not take would produce two identical arms and
    # an A/B that reads as "no effect".
    which = subprocess.run(
        [sys.executable, "-c", "import insitubatch; print(insitubatch.__file__)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(f"children import insitubatch from: {which}\n")
    seen_geom: set[str] = set()
    with args.out.open("a") as out_fh:
        for i, cfg in enumerate(configs):
            # One ceiling per geom (compute identity); the report joins every config to it.
            ceiling = cfg["geom"] not in seen_geom
            seen_geom.add(cfg["geom"])
            for r in range(args.repeats):
                # The ceiling (compute-only) is geom-invariant, so collect it on the first repeat.
                do_ceiling = ceiling and r == 0
                cmd = _command(
                    cfg,
                    url_prefix=args.url_prefix,
                    pin=args.pin,
                    device=args.device,
                    epochs=args.epochs,
                    n_steps=args.n_steps,
                    ceiling=do_ceiling,
                )
                label = f"[{i + 1}/{len(configs)}] {cfg['sweep']} {cfg['geom']}"
                extra = {
                    k: cfg[k] for k in ("max_inflight", "sample_chunk", "inner_chunk") if k in cfg
                }
                if cfg.get("batch_size") is not None:
                    extra["MiB/buffer"] = round(buffer_mib(cfg["size"], cfg["batch_size"]), 1)
                    extra["MiB/step"] = round(step_mib(cfg["size"], cfg["batch_size"]), 1)
                print(f"=== {label} repeat {r + 1}/{args.repeats} {extra} ceiling={do_ceiling} ===")
                _run_config(cmd, cfg, r, out_fh, env=env, arm=arm)
    print(f"\nwrote sweep rows -> {args.out}")
    print(f"aggregate with: uv run python -m bench.advection_report --in {args.out}")


if __name__ == "__main__":
    main()
