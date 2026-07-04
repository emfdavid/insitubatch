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

The compute-only ceiling depends only on the field geometry (the conv cost), not on
inflight / chunking / cache, so ``--ceiling`` is run exactly once per distinct ``geom`` and the
report matches every insitu config to its geom's ceiling.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path(__file__).parent / "results" / "advection_sweep.jsonl"

# n_steps for the synthetic sweep stores: enough chunks to split/shuffle and to make cloud IO
# non-trivial, small enough to generate quickly. Overridable for a heavier run.
SYNTH_STEPS = 4000


def _synth(size: int, sample_chunk: int, inner_chunk: int | None) -> dict[str, Any]:
    """A synthetic-store config: its geom (compute identity) is the field size alone."""
    return {
        "source": "synthetic",
        "size": size,
        "sample_chunk": sample_chunk,
        "inner_chunk": inner_chunk,
        "geom": f"synth{size}",
    }


def _configs(sweeps: set[str], *, wb2_range: str) -> Iterator[dict[str, Any]]:
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


def _command(
    cfg: dict[str, Any],
    *,
    url_prefix: str,
    device: str,
    epochs: int,
    n_steps: int,
    ceiling: bool,
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
        url = f"{url_prefix}_{cfg['geom']}_c{cfg['sample_chunk']}_i{inner}.zarr"
        cmd += ["--url", url, "--n-steps", str(n_steps)]
        cmd += ["--size", str(cfg["size"]), "--sample-chunk", str(cfg["sample_chunk"])]
        if cfg["inner_chunk"] is not None:
            cmd += ["--inner-chunk", str(cfg["inner_chunk"])]
    if cfg.get("sample_range"):
        cmd += ["--sample-range", cfg["sample_range"]]
    if cfg.get("max_inflight") is not None:
        cmd += ["--max-inflight", str(cfg["max_inflight"])]
    if ceiling:
        cmd += ["--ceiling"]
    return cmd


def _run_config(cmd: list[str], cfg: dict[str, Any], repeat: int, out_fh: Any) -> None:
    """Run one child, tag its rows with the config + repeat, append to the combined file."""
    with tempfile.NamedTemporaryFile("r+", suffix=".jsonl") as tmp:
        subprocess.run([*cmd, "--metrics-out", tmp.name], check=True)
        tmp.seek(0)
        for line in tmp:
            if not line.strip():
                continue
            row = json.loads(line)
            row["config"] = cfg
            row["repeat"] = repeat
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
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    sweeps = {s.strip() for s in args.sweeps.split(",") if s.strip()}
    unknown = sweeps - {"inflight", "size", "chunk", "inner"}
    if unknown:
        p.error(f"unknown sweeps: {sorted(unknown)}")

    configs = list(_configs(sweeps, wb2_range=args.wb2_range))
    # A 0.8/0.1/0.1 split needs >=3 sample-axis chunks, so the synthetic trajectory must be a
    # few chunks long; catch a too-short --n-steps here rather than as an empty-val crash later.
    for cfg in configs:
        if cfg["source"] == "synthetic" and args.n_steps < 3 * cfg["sample_chunk"]:
            p.error(
                f"--n-steps {args.n_steps} too short for sample_chunk {cfg['sample_chunk']} "
                f"(need >= 3 chunks to split; use --n-steps >= {3 * cfg['sample_chunk']})"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
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
                    device=args.device,
                    epochs=args.epochs,
                    n_steps=args.n_steps,
                    ceiling=do_ceiling,
                )
                label = f"[{i + 1}/{len(configs)}] {cfg['sweep']} {cfg['geom']}"
                extra = {
                    k: cfg[k] for k in ("max_inflight", "sample_chunk", "inner_chunk") if k in cfg
                }
                print(f"=== {label} repeat {r + 1}/{args.repeats} {extra} ceiling={do_ceiling} ===")
                _run_config(cmd, cfg, r, out_fh)
    print(f"\nwrote sweep rows -> {args.out}")
    print(f"aggregate with: uv run python -m bench.advection_report --in {args.out}")


if __name__ == "__main__":
    main()
