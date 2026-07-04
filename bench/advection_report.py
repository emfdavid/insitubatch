"""Aggregate an advection sweep JSONL into the finding: median +/- IQR tables and one figure
per sweep.

    uv run python -m bench.advection_report --in bench/results/advection_sweep.jsonl \
        --out bench/figures

Reads rows written by ``bench.advection_sweep`` (each an ``examples._forecast_metrics``
row plus a ``config`` dict and ``repeat``). For every config it takes the **steady** epochs
(epoch > 0, dropping the cold first pass), medians per repeat, then reports the
median and inter-quartile range across repeats -- so no single run carries the claim.

The compute-only ceiling is joined per ``geom`` (field geometry = the conv cost), giving each
insitu config its ``% of ceiling``. Prints a Markdown table per sweep (paste-ready for
docs/benchmarks.md) and, if plotly is present, writes an interactive HTML figure per sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load(path: str | Path) -> pd.DataFrame:
    """Flatten the nested ``config`` dict into columns and keep steady (epoch>0) rows."""
    raw = pd.read_json(path, lines=True)
    cfg = pd.json_normalize(raw["config"]).add_prefix("cfg_")
    df = pd.concat([raw.drop(columns=["config"]), cfg], axis=1)
    steady = df[df["epoch"] > 0].copy()
    # Fall back to all rows if a run only had one epoch (no warm rows to keep).
    return steady if not steady.empty else df


def _per_repeat(df: pd.DataFrame, run: str, keys: list[str]) -> pd.DataFrame:
    """Median throughput/stall per (config, repeat) for one run, over its steady epochs."""
    sub = df[df["run"] == run]
    return (
        sub.groupby([*keys, "repeat"], dropna=False)
        .agg(samples_per_s=("samples_per_s", "median"), stall=("data_stall_fraction", "median"))
        .reset_index()
    )


def _iqr(s: pd.Series) -> float:
    return float(s.quantile(0.75) - s.quantile(0.25))


def summarize(df: pd.DataFrame, sweep: str) -> pd.DataFrame:
    """One row per config of ``sweep``: median +/- IQR samples/s + stall, and % of ceiling."""
    sdf = df[df["cfg_sweep"] == sweep]
    # The knob that varies in this sweep (for a readable leading column).
    knob = {
        "inflight": "cfg_max_inflight",
        "size": "cfg_size",
        "chunk": "cfg_sample_chunk",
        "inner": "cfg_inner_chunk",
    }[sweep]
    keys = [knob, "cfg_geom"]

    insitu = _per_repeat(sdf, "insitu", keys)
    agg = (
        insitu.groupby(keys, dropna=False)
        .agg(
            samples_per_s=("samples_per_s", "median"),
            samples_iqr=("samples_per_s", _iqr),
            stall=("stall", "median"),
            n=("repeat", "count"),
        )
        .reset_index()
    )

    # Ceiling per geom (median over its repeats/epochs), joined for % of ceiling.
    ceil = (
        df[df["run"] == "ceiling"]
        .groupby("cfg_geom")["samples_per_s"]
        .median()
        .rename("ceiling_samples_per_s")
    )
    agg = agg.join(ceil, on="cfg_geom")
    agg["pct_of_ceiling"] = 100 * agg["samples_per_s"] / agg["ceiling_samples_per_s"]
    return agg.sort_values(knob).reset_index(drop=True)


def to_markdown(agg: pd.DataFrame, sweep: str) -> str:
    knob = {
        "inflight": "max_inflight",
        "size": "size",
        "chunk": "sample_chunk",
        "inner": "inner_chunk",
    }[sweep]
    knob_col = f"cfg_{knob}"
    lines = [
        f"### {sweep}",
        f"| {knob} | samples/s (med) | IQR | stall | % of ceiling | n |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in agg.iterrows():
        val = "default" if pd.isna(r[knob_col]) else int(r[knob_col])
        pct = "-" if pd.isna(r["pct_of_ceiling"]) else f"{r['pct_of_ceiling']:.1f}%"
        lines.append(
            f"| {val} | {r['samples_per_s']:.1f} | ±{r['samples_iqr']:.1f} | "
            f"{r['stall']:.1%} | {pct} | {int(r['n'])} |"
        )
    return "\n".join(lines)


def _figure(agg: pd.DataFrame, sweep: str, out_dir: Path) -> str | None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    knob = {
        "inflight": "max_inflight",
        "size": "size",
        "chunk": "sample_chunk",
        "inner": "inner_chunk",
    }[sweep]
    x = agg[f"cfg_{knob}"].fillna(-1).astype(int).astype(str).replace("-1", "default")
    fig = go.Figure()
    fig.add_bar(
        x=x, y=agg["samples_per_s"], name="samples/s", error_y={"array": agg["samples_iqr"]}
    )
    fig.add_scatter(x=x, y=agg["stall"] * 100, name="stall %", yaxis="y2", mode="lines+markers")
    fig.update_layout(
        title=f"advection sweep: {sweep}",
        xaxis_title=knob,
        yaxis_title="samples/s (median, IQR)",
        yaxis2={"title": "stall %", "overlaying": "y", "side": "right", "rangemode": "tozero"},
        legend={"orientation": "h"},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"advection_{sweep}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return str(path)


def main() -> None:
    p = argparse.ArgumentParser(description="aggregate the advection sweep JSONL")
    p.add_argument("--in", dest="inp", required=True, help="sweep JSONL from bench.advection_sweep")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "figures")
    args = p.parse_args()

    df = load(args.inp)
    for sweep in ["inflight", "size", "chunk", "inner"]:
        if not (df["cfg_sweep"] == sweep).any():
            continue
        agg = summarize(df, sweep)
        print("\n" + to_markdown(agg, sweep) + "\n")
        fig = _figure(agg, sweep, args.out)
        if fig:
            print(f"figure -> {fig}")


if __name__ == "__main__":
    main()
