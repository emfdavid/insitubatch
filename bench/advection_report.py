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
docs/benchmarks.md).

The figure metric is chosen **per sweep**, because they don't share a signal (if plotly is
present, one interactive HTML each):

* ``size`` -- a *compute* knob (field size sets the conv cost), so it moves **steady
  throughput**: bars = samples/s (+IQR), line = steady stall %.
* ``inflight`` / ``chunk`` / ``inner`` -- *IO* knobs. Once the cross-epoch cache is warm every
  read is served from RAM, so steady throughput is flat **by construction**; the knob only
  acts on the **cold first-fill**. So these plot bars = epoch-0 time-to-first-batch (ms), line
  = epoch-0 stall %. (The tables stay on steady epochs -- that half is a throughput/ceiling
  claim; only the figure follows the varying signal.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# The knob that varies in each sweep; its column is ``cfg_{_KNOB[sweep]}``.
_KNOB = {
    "inflight": "max_inflight",
    "size": "size",
    "chunk": "sample_chunk",
    "inner": "inner_chunk",
}
# Which signal the figure follows: steady throughput vs the cold first-fill (see module docstring).
_FIG_KIND = {"size": "throughput", "inflight": "cold", "chunk": "cold", "inner": "cold"}


def load(path: str | Path) -> pd.DataFrame:
    """Flatten the nested ``config`` dict into columns (all epochs; consumers filter)."""
    raw = pd.read_json(path, lines=True)
    cfg = pd.json_normalize(raw["config"]).add_prefix("cfg_")
    return pd.concat([raw.drop(columns=["config"]), cfg], axis=1)


def _steady(df: pd.DataFrame) -> pd.DataFrame:
    """Keep steady (epoch>0) rows; fall back to all rows if a run had a single epoch."""
    steady = df[df["epoch"] > 0].copy()
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
    """One row per config of ``sweep``: median +/- IQR samples/s + stall, and % of ceiling.

    Steady (epoch>0) only; ``df`` may span all sweeps -- the ceiling is joined per ``geom`` so
    a config uses its geom's ceiling even when that ceiling was collected under another sweep."""
    df = _steady(df)
    sdf = df[df["cfg_sweep"] == sweep]
    knob = f"cfg_{_KNOB[sweep]}"  # the varying knob's column (readable leading column)
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
    knob = _KNOB[sweep]
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


def _xlabels(col: pd.Series) -> pd.Series:
    """Knob values as categorical x labels, ``NaN`` -> ``default`` (the unthrottled engine)."""
    return col.fillna(-1).astype(int).astype(str).replace("-1", "default")


def _cold_start(df: pd.DataFrame, sweep: str) -> pd.DataFrame:
    """Per-knob cold first-fill (epoch 0): median TTFB (ms) + stall over repeats, knob-sorted."""
    knob = f"cfg_{_KNOB[sweep]}"
    cold = df[(df["cfg_sweep"] == sweep) & (df["run"] == "insitu") & (df["epoch"] == 0)]
    return (
        cold.groupby(knob, dropna=False)
        .agg(ttfb_ms=("ttfb_ms", "median"), stall=("data_stall_fraction", "median"))
        .reset_index()
        .sort_values(knob, na_position="last")
        .reset_index(drop=True)
    )


def _figure(df: pd.DataFrame, agg: pd.DataFrame, sweep: str, out_dir: Path) -> str | None:
    """One HTML per sweep, plotting the signal that actually varies (see ``_FIG_KIND``)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    knob = _KNOB[sweep]
    fig = go.Figure()
    if _FIG_KIND[sweep] == "throughput":
        # Compute knob: it moves steady throughput.
        x = _xlabels(agg[f"cfg_{knob}"])
        fig.add_bar(
            x=x, y=agg["samples_per_s"], name="samples/s", error_y={"array": agg["samples_iqr"]}
        )
        fig.add_scatter(x=x, y=agg["stall"] * 100, name="stall %", yaxis="y2", mode="lines+markers")
        y_title, y2_name, subtitle = "samples/s (median, IQR)", "stall %", "steady throughput"
    else:
        # IO knob: steady throughput is cache-flat, so plot the cold first-fill it does move.
        cold = _cold_start(df, sweep)
        x = _xlabels(cold[f"cfg_{knob}"])
        fig.add_bar(x=x, y=cold["ttfb_ms"], name="cold TTFB (ms)")
        fig.add_scatter(
            x=x, y=cold["stall"] * 100, name="epoch-0 stall %", yaxis="y2", mode="lines+markers"
        )
        y_title = "cold time-to-first-batch (ms)"
        y2_name, subtitle = "epoch-0 stall %", "cold first-fill"
    fig.update_layout(
        title=f"advection sweep: {sweep} — {subtitle}",
        xaxis_title=knob,
        yaxis_title=y_title,
        yaxis2={"title": y2_name, "overlaying": "y", "side": "right", "rangemode": "tozero"},
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
        fig = _figure(df, agg, sweep, args.out)
        if fig:
            print(f"figure -> {fig}")


if __name__ == "__main__":
    main()
