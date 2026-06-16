"""Plot the benchmark JSONL into interactive Plotly graphs (one HTML each).

    uv run python -m bench.plot --in bench/results/suite.jsonl --out bench/figures

Builds whichever of the planned graphs (benchmark_plan.md, G1-G7) the data
supports — a graph whose axis doesn't vary is skipped. Repeats are aggregated by
median; the DataLoader engines are compared at their **best** num_workers (so the
baselines are tuned, not strawmanned). HTML is dependency-free; for static
PNG/SVG add kaleido and `fig.write_image(...)` later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def load(jsonl_path: str | Path) -> pd.DataFrame:
    df = pd.read_json(jsonl_path, lines=True)
    if not df.empty:
        df["engine_label"] = df.apply(
            lambda r: (
                f"{r.engine}+{r.cache}" if r.engine == "insitu" and r.cache != "none" else r.engine
            ),
            axis=1,
        )
    return df


def _median(df: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    keys = [k for k in keys if k in df.columns]
    return df.groupby(keys, as_index=False)[value].median()


def _best(df: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    """Median over repeats, then the BEST num_workers per group (tuned baseline)."""
    m = _median(df, value, [*keys, "num_workers"])
    return m.groupby([k for k in keys if k in m.columns], as_index=False)[value].max()


def build_figures(df: pd.DataFrame) -> dict[str, object]:
    figs: dict[str, object] = {}
    if df.empty:
        return figs
    warm = df[df["epoch"] == int(df["epoch"].max())]
    base = warm[warm["compute_ms"] == warm["compute_ms"].min()]  # pure-IO slice

    # G1 — throughput vs sample-chunk size (best workers per engine; memory = ceiling)
    if df["sample_chunk"].nunique() > 1:
        d = _best(base, "samples_per_s", ["engine_label", "sample_chunk", "storage"])
        figs["g1_throughput_vs_chunk"] = px.line(
            d,
            x="sample_chunk",
            y="samples_per_s",
            color="engine_label",
            facet_col="storage" if d["storage"].nunique() > 1 else None,
            log_x=True,
            markers=True,
            title="G1 Throughput vs sample-chunk size (memory = in-memory ceiling)",
        )

    # G2 — ablation bars (best workers per engine)
    d = _best(base, "samples_per_s", ["engine_label", "sample_chunk"])
    figs["g2_ablation"] = px.bar(
        d,
        x="engine_label",
        y="samples_per_s",
        facet_col="sample_chunk" if d["sample_chunk"].nunique() > 1 else None,
        title="G2 Ablation - throughput by engine (best num_workers)",
    )

    # G3 — throughput vs compute_ms (prefetch overlap; best workers per engine)
    if df["compute_ms"].nunique() > 1:
        d = _best(warm, "samples_per_s", ["engine_label", "compute_ms"])
        figs["g3_throughput_vs_compute"] = px.line(
            d,
            x="compute_ms",
            y="samples_per_s",
            color="engine_label",
            markers=True,
            title="G3 Throughput vs per-batch compute (prefetch overlap)",
        )

    # G4 — cache cold vs warm across epochs (insitu only)
    insitu = df[(df["engine"] == "insitu") & (df["compute_ms"] == df["compute_ms"].min())]
    if not insitu.empty and insitu["epoch"].nunique() > 1:
        d = _median(insitu, "samples_per_s", ["cache", "epoch", "sample_chunk"])
        figs["g4_cache_epochs"] = px.line(
            d,
            x="epoch",
            y="samples_per_s",
            color="cache",
            markers=True,
            facet_col="sample_chunk" if d["sample_chunk"].nunique() > 1 else None,
            title="G4 Cache: cold (epoch 0) vs warm (epoch 1+)",
        )

    # G5 — peak RSS by engine
    d = _best(base, "peak_rss_mb", ["engine_label", "sample_chunk"])
    figs["g5_peak_memory"] = px.bar(
        d,
        x="engine_label",
        y="peak_rss_mb",
        facet_col="sample_chunk" if d["sample_chunk"].nunique() > 1 else None,
        title="G5 Peak RSS by engine (MB)",
    )

    # G6 — time-to-first-batch by engine
    d = _best(base, "ttfb_ms", ["engine_label", "sample_chunk"])
    figs["g6_ttfb"] = px.bar(
        d,
        x="engine_label",
        y="ttfb_ms",
        facet_col="sample_chunk" if d["sample_chunk"].nunique() > 1 else None,
        title="G6 Time-to-first-batch (ms)",
    )

    # G7 — DataLoader tuning curve: throughput vs num_workers (workers/xbatcher)
    dl = base[base["engine"].isin(["workers", "xbatcher"])]
    if not dl.empty and dl["num_workers"].nunique() > 1:
        d = _median(dl, "samples_per_s", ["engine_label", "num_workers", "sample_chunk"])
        figs["g7_worker_tuning"] = px.line(
            d,
            x="num_workers",
            y="samples_per_s",
            color="engine_label",
            markers=True,
            facet_col="sample_chunk" if d["sample_chunk"].nunique() > 1 else None,
            title="G7 Baseline tuning - throughput vs num_workers",
        )
    return figs


def write_figures(figs: dict[str, object], outdir: str | Path, *, cdn: bool = False) -> list[Path]:
    # cdn=True loads plotly.js from a CDN instead of inlining ~3.5 MB per file,
    # so the figures are small enough to commit and embed (the docs site iframes them).
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    plotlyjs = "cdn" if cdn else True
    paths = []
    for name, fig in figs.items():
        p = out / f"{name}.html"
        fig.write_html(p, include_plotlyjs=plotlyjs)  # type: ignore[attr-defined]
        paths.append(p)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="infile", default="bench/results/suite.jsonl")
    p.add_argument("--out", dest="outdir", default="bench/figures")
    p.add_argument(
        "--cdn", action="store_true", help="load plotly.js from CDN (small files for the docs site)"
    )
    a = p.parse_args()
    paths = write_figures(build_figures(load(a.infile)), a.outdir, cdn=a.cdn)
    print(f"wrote {len(paths)} figures -> {a.outdir}")
    for p_ in paths:
        print(f"  {p_}")


if __name__ == "__main__":
    main()
