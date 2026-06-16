"""One-command benchmark runner.

    uv run python -m bench                 # quick local smoke grid -> JSONL + table
    uv run python -m bench --full          # the chunk-size spectrum, more epochs
    uv run python -m bench --url-prefix s3://bucket/era5   # against pre-generated S3 data

Sweeps {chunk size x engine x cache x repeat} x epochs, appends one JSONL row per
(engine, config, epoch), and prints a table. See benchmark_plan.md.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .engines import Cfg, run
from .make_dataset import make_family
from .result import Result, append_jsonl

DEFAULT_OUT = Path(__file__).parent / "results" / "suite.jsonl"


def run_suite(
    *,
    out: str | Path = DEFAULT_OUT,
    data_dir: str | Path | None = None,
    url_prefix: str | None = None,
    storage: str = "file",
    chunk_sizes: Sequence[int] = (1, 8),
    engines: Sequence[str] = ("naive", "workers", "xbatcher", "insitu", "memory"),
    caches: Sequence[str] = ("none", "memory", "disk"),
    n_samples: int = 128,
    inner: tuple[int, ...] = (16, 16),
    batch_size: int = 16,
    block_chunks: int = 8,
    num_workers: int = 2,
    epochs: int = 2,
    compute_ms: float = 0.0,
    repeats: int = 1,
    request_payer: bool = False,
    verbose: bool = True,
) -> list[Result]:
    store_kwargs = {"request_payer": True} if request_payer else {}
    if url_prefix:
        urls = {spc: f"{url_prefix}_c{spc}.zarr" for spc in chunk_sizes}
    else:
        ddir = Path(data_dir or tempfile.mkdtemp(prefix="insitubatch-bench-data-"))
        urls = make_family(
            f"file://{ddir}/era5", tuple(chunk_sizes), n_samples=n_samples, inner=inner
        )

    scratch = Path(tempfile.mkdtemp(prefix="insitubatch-bench-cache-"))
    results: list[Result] = []
    if verbose:
        print(
            f"{'engine':8s} {'cache':6s} {'chunk':>5s} {'ep':>2s} "
            f"{'samp/s':>10s} {'MB/s':>8s} {'ttfb_ms':>8s} {'rssMB':>7s}"
        )

    for spc, url in urls.items():
        for engine in engines:
            engine_caches = caches if engine == "insitu" else ("none",)
            for cache in engine_caches:
                for rep in range(repeats):
                    cfg = Cfg(
                        engine=engine,
                        url=url,
                        storage=storage,
                        sample_chunk=spc,
                        batch_size=batch_size,
                        block_chunks=block_chunks,
                        num_workers=num_workers,
                        cache=cache,
                        compute_ms=compute_ms,
                        epochs=epochs,
                    )
                    cdir = scratch / f"{engine}_{spc}_{cache}_{rep}"
                    try:
                        rows = run(cfg, cache_dir=str(cdir), store_kwargs=store_kwargs)
                    except Exception as exc:  # noqa: BLE001 - skip a missing/failing engine
                        if verbose:
                            print(f"  skip {engine}/{cache}/c{spc}: {type(exc).__name__}: {exc}")
                        continue
                    for r in rows:
                        append_jsonl(out, r)
                        results.append(r)
                        if verbose:
                            print(
                                f"{r.engine:8s} {r.cache:6s} {r.sample_chunk:5d} {r.epoch:2d} "
                                f"{r.samples_per_s:10.1f} {r.mb_per_s:8.1f} "
                                f"{r.ttfb_ms:8.1f} {r.peak_rss_mb:7.0f}"
                            )
    if verbose:
        print(f"\nwrote {len(results)} rows -> {out}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--data-dir", default=None, help="where to write local datasets (default: temp)")
    p.add_argument("--url-prefix", default=None, help="pre-generated data: <prefix>_c<spc>.zarr")
    p.add_argument("--storage", default="file", choices=["file", "s3"])
    p.add_argument("--full", action="store_true", help="chunk-size spectrum + larger grid")
    p.add_argument("--engines", default=None, help="comma list of engines to run")
    p.add_argument("--chunk-sizes", default=None, help="comma list, e.g. 1,2,4,8,16,32")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--compute-ms", type=float, default=0.0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--request-payer", action="store_true")
    p.add_argument("--plot", action="store_true", help="render Plotly graphs after the run")
    p.add_argument("--fig-dir", default="bench/figures")
    a = p.parse_args()

    kw: dict = dict(
        out=a.out,
        data_dir=a.data_dir,
        url_prefix=a.url_prefix,
        storage=a.storage,
        epochs=a.epochs,
        compute_ms=a.compute_ms,
        repeats=a.repeats,
        num_workers=a.num_workers,
        request_payer=a.request_payer,
    )
    if a.full:
        kw.update(chunk_sizes=(1, 2, 4, 8, 16, 32), n_samples=512, inner=(64, 64), batch_size=32)
    if a.engines:
        kw["engines"] = tuple(a.engines.split(","))
    if a.chunk_sizes:
        kw["chunk_sizes"] = tuple(int(x) for x in a.chunk_sizes.split(","))
    run_suite(**kw)

    if a.plot:
        from .plot import build_figures, load, write_figures

        paths = write_figures(build_figures(load(a.out)), a.fig_dir)
        print(f"wrote {len(paths)} figures -> {a.fig_dir}")


if __name__ == "__main__":
    main()
