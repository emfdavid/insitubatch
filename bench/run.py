"""One-command benchmark runner.

    uv run python -m bench                 # quick local smoke grid -> JSONL + table
    uv run python -m bench --full --plot   # chunk-size spectrum + worker/compute sweeps
    uv run python -m bench --url-prefix s3://bucket/era5 --request-payer   # S3 data

Sweeps {chunk size x engine x cache x num_workers x compute_ms x repeat} x epochs,
appends one JSONL row per (engine, config, epoch), prints a table. The num_workers
sweep applies only to the DataLoader engines (workers, xbatcher) — tune them to
their best (benchmark_plan.md) so the comparison isn't a strawman; the compute_ms
sweep feeds the prefetch-overlap graph (G3).
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
_DATALOADER_ENGINES = {"workers", "xbatcher"}


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
    epochs: int = 2,
    worker_sweep: Sequence[int] = (0,),
    compute_ms_sweep: Sequence[float] = (0.0,),
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
            f"{'engine':8s} {'cache':6s} {'chunk':>5s} {'nw':>3s} {'cms':>5s} {'ep':>2s} "
            f"{'samp/s':>10s} {'MB/s':>8s} {'ttfb_ms':>8s} {'rssMB':>7s}"
        )

    for spc, url in urls.items():
        for engine in engines:
            engine_caches = caches if engine == "insitu" else ("none",)
            nw_values = tuple(worker_sweep) if engine in _DATALOADER_ENGINES else (0,)
            for cache in engine_caches:
                for nw in nw_values:
                    for cms in compute_ms_sweep:
                        for rep in range(repeats):
                            cfg = Cfg(
                                engine=engine,
                                url=url,
                                storage=storage,
                                sample_chunk=spc,
                                batch_size=batch_size,
                                block_chunks=block_chunks,
                                num_workers=nw,
                                cache=cache,
                                compute_ms=cms,
                                epochs=epochs,
                            )
                            cdir = scratch / f"{engine}_{spc}_{cache}_w{nw}_c{cms}_{rep}"
                            try:
                                rows = run(cfg, cache_dir=str(cdir), store_kwargs=store_kwargs)
                            except Exception as exc:  # noqa: BLE001 - skip a failing/missing engine
                                if verbose:
                                    print(
                                        f"  skip {engine}/{cache}/c{spc}/w{nw}: "
                                        f"{type(exc).__name__}: {exc}"
                                    )
                                continue
                            for r in rows:
                                append_jsonl(out, r)
                                results.append(r)
                                if verbose:
                                    print(
                                        f"{r.engine:8s} {r.cache:6s} {r.sample_chunk:5d} "
                                        f"{r.num_workers:3d} {r.compute_ms:5.0f} {r.epoch:2d} "
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
    p.add_argument("--full", action="store_true", help="chunk-size spectrum + sweeps")
    p.add_argument("--engines", default=None, help="comma list of engines to run")
    p.add_argument("--chunk-sizes", default=None, help="comma list, e.g. 1,2,4,8,16,32")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--num-workers", default="0", help="DataLoader worker counts to sweep (comma)")
    p.add_argument("--compute-ms", default="0", help="per-batch compute ms to sweep (comma)")
    p.add_argument("--repeats", type=int, default=1)
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
        repeats=a.repeats,
        request_payer=a.request_payer,
        worker_sweep=tuple(int(x) for x in a.num_workers.split(",")),
        compute_ms_sweep=tuple(float(x) for x in a.compute_ms.split(",")),
    )
    if a.full:
        kw.update(
            chunk_sizes=(1, 2, 4, 8, 16, 32),
            n_samples=512,
            inner=(64, 64),
            batch_size=32,
            worker_sweep=(2, 4, 8),
            compute_ms_sweep=(0.0, 10.0),
        )
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
