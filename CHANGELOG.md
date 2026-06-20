# Changelog

## 0.0.2

First results on real cloud IO, and the tuning model behind them.

- **Benchmarked on S3** (`c6id.8xlarge`, in-region): ~8× throughput and ~10× lower
  time-to-first-batch vs a *tuned* xbatcher/worker `DataLoader` baseline (swept to
  32 workers). The ~8× ≈ `sample_chunk` — the map-style baseline re-decodes a whole
  chunk per sample; insitubatch reads each chunk once.
- **Read concurrency follows `block_chunks`** (`max_inflight` defaults to it) — the
  fix for the throughput wall (it was concurrency, not decode or bandwidth).
  Saturates ~85% of the raw-GET ceiling.
- **Bounded decode pool** (`IOConfig.decode_threads`) and a **`read_concurrency`**
  inner-fan-out knob.
- **One-block read-ahead** so block-boundary IO overlaps the per-batch compute.
- **Inner (spatial) chunking** supported end-to-end; `make_dataset --inner-chunks`.
- **Examples**: a WeatherBench2 dataloader (insitubatch) and the xbatcher stack
  with a `spawn`/`forkserver`/`forkserver-preload` startup comparison.
- **Docs site** (MkDocs → GitHub Pages): architecture, benchmarks, tuning,
  WeatherBench2 walkthrough, API reference.
- **Bench/diagnostics**: `block_chunks` axis, `--max-batches`, `--caches`, S3
  warm-up, a progress counter, an `RssAnon`/`RssFile` memory split, and the
  `probe_decode` network-vs-decode diagnostic.
- **V2 decoupled fetch scheduler** designed and de-risked (one concurrency budget
  over inner+outer chunks; buffer-as-cache) — not yet built.

## 0.0.1

Initial release — PyPI name claim; core async engine.
