# Changelog

## Unreleased

The V2 decoupled fetch scheduler (M1.6, B1) is now the training engine; the v1
shuffle-block path is retired.

- **`Scheduler` + `ChunkPool` replace the v1 reader+buffer** on the training path.
  Reads are flattened to *stored chunks* (`(outer, inner)` tiles) under one
  `max_inflight` budget — no nested inner/outer concurrency caps. Decoded tiles
  scatter into pre-allocated outer-chunk slots (disjoint, lock-free copies);
  residency is decoupled at `resident_cap = 2*block_chunks`. **Read concurrency
  (`max_inflight`) and shuffle span / residency (`block_chunks`) are now
  independent dials.**
- **Free-threading-ready:** pool readiness is published through a lock (not the
  GIL), so the disjoint-scatter design is correct on 3.13t as well as the GIL build.
- **`AsyncChunkReader` kept** as the streaming-chunk primitive (used by
  `fit_standard_scaler`); only the v1 *training* path was removed.
- **`__version__`** now derives from package metadata (pyproject is the single
  source of truth).
- **Breaking (pre-1.0):** `buffer.py` (`ShuffleBlockBuffer`, `BufferConfig`)
  removed; `InSituDataset(cache=...)` removed — B1 is read-once. Cross-epoch reuse
  returns in B2 as a `ChunkPool` policy (same `ChunkCache` protocol);
  `MemoryCache`/`DiskCache` remain. Observability attr `buffer_peak` →
  `resident_peak`. New exports: `Scheduler`, `SchedulerConfig`, `ChunkPool`,
  `StoredChunkRead`, `build_stored_chunk_reads`.

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
