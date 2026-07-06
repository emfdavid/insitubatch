# Changelog

## 0.1.0 — 2026-07-06

**The sample-geometry generalization + a stable public API.** insitubatch is no longer
weather-only: the sample axis is now a *role*, not a fixed dimension, validated cross-domain
against a real bio-imaging store — and the headline classes are exposed at the package root
so the surface is exactly `insitubatch.__all__`.

- **Arbitrary sample axis** — `open_geometries(store, sample_axis=k)` lets *any single*
  physical axis be the sample axis (e.g. sample over `Z` of an OME-NGFF `(T,C,Z,Y,X)` stack),
  by keeping `shape`/`chunks` in physical order and confining one physical↔logical permutation
  to the scheduler. The common (axis-0) path is unchanged.
- **Per-variable sample-axis chunk size** — co-registered variables may chunk the sample axis
  *differently* (the OME-NGFF raw image Z-chunk 1 + label mask Z-chunk 30 pairing) as long as
  they share its *length*. The manifest defines a reference anchor grid; each variable maps
  global anchors onto its own chunk grid. Composes with windowing and arbitrary axes; covered
  including the uneven-tail case where a coarse chunk runs out at the end of the axis.
- **Cross-domain example** — `examples/microscopy/`: OME-NGFF cell segmentation streamed from
  the public IDR store, raw + mask co-batched over `Z` with no reshard, a tiny CNN beating an
  Otsu-threshold baseline. Proves the geometry generalizes beyond weather.
- **Public API surface** — `InSituDataset` and the framework adapters (`to_torch`, `to_jax`,
  `to_tf`, `as_torch`, `as_tf_dataset`) are now re-exported from the top-level package (added
  to `__all__`); import them from `insitubatch`, not submodules. Re-exports are identity and
  the adapters still import their framework lazily, so importing `insitubatch` pulls in none.
- **Advection GPU benchmark** — a stall/ceiling sweep (`bench/advection_sweep.py`) and results:
  the loader holds 94–98% of the in-memory compute ceiling on the advection forecast, i.e. it
  keeps a GPU fed; two-regime framing and figures on the benchmarks page.
- **Docs** — the sample-geometry *axis-role contract* (architecture) and *how the ladder
  evolved* (DESIGN); cross-domain use-case tables; a radio-astronomy (xradio MSv4) mapping for
  astrophysics readers.

## 0.0.3 — 2026-06-29

First **Alpha** release. Headline: the V2 decoupled fetch scheduler + the `ChunkPool`
cache are now the engine, the **torch / JAX / TF** surfaces ship with runnable
three-framework [examples](examples/), and the first real-cloud benchmark round is
published. The pre-1.0 API changes below (the v1 reader/buffer/cache stack removed in
favor of `Scheduler` + `ChunkPool`) are also called out in the GitHub release notes.

The V2 decoupled fetch scheduler (M1.6, B1) is now the training engine; the v1
shuffle-block path is retired. **Acceptance passed** on S3 (c6id.8xlarge,
fat-spatial): **1052 MB/s at `max_inflight=32`**, beating the 930 MB/s v1 peak at
the same `block_chunks=2` memory, with residency **flat at 4 chunks across the
whole `max_inflight` sweep** (8→128) — concurrency dialed independently of memory,
no oversubscription collapse.

- **`Scheduler` + `ChunkPool` replace the v1 reader+buffer** on the training path.
  Reads are flattened to *stored chunks* (`(outer, inner)` tiles) under one
  `max_inflight` budget — no nested inner/outer concurrency caps. Decoded tiles
  scatter into pre-allocated outer-chunk slots (disjoint, lock-free copies);
  residency is decoupled at `resident_cap = 2*block_chunks`. **Read concurrency
  (`max_inflight`) and shuffle span / residency (`block_chunks`) are now
  independent dials.**
- **B2 — the pool is the cache.** `ChunkPool` gains a **byte budget + pin/unpin +
  LRU** and an optional **mmap backing** (`np.lib.format.open_memmap` direct-scatter
  on NVMe — no `np.save` copy). One machinery: a small budget is read-once; a large
  budget retains drained chunks for **cross-epoch decode-once reuse** (the scheduler
  skips fetch+decode+transform for a still-resident chunk). The pool is now
  dataset-owned (persists across epochs); `InSituDataset` gains `cache_dir` and
  `cache_budget_bytes`, and `close()`. B1's `resident_cap` admission is **unified**
  into the budget (admission evicts unpinned-LRU; consumer `unpin` replaces `evict`).
- **Free-threading-ready:** pool readiness is published through a lock (not the
  GIL), so the disjoint-scatter design is correct on 3.13t as well as the GIL build.
  Validated GIL-free incl. the new pin/LRU admission.
- **Bad/truncated chunks** (`on_bad_chunk`, default `"raise"`): real GRIB-under-zarr
  archives (HRRR) have corrupt stored chunks. `"nan"` fills a failed tile with NaN
  (float) or the fill value instead of poisoning the epoch — the caller then handles
  NaN with a `chunk_transform`. The corrupt reads are listed in `ds.bad_chunks` (the
  `(array, chunk_index, inner_coord)` tiles) for logging/quarantine. A failure
  *during scatter* still poisons (a genuine bug, not a bad chunk).
- **Sample-axis subsetting:** `split_by_chunk(..., sample_range=(start, stop))`
  restricts a split to a contiguous window of the sample (time) axis — train on a
  date range of a long archive. Chunk-aligned (snaps outward to chunk bounds; whole
  chunks only). Docs show defining the window with the xarray API (`xds.sel(time=...)`)
  and translating it — xarray stays off the hot path.
- **Scaler-over-the-loader example** (`examples/fit_scaler.py`): fit a
  `sklearn.StandardScaler` with `partial_fit` while iterating once — the pass decodes +
  **caches** the raw chunks (the fit *is* the warm-up), then the fitted scaler attaches
  as a `batch_transform`. The cache stays raw/reusable; training reads decode-once
  (~20× warm vs cold even on `file://`). The familiar, cache-friendly alternative to
  the chunk-stage scaler.
- **`__version__`** now derives from package metadata (pyproject is the single
  source of truth).
- **Breaking (pre-1.0):** `buffer.py` (`ShuffleBlockBuffer`, `BufferConfig`)
  removed; the v1 `InSituDataset(cache=...)` reader intercept removed — caching is
  now the `ChunkPool` policy (`cache_dir` / `cache_budget_bytes`). `Scheduler` takes
  a caller-owned `pool`; `SchedulerConfig.resident_cap` removed (the budget governs
  residency). Observability attr `buffer_peak` → `resident_peak`. New exports:
  `Scheduler`, `SchedulerConfig`, `ChunkPool`, `StoredChunkRead`,
  `build_stored_chunk_reads`. `cache.py` (`ChunkCache`/`MemoryCache`/`DiskCache`)
  removed — the pool subsumes it.
- **Breaking (pre-1.0): the v1 streaming-reader stack is gone.** `fit_standard_scaler`
  removed (fit over the loader with sklearn `partial_fit` instead — see above);
  `io.py` (`AsyncChunkReader`, `IOConfig`) and the v1 read-plan (`build_read_plan`,
  `ReadPlan`, `dedup_ratio`) removed — they were only used by that fitter and the v1
  reader. `StandardScaler` stays as the chunk-stage applier (pass your own stats).

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
