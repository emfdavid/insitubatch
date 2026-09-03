# Changelog

## Unreleased

- **`InSituDataset.last_pass` — which stage was the bottleneck, and what to do about it.**
  Every producer-side problem presents the same way, as an empty batch queue: slow storage,
  a saturated decode pool, and a residency budget too small to admit the next chunk are
  indistinguishable from the consumer's seat, and they want opposite fixes. Turning the
  wrong knob is the default outcome. `last_pass` is a `PassStats` carrying sampled depths
  (batch queue, in-flight permits, decode-pool depth, residency vs budget) and cumulative
  time by stage, and `last_pass.limiting_stage` applies a rule to them and names *one*
  stage. The per-epoch INFO line ends in `limited by: <stage> -- <what to do>`.

  The counter worth having is `admission_parked_s`. A merely budget-starved loader looks
  exactly like slow storage — which is how the pre-#39 deadlock presented, "slow storage
  rather than an error" — and this is the only number that separates them. It is the
  non-terminal neighbour of the state the loader raises `residency budget exhausted` on,
  made visible before it becomes provably fatal. `inflight_peak` is also reachable at last:
  it lived on the `Scheduler`, which is created inside the pass and torn down with it, so
  nothing ever copied it out.

  **Two clocks, deliberately.** Waiting uses `perf_counter`, because there the waiting is
  the quantity; in-thread cost uses `thread_time`, because a wall clock around a thread hop
  measures GIL wait rather than work — that is how our scatter memcpy once read as 51% of
  the hot path when its real share is 7.7-10.1%. A stage timer that over-attributes is
  worse than no timer: it sends people to optimize a stage that was never the problem.
  `fetch_wait_s` is therefore summed across tiles in flight and exceeds wall time by design.

  **The collector takes no lock.** Every counter has exactly one writing thread, and decode
  — the one stage that runs on the pool's threads — measures its own `thread_time` and
  *returns* it, so the add happens back on the loop. Measured cost, interleaved against
  `main` with a same-vs-same NULL control on an 8-vCPU box reading a local NVMe store: a
  null on ordinary geometry (32-sample chunks, one tile each, 128 tiles/pass), and **~2-5%
  on a deliberately tile-heavy shape** (4-sample chunks on a 4x4 inner grid, 16,384
  tiles/pass) where the NULL's own worst pair was 6%. The cost is per tile, so it is
  largest exactly where tiles are smallest — which is also where you are least likely to be
  IO-bound.

- **`InSituDataset.describe()` / `.print_summary()` — what this configuration will cost,
  before it costs it.** The layout facts that decide whether a run is fast or twice as slow
  were only discoverable by running it: a 360-byte gather run (1.8-2.2x on the placement
  path), a chunk grid that does not divide the array so residency is 1.25x what the logical
  chunk size predicts, `block_chunks` silently widened to fit a batch, a budget sized for one
  iteration when you meant to `zip(ds.train, ds.val)`. Each of those cost someone an hour to
  find out, and none of them needs a byte read to know. The report answers from geometry and
  configuration alone — it opens no store and runs no pass, which is the whole point, because
  a report that had to touch the store would be useless in exactly the situation you want it.
  It is on demand: construction stays quiet, because layout advice on every dataset you build
  teaches people to skim our logs.

  The working-set formula the report prints is now the same function the engine sizes its
  automatic budget with (`summary.working_set_bytes`) rather than a second copy — a report
  that predicted a different number from the one the loader uses would be worse than none.
  The memory block ends in an ESTIMATED row: accounted x 1.25 for glibc's retention of freed
  slot buffers, re-measured on the chunked-slot pool (#36; it was ~1.85x before that work).
  Runtime observability — queue depths, per-stage timing, cache hits, peak residency — is a
  different surface with a different cost model and is tracked separately (#45).

- **The pool's buffer unit is now the stored chunk, and the scheduler runs on zarr's loop.**
  Two changes that had to land together, because both rewrite `Scheduler._one`. A slot used
  to be one assembled ndarray that every decoded tile was memcpy'd into; it is now the
  decoded tiles themselves, adopted by reference, with placement deferred to `gather`. And
  orchestration moved from a private per-pass event loop onto zarr's process-global one.
  What made the second possible was `zarr.core.chunk_utils.ChunkTransform.decode_chunk`
  (new in zarr 3.3.0) being *synchronous*: the async codec pipeline dispatches to the loop's
  default executor, so keeping our own decode pool meant claiming that slot — which on a
  borrowed loop retunes zarr's concurrency process-wide and then breaks it on close. Calling
  a sync decode ourselves means we never touch it.

  Deleted: the fsspec bridge (`_io`, `_foreign_loop`, `_fsspec_io_loop`) — gcsfs and obstore
  are now the same inline `await`, because we are already on the loop an async fsspec session
  binds to; the per-pass loop, scheduler thread and decode pool (three of four execution
  contexts were rebuilt every epoch); the second executor hop; and the fill-path memcpy.
  Verified byte-identical on real GCS against both backends, on our zarr-v3 store and on
  WeatherBench2's zarr-v2.

  **Sold as simplicity, not throughput.** Wall clock is a null in every warm configuration
  measured on 4 and 16 vCPU; the deleted memcpy is 7.7–10.1% of process CPU and never
  surfaced, because it was not the bottleneck. Chunked `gather` is *faster* at coarse tile
  grids (0.83–1.02x at 1–16 tiles per chunk) and slower only at the fine end (1.84–2.20x at
  256), which restates known chunking guidance rather than indicting the design.

- **A scheduler may no longer tear down anything it shares.** `close()` used to cancel every
  task on its loop via `asyncio.all_tasks`, then stop and close the loop, then shut down the
  decode pool. All three were correct while the loop was private and destructive the moment
  it was not: cancelling the whole task set raises `CancelledError` inside unrelated `zarr`
  reads in other threads, stopping the loop hangs every later `zarr.core.sync.sync()` call in
  the process, and shutting the shared pool down makes the next scheduler raise `cannot
  schedule new futures after shutdown`. The failures are races — the same arm produced them
  2/3, 3/3 or 0/3 across runs — which is why they are pinned by tests rather than left to
  review. A scheduler now cancels only the tasks it created and tears down nothing else.

- **Persisted chunks are stored tile-major.** A cache entry's `.npy` now holds
  `(n_tiles, *tile_shape)` — each stored tile contiguous, in `inner_index` order — instead of
  one assembled array. File count per chunk is unchanged. This is what keeps one code path:
  a revived chunk comes back as zero-copy views of the mapping, tiled exactly like a freshly
  fetched one, so `gather` never asks "assembled or tiled?". Contiguous-on-disk tiles also
  make a future `decode(out=)` usable on the mmap tier, not just the heap.
  **Breaking:** the manifest format is bumped to 3, so an existing `cache_dir` is rejected
  with the usual stale-cache error and rebuilt (`reset_stale_cache=True`, or delete it). A
  version-2 file read as tile-major would be plausible-looking garbage, so this is refused
  rather than reinterpreted.

- **A ragged chunk grid is now charged for the padding it actually holds.** A stored chunk
  decodes whole, and we keep it whole rather than clipping edge tiles — one buffer unit, one
  shape, which a fixed-shape arena will later need to reuse anything. So residency is
  `n_tiles * chunk_shape`, which exceeds the assembled `slot_shape` wherever the grid does
  not divide the extent: 1.248x on ERA5 721x1440 at 180x360, 1.997x on a short final outer
  chunk. The budget charges that, including on the `chunk_transform` path, which holds the
  source tiles for its whole fill and only collapses to the assembled output at completion.
  Charging the assembled size instead — as the old design did — would let a pool told
  2048 MiB resident ~2560 MiB while reporting 2048. Invisible on a grid that divides evenly,
  which is every store our benchmarks used.

  A `chunk_transform` still receives the **logical** chunk: assembly clips every tile to its
  in-bounds region first, so user code never sees stored-chunk padding, never masks an edge,
  and never has to know the chunk grid. `DecodedChunk` now says so outright, since the pool
  holding padded tiles internally makes the distinction newly worth stating.

- **`decode_threads` is now a process-wide setting, and says so when ignored.** The decode
  pool outlives every scheduler, so it is sized by the first dataset built in the process; a
  later, different value is ignored with a warning rather than silently. Thread count is a
  property of the machine, not of a dataset, so one number per process is the honest shape —
  and two datasets in one process should not run two pools competing for the same cores.

- **Under-sizing the budget for concurrent iterations now says so.** One `InSituDataset`
  owns one chunk pool and every active iteration shares it — `zip(ds.train, ds.val)`, or
  two `DataLoader`s — but each holds its *own* chunk references, so residency is the sum
  of their working sets, not the maximum. The auto-sized default covers one iteration and
  deliberately stays that way: the engine cannot know how many you intend to run, and
  guessing high would cost memory in the single-iteration case that is almost every case.
  What was missing was the diagnostic. Starvation previously advised "raise
  cache_budget_bytes, or lower batch_size / block_chunks" — correct but not actionable
  when *every* resident chunk is legitimately referenced and the caller has no way to see
  why. It now names how many iterations are sharing the pool, and the pattern that
  produces that. Owners count from mint to release rather than from their first pin,
  because the iteration that starves before it can pin anything is exactly the one that
  needs naming. `docs/tuning.md` gains the sizing rule.

- **The chunk pool's "safe to take away" predicate is now true, not approximately true.**
  Eviction eligibility was spread across five loosely-coupled fields, and each one could
  lie. `fail()` had to set `ready = True` to wake a waiter — the only lever available —
  which simultaneously declared a half-written slot a finished cache entry while sibling
  tile tasks were still writing into it, and left the poisoned slot resident so the *next*
  epoch re-raised the stale error forever instead of refetching (#33). `unpin_all()`
  cleared the pin map globally, so with two producers over one pool (`zip(ds.train,
  ds.val)`, a documented configuration) one iteration's epoch boundary stripped the
  other's pins and its in-use chunks became eviction candidates mid-gather (#34). And
  `claimed` was a single bool, so one iteration's claim satisfied another's `wait_ready`
  — that iteration then gathered a chunk it never referenced and its release decremented
  someone else's count (#35). All three produced *plausible* data, which throughput,
  shapes and smoke tests all pass; only byte fingerprints catch them.
  A slot now carries one explicit `SlotState` (`FILLING → ASSEMBLED → READY`, `FAILED`
  terminal) advanced in exactly one place, plus two counters that answer one question
  each: `writers` (tile tasks *running*, so eviction is never racing a live write) and
  `pending` (tiles not yet delivered, so completeness is separate from quiescence).
  References are owner-scoped, and a reference *is* that owner's claim, so `claimed` is
  gone. `Scheduler` takes the pool's obligation off the caller: every tile write happens
  inside `pool.tile_write`, whose scope releases on **every** exit path — including
  cancellation at an `await`, which no explicit call site can cover.
  `ASSEMBLED` is a real state, not a formality: the chunk transform and the persist
  write-back run outside the lock between the last tile landing and the slot being
  published, and a predicate derived only from a tile counter would call that window
  evictable.

- **Added: `insitubatch.print_debug_info()` — one paste instead of a dozen version questions.**
  Reports the storage stack (zarr / obstore / numpy / xarray), whichever framework adapter is
  actually installed, and the **free-threading state** — both the build flag and whether an
  import has since switched the GIL back on, which are not the same thing and have explained
  more than one "works for me". Nothing is imported to report on it (versions come from
  distribution metadata), because importing torch and JAX in one process crashes — a debug
  helper that took the process down while someone was reporting a bug would be worse than
  none. `debug_info()` returns the same facts as a dict. The new bug and performance issue
  forms ask for its output.

- **Fixed: a batch wider than a shuffle-block deadlocked the loader.** A batch draws from
  every block it spans and holds them until it has gathered, so `batch_size >
  2 × block_chunks × samples-per-chunk` needed more blocks resident than the budget floor
  provides: the fetch driver parked on admission, the consumer parked waiting for a chunk that
  could never be admitted, and **no batch was delivered at all**. It bit one-sample-per-chunk
  stores at the shipped defaults (`batch_size=64`, `block_chunks=16`) — per-frame and
  per-spectrum archives — and presented as slow storage rather than an error.
  `InSituDataset` now raises `block_chunks` to `⌈batch_size / samples-per-chunk⌉` (capped at
  the array's chunk count) and logs when it does, so a block always holds a batch. Coarsely
  chunked stores are unaffected: the requested `block_chunks` is a floor, never lowered.
- **Added: admission starvation raises instead of hanging.** If a working set exceeds its
  budget anyway, the scheduler now detects the *provably* terminal state — nothing in flight,
  every resident slot pinned, and a consumer blocked in `wait_ready` — and raises with the
  residency arithmetic and the offending chunk. The test is structural, not a timeout, so a
  merely slow consumer is never mistaken for a deadlock.
- **Project: a contributor and governance model, adopted before it is strictly needed.**
  insitubatch is maintained by one person today and that is a transitional state, so the rules
  are now written down rather than improvised at the moment they are first contested. A
  [contributing guide](https://emfdavid.github.io/insitubatch/contributing/) puts the
  load-bearing scope limits up front — the four things that will not land, and *why* — so a
  contributor can tell in a minute whether their change is compatible instead of finding out in
  review. `GOVERNANCE.md` adapts the Zarr Project's affiliated-project template: a merit-based
  core-developer group, lazy consensus with a vote as last resort, and a stated intent to seek
  Zarr Affiliated Project status. Plus Contributor Covenant 3.0, a security policy, and
  issue/PR templates — including a **performance-report form** that asks for the chunk
  geometry, loader config, hardware and a control run, because a throughput report without
  those is not actionable. AI-assisted contributions are welcome under an explicit
  *accountability* rule rather than an authorship one.

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
