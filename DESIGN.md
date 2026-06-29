# insitubatch — design

> Train in place on n-dimensional cloud tensors. The loader-orchestration layer
> on top of solved async IO (obstore / zarr v3 / icechunk), with a Python hot
> path that scales with **chunks, not samples**.

## The thesis

The hard part of feeding a GPU from a cloud Zarr archive is **not** raw IO speed
anymore. obstore / icechunk / tensorstore already saturate the NIC (~37 Gbps read
on a large EC2 box; flat throughput from 800 KB to 50 MB chunks). The unsolved
part is the **training-loader orchestration that consumes that fast IO** —
read-planning, chunk-aligned splits, shuffle, bounded buffering, batch assembly,
framework handoff (torch today; JAX/TF planned) — without the per-chunk **Python
tax** throttling everything.

Evidence the IO race is over and the loader race is open:

- **light-speed-io** (Rust, io_uring, object_store; 11.2 GiB/s local) — *paused*,
  no Python API ever shipped.
- **hypergrib** (GRIB-native virtual index, ~20 GB/s) — *paused* because
  "Dynamical.org, Icechunk, modern Zarr now address most of it."
- David's own Pangeo benchmark — obstore beats the zarr3 Python backend as chunk
  count rises, but **Python per-chunk overhead bounds the minimum time**.

So insitubatch is **not** a faster IO library. It *stands on* obstore and the
zarr v3 async store and builds the layer those projects stopped one step short of.

## What it is, by contrast

| Neighbor | Why insitubatch is different |
|---|---|
| **MosaicML Streaming / WebDataset** | They require **resharding** into a sample-oriented format (MDS/tar) — a full ETL copy, and a "sample" becomes an opaque blob. insitubatch trains **in place** on the existing ndim Zarr; splits/shuffle/batches live in **coordinate space**. |
| **xbatcher + DataLoader (Earthmover stack)** | xbatcher (Earthmover / pangeo) is the standard, elegant way to *define* ndim batches; its torch-worker engine (N **processes**) is strongest at the GRIB / one-sample-per-chunk end, and elsewhere pays worker cold start, memory ∝ workers, no shared cache, and per-sample decode. insitubatch keeps those same ndim batch semantics and offers a *different engine* — one async loop — that wins **cold start** (inference) and **memory + cache + ops** across the chunk spectrum (training). Complementary tools; pick by regime (and we tune xbatcher well before any comparison). |
| **DALI / kvikio / nvCOMP** | The GPU compute/decompress path — a *peer* we interop with (cupy→dlpack→torch, optional nvCOMP), not the orchestration. |
| **anemoi-datasets** | Weather-locked, opinionated schema. We are general ndim arrays. |
| **dask / Ray Data** | General compute schedulers. We deliberately keep dask **off the hot path** (its nested thread pools inside forked workers are the problem). |

## The core inversion

Classic `DataLoader`: N OS-process workers each run a **synchronous** `__getitem__`.
Three frictions with cloud Zarr:

1. **No shared chunk cache** across workers → one chunk fetched + decompressed
   once *per worker* whose samples land in it.
2. **Sync `getitem` can't drive async obstore** → you can't fan out 200 concurrent
   range reads from inside a worker.
3. **dask thread pool nested in each worker** → procs × threads oversubscription,
   slow fork startup, fat memory.

**Inversion:** make the async IO loop the *driver* and batch assembly the
*consumer*. Parallelism moves out of `num_workers` into one asyncio event loop +
a bounded decode/shuffle buffer. Torch runs `num_workers=0`, `batch_size=None`.

## The central abstraction: the read plan

The unit of work is neither *sample* nor *chunk* — it's a **read plan**:
required samples → **deduplicated** set of chunk reads + a gather map back to
samples. This makes the whole spectrum one code path:

```
fat chunks  ──────────────────────────────► GRIB-per-timestep (degenerate)
many samples / chunk                         one sample / chunk
dedup collapses N samples → 1 read           no dedup; B samples = B reads
shared-cache + intra-chunk shuffle win       async fan-out is the whole game
```

`build_stored_chunk_reads()` is vectorized: Python touches **O(reads)**, never
O(samples).

## Sample geometry (v1 contract)

- **v1:** a sample is a slice along the **outer dimension** (axis 0; time for
  ERA5/HRRR) that **does not cross a chunk boundary**. This keeps gathers to one
  coalesced copy per chunk and preserves partial zero-copy.
- **must support now:** the degenerate end — **one slice per chunk**
  (GRIB-per-timestep). Same scheduler, fan-out ratio just slides to 1:1.
- **inner (spatial) chunking is supported** — and is how you get concurrency in
  the *fat* outer-chunk regime. The reader fetches each outer chunk with a
  full-inner `getitem`, so if the inner dims are chunked (the ARCO/ERA5 norm) zarr
  fans the read across the spatial grid; with few outer chunks, the grid is what
  keeps reads parallel. So "inner dims single-chunk" is a simplification, not a
  requirement. (Concurrency then has two dials: our `block_chunks`/`max_inflight`
  on the outer axis, and zarr's `async.concurrency` on the inner grid per read.)
- **later (opt-in):** windows spanning *n* outer chunks, and pushing a spatial
  *sub-selection* into the read (read only the inner chunks covering a crop, rather
  than reading the full field and cropping in a `batch_transform`) — both trade
  zero-copy for flexibility.

GRIB / NetCDF are consumed via a **virtual-zarr** view (virtualizarr / kerchunk /
icechunk) so the engine only ever speaks zarr-async — we never parse GRIB.

## Splits

Done **ahead of time, at chunk granularity** along the sample axis:

- prevents **leakage** (temporally adjacent, autocorrelated samples can't straddle
  train/val), and
- keeps reads **chunk-aligned** (a read serves exactly one split; no half-chunk waste).

Persisted as a `SplitManifest` (JSON) for reproducibility. Default contiguous
blocks (safest for time series); optional chunk-shuffle for exchangeable samples.

## Shuffle (the interesting compromise)

Global shuffle ⊥ chunk-aligned reads. Two-level approximation, after MosaicML
Streaming's `py1e`/`py1br`:

1. **Chunk permutation** — shuffle the order chunks are scheduled per epoch,
   keyed on `(seed, epoch)` only (canonical: hardware-independent, resumable).
2. **Shuffle-block buffer** — hold a window of `block_chunks` decoded chunks and
   draw batches across the window.

`block_chunks ≳ 10×` samples-per-chunk ≈ global quality; `block_chunks` is the
single **quality ↔ memory** knob. `shuffle_quality()` scores a draw order so the
knob can be tuned empirically. `shuffle=False` (eval / inference / reconstruction)
swaps in `sequential_order` — chunks and samples in order, no permutation. Both
order functions size a short final chunk correctly (no out-of-range draws).

## Memory model

v1 peak residency ≈ `block_chunks × outer_chunk_nbytes` (the shuffle/assembly
window) + the prefetch queue (`prefetch_depth` batches) + the chunk-cache budget
(M-C, RAM optionally spilled to NVMe). Because read concurrency follows
`block_chunks` (a block fetches its chunks), v1 **couples three things into one
knob**: read concurrency, residency, and shuffle span. Every term is still a
tunable cap that never scales with batch size or epoch length — but the coupling is
what V2 breaks.

## V2 — decoupled fetch scheduler (M1.6)

v1 fetches one *outer* chunk per `arr.getitem` (zarr stitches the inner grid),
under two nested caps: `max_inflight` (outer) and zarr's `async.concurrency`
(inner). That double-quantizes — a 15-tile field at inner-cap 10 takes 2 waves ≈
half rate — and pins read concurrency to `block_chunks`.

V2 flattens the unit of work to the **stored chunk** `(outer_id, inner_coord)` and
runs a single budget over it:

- **ReadPlan v2**: deduped stored-chunk reads in draw/priority order.
- **One semaphore = `max_inflight`** — a slot held from fetch-start to
  scatter-done, spanning all stages, across inner *and* outer. No nested caps, no
  double sawtooth; total concurrency is dialed directly.
- **Scatter-assemble**: each decoded tile is copied into its outer chunk's array
  (pool threads write *disjoint* regions — no data lock, only an atomic completion
  counter); the tile frees after the copy.
- **Two explicit caps**: ≤ `block_chunks` (+read-ahead) outer chunks resident
  (shuffle window); ≤ `max_inflight` inner tiles in flight (pipeline).

### The pipeline: three GIL-free stages

All three steps release the GIL — **fetch** (obstore/tokio, Rust), **decode**
(numcodecs C), **transform** (vectorized numpy) — so all three run *off* the loop,
in the bounded pool; the loop only does async fetch + scheduling. Fuse
decode→transform→scatter into **one** pool task per chunk (one GIL-release window,
no inter-stage handoff). Transform granularity is the wrinkle, since fetch is at
*tile* granularity but the v1 `chunk_transform` contract is *per outer chunk*:

- **elementwise** transforms (e.g. `StandardScaler`) may fuse per tile — earliest
  overlap, lowest peak.
- **spatial** transforms (regrid, anything crossing tiles) run on the *assembled*
  outer chunk — a completion-triggered pool task. (Default; opt into per-tile
  fusion via a `chunk_transform` that declares itself elementwise.)

Memory model (the user-facing invariant):

```
in-flight ≈ max_inflight × (sample_chunk · ∏inner_chunk · itemsize)  + transform scratch
resident  ≈ block_chunks × (sample_chunk · ∏inner_shape · itemsize)   # the ChunkPool
queue     ≈ prefetch_depth × batch_bytes

read_concurrency = max_inflight     (independent of block_chunks)
shuffle_span     = block_chunks
```

So `block_chunks` sets shuffle quality + residency; `max_inflight` sets network
saturation — separately. (Transform output may differ in shape — regrid — so the
pool slot is sized to the *output*; the input tile frees after.) The honest limit:
in-flight memory is `max_inflight × stored_chunk` — cheap when data is inner-chunked
(small tiles), but for *single-inner* data the stored chunk **is** the outer chunk,
so concurrency and memory collapse back together (no scheduler escapes it; rechunk
spatially or shrink `sample_chunk`). See [docs/tuning.md](docs/tuning.md).

### The buffer is the cache (ChunkPool)

Once V2 manages the slots itself, the assembly buffer **is** a pool of prepped
(decoded + chunk-transformed) chunks — exactly what a chunk cache stores. So
the shuffle-block buffer and the chunk cache collapse into one **`ChunkPool`**
parameterized by:

- **backing** — heap (numpy) *or* mmap'd `.npy` on NVMe; the scatter writes straight
  into the slot either way. mmap keeps **anon** (real pressure) low while pages stay
  reclaimable (the anon/file split we measure).
- **policy** — a byte budget + eviction. Epoch-last-use with `budget = block_chunks`
  → today's read-once buffer; large budget + LRU → cross-epoch cache. *Same machinery* —
  caching stops being an optional intercept and becomes "raise the budget, switch to
  LRU, pick mmap backing."

Caveats: (1) **mmap isn't free for read-once** — scattering into mmap is NVMe write
traffic even when never reused, so default the pool to **heap** for streaming and
use mmap only to spill a working set past RAM or for cross-epoch reuse. (2)
**cross-*run* persistence is still extra** — intra-run/cross-epoch reuse is
intrinsic (just don't evict), but surviving process exit needs a stable content key
+ index rebuild on reopen (the deferred cache item below).

Internals are **validated** (`bench/spike_v2_decode.py`, zarr 3.2): key via
`chunk_key_encoding.encode_chunk_key`, bytes via `store.get`, decode via
`codec_pipeline.decode([(buf, ArraySpec)])`, scatter into the outer array — matches
`arr.getitem` exactly for single-inner and spatial layouts incl. partial edge chunks.

### Phasing

- **B1 ✅** — V2 scheduler + a **heap `ChunkPool`** subsuming `ShuffleBlockBuffer`
  (cache off, read-once). Landed the throughput/memory decoupling first.
- **B2 ✅** — `ChunkPool` gains a **byte budget + pin/unpin + LRU** and an optional
  **mmap backing** (`open_memmap` direct-scatter). One machinery: a small budget is
  read-once; a large budget (`cache_budget_bytes`) retains drained chunks for
  cross-epoch decode-once reuse (the scheduler skips fetch+decode+transform on a
  still-resident chunk). The pool is dataset-owned (persists across epochs); B1's
  `resident_cap` admission unified into the budget (consumer `unpin` replaces
  `evict`, eviction is unpinned-LRU). Remaining: cross-*run* persistence (stable
  content key + index rebuild on reopen).

Demonstrate: on `era5_fatspatial`, plot throughput **and** peak heap vs concurrency
— v1 (concurrency = `block_chunks`) rises in both; V2 (`block_chunks` fixed small,
`max_inflight` swept) reaches the ~1 GB/s knee at flat, low memory. Plus the
measured de-quantization (inner-grid sweep at a fixed budget: v1 sawtooth, V2 flat).

### v1 baseline & acceptance (exp_c — S3, c6id.8xlarge)

Fat data (`sample_chunk=200`, ~830 MB outer chunks), insitu MB/s, median of 5:

| block_chunks (≈ resident) | single-inner | spatial (grid 15, read_conc 16) |
|---|---:|---:|
| 2 (~3.3 GB) | 76 | **930** |
| 4 (~6.6 GB) | 120 | 871 |
| 8 (~13 GB) | 178 | 724 (oversubscribed) |

Spatial peaks at `bc=2` (~30 in-flight ≈ the network knee) and *falls* as `bc`
rises — the nested caps overshoot. **V2 acceptance test:** match ~930 MB/s at
`block_chunks ≤ 2` with a single `max_inflight ≈ 32` budget, and show throughput
*flat* (not falling) + memory flat as `max_inflight` rises past the knee. Datasets
persist in S3 (`era5_fat.zarr`, `era5_fatspatial.zarr` under the bench bucket);
re-probe with:

```
uv run python -m bench.probe_decode --url s3://$BUCKET/era5_fatspatial.zarr --var t2m \
  --max-chunks 16 --repeats 5 --decode-threads 8 --block-chunks 2 --max-inflight 8,16,32,64 --no-raw
```

Section 1b sweeps `max_inflight` at fixed `block_chunks=2`: V2 passes if throughput
rises to the network knee and stays *flat* (not falling, as v1's nested caps did
when oversubscribed) while `resident` stays pinned at `2*block_chunks` independent
of `max_inflight`.

**V2 result (✅ passed, c6id.8xlarge, fat-spatial, median of 5):**

| `max_inflight` | 8 | 16 | **32** | 64 | 128 |
|---|---:|---:|---:|---:|---:|
| MB/s | 388 | 788 | **1052** | 970 | 970 |
| `resident` (chunks) | 4 | 4 | 4 | 4 | 4 |

V2 **beats** the v1 spatial peak (1052 vs 930) at the *same* low memory (`bc=2`,
`resident=4 ≈ 3.3 GB`), and — the thesis — residency is **flat at `2*block_chunks`
for every `max_inflight`**, so concurrency is now dialed independently of memory.
Past the knee (`mi≈32`) throughput settles to a **stable 970 plateau** (64 and 128
identical) instead of v1's collapse to 724 under oversubscription. The ~8% settle
from the 1052 peak is benign oversubscription, not a sawtooth; the sweet spot is
`max_inflight ≈ 32` (the ~30-in-flight network knee).

**Confirmed post-B2 (the admission rewrite `resident_cap` → byte-budget pin/LRU did
not regress the thesis).** Re-running on the same box: throughput flat at the plateau
across the sweep — `981 / 981 / 988` MB/s at `mi = 32 / 64 / 128` — with `resident = 4`
at *every* `max_inflight`. The plateau sits a touch below the original 1052, inside
the cold-S3 run-to-run spread (the small sample is noisy by design); the shape — rise
to the knee, flat after, residency pinned — is intact.

**Cache (B2), on S3:** with `cache_budget_bytes` large enough to hold the probed
window (mmap on NVMe), epoch 0 cold `930` MB/s → epoch 1 **cached `2314` MB/s
(2.5×)** — the cached epoch is served from the pool with no GET and no decode. The
multiple grows with colder S3 or heavier decode; 2.5× is the conservative read.

### B1 task list

1. ✅ **`build_stored_chunk_reads`** — deduped stored-chunk reads `(outer, inner)` in
   draw/priority order (`plan.py`).
2. ✅ **`Scheduler`** — one `asyncio.Semaphore(max_inflight)` over fetch+decode+scatter
   tasks; chunk-major priority; scatter into pre-allocated outer arrays (disjoint
   writes, lock-published completion counter). Residency admission is a second
   semaphore over outer positions. Spike `bench/spike_v2_decode.py` validated the
   fetch+decode path. (Per-tile transform fusion deferred; transforms run on the
   assembled chunk — see the pipeline section.)
3. ✅ **Heap `ChunkPool`** — allocate / scatter / wait_ready / gather / evict / fail;
   residency window = `2*block_chunks`; subsumes `ShuffleBlockBuffer` (deleted).
4. ✅ **Wire into `source.py`** — V2 *replaces* v1 (no flag; the recorded exp_b/exp_c
   baselines are the A/B reference). The v1 streaming reader (`AsyncChunkReader`) and
   read plan (`build_read_plan`) were later removed as dead.
5. ✅ **Validate** — local parity green (`test_pool`, `test_scheduler`, `test_source`)
   and exp_c acceptance **passed** on fat-spatial S3 (1052 MB/s at `mi=32`, beats the
   930 v1 peak; residency flat at `2*block_chunks` across the sweep — see the result
   table above). Next: B2 (mmap backing + LRU; the pool subsumes the cache).

## Module map

| Module | Role |
|---|---|
| `types.py` | `ArrayGeometry`, `ChunkRead`, `DecodedChunk`, `Batch` |
| `split.py` | chunk-aligned `SplitManifest`, `split_by_chunk` (+ `sample_range` subsetting) |
| `store.py` | `store_from_url` shim (local↔S3 via obstore) + geometry introspection |
| `shuffle.py` | chunk permutation + shuffle-block / sequential order + quality metric |
| `plan.py` | `build_stored_chunk_reads` — a draw order's chunks → deduped stored-chunk (tile) reads for the scheduler |
| `scheduler.py` | `Scheduler` — one `max_inflight` budget over stored-chunk reads; fetch→decode→scatter; residency admission |
| `pool.py` | `ChunkPool` — the assembly buffer *and* the cache: byte budget + pin/unpin + LRU, heap or mmap'd-`.npy` backing (try_admit/scatter/wait_ready/gather/unpin) |
| `source.py` | `InSituDataset` — framework-neutral iterable of numpy `Batch`; prefetch producer over the scheduler+pool, block-granular eviction (inherits nothing) |
| `frameworks.py` | thin optional DLPack adapters over the numpy `Batch`: `as_torch` (DataLoader subclass), `to_jax`, `to_tf` / `as_tf_dataset` (from_generator) |
| `transforms.py` | chunk/batch transform hooks, `StandardScaler` (chunk-stage applier; fit over the loader — see `examples/fit_scaler.py`) |

## Open questions / spikes

- **Decompression — resolved stance (was "the next wall").** The chunk cache
  changes the calculus: for reuse-heavy workloads (multi-epoch / fat-chunk / HPO /
  scoring) decode is paid *once* per chunk then served from the host cache, so it
  is a warm-up cost, not a steady-state wall. It only stays a per-step wall in the
  cold / streaming / doesn't-fit-cache regime. **Decision: the default chunk stage
  is firmly CPU** (numcodecs decode + vectorized chunk_transform, GIL-released,
  threaded → overlaps IO) feeding a **host** cache (RAM→NVMe, cheap + spillable).
  GPU decode (nvCOMP) is a *separate* **Config B (Phase-2, GPU-native)** path —
  obstore/kvikio(+GDS) → GPU → nvCOMP → cupy → DLPack — for cold-streaming on GPU
  boxes. The two are largely **mutually exclusive** within one pipeline (host
  cache wants host-resident chunks; GPU decode wants GPU-resident chunks), so this
  is a config choice by workload, not a competing implementation. The remaining
  spike (folded into the M1 codec sweep): measure the CPU chunk-stage ceiling
  (`n_cores × (decode + transform)`) vs NIC throughput — that ratio decides *when*
  a workload must switch to Config B.
- **GIL**: even with Rust IO, Python decode/assembly can choke — so the standing
  rule is **chunk transforms must be vectorized numpy** (numcodecs C codecs and
  big-array numpy ops release the GIL; a pure-Python transform would serialize and
  kill the threaded overlap). Treat free-threaded 3.13t as *upside*, not a
  dependency; still must win on stock CPython via async + coalescing.
  - **Validated on 3.13t (B1):** the engine runs correctly GIL-free — the pool's
    disjoint lock-free scatter + lock-published readiness hold under true parallel
    execution (`test_pool_concurrent_scatter_is_race_free`, the `test-freethreaded`
    CI job). **Caveat (running FT cleanly is gated upstream, not by us):** `numcodecs`
    has not yet declared itself GIL-safe, so importing the codec stack *re-enables*
    the GIL on 3.13t. We override with `PYTHON_GIL=0` (its codecs already release the
    GIL, so this is safe in practice). What numcodecs shipping `Py_MOD_GIL_NOT_USED`
    would buy is running FT *without* the override — **not a speedup**: throughput is
    GIL-independent by design (fetch/decode/scatter all release the GIL, scheduling is
    one asyncio loop), so 3.13t *matches* the GIL build rather than beating it. Decode
    even parallelizes under the GIL (zstd releases it — the `decode_threads` sweep
    scales on the GIL build). The 3.13t value is correctness + future-proofing; *not
    depending* on the GIL is the win.
- **Cross-variable derived fields** — reads already co-schedule per-variable
  chunkings (the plan keys each variable by its own chunk size); the open
  part is a *cached* derived variable (e.g. windspeed), which needs sample-axis
  aligned inputs (deferred — see the limitations in docs/architecture.md).
- **Determinism + resumption** across epochs and DDP ranks (canonical-node style;
  `state_dict` à la torchdata `StatefulDataLoader`).
- **DDP**: shard *chunks* across ranks.

## Status

**Maturity: Alpha** — validated on real cloud IO; API is pre-1.0 (breaking changes
allowed). This section is the single source of truth for status; other pages link here.

**Phase 1 (real S3) validated.** Run on a `c6id.8xlarge` against in-region S3
(ERA5-shaped `721×1440` fields, `sample_chunk=8`, warm), insitubatch delivers
**~8× the throughput** of a tuned `xbatcher`/worker `DataLoader` baseline (swept to
32 workers) and **~10× lower** time-to-first-batch — the map-style baseline
re-decodes a whole chunk per sample (the ~8× ≈ `sample_chunk`); insitubatch reads
each chunk once. It saturates ~85% of the raw-GET ceiling at `block_chunks=32`. See
[the benchmarks page](https://emfdavid.github.io/insitubatch/benchmarks/).

The diagnosis that got there: the throughput wall was **read concurrency** (pinned
at a fixed `max_inflight`), not decode or bandwidth. Fixes shipped — concurrency
follows `block_chunks`, a bounded decode pool, S3 warm-up before timing,
inner-chunk support, and one-block read-ahead so block-boundary IO overlaps compute
(M1.5). The probe (`bench/probe_decode.py`) separates network vs decode on any
store.

Built so far: planner, chunk-aligned splits, async obstore reads + bounded decode
pool, coalesced gather, the **torch / JAX / TF framework surfaces (M3)** — DLPack
`to_jax` / `to_tf` / `as_tf_dataset` over the numpy `Batch`, exercised by the
three-framework advection examples — **chunk/batch transforms + `StandardScaler`
(M-T)**, **prefetch (M1.5)**, runnable examples + a published docs site, and the
**V2 decoupled fetch scheduler (M1.6, B1)** — `Scheduler` + `ChunkPool` are now the
training engine (one `max_inflight` budget over stored chunks, residency decoupled
at `2*block_chunks`); the v1 shuffle-block buffer is retired. **Acceptance passed**
on fat-spatial S3: 1052 MB/s at `max_inflight=32` (beats the 930 v1 peak) with
residency flat across the concurrency sweep. **B2 done** — the `ChunkPool` is now
the cache too (byte budget + pin/unpin + LRU, heap or mmap backing; cross-epoch
decode-once reuse via `cache_dir`/`cache_budget_bytes`). **Not yet built:** cross-*run*
cache persistence, `Regrid` + the GPU/device transform stage (M2), bounded read-ahead (M-RA).

## Roadmap / milestones

Perf track (the core thesis):
- **M0 — local proof** ✅ real obstore IO, naive baseline, ~2.8× on GRIB regime.
- **M1 — CPU EC2 / S3** run the harness against real S3 (us-east-1, c7i/m7i,
  Spot); decode-codec sweep to measure the CPU chunk-stage ceiling vs NIC (the
  one remaining decompression spike — see Open questions).
- **M1.5 — prefetch** ✅ background producer + bounded queue (`prefetch_depth`)
  overlap IO/decode/assembly with the consumer step; backpressure + early-exit
  cleanup; tests assert the producer runs ahead. The producer walks shuffle-blocks
  and reads **one block ahead** (a read-ahead thread fetches block N+1 while the
  consumer drains block N), so block-boundary IO overlaps the per-batch compute
  instead of stalling. Validated on WeatherBench2/GCS: at a realistic train step
  the per-block sawtooth disappears (boundary waits 0.1 ms); at *zero* compute the
  loader is IO-throughput-bound and the stall is only smoothed, not removed — that
  is the network ceiling, not a scheduling bug.
- **M1.6 — decoupled fetch scheduler (V2). ✅ B1 (acceptance passed: 1052 MB/s, residency flat).**
  Flatten to stored-chunk granularity with one `max_inflight` budget over
  inner+outer reads (full design above): decouples read concurrency from
  residency/shuffle and kills the nested-cap sawtooth. `Scheduler` + `ChunkPool`
  replace `AsyncChunkReader`/`ShuffleBlockBuffer` on the training path; the
  zarr-internals path (fetch encoded bytes → `codec_pipeline` decode → scatter,
  first proven in `bench/spike_v2_decode.py`) is the live engine, covered by
  `test_pool`/`test_scheduler` parity + bound tests. Supersedes the one-block
  look-ahead in `source.py`. exp_c fat-spatial acceptance passed (see the result
  table). Remaining: B2 (mmap/LRU pool backing).
- **M-RA — bounded read-ahead (decouple prefetch depth from the cache budget).** Today the
  scheduler's read-ahead depth is gated *only* by the pool's byte budget: admission
  (`try_admit`) parks just when the budget is full. A large `cache_budget_bytes`
  (decode-once retention, e.g. `--cache-resident`) removes that backpressure, so the driver
  eagerly admits + fetches the *whole* resident set at once. On a high-latency /
  bandwidth-limited network this starves the current block — the first batch competes with
  the whole-subset prefetch, inflating cold TTFB (observed on the WeatherBench2 Arraylake
  example: ~7 s without the cache → ~50 s with `--cache-resident`, since `max_inflight`
  worth of fetches fan out across the entire split). As a stopgap the example uses a low
  `--max-inflight` so it doubles as a read-ahead throttle, but that couples two dials.
  **Fix:** a fetch-ahead cap — a semaphore released on the consumer's `unpin` — that bounds
  how far ahead of the consumer the driver admits, *independent* of the retention budget.
  Then the cache retains consumed chunks (fast warm epochs) without eager whole-split
  fetching (fast cold TTFB), and `max_inflight` goes back to being purely the concurrency
  dial. Applies to both the V2 main engine and the M-W branch (whose residency rework did
  not add a read-ahead cap either).
- **M2 — GPU full scale** kvikio/cupy/nvCOMP, dlpack→torch; prove GPU saturation
  with bounded host memory.

Engine track (make it real for models — see [docs/architecture.md](docs/architecture.md)):
- **M-T — transforms.** ✅ `chunk_transform` + `batch_transform` hooks wired,
  `StandardScaler` (chunk-stage applier) + the recommended **fit-over-the-loader**
  pattern (sklearn `partial_fit`, warms the cache — `examples/fit_scaler.py`), incl.
  cross-variable windspeed at the batch stage. Pending: `Regrid` (precomputed
  weights) and `device_transform` (with the M3 adapters).
  Scope limits hold: chunk transforms are single-variable/single-chunk;
  cross-variable (e.g. windspeed) is batch-stage and uncached; cross-chunk is not
  v1.
- **M-C — chunk cache.** ✅ Caching **is** the `ChunkPool` policy (B2): a byte
  budget + pin/unpin + LRU, with heap or mmap'd-`.npy` backing (NVMe page cache,
  bounded working set). Raise `cache_budget_bytes` past the working set and drained
  chunks are retained for cross-epoch **decode-once** reuse (the scheduler skips
  fetch+decode+transform on a hit); `cache_dir` spills to NVMe. The v1 `cache=`
  intercept and the standalone cache classes were retired — caching is no longer a
  separate intercept, it became "don't evict." Deferred: an L1/L2 (RAM+NVMe) tier and
  cached cross-variable derived variables (cross-*run* persistence is M-C2).
- **M-C2 — cross-run persistent cache.** The mmap'd `.npy` slots under `cache_dir`
  already *survive* the process (they are files on NVMe), but nothing rebuilds the
  index to reuse them — so each new run re-fetches + re-decodes from S3. Add a
  **persisted slot index** (`(array, chunk_index) → file`, with shape/dtype and the
  chunk-transform version) rebuilt on startup, gated by a **content fingerprint**
  (store URL + zarr metadata + codec + chunk-transform identity) so stale slots are
  invalidated when the data or pipeline changes. Then decode-once amortizes across
  *runs*, not just epochs — a relaunched job, a hyperparameter sweep, or several
  processes on one box read decoded chunks straight from NVMe (no S3, no decode).
  Scope: a read-through cache keyed by fingerprint; a shared/networked cache tier and
  the L1/L2 split above are later.
- **M-W — windowed / multi-offset sampling (sample geometry v2).** The forecasting
  unlock, and a prerequisite for the canonical WeatherBench examples and the M4
  "around their models" play. Today a batch draws **one shared time index for all
  variables**, so input@t / target@t+1 is inexpressible; shuffle destroys temporal
  adjacency (so a `batch_transform` can't pair times either); and a precomputed
  shifted-target var means **resharding — impossible on read-only public ERA5/HRRR**.
  Fix: a sample becomes an **anchor `t` + a window spec** mapping roles
  (`input`/`target`) to sets of time-offsets; each `(var, t+offset)` stays a
  *within-chunk single-index read* (the no-cross-chunk-**read** invariant holds — what
  generalizes is that one *sample* may now reference several chunks). Gather runs per
  role; `Batch` grows a lead axis (`x: {var:(B,L_in,*inner)}`,
  `y: {var:(B,L_out,*inner)}`); anchors with any offset outside `[0,T)` are dropped at
  manifest build. Shuffle/`split_by_chunk` operate on **anchors** (with a guard band so
  a target can't leak across a split boundary); residency/pinning is **per-window** (a
  sample pins the union of chunks its offsets touch — the block budget sizes to the
  span), and offsets that share a chunk decode once (overlapping windows → strong
  cross-sample cache reuse). The lead axis flows through `batch_transform` → the DLPack
  adapters unchanged.

  **Scope decision: multi-offset sampling is sufficient for Earth2Studio and for
  training forecasters; promoting a window to a single cross-chunk contiguous-slab read
  is an explicit non-goal.** Every temporal access in E2S/forecasting is a *finite set
  of discrete offsets from an anchor* — rollout reads only the initial condition
  (FourCastNet/Pangu/SFNO: 1 step; GraphCast: 2 = offsets `{-1,0}`), the model generates
  futures; scoring reads ground truth at enumerable lead times; training pairs an input
  history with a target horizon. And multi-offset sampling already *expresses a
  contiguous window* as the offset range `0..L-1` over the **same** chunks a slab read
  would touch (we decode chunk-granular, so L point-gathers from a decoded chunk are
  cheap) — so a cross-chunk slab read buys only marginal gather-ergonomics for very long
  windows. Recorded so it isn't relitigated. Validation: a synthetic *advected-field*
  forecast where a tiny model beats persistence, then the WeatherBench framework
  examples ride on top.

  **As built (PR #4, through Phase 3) — deviations from the sketch above.** The engine
  stayed simpler than the role/lead-axis design:
    - A variable is a `(label, path, offset)` **view**; a window is just several views
      of one array (`{"x": g, "y": g.shift(1)}`). `Batch` stays a **flat `{label:
      array}` dict** — *no* lead axis, *no* `input`/`target` roles in the engine. It
      carries an `offsets` metadata dict; `Batch.read_indices(label)` and
      `Batch.stack(labels)` are the projection helpers (the user composes the window).
    - **No guard band.** The engine enforces *range validity only* (`anchor+offset ∈
      [0,T)`, `valid_anchor_range` drops edge anchors); offset 0 is not special
      ("relativity"). Partitioned anchors give disjoint train/val labels for any offset
      (injective translation), and the shared boundary timestep is only ever a val
      *input*, never scored — so a guard band is unnecessary. Task meaningfulness
      (non-degenerate offset choices) is the **user's** responsibility, not the engine's.
    - Residency is **reference-counted** (`pool._pinned: dict[key,int]`, not a boolean
      set) because a windowed chunk can feed several blocks; a per-epoch `claimed` flag
      orders pin→consume→release across the epoch boundary (fixes a cross-epoch
      gather/leak race). windowed+shuffle holds the **whole split** resident (shuffle
      spills reads into arbitrary blocks); bounded-residency windowed+shuffle (per-block
      re-fetch) is deferred. Decode-once across views holds (slots key on `path`).
    - **Splits are iterable views, not a constructor arg.** `InSituDataset` is
      split-agnostic; you iterate `ds.train` (shuffled) / `ds.val` / `ds.test` / `ds.all`
      (deterministic), all sharing **one** `ChunkPool`. This matters *because* of windows:
      a windowed read near a split boundary spills into the adjacent split's chunk, so the
      splits' *chunk reads* overlap even though no sample leaks (the disjoint-anchor proof
      is about samples, not chunks) -- one shared pool decodes those boundary chunks once,
      where separate per-split datasets would decode them twice. Per-split *config* (e.g.
      train-only augmentation) is intentionally not supported on one instance -- make a
      second dataset for that; the default path stays clean.

  **Deferred — refcount hot-path cost (accept for now).** The refcounted residency +
  offset-aware `gather` add ~**3.5%** to the *non-windowed* path at the **GRIB-end local
  worst case** (one sample/chunk, ~36 KB blobs where decode is ≈ free, GIL build) — not
  residency (`resident` unchanged) and not the per-block setup (a fast-path there moved
  nothing); it is per-batch gather + per-chunk `dict`/`claimed` bookkeeping under
  `pool._cv`. It shrinks toward noise with real chunk sizes (decode/IO dominate) and is
  invisible on blob-store datasets. The deliberate microscope for this — and for any
  future *locking* work — is a **tiny local store with negligible decode** under
  **free-threading** (`PYTHON_GIL=0`, where `pool._cv` is the only contention point);
  the harness recipe is in [bench/benchmark_plan.md](bench/benchmark_plan.md)
  ("Windowed-sampling overhead"). Correctness was the bar for PR #4; the lever, if real-
  `c1`-on-S3 ever shows it matters, is a non-windowed `gather` fast-path.

  **Phase 4 (examples) — the advected-field forecast. ✅ implemented (`examples/advection/`).**
  One `InSituDataset` feeds three framework training loops, showing the numpy `Batch` +
  DLPack thesis end to end; on the synthetic field the same tiny CNN beats persistence
  ~1.6× in torch, JAX, and TF (identical dataset + architecture). Plan as built:
    - **Task:** a *direct* 24-hour forecast. Inputs `{t2m, u10, v10}` at anchor `t`
      (cross-variable, same index) → target `t2m` at `t + 24h` (`g.shift(horizon)`):
      `horizon = 24` steps on the 1-hourly synthetic store, `4` on the 6-hourly
      WeatherBench2 store. The 24 h displacement is what makes advection (and the wind
      inputs) *matter* vs persistence. The input window is assembled with
      `Batch.stack(["t2m","u10","v10"])` → `(B, 3, H, W)`; the target is
      `batch.arrays["t2m_next"]`.
    - **No autoregressive rollout.** Chaining `n+1 → n+2 …` is a modeling/operational
      choice, not the data layer's job — insitu delivers `(inputs@t, target@t+h)` pairs;
      composing forecasts at serving time is the user's. Inference is a single forward
      pass: forecast 24 h from the last available data.
    - **One dataset → torch / JAX / TF.** `examples/advection/data.py` builds the store +
      the single `forecast_dataset()`; `train_{torch,jax,tf}.py` each define the *same*
      tiny CNN (3→1 channels) and a framework-native loop over the shared dataset via the
      adapters (`as_torch` / `to_jax` / `as_tf_dataset`). Near-identical files differing
      only in framework calls — the parallelism is the message. Shared numpy eval
      (`rmse`, persistence baseline) in `data.py`.
    - **Two stores, honest framing.** Synthetic advected field (seeded, time-constant
      spatially-varying wind; tests + the pedagogical "beats persistence" result, true by
      construction). WeatherBench2 6-hourly ARCO (`--wb2`, the real cloud store the
      existing `wb2_*` examples use) runs the *same code* on real ERA5 — claim "same
      pipeline, real data, no reshard", **not** SOTA skill (t2m persistence at a 24 h
      multiple is a strong baseline; don't over-claim). A third real source may be
      swapped in by URL + variable names.
    - **Tests:** torch is in CI (`--extra torch`) — test data-gen advects, the dataset
      yields the right windowed pairs, and torch training beats persistence on the
      synthetic. JAX/TF gated with `importorskip` (their extras), mirroring
      `test_jax.py` / `test_tf.py`.

Reach track (broaden + make a splash):
- **M3 — framework surfaces (done).** The core `Batch` is numpy and the engine
  inherits no framework; handoff is thin DLPack adapters in `frameworks.py`
  (`as_torch` / `to_jax` / `to_tf` + `as_tf_dataset`), each an optional extra.
  The shapes are the ecosystems': torch needs a `Dataset` subclass, JAX iterates
  directly (no native loader; the weather/climate frontier — GraphCast, NeuralGCM
  — is JAX/torch, not TF), TF wraps via `tf.data.from_generator`. Same async
  engine, multiple framework fronts. Next: `device_transform` (on-device, post
  DLPack) and `prefetch(AUTOTUNE)` tuning land with the GPU path (M2).
- **M4 — NVIDIA Earth2Studio target** (grounded in `data/arco.py`, `run.py`,
  `data/utils.py`). Their pipeline is `DataSource → xr.DataArray → fetch_data →
  prep_data_array → (torch.Tensor, coords) → model.create_iterator`. xarray is
  load-bearing down to `prep_data_array`. Two integrations, only one is ours
  (details in [docs/architecture.md](docs/architecture.md)):
    - **Inside their inference loop = obstore, not insitubatch.** ARCO is already
      zarr-v3-async; only the store backend differs (`FsspecStore(gcsfs/MSC)` vs
      `ObjectStore(obstore gs://)`). A cold-cache backend-swap benchmark is a
      clean **obstore** win (ARCO is `...chunk-1.zarr-v3` = our GRIB regime).
      insitubatch building `xr.DataArray` would just reimplement their
      lexicon/coords/regrid — not worth it.
    - **Around their models = the insitubatch play.** For training / fine-tuning
      / big batched hindcast & scoring, feed `prognostic.create_iterator(x,
      coords)` tensor batches straight from insitubatch (zarr → DLPack → torch),
      bypassing DataSource/fetch_data/xarray. `coords` is a light OrderedDict,
      not the xarray machinery. This is the "closer to the GPU" headline.
      **Depends on M-W** (multi-offset sampling) for the input-history / lead-time
      supervision — which is sufficient here (no cross-chunk window reads needed).
  Honesty bar: NVIDIA prefers **MSC** (fsspec-based; obstore can still beat it
  cold) and caches via `AsyncCachingFileSystem`, so target MSC *cold-cache* on an
  IO-bound workload (scoring/hindcast/large or lagged ensembles, NOT a single-IC
  ensemble, which is rollout-bound). GFS/GRIB: later, our degenerate sweet spot
  via virtual-zarr.

Backlog (deferred, unscheduled):
- **Bad/corrupt chunks — Phase 2.** Phase 1 shipped (`on_bad_chunk="raise"|"nan"`,
  `ds.bad_chunks`). Deferred: a bad-chunk **registry** + an optional pre-scan
  validation pass that records bad chunks into the `SplitManifest` so training
  excludes them up front; and an **end-to-end HRRR (GRIB-under-zarr) showcase**
  (`on_bad_chunk="nan"` + a fill `chunk_transform`) once a public HRRR-via-virtualizarr
  store is handy. (Sample-level NaN drop stays out — it breaks the fixed-shape gather;
  chunk-level manifest exclusion is the sane "drop.")
- **Time-coordinate helper.** `split_by_chunk(sample_range=…)` (chunk-aligned) shipped
  with the documented xarray pattern; a no-xarray
  `samples_for_time_range(url, "time", start, stop) → indices` reading the 1-D coord
  via the store is a small nice-to-have.
- **Spatial / inner-dimension subsetting.** Outer (time) subsetting shipped
  (`sample_range`); selecting a **lat/lon region or a level subset** still reads the
  whole field and crops at the batch stage (wasteful IO). The on-thesis fix selects
  only the stored chunks intersecting the inner region in the plan (coordinate-space,
  like the outer split), so IO scales with the subregion, not the full grid.
- **FT bench methodology.** Prewarm + controlled `decode_threads` sweep landed in the
  bench refresh; remaining is reporting the **p50/p95 distribution** (not just
  median/min-max) for the noisy free-threaded probe points.
