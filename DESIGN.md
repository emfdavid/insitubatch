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
| **xbatcher + DataLoader (Earthmover stack)** | Great batch *definition*; weak *engine* — rides torch worker processes (no async, no shared cache, redundant reads). We keep ndim-native batch semantics, replace the engine with async-IO-as-driver. |
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

`build_read_plan()` is vectorized: Python touches **O(reads)**, never O(samples).

## Sample geometry (v1 contract)

- **v1:** a sample is a slice along the **outer dimension** (axis 0; time for
  ERA5/HRRR) that **does not cross a chunk boundary**. This keeps gathers to one
  coalesced copy per chunk and preserves partial zero-copy.
- **must support now:** the degenerate end — **one slice per chunk**
  (GRIB-per-timestep). Same scheduler, fan-out ratio just slides to 1:1.
- **later (opt-in):** windows spanning *n* chunks, trading zero-copy for
  flexibility.

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

Peak residency ≈ `max(max_inflight, block_chunks) × chunk_nbytes` — **independent
of batch size and epoch length**. Prefetch (M1.5) adds a bounded queue
(`prefetch_depth` batches) and the chunk cache (M-C) adds an explicit, bounded
budget (RAM, optionally spilled to NVMe). Every term is a tunable cap; none scale
with batch size or epoch length. That directly answers the "low overhead vs
chunk/batch size" goal.

## Module map

| Module | Role |
|---|---|
| `types.py` | `ArrayGeometry`, `ChunkRead`, `DecodedChunk`, `Batch` |
| `plan.py` | `ReadPlan`, `build_read_plan` (samples → deduped reads) |
| `split.py` | chunk-aligned `SplitManifest`, `split_by_chunk` |
| `store.py` | `store_from_url` shim (local↔S3 via obstore) + geometry introspection |
| `io.py` | `AsyncChunkReader` — one event loop, bounded fan-out, real zarr-async reads |
| `shuffle.py` | chunk permutation + shuffle-block / sequential order + quality metric |
| `buffer.py` | `ShuffleBlockBuffer` — residency + coalesced batch gather |
| `source.py` | `InSituDataset` (IterableDataset) — prefetch producer, last-use eviction, optional torch handoff |
| `transforms.py` | chunk/batch transform hooks, `StandardScaler`, `fit_standard_scaler` (Regrid + device stage: follow-up) |
| `cache.py` | `ChunkCache` protocol + `MemoryCache` (heap) / `DiskCache` (mmap NVMe), byte-LRU of prepped chunks keyed `(array, chunk_index)` |

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
- **Cross-variable derived fields** — reads already co-schedule per-variable
  chunkings (`build_read_plan` keys each variable by its own chunk size); the open
  part is a *cached* derived variable (e.g. windspeed), which needs sample-axis
  aligned inputs (deferred — see the limitations in docs/architecture.md).
- **Determinism + resumption** across epochs and DDP ranks (canonical-node style;
  `state_dict` à la torchdata `StatefulDataLoader`).
- **DDP**: shard *chunks* across ranks.

## Status

**Phase 0 complete (local).** Real obstore-backed zarr v3 async reads are wired
end-to-end via `store_from_url` (one URL → local `file://` now, `s3://` later).
`bench/make_dataset.py` generates datasets; the one-command `bench` suite
(`uv run python -m bench`) compares insitubatch against the baselines (naive,
workers, xbatcher, memory/ceiling) and logs JSONL + Plotly graphs.

Early signal on **local disk**: the degenerate GRIB-per-timestep regime (1
sample/chunk) is already **~2.8× faster** than naive sync via async fan-out, with
bounded memory. The fat-chunk regime is overhead-bound locally (no latency to
hide) — its win is expected to appear on S3.

Prefetch (M1.5) is implemented: a background producer assembles batches ahead of
the consumer through a bounded queue (`prefetch_depth`), overlapping IO + decode +
assembly with the compute step. Batch-granularity; chunk-granularity look-ahead is
a later refinement. See [docs/architecture.md](docs/architecture.md).

Built so far: planner, chunk-aligned splits, async obstore reads, shuffle-block
buffer, coalesced gather, torch surface, **chunk/batch transforms +
`StandardScaler`/`fit_standard_scaler` (M-T)**, **prefetch (M1.5)**, **chunk cache
(M-C)**. **Not yet built:** `Regrid` + the GPU/device transform stage (M2),
JAX/TF surfaces (M3). Next is **Phase 1** — run the harness on a CPU EC2 instance
against S3 (us-east-1, c7i/m7i, Spot) with the decode-codec sweep.

## Roadmap / milestones

Perf track (the core thesis):
- **M0 — local proof** ✅ real obstore IO, naive baseline, ~2.8× on GRIB regime.
- **M1 — CPU EC2 / S3** run the harness against real S3 (us-east-1, c7i/m7i,
  Spot); decode-codec sweep to measure the CPU chunk-stage ceiling vs NIC (the
  one remaining decompression spike — see Open questions).
- **M1.5 — prefetch** ✅ background producer + bounded queue (`prefetch_depth`)
  overlap IO/decode/assembly with the consumer step; backpressure + early-exit
  cleanup; tests assert the producer runs ahead. Batch-granularity (chunk-level
  look-ahead is a later refinement).
- **M2 — GPU full scale** kvikio/cupy/nvCOMP, dlpack→torch; prove GPU saturation
  with bounded host memory.

Engine track (make it real for models — see [docs/architecture.md](docs/architecture.md)):
- **M-T — transforms.** ✅ `chunk_transform` + `batch_transform` hooks wired,
  `StandardScaler` + `fit_standard_scaler` (one streaming pass with our own
  reader), 6 tests incl. cross-variable windspeed at the batch stage. Pending:
  `Regrid` (precomputed weights) and `device_transform` (with the M3 adapters).
  Scope limits hold: chunk transforms are single-variable/single-chunk;
  cross-variable (e.g. windspeed) is batch-stage and uncached; cross-chunk is not
  v1.
- **M-C — chunk cache.** ✅ Pluggable byte-bounded LRU of *prepped* chunks
  (`ChunkCache` protocol): **`MemoryCache`** (heap) and **`DiskCache`** (mmap'd
  `.npy` on local NVMe — RAM footprint becomes reclaimable page cache, working set
  stays bounded). Caller-owned, passed via `cache=`; intercepted in
  `_fetch_and_decode`; a hit skips fetch + decode + transforms. Generalizes the
  epoch buffer into the dedup→buffer→cache continuum; unlocks multi-epoch /
  fat-chunk / scoring / HPO reuse. Tests assert decode-once across epochs for both
  backends. Deferred: cross-*run* index rebuild + content fingerprint, an L1/L2
  (RAM+NVMe) tier, and cached cross-variable derived variables.

Reach track (broaden + make a splash):
- **M3 — framework surfaces.** The core `Batch` is numpy; frameworks are thin
  DLPack adapters, never core deps. Add **JAX first** (no native loader; the
  weather/climate frontier — GraphCast, NeuralGCM — is JAX/torch, not TF), then
  a **TF** surface via `tf.data.from_generator` + `prefetch(AUTOTUNE)`
  opportunistically. Same async engine, multiple framework fronts.
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
  Honesty bar: NVIDIA prefers **MSC** (fsspec-based; obstore can still beat it
  cold) and caches via `AsyncCachingFileSystem`, so target MSC *cold-cache* on an
  IO-bound workload (scoring/hindcast/large or lagged ensembles, NOT a single-IC
  ensemble, which is rollout-bound). GFS/GRIB: later, our degenerate sweet spot
  via virtual-zarr.
