# Architecture: loaders, parallelism, and prefetch

This doc contrasts the classic worker-based data loader with insitubatch's
async-driven engine, then specifies the prefetch pipeline and the Earth2Studio
integration surfaces. For the why behind the project see [DESIGN.md](../DESIGN.md).

## The inversion in one line

Classic `DataLoader`: parallelism lives in **`num_workers` OS processes**, each
running a *synchronous* `__getitem__`. insitubatch: parallelism lives in **one
async event loop**; batch assembly is the consumer. That move is what unlocks
async obstore, a shared chunk cache, bounded memory, and prefetch overlap.

## Classic worker-based loader

```mermaid
flowchart TB
    SRC[("cloud zarr")]:::cloud
    subgraph MAIN["main process"]
        SMP["sampler: shuffled indices"] --> IQ["index queue"]
        COL["collate + pin_memory"] --> GPU[("GPU step")]
    end
    subgraph WK["worker processes — num_workers, prefetch_factor"]
        W1["worker 1 · __getitem__ (sync)<br/>read chunk + decode"]
        W2["worker 2 · __getitem__ (sync)<br/>read chunk + decode"]
        WN["worker N · __getitem__ (sync)<br/>read chunk + decode"]
    end
    IQ -->|"fork + IPC"| W1 & W2 & WN
    SRC -->|"each worker re-reads<br/>the SAME chunks · no shared cache"| W1 & W2 & WN
    W1 & W2 & WN -->|"pickle / IPC<br/>(this queue is the prefetch buffer)"| COL
    classDef cloud fill:#cfe8ff,stroke:#379a4a;
```

Frictions against cloud ndim zarr:

- **No shared chunk cache** — a chunk is fetched + decompressed once *per worker*
  whose samples land in it.
- **Sync `getitem` can't drive async obstore** — no way to fan out concurrent
  range reads from inside a worker.
- **dask thread pool nested in each worker** — procs × threads oversubscription,
  slow fork startup, fat memory.
- Note: the worker model *does* prefetch (via `prefetch_factor` — workers run
  ahead into the IPC result queue). Prefetch is not the differentiator; **how**
  we prefetch is.

## insitubatch async-driven pipeline

```mermaid
flowchart TB
    SRC[("cloud zarr · gs:// / s3://")]:::cloud
    subgraph PLAN["planner — per epoch"]
        PERM["chunk permutation + shuffle-block order"] --> RP["read plan:<br/>samples → DEDUPED chunk reads"]
    end
    subgraph LOOP["single async event loop · one thread"]
        PROD["prefetch producer<br/>looks ahead d batches"] --> SEM["bounded fan-out · max_inflight"]
        SEM --> OBS["obstore get_ranges_async"]
        OBS --> DEC["decode · GIL released"]
        DEC --> CT["chunk_transform<br/>(vectorized)"]
    end
    CT --> CACHE["chunk cache<br/>RAM → NVMe (optional)"]
    CACHE --> BUF["shuffle-block buffer<br/>bounded residency"]
    BUF --> GTH["gather + batch_transform"]
    GTH --> PQ["prefetch queue · depth d"]
    PQ --> GPU[("device + device_transform<br/>→ model step")]
    SRC -->|"each chunk read ONCE"| OBS
    RP --> PROD
    GPU -.->|"pulls"| PQ
    PROD -.->|"refills ahead of consumption"| PQ
    classDef cloud fill:#cfe8ff,stroke:#379a4a;
```

*Target pipeline.* Built today: planner, bounded fan-out, obstore reads, decode,
chunk/batch transforms (M-T), chunk cache (M-C), buffer, gather, prefetch overlap
(M1.5), torch surface. Planned: `Regrid` + `device_transform` (M2/M3).

Properties: parallelism in the loop (not processes); each chunk read once and
amortized across every sample that touches it; residency bounded by
`max(max_inflight, block_chunks)` chunks + the prefetch queue (depth `d`) + the
cache budget — every term a tunable cap, none scaling with batch size or epoch
length.

## Prefetch

`source.InSituDataset.__iter__` runs a **background producer thread** that
assembles batches ahead of the consumer:

- ✅ **Intra-batch concurrency** — a batch's missing chunks are fetched
  concurrently via the async loop (`read_plan` fan-out under `max_inflight`). This
  is what won the ~2.8× on the GRIB regime locally.
- ✅ **Inter-batch overlap (M1.5)** — the producer assembles batches
  N+1..N+`depth` while the caller works on batch N; the consumer just drains a
  bounded queue. (A pre-M1.5 demand-driven loop left the event loop idle during
  the compute step.)

### Design (producer/consumer pipeline)

```mermaid
flowchart LR
    subgraph PRODUCER["producer thread"]
        WALK["walk draw order"] --> ASM["assemble batch<br/>(async read + gather + transforms)"]
    end
    ASM --> Q["bounded queue · maxsize d"]
    Q --> CONS["consumer __iter__<br/>pops finished batches"]
    CONS --> STEP[("train / infer step")]
    Q -.->|"full ⇒ producer blocks<br/>(backpressure)"| ASM
```

- **Producer** walks the draw order ahead of the consumer, keeping "every chunk
  needed for the next `d` batches is resident-or-in-flight," assembles batches as
  chunks land, and pushes them to a bounded `queue.Queue(maxsize=d)`.
- **Consumer** (`__iter__`) just pops finished batches → the train/infer step
  overlaps with IO+decode+assembly of the next `d` batches.
- **Backpressure / memory bound** — queue depth `d` + buffer `block_chunks` cap
  residency; a full queue pauses scheduling.
- **Granularity (v1: per batch)** — the producer assembles whole batches ahead.
  Chunk-granularity look-ahead (reads for N+2 starting before N+1 is assembled) is
  a later refinement.
- **Lifecycle** — early consumer exit sets a stop flag and drains the queue so a
  producer parked on a full `put` can exit before the reader is closed.
- **Knobs:** `prefetch_depth` (queue depth `d`), `max_inflight`, `block_chunks`.

Same shape as `torchdata.nodes.Prefetcher`, but async-native. This is what turns a
throughput win into a *GPU-fed* win.

## Transforms — three stages, placed by cost

Models need preprocessing (at minimum scaling; often regridding). The interesting
question is *where* a transform runs, because placement is a performance lever
tied to the core principle (Python work scales with **chunks, not samples**).

```
read → decode ─►[chunk_transform]─► buffer → gather ─►[batch_transform]─► DLPack ─►[device_transform]─► model
                   O(chunks)                              O(batches)                    O(batches), on GPU
                   amortized over every                   needs the                     cheap-on-device
                   sample in the chunk                    assembled batch               ops
```

1. **`chunk_transform(DecodedChunk) -> DecodedChunk`** — per-chunk, on the decode
   thread pool, **before** shuffle/gather. Amortized over every sample that draws
   from the chunk. Home for per-element, sample-order-independent ops: **scaling /
   normalization, unit conversion, dtype cast, chunk-local regrid.** Sees one
   variable, one chunk (`chunk.read.array` gives the variable).
2. **`batch_transform(Batch) -> Batch`** — per-batch, after gather. For ops that
   need the assembled batch: **cross-variable derived fields, channel stacking,
   per-sample random augmentation/crops, collation to model layout.**
3. **`device_transform`** — in the framework adapter, after DLPack, on-GPU,
   overlapping compute. For ops cheap on device (GPU normalization, batched
   interpolation, FFTs).

**Placement principle:** push each transform as early and as shared (per-chunk) as
possible; move later only when it needs the batch, per-sample randomness, or is
cheaper on-device. A per-sample transform in `__getitem__` (the torch way) redoes
work for every reused sample — we refuse that by default.

**Free advantage:** parallelism is in one event loop, not worker processes, so
transforms need **not** be picklable — stateful normalizers, closures, GPU objects
all work. torch's DataLoader forces picklable transforms across `fork`.

### Standard scaler — pre-fit GLOBAL stats (not per-chunk)

```python
@dataclass
class StandardScaler:
    """Global per-variable (optionally per-level) standardization with PRE-FIT,
    FIXED stats. Applied identically to every chunk — never recomputed per chunk."""
    mean: dict[str, np.ndarray]   # per var, shaped to broadcast: surface (1,1); per-level (level,1,1)
    std:  dict[str, np.ndarray]
    eps: float = 1e-8
    def __call__(self, chunk: DecodedChunk) -> DecodedChunk:
        m, s = self.mean[chunk.read.array], self.std[chunk.read.array]
        chunk.data = (chunk.data - m) / (s + self.eps)
        return chunk
```

Fit it **with our own infra** — one streaming, bounded-memory pass over the
training split, reusing the read plan + async reader (no separate Dask job):

```python
def fit_standard_scaler(url, manifest, geometries, split=SplitName.TRAIN,
                        keep_axes=("level",)) -> StandardScaler:
    """Per-variable sum / sumsq / count, reducing over the sample axis (+ spatial),
    keeping `keep_axes`. (Production: use Welford / a shifted mean for stability.)"""
    sums, sqs, counts = {}, {}, {}
    for var, geom in geometries.items():
        plan = build_read_plan(manifest.sample_indices(split, geom), {var: geom})
        with AsyncChunkReader(url, {var: geom}) as reader:
            for chunk in reader.read_plan(plan):
                x = chunk.data.astype("f8"); axes = _reduce_axes(geom, keep_axes)
                sums[var]   = sums.get(var, 0)   + x.sum(axes, keepdims=True)
                sqs[var]    = sqs.get(var, 0)    + (x * x).sum(axes, keepdims=True)
                counts[var] = counts.get(var, 0) + _n_reduced(x, axes)
    mean = {v: sums[v] / counts[v] for v in sums}
    std  = {v: np.sqrt(np.maximum(sqs[v] / counts[v] - mean[v] ** 2, 0)) for v in sums}
    return StandardScaler(mean, std)
```

### Regrid — precomputed weights, placement by regime

```python
@dataclass
class Regrid:
    """Bilinear lat/lon → target grid. Chunk-local (spatial dims whole per chunk).
    Weights computed ONCE; apply is a vectorized sparse gather. inner_shape changes
    consistently across chunks."""
    src_lat, src_lon, dst_lat, dst_lon: np.ndarray
    def __post_init__(self):
        self._idx, self._w = _bilinear_weights(self.src_lat, self.src_lon,
                                               self.dst_lat, self.dst_lon)
    def __call__(self, chunk: DecodedChunk) -> DecodedChunk:
        chunk.data = _apply_weights(chunk.data, self._idx, self._w)
        return chunk
```

- **Fat chunks** → `chunk_transform` (amortized over the chunk's samples).
- **ARCO `chunk-1`** → reuse the same weights as a sparse tensor in a
  `device_transform` (batched on GPU), since per-chunk == per-sample there.

## The caching continuum

**The cache boundary IS the chunk-transform boundary.** The read plan keys every
chunk as `(array, chunk_index)`, and `chunk_transforms` are deterministic and
applied before shuffle — so the cache stores the **decoded + scaled + regridded**
array, and a hit skips fetch *and* decode *and* normalize *and* regrid (not just
bytes). One localized interception at the existing keyed boundary:

```python
async def _fetch_and_decode(self, read: ChunkRead) -> DecodedChunk:
    key = (read.array, read.chunk_index, self._pipeline_fingerprint)
    hit = self._cache.get(key)
    if hit is not None:
        return hit                          # skips fetch + decode + scale + regrid
    chunk = await self._raw_fetch_decode(read)
    for t in self._chunk_transforms:        # deterministic, pre-shuffle
        chunk = t(chunk)
    self._cache.put(key, chunk)             # store the PREPPED chunk
    return chunk
```

`batch_transforms` (per-sample / random, post-shuffle) are applied *after*
retrieval and never cached. The three-stage split and the cache are the same idea
seen twice: **chunk_transforms are exactly the deterministic prefix safe to
persist.** The `fingerprint` invalidates the cache when stats or the transform
list change.

Dedup → buffer → cache is **one continuum**, all keyed `(array, chunk_index)`, all
storing post-chunk-transform arrays:

| layer | reuse scope | backing |
|---|---|---|
| read-plan dedup | within a request | — |
| shuffle-block buffer | within an epoch | RAM (bounded) |
| chunk cache | across epochs & runs | RAM LRU → optional NVMe/zarr spill |

The current buffer is the epoch-scoped special case; the cache generalizes it with
an LRU eviction policy. v1 is **RAM, cross-epoch** (`cache_chunks`, default off);
the NVMe spill tier + content fingerprint (cross-*run*) are deferred. Note this
caches the *prepped* representation — strictly stronger than MSC's raw-byte NVMe
cache for an ML pipeline.

**Heavy-reuse tasks unlocked:** multi-epoch training (epoch 0 warms it); the
fat-chunk regime (one chunk → many batches); scoring/verification (reference
chunks reused across metrics, lead times, models); HPO/sweeps (disk tier amortizes
prepped chunks across *runs*); datasets that fit in RAM/NVMe (effectively
in-memory at GPU-fed speed after the first pass).

### Future: persistent (NVMe) tier

v1 is RAM-only and cross-epoch. A persistent tier would extend reuse across
*runs* (HPO sweeps, restarts, multi-job) and feed the GDS path. Design notes:

- **The key needs a fingerprint.** RAM keys on `(array, chunk_index)` because one
  cache instance == one fixed pipeline. A cross-run key must add a fingerprint of
  (a) source identity (store URL + array + chunk version/etag) and (b) the
  chunk-transform pipeline (stats + transform list), so changed data *or*
  transforms invalidate. Fingerprinting arbitrary callables is the hard part —
  require transforms to expose a stable `version` / config hash (e.g. hash the
  `StandardScaler` stats).
- **Two levels are worth considering.** A *raw-decoded* disk tier keyed by source
  identity only (decode = the expensive cloud + decompress step) beneath the
  *prepped* RAM tier. Transform experimentation then reuses decoded chunks without
  a fingerprint, recomputing only the cheap transforms.
- **On-disk format.** Prepped arrays on local NVMe, `mmap` on read (near
  zero-copy; aligns with kvikio/GDS NVMe→GPU in M2). Packed store (lmdb/zarr) vs
  many small files trades write-amplification for simplicity.
- **Two-tier eviction.** RAM LRU (chunk count) backed by NVMe (byte budget); a RAM
  miss checks disk → `mmap`-load → promote.
- **Concurrency.** Cross-process reuse needs atomic writes / immutable files +
  light locking; readers `mmap` immutable entries.
- **GDS synergy.** A persistent NVMe tier of prepped chunks is the natural feed
  for the Phase-2 kvikio/GDS NVMe→GPU path — ties M-C's disk tier to M2.

## Earth2Studio integration

A raw `zarr.storage.ObjectStore(obstore ...)` swap in their ARCO source delivers
*faster bytes* — that's an **obstore** win, not an insitubatch one. insitubatch
earns its place by adding what obstore alone does not:

1. **Bounded fan-out** — their `gather(*tasks)` is *unbounded*; a hindcast /
   scoring request over thousands of timesteps spawns thousands of concurrent
   getitems. `max_inflight` sustains throughput at bounded memory.
2. **Read-plan dedup across a request** — ensembles (many members), multiple lead
   times, and overlapping verification windows touch the same chunks repeatedly;
   their per-`(time, var)` task model re-reads them, our plan collapses to one
   read each.
3. **Prefetch overlap** — for sequential inference (rolling through init times;
   autoregressive rollout pulling forcings each step), prefetch the next step's
   inputs *during* the current step's compute. This hides the IO time observed in
   real ensemble runs (e.g. StormCast).
4. **For training on big hindcasts (the real target)** — the whole loader: split,
   shuffle, prefetch, bounded memory.

### Where tensors are born (grounded in `run.py` / `data/utils.py`)

```
DataSource.__call__ ──► xr.DataArray         # ARCO: zarr-async + fsspec/gcsfs
        ▼
fetch_data(source, time, variable, lead_time, device)
        ▼
prep_data_array(da, device) ──► (torch.Tensor, coords)
        ▼
prognostic.create_iterator(x, coords) ──► rollout yields (torch.Tensor, coords)
```

xarray is **load-bearing all the way down to `prep_data_array`** — it carries
their lexicon/vocabulary, lat/lon coords, and optional regridding (`interp_to`).
The torch tensor only materializes at the end. So inside their inference loop
there is **no public "give me a torch batch from cloud" hook**; the only clean
public seam is the `xr.DataArray` DataSource.

### Two integration philosophies — only one is ours

- **Inside their inference loop → obstore store-swap (NOT insitubatch).** Keep
  their xarray DataSource; swap `FsspecStore(gcsfs/MSC)` → `ObjectStore(obstore)`
  for faster bytes. Having insitubatch *build* `xr.DataArray` would add a
  conversion and force us to reimplement their lexicon/coords/regrid machinery.
  That is an **obstore** contribution, not an insitubatch one.

- **Around their models → insitubatch delivers tensor batches (the real play).**
  For training, fine-tuning, and big batched hindcast/scoring, skip
  `DataSource`/`fetch_data`/xarray entirely: insitubatch reads ARCO / your zarr →
  DLPack → `(torch.Tensor, coords)` and feeds `prognostic.create_iterator(x,
  coords)` directly. The `coords` we supply is a light `OrderedDict` of
  coordinate arrays (variable names, lat/lon, lead/time) — metadata, not the
  xarray machinery. This is "closer to the GPU," many-samples-through-the-GPU, and
  exactly what our infra does.

> insitubatch never builds xarray. Stay-in-their-loop = obstore store-swap;
> batched workloads = insitubatch drives their *model* with tensor batches.

(Tell: `fetch_data(legacy=False)` already returns a **cupy-backed** `xr.DataArray`
for CUDA — NVIDIA themselves reaching for GPU-resident arrays. A fully
tensor-native fast path in E2S is conceivable later, but it is a larger change to
their framework, not v1.)

### Positioning vs NVIDIA MSC and GDS

- **MSC (Multi-Storage Client)** is an `fsspec` client (integrates with
  Zarr/Xarray via `msc://`). Its value is multi-backend access + caching (incl.
  local NVMe) + observability — not a faster cold-read primitive. Because it
  routes through fsspec, obstore can still beat it on **cold raw-read
  throughput**. MSC shines with big infra + a warm NVMe cache; insitubatch's
  niche is **cold-cache / streaming / commodity-infra / bounded-memory**.
- **GDS (GPUDirect Storage / cuFile)** is a separate path — direct DMA from
  NVMe/NVMe-oF into GPU memory. No evidence MSC uses GDS. GDS is where our
  **Phase 2 kvikio path** lives — a GPU-direct route MSC doesn't natively
  provide. (From docs, not MSC source — confirm before using in a public claim.)

## What this does NOT do (scope boundaries)

These are deliberate v1 boundaries — the design is honest about them rather than
pretending to be a general compute graph.

- **`chunk_transform` sees ONE variable and ONE chunk.** It cannot combine
  variables. So `windspeed = sqrt(U10² + V10²)` is **not** a chunk transform.
  - It *is* a **`batch_transform`** — the `Batch` holds all variables aligned on
    the sample axis (`batch.arrays["u10"]`, `batch.arrays["v10"]`), so derived
    cross-variable fields compute cleanly there. **Caveat:** batch transforms run
    *after* the cache, so a derived field is recomputed per batch/draw, not
    cached. A cached cross-variable **derived variable** (compute once from
    co-scheduled input chunks, store as a pseudo-chunk keyed like any other) is a
    deliberate **future** feature, not v1.
- **No cross-chunk / cross-sample-boundary ops.** A sample is a slice of the outer
  (sample) axis that does **not span a chunk boundary** (the v1 contract). So
  temporal stencils or windows that straddle two time-chunks (e.g. finite
  differences across the seam, or a 6-step window crossing chunk edges) are not
  supported. Windows spanning *n* chunks are a future opt-in that trades away
  zero-copy.
- **Not a compute framework.** No general task graph, no cross-chunk reductions on
  the hot path, no lazy dask-style evaluation. `fit_standard_scaler` is a
  hand-rolled streaming reduction, not a generic groupby — by design (dask on the
  hot path is the thing we route around).
- **Shuffle is approximate**, not global — chunk permutation + shuffle-block
  (`block_chunks` is the quality↔memory knob). Exact global shuffle is
  incompatible with chunk-aligned, low-copy reads.
- **Variables must share the sample axis** (same length and chunk size) — an
  enforced v1 invariant; `InSituDataset` raises `ValueError` otherwise. The draw
  order and gather use one chunk size for all variables. (`build_read_plan` can
  map per-variable chunkings, but the iteration layer does not yet; lifting this
  is future work.)

Rule of thumb: **per-variable, per-chunk, deterministic → chunk stage (cacheable).
Cross-variable or per-sample-random → batch stage (not cached). Cross-chunk →
not v1.**
