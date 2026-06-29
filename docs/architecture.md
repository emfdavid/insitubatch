# Architecture: loaders, parallelism, and prefetch

This doc contrasts the classic worker-based data loader with insitubatch's
async-driven engine, then specifies the prefetch pipeline and the Earth2Studio
integration surfaces. For the why behind the project see
[DESIGN.md](https://github.com/emfdavid/insitubatch/blob/main/DESIGN.md).

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
- **The fork-safety tax** — a modern object store (obstore) runs a Rust **tokio**
  runtime. `fork` (the Linux default for workers) copies only the calling thread
  and leaves that runtime's threads dead with their locks held, so the *first read*
  in a forked worker **deadlocks**. Every escape is a cost the process model
  imposes: `spawn` (relaunch the interpreter per worker), `forkserver` (keep a
  pristine pre-fork server around), or a stack that rebuilds its loop on a PID
  change (s3fs/gcsfs) — and even that only if the store is *reopened in the worker*,
  never inherited across the fork. An obstore handle opened pre-fork deadlocks; a
  gcsfs one raises `Future attached to a different loop`. We hit both: the obstore
  `workers` baseline hung under fork, and the gcsfs xbatcher example raised — so
  fork is off the table for async cloud stores.
- Note: the worker model *does* prefetch (via `prefetch_factor` — workers run
  ahead into the IPC result queue). Prefetch is not the differentiator; **how**
  we prefetch is.

> **Why this is the argument for the single loop.** Every friction above — no
> shared cache, sync IO that can't drive obstore, thread oversubscription, the
> fork-safety tax — follows from putting parallelism in OS *processes*.
> insitubatch drives one in-process event loop (`num_workers=0`): there is no
> fork, so there is no fork-safety tax, no per-worker runtime to relaunch, and the
> chunk cache and the obstore runtime are simply shared. The deadlock we hit
> benchmarking the baseline is a symptom of the very thing we replace.

### Startup latency — the inference angle

Training amortizes worker spin-up over many epochs, so the start-method tax above
mostly disappears. **Inference does not**: you typically make a single pass from a
cold loader, and a long-lived server holding a `DataLoader` open (pinned workers,
held file handles) is rare. There, time-to-first-batch is dominated by process
startup. The worker model's best case is `forkserver` with
`set_forkserver_preload([...])` — heavy imports paid once in the server, forked
workers skip them. insitubatch's first batch is just the first read: no processes
to start. The two runnable examples make this concrete and measurable:
[`examples/wb2_dataloader.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/wb2_dataloader.py)
(insitubatch) and
[`examples/wb2_xbatcher.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/wb2_xbatcher.py)
(`--compare` prints TTFB across `spawn` / `forkserver` / `forkserver-preload`),
both on the public WeatherBench2 ERA5.

## insitubatch async-driven pipeline

```mermaid
flowchart TB
    SRC[("cloud zarr · gs:// / s3://")]:::cloud
    subgraph PLAN["planner — per epoch"]
        PERM["chunk permutation + shuffle-block order"] --> RP["read plan:<br/>samples → deduped STORED-chunk (tile) reads"]
    end
    subgraph LOOP["scheduler · single async event loop · one thread"]
        PROD["prefetch producer<br/>looks ahead d batches"] --> SEM["one budget · max_inflight tiles"]
        SEM --> SKIP{"pool already<br/>holds it?"}
        SKIP -->|"hit"| POOL
        SKIP -->|"miss"| OBS["obstore get (stored chunk)"]
        OBS --> DEC["decode · GIL released"]
        DEC --> CT["chunk_transform (vectorized)"]
        CT --> SCAT["scatter tile → outer-chunk slot"]
    end
    SCAT --> POOL["ChunkPool — byte budget + pin/LRU<br/>buffer AND cache · heap or mmap NVMe"]
    POOL --> GTH["gather + batch_transform"]
    GTH --> PQ["prefetch queue · depth d"]
    PQ --> GPU[("device + device_transform<br/>→ model step")]
    SRC -->|"each stored chunk read ONCE"| OBS
    RP --> PROD
    GPU -.->|"pulls"| PQ
    PROD -.->|"refills ahead of consumption"| PQ
    classDef cloud fill:#cfe8ff,stroke:#379a4a;
```

The unit of work is the **stored chunk** (an `(outer, inner)` tile), and one
`max_inflight` budget spans every fetch — so read concurrency is one dial, with no
nested inner/outer caps. Each decoded tile is scattered into its outer chunk's slot
in the **ChunkPool**, which is the assembly buffer *and* the cache in one: a byte
budget with pin/unpin + LRU, backed by heap or an mmap'd `.npy` on NVMe. Before
fetching, the scheduler asks the pool whether it already holds a chunk — a hit
(cross-epoch, since the pool persists) skips fetch + decode + transform entirely.

Properties: parallelism in the loop (not processes); each stored chunk read once
and amortized across every sample that touches it; **read concurrency
(`max_inflight`) and residency (the pool's byte budget) are independent dials**;
total memory is the budget + the prefetch queue (depth `d`) + the in-flight tiles —
every term a tunable cap, none scaling with batch size or epoch length.

## Prefetch

`source.InSituDataset.__iter__` runs a **background producer thread** that
assembles batches ahead of the consumer:

- ✅ **Intra-batch concurrency** — a batch's missing chunks are fetched
  concurrently via the async loop (stored-chunk fan-out under `max_inflight`).
- ✅ **Inter-batch overlap** — the producer assembles batches N+1..N+`depth`
  while the caller works on batch N; the consumer just drains a bounded queue. (A
  demand-driven loop would leave the event loop idle during the compute step.)

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

- **Producer** starts the scheduler over the epoch's chunks, then per shuffle-block
  waits the block assembled, gathers its batches, and unpins it, pushing batches to
  a bounded `queue.Queue(maxsize=d)`.
- **Consumer** (`__iter__`) just pops finished batches → the train/infer step
  overlaps with IO+decode+assembly of the next `d` batches.
- **Backpressure / memory bound** — queue depth `d` + the pool's byte budget cap
  residency; a full queue pauses the consumer, a full budget pauses admission.
- **Continuous fetch** — the scheduler keeps `max_inflight` tiles in flight across
  block boundaries, and the budget (sized to ~two blocks) lets it admit the next
  block while the current one drains, so block-boundary IO overlaps compute. (At
  zero per-batch compute the loader is IO-throughput-bound, so the boundary is only
  smoothed, not removed — the network ceiling, not a scheduling gap.)
- **Lifecycle** — early consumer exit sets a stop flag and drains the queue so a
  producer parked on a full `put` can exit before the scheduler is closed.
- **Knobs:** `prefetch_depth` (queue depth `d`), `max_inflight`, `block_chunks`,
  `cache_budget_bytes`.

Same shape as `torchdata.nodes.Prefetcher`, but async-native. This is what turns a
throughput win into a *GPU-fed* win.

## Trade-offs: chunk size, shuffle window, concurrency, batch size

Four dials shape throughput and memory, and the engine keeps them as independent as possible
so you can move one without paying on the others. This is the model behind them; for the
values to actually set, see [Tuning](tuning.md).

| dial | what it trades | bounded by |
|---|---|---|
| **stored-chunk size** (`inner_chunks`, write time) | fetch granularity ↔ per-request overhead | a tile of ~10–50 MB |
| **read concurrency** (`max_inflight`) | network saturation ↔ in-flight memory | the store's raw-GET knee |
| **shuffle window** (`block_chunks`) | shuffle quality ↔ resident memory | RAM / cache budget |
| **batch size** (`batch_size`) | the model's step size | the window's sample pool |

**Chunk size is the amortization lever.** insitu reads each stored chunk once and gathers
every sample inside it, so the work saved versus a per-sample `__getitem__` grows with
samples-per-chunk. Fat chunks amortize more; the one-sample-per-chunk (GRIB) end has nothing
to amortize. Chunk size also sets the memory *unit* — residency is counted in whole outer
chunks.

**Stored-chunk size decouples concurrency cost from chunk size.** When a chunk is split into
an inner grid of tiles, a read fetches a *tile*, not the whole chunk, so raising
`max_inflight` costs tile-sized memory, not chunk-sized. That is why "fat, spatial" is the
sweet spot and "fat, single inner" is not: with one tile per chunk the two collapse and
concurrency costs full chunks. The [decoupled scheduler](#insitubatch-async-driven-pipeline)
is what makes read concurrency and residency independent dials in the first place.

**Batch size is largely orthogonal to IO.** A batch is a vectorized gather from the resident
window, so batch size sets the step the model sees, not the read pattern — as long as the
window's pool (`block_chunks × samples-per-chunk`) stays well above it.

### Why the block-local shuffle is enough

The shuffle is **approximate, not global**: chunks are permuted each epoch, and within a
window of `block_chunks` chunks all samples are shuffled together (the [scope
boundary](#what-this-does-not-do-scope-boundaries) on exact shuffle explains why a global
shuffle is incompatible with chunk-aligned, low-copy reads). Two things make that converge to
a full shuffle in practice:

1. **Within an epoch**, each batch is a uniform draw from the window's pool of
   `block_chunks × samples-per-chunk` samples. Keep that pool well above `batch_size` and a
   single batch is already well-mixed locally.
2. **Across epochs**, the per-epoch chunk permutation re-randomizes *which* chunks share a
   window, so any two samples' chunks eventually co-occur. Over a run the set of samples a
   given sample is ever batched with approaches the whole dataset.

So even a modest window asymptotes quickly toward a global shuffle over the many epochs
training actually runs — at memory cost `O(block_chunks)`, not `O(dataset)`.
[`shuffle_quality`](api.md) scores an emitted order against a perfect global shuffle if you
want to see it on your own data.

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

Runnable side-by-side example:
[`examples/transforms.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/transforms.py)
— a Kelvin→Celsius `chunk_transform` (one variable, cached) and a cross-variable windspeed
`batch_transform` (needs the assembled batch, uncached).

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

Pre-fit the stats however you like and pass them in. The recommended way is to fit
*over the loader itself* with scikit-learn's incremental `StandardScaler.partial_fit`
— covered next — which also warms the cache. `StandardScaler` above is then the
*chunk*-stage applier, for when you want the normalization cached with the decoded
chunk; the fit pass and the apply stage are independent.

**Alternative: fit at the *batch* stage with community tooling, warming the cache.**
Standardization is elementwise, so per-chunk and per-batch are identical — which means
you can also fit it *over the loader itself*: iterate once with no scaler (decoding +
**caching** the raw chunks) while a `sklearn.preprocessing.StandardScaler.partial_fit`
(or `dask_ml`) accumulates per-variable stats, then attach the fitted scaler as a
`batch_transform`. The cache then holds **raw** chunks — normalization-agnostic and
reusable across experiments — and the fit pass *is* the warm-up; training reads
decode-once. It also composes cleanly with a preceding `chunk_transform` (a regrid),
since the fit sees the chunk stage's output. Runnable:
[`examples/fit_scaler.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/fit_scaler.py).

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

## Bad / corrupt chunks

Real archives — especially GRIB-under-zarr (HRRR) — ship the occasional truncated or
corrupt stored chunk. By default a decode failure **fails fast** (`on_bad_chunk="raise"`).
Set `on_bad_chunk="nan"` and a bad (or missing) tile is filled with NaN (float dtypes)
or the array's fill value instead of poisoning the epoch — the outer chunk assembles
with a hole that you repair with an ordinary `chunk_transform`:

```python
def fill_nan(chunk):                      # your policy: climatology, interpolate, ...
    np.nan_to_num(chunk.data, copy=False, nan=0.0)
    return chunk

ds = InSituDataset(url, manifest, on_bad_chunk="nan", chunk_transforms=[fill_nan])
for batch in ds.train:
    ...
print(ds.bad_chunks)   # the (array, chunk_index, inner_coord) reads that were bad this epoch
```

Granularity is the **stored** chunk (tile), so one corrupt inner tile NaNs only its
region of the outer chunk, not the whole field. A failure *during scatter* (a genuine
bug, not a bad chunk) still poisons — the policy only covers fetch/decode. Dropping
NaN-containing *samples* is deliberately not automatic (it would break the fixed-shape
vectorized gather); exclude known-bad chunks at the split/manifest level instead
(`ds.bad_chunks` gives you the list to quarantine).

## Splits — chunk-aligned and leakage-safe

Train/val/test are partitioned **ahead of time, at chunk granularity** along the sample
axis — `split_by_chunk` assigns whole chunks to each split, never individual samples. Two
reasons:

- **No leakage.** A sample never straddles a split boundary, and the temporally adjacent,
  autocorrelated samples inside a chunk can't land on opposite sides of train/val.
- **Reads stay chunk-aligned.** Every read serves exactly one split — no half-chunk waste,
  no sample shared between two splits.

The result is a `SplitManifest` (which chunk indices belong to each split), persisted as JSON
for reproducibility; the dataset's views (`ds.train` / `ds.val` / `ds.test`) read from it.

**`contiguous` is the decision to get right.** By default (`contiguous=True`) each split is a
*contiguous block* of chunks — the safe choice for time series, where a randomly interleaved
split still leaks through autocorrelation across chunk boundaries (a val chunk wedged between
two train chunks shares its neighbours' weather). Set `contiguous=False` only when samples are
**exchangeable** (independent scenes); it shuffles chunks before partitioning.

```python
from insitubatch import open_geometries, split_by_chunk

geom = open_geometries(url)[var]
# time series (default): contiguous blocks, no cross-boundary leakage
manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))
# independent scenes: shuffle chunks before splitting
manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1), contiguous=False)
```

`fractions` are fractions of *chunks*, not samples (for many chunks the two converge). See
`split_by_chunk` in the [API reference](api.md) and "Splits" in
[DESIGN.md](https://github.com/emfdavid/insitubatch/blob/main/DESIGN.md) for the rationale.

### Subsetting to a window — define it with xarray

The `SplitManifest` records *which sample-axis chunks* belong to each split, so to
train on a window of a long archive you just restrict the manifest. `split_by_chunk`
takes a `sample_range=(start, stop)` of sample indices and keeps the chunks overlapping
it before partitioning. And because you probably think in *time*, not indices, you can
define the window with the **xarray API you already know** and translate it — xarray is
used only for this off-hot-path planning step; the engine itself never touches xarray:

```python
import xarray as xr
from insitubatch import open_geometries, split_by_chunk, store_from_url
from insitubatch.source import InSituDataset

# define the window in xarray
xds = xr.open_zarr(store_from_url(url))
sel = xds.sel(time=slice("2020-01-01", "2021-01-01"))
times = xds.indexes["time"]
i0 = times.get_loc(sel.time.values[0])
i1 = times.get_loc(sel.time.values[-1]) + 1  # half-open

# pure zarr/numpy from here
geom = open_geometries(url)[var]
manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1), sample_range=(i0, i1))
ds = InSituDataset(url, manifest, ...)
```

**Limitation — chunk-aligned and contiguous.** The selection snaps *outward* to chunk
boundaries: a window that starts or ends mid-chunk pulls in that whole partial edge
chunk, because splits are chunk-granular (you subset whole chunks, never individual
samples). For day/hour chunks against a multi-month window that's effectively exact.
It is **only** for a single contiguous window — scattered/boolean selections (e.g.
"summers only") don't map cleanly, since straddling chunks would silently re-add the
samples you meant to drop. Subsetting the *inner* (spatial) axes is a separate, later
feature.

## The caching continuum

**The cache boundary IS the chunk-transform boundary.** Every chunk is keyed
`(array, chunk_index)`, and `chunk_transforms` are deterministic and applied before
shuffle — so what's worth keeping is the **decoded + scaled + regridded** array, and
a hit skips fetch *and* decode *and* normalize *and* regrid (not just bytes).
`batch_transforms` (per-sample / random, post-shuffle) run after and are never
cached. So `chunk_transforms` are exactly the deterministic prefix safe to persist.

Dedup → buffer → cache is **one continuum** — and in the engine it is literally one
object, the `ChunkPool`, parameterized by a byte budget:

| layer | reuse scope | how |
|---|---|---|
| read-plan dedup | within a request | a chunk's tiles are fetched once, scattered into one slot |
| assembly buffer | within an epoch | a small budget (the working set, ~2 blocks) |
| cache | across epochs | a large budget retains drained chunks |

A chunk is **pinned** while the current epoch needs it; once its shuffle-block is
drained it becomes **unpinned** — LRU-evictable but not dropped. The pool drops
unpinned chunks only under budget pressure (evicting LRU to admit a miss). With a
small budget that is prompt — the read-once buffer, where each chunk is still
read+decoded **once per epoch** (a naive per-batch eviction would re-read chunks
whose samples scatter across a shuffle block). Raise the budget past the working set
and drained chunks linger, so a still-resident prepped chunk is a cross-epoch hit:
the same machinery becomes the cache by *"don't evict."*

Backing is **heap or mmap** (`cache_dir` → mmap'd `.npy` on local NVMe): the scatter
writes straight into the slot either way, so a hit needs no copy out of a separate
cache. mmap makes the footprint reclaimable kernel page cache, bounded on disk by
bytes, so the working set stays bounded. Caching the *prepped* representation is
strictly stronger than a raw-byte NVMe cache for an ML pipeline. The default budget
is the working set (read-once); raise `cache_budget_bytes` to cache.

**Heavy-reuse tasks unlocked:** multi-epoch training (epoch 0 warms it); the
fat-chunk regime (one chunk → many batches); scoring/verification (reference chunks
reused across metrics, lead times, models); datasets that fit in RAM/NVMe
(effectively in-memory at GPU-fed speed after the first pass).

### Two cache models: chunks vs batches

insitubatch and xbatcher both cache — they cache *different things*, and each choice
buys something real. [xbatcher](https://github.com/xarray-contrib/xbatcher) serializes
**assembled batches** to a zarr store; insitubatch retains **decoded, chunk-transformed
chunks** in the pool. Caching whole batches is a deliberate design choice, not a
shortcut: it is exactly what lets xbatcher's cache survive process exit today.

| dimension | insitubatch — chunk pool | xbatcher — batch cache |
|---|---|---|
| unit cached | decoded + chunk-transformed **chunk** (deduped) | **assembled batch**, in batch layout |
| extra copy | none — `gather` views the slot in place | a separate materialized copy |
| key | `(array, chunk_index)` | batch index → zarr store |
| backing | heap or mmap'd `.npy` (reclaimable NVMe page cache) | zarr store (local dir or cloud) |
| cross-epoch reuse | intrinsic (just don't evict) | yes |
| cross-run persistence | **not yet** (see below) | **yes** — survives restarts |
| shuffle | sits *before* shuffle: sample→batch membership re-drawn every epoch | batch composition frozen; only batch *order* reshuffles |
| sweet spot | many samples per chunk, fat-chunk, multi-epoch, scoring reuse | one-sample-per-chunk, stable batch defs reused across runs |

The two rows that decide it: insitu avoids the second copy and keeps a stronger
per-epoch shuffle (the cache is upstream of shuffling), while xbatcher's batch cache
**persists across runs** and insitu's does not — yet. Closing that one gap is the
cross-run work below.

### Cross-run persistence (planned)

Intra-run cross-epoch reuse is intrinsic (just don't evict). Surviving process exit
is the one place the batch-cache model leads today, and it is the gap this closes — a
deduped decoded-chunk tier on NVMe rather than a separate materialized batch copy:

- **A content key.** Within a run the key is `(array, chunk_index)` because one pool
  == one fixed pipeline. A cross-run key must add a fingerprint of (a) source
  identity (store URL + array + chunk version/etag) and (b) the chunk-transform
  pipeline (stats + transform list), so changed data *or* transforms invalidate.
  Require transforms to expose a stable `version` / config hash.
- **Index rebuild on reopen.** The slot files persist; a dir scan on init (parse
  `(array, chunk)` + size) recovers entries written by earlier runs.
- **A raw-decoded tier** keyed by source identity only (decode being the expensive
  cloud + decompress step), beneath the prepped tier, would let transform
  experimentation reuse decoded chunks without a fingerprint.
- **GDS synergy.** A persistent NVMe tier of prepped `.npy` chunks is the natural
  feed for the kvikio/GDS NVMe→GPU path; cross-process reuse then needs atomic
  writes / immutable files + light locking.

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
their framework, and out of scope today.)

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

These are deliberate current boundaries — the design is honest about them rather than
pretending to be a general compute graph.

- **`chunk_transform` sees ONE variable and ONE chunk.** It cannot combine
  variables. So `windspeed = sqrt(U10² + V10²)` is **not** a chunk transform.
  - It *is* a **`batch_transform`** — the `Batch` holds all variables aligned on
    the sample axis (`batch.arrays["u10"]`, `batch.arrays["v10"]`), so derived
    cross-variable fields compute cleanly there. **Caveat:** batch transforms run
    *after* the cache, so a derived field is recomputed per batch/draw, not
    cached. A cached cross-variable **derived variable** (compute once from
    co-scheduled input chunks, store as a pseudo-chunk keyed like any other) is a
    deliberate **future** feature.
- **No cross-chunk / cross-sample-boundary ops.** A sample is a slice of the outer
  (sample) axis that does **not span a chunk boundary** (the current contract). So
  temporal stencils or windows that straddle two time-chunks (e.g. finite
  differences across the seam, or a 6-step window crossing chunk edges) are not
  supported. Windows spanning *n* chunks are a future opt-in that trades away
  zero-copy.
- **Not a compute framework.** No general task graph, no cross-chunk reductions on
  the hot path, no lazy dask-style evaluation — by design (dask on the hot path is
  the thing we route around). Reductions like fitting a scaler run *over the loader*
  (e.g. sklearn `partial_fit`), not as a graph.
- **Shuffle is approximate**, not global — chunk permutation + shuffle-block
  (`block_chunks` is the quality↔memory knob). Exact global shuffle is
  incompatible with chunk-aligned, low-copy reads.
- **Variables must share the sample axis** (same length and chunk size) — an
  enforced invariant; `InSituDataset` raises `ValueError` otherwise. The draw order
  and gather use one chunk size for all variables; lifting this to per-variable
  chunkings is future work.

Rule of thumb: **per-variable, per-chunk, deterministic → chunk stage (cacheable).
Cross-variable or per-sample-random → batch stage (not cached). Cross-chunk →
not supported.**
