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
torch handoff — without the per-chunk **Python tax** throttling everything.

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
knob can be tuned empirically.

## Memory model

Peak residency ≈ `max(max_inflight, block_chunks) × chunk_nbytes` — **independent
of batch size and epoch length**. That directly answers the "low overhead vs
chunk/batch size" goal.

## Module map

| Module | Role |
|---|---|
| `types.py` | `ArrayGeometry`, `ChunkRead`, `DecodedChunk`, `Batch` |
| `plan.py` | `ReadPlan`, `build_read_plan` (samples → deduped reads) |
| `split.py` | chunk-aligned `SplitManifest`, `split_by_chunk` |
| `io.py` | `AsyncChunkReader` — one event loop, bounded fan-out *(IO wiring stubbed)* |
| `shuffle.py` | chunk permutation + shuffle-block order + quality metric |
| `buffer.py` | `ShuffleBlockBuffer` — residency + coalesced batch gather |
| `source.py` | `InSituDataset` (IterableDataset), optional torch handoff |

## Open questions / spikes

- **Decompression is the next wall** once obstore saturates the NIC: CPU
  Blosc/zstd (GIL-released) vs GPU nvCOMP. Early benchmark needed.
- **GIL**: even with Rust IO, Python decode/assembly can choke. Treat
  free-threaded 3.13t as *upside*, not a dependency; still must win on stock
  CPython via async + coalescing.
- **Multi-variable co-scheduling** when variables have different chunkings.
- **Determinism + resumption** across epochs and DDP ranks (canonical-node style;
  `state_dict` à la torchdata `StatefulDataLoader`).
- **DDP**: shard *chunks* across ranks.

## Status

Pre-alpha **skeleton**. Control flow and abstractions are in place and import
cleanly; the live store wiring in `io.py::_fetch_and_decode` and the GPU path are
stubbed at marked TODOs. First milestone: replace the stub with a real
obstore-backed zarr async read and benchmark throughput vs a classic
`DataLoader` baseline on a sample archive.
