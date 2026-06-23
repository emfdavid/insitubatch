# Benchmarks

Live Plotly figures (hover, zoom, toggle traces) from the benchmark suite. The
suite isolates each optimization against good-faith, **tuned** baselines; the full
dataset matrix, run commands, and the win-claim gate are in the
[benchmark plan](https://github.com/emfdavid/insitubatch/blob/main/bench/benchmark_plan.md).

The post-V2 architecture tells **four stories**. Story 1 has first real numbers
from an S3 run; 2–4 and the framing sections fill in as the matrix runs on the box.

!!! note "Real run — first results (`exp_b`)"
    Story 1 below is from a **real S3 run** on the
    [`c6id.8xlarge`](https://github.com/emfdavid/insitubatch/blob/main/bench/ops_aws.md)
    (32 vCPU, in-region S3), not a laptop: ERA5-shaped data, `721×1440` fields
    (4.15 MB/sample), `sample_chunk=8`, a bounded slice (128 batches/config) with a
    warm-up burst first to clear S3 cold-start. Both stacks are **tuned** — insitu
    swept over `block_chunks`, the worker baselines over `num_workers` up to 32 (=
    vCPUs) — so this is not a strawman.

## The comparison set

| Engine | What it is | Role |
|---|---|---|
| `insitu` | insitubatch: one async event loop, prefetch, shared cache | the system under test |
| `naive` | sequential synchronous reads, one sample at a time | the floor |
| `workers` | map-style `Dataset` + `DataLoader(num_workers=N)` | the realistic baseline |
| `xbatcher` | `xbatcher.BatchGenerator` + `DataLoader` (the Earthmover stack) | the credibility bar |
| `memory` | data preloaded into RAM, compute only | the in-memory ceiling |

Each engine is reported at its **tuned** optimum (insitu over `block_chunks`, the
DataLoader baselines over `num_workers`).

---

## Story 1 — chunks, not samples

insitubatch reads each stored chunk **once** and vector-gathers every sample inside
it; a map-style `__getitem__` decodes the whole containing chunk to return **one**
sample. At `sample_chunk=8`, insitubatch delivers **~8× the throughput** of the
best-tuned baseline and reaches its first batch **~10× sooner**:

| stack (tuned) | throughput | time-to-first-batch |
|---|---:|---:|
| **insitubatch** (`bc=64`) | **1172 MB/s** | ~1 s |
| xbatcher + DataLoader (`nw=32`) | 146 MB/s | ~10 s |
| map-style workers (`nw=32`) | 125 MB/s | ~10–13 s |

The 8× is exactly `sample_chunk`: both stacks move ~the same *raw* bytes/s
(~1.1 GB/s, the network ceiling), but the map-style stack wastes 7/8 of its bytes
re-decoding the same chunk for each sample. The gap therefore grows linearly with
chunk size (and shrinks to ~1× at GRIB/chunk=1) — which the chunk-size spectrum
(below) is built to show.

### Throughput by engine (tuned)

<iframe src="figures/g2_ablation.html" width="100%" height="480" frameborder="0"></iframe>

### Baseline tuning curve

Throughput vs `num_workers` for the DataLoader baselines — this is what "tuned to
their best" means, and where the best-of points above come from. They keep scaling
toward ~32 workers and still top out ~8× below insitu.

<iframe src="figures/g7_worker_tuning.html" width="100%" height="480" frameborder="0"></iframe>

### Time-to-first-batch

Worker spin-up (32 processes + cold reads) vs the event loop's first read.

<iframe src="figures/g6_ttfb.html" width="100%" height="480" frameborder="0"></iframe>

### Throughput vs the chunk-size spectrum

_Pending the `era5_c{1..32}` family run (story-1 spectrum command in the plan)._ The
8× is predicted to grow with chunk size; this is the figure that makes that claim
falsifiable across the GRIB→fat spectrum. Renders as `figures/g1_throughput_vs_chunk.html`.

---

## Story 2 — the V2 decoupling

Read concurrency (`max_inflight`) is decoupled from residency/shuffle
(`block_chunks`): throughput climbs to the knee then stays flat, and **memory stays
flat**, as `max_inflight` rises at fixed `block_chunks`. In the zero-compute case it
shows the **sawtooth** — peaks where `max_inflight` evenly divides the tiles per
batch, which is why the fat regime needs inner (spatial) chunking to restore
concurrency (see [Architecture](architecture.md)).

_Pending the `probe_decode` `max_inflight` sweep across the spatial-grid datasets
(`era5_fat_g4/g16/g36`) and the `--block-chunks` memory-flatness suite run._

---

## Story 3 — the cache (decode-once across epochs)

With a large cache budget, epoch-2 reads come from the pool (heap or mmap'd `.npy`)
instead of re-fetching and re-decoding, so warm ≫ cold. The probe's cross-epoch test
showed ~2.5× cold→warm at `sample_chunk=8`; the epoch-over-epoch suite figure
renders as `figures/g4_cache_epochs.html`.

_Pending the `epochs≥2` suite run with `--cache-dir` on instance-store NVMe._

---

## Story 4 — efficiency vs the ceiling

The honest "how much of the NIC are we keeping" number: insitu's decoded MB/s as a
**% of the raw-GET ceiling** (obstore reading the same bytes with no decode/gather).
Story 1 already shows both stacks pinned at ~1.1 GB/s raw on the `c6id.8xlarge`, so
the headline is decode/gather overhead, not network.

_Pending the `probe_decode` raw-GET section on S3 and on **S3 Express One Zone**
(`--s3-express`), where single-digit-ms GETs stress the loader hardest._

---

## Free-threading readiness

insitubatch's throughput is **GIL-independent by design**: the heavy work already runs
outside the GIL — fetch (obstore/Rust), decode (numcodecs zstd, C), scatter/gather
(vectorized numpy) — and scheduling is a single asyncio loop, so there is no GIL-held
hot path for free-threading to accelerate. (Decode even parallelizes *under* the GIL:
zstd releases it, so the `decode_threads` sweep scales on the GIL build too.) On the
3.13 free-threaded build the engine is **correct** — the scatter is disjoint and
readiness is published under the lock, so the lock, not the GIL, is the happens-before
edge — and it runs at the **same speed** as the GIL build.

So the free-threading story is **correctness + future-proofing, not a speedup** — and
*not depending* on the GIL being gone is a stronger position than needing it. numcodecs
re-enabling the GIL on import only affects Python-level parallelism, which our
GIL-releasing hot path doesn't use. The panels are a no-regression check (control
`decode_threads ≤ cores`, report p50/p95).

_Pending the `PYTHON_GIL=0` probe panels (a no-regression check vs the GIL build)._

---

## Deferred

- **Prefetch overlap vs per-batch compute** (`g3_throughput_vs_compute.html`) — needs
  a `compute_ms` sweep; insitu stays GPU-fed while baselines stall once IO-bound.
- **Resident memory by engine** (`g5_peak_memory.html`) — needs per-config process
  isolation; in a single-process run `ru_maxrss` is a monotonic high-water and heap
  isn't returned between configs, so the measurement is confounded.

## Reproduce

Full dataset matrix and per-story commands are in the
[benchmark plan](https://github.com/emfdavid/insitubatch/blob/main/bench/benchmark_plan.md).
The story-1 slice on a pre-generated S3 family:

```bash
# tune the baselines once, then the spectrum at the tuned best
uv run python -m bench --url-prefix "s3://$BUCKET/era5" --storage s3 \
  --out bench/results/story1_spectrum.jsonl --fig-dir bench/figures/story1 \
  --engines naive,workers,xbatcher,insitu --chunk-sizes 1,2,4,8,16,32 \
  --num-workers 32 --max-batches 64 --repeats 3 --warmup-batches 32 --plot

# (re)build the embeddable, CDN-loaded figures used on this page
uv run python -m bench.plot --in bench/results/story1_spectrum.jsonl --out docs/figures --cdn
```
