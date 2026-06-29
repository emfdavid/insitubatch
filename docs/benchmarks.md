# Benchmarks

Live Plotly figures (hover, zoom, toggle traces) from the benchmark suite. The
suite isolates each optimization against good-faith, **tuned** baselines; the full
dataset matrix, run commands, and the win-claim gate are in the
[benchmark plan](https://github.com/emfdavid/insitubatch/blob/main/bench/benchmark_plan.md).

## Who wins where

insitubatch is the **batteries-included** choice across two operating points:

- **Inference (cold start).** It pays no worker-pool startup — first batch in ~0.2 s vs
  the worker stacks' 1–15 s — and a production inference service can't keep a hot 32-worker
  pool alive anyway. insitu wins cold-start at **every** chunk size measured.
- **Training (across the chunk spectrum).** It uses **8–25× less memory** (one process, not
  32), reads each chunk **once** (vs the baselines' per-sample re-decode), and brings a
  **cross-epoch cache** (warm epochs at ~4 GB/s) the worker stacks don't have. Throughput
  beats the best-tuned baseline from `c2` up — to **~25×** at fat chunks.

The honest exception is the **GRIB end (`c1`, one sample per chunk)**: there xbatcher is
~30% faster on *single-pass* throughput, because with nothing to amortize per chunk insitu's
read-once advantage doesn't apply. The trade-offs are memory (~25×) and cold start (~6×
TTFB), and since the worker stacks re-read every epoch, multi-epoch training flips back to
insitu via the cross-epoch cache (~9× warm). xbatcher is a well-established *batch
definition* on a worker-process *engine*; insitu keeps the same ndim batch semantics with
one async loop, so its edge grows with samples-per-chunk.

!!! note "Real run"
    Numbers below are from a **real S3 run** on a
    [`c6id.8xlarge`](https://github.com/emfdavid/insitubatch/blob/main/bench/ops_aws.md)
    (32 vCPU, in-region S3 + S3 Express), not a laptop: coarsened ERA5 `361×720`
    fields, the `era5_c{1..32}` chunk-size family plus fat-spatial grids, ≥3 repeats
    with a warm-up burst to clear S3 cold-start. Both baselines are **tuned**
    (`num_workers` swept to 32 = vCPUs); insitu is swept over `block_chunks` — so neither
    side is under-tuned.

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
sample. So the win *grows with samples-per-chunk* — throughput vs the best-tuned
baseline (MB/s, warm), across the chunk spectrum:

| sample_chunk | 1 | 2 | 4 | 8 | 16 | 32 |
|---|--:|--:|--:|--:|--:|--:|
| **insitu** | 283 | 483 | 514 | **630** | 617 | **657** |
| xbatcher (tuned) | 372 | 432 | 242 | 101 | 55 | 26 |
| workers (tuned) | 292 | 420 | 216 | 85 | 43 | 31 |
| naive | 30 | 43 | 56 | 31 | 20 | 21 |
| **insitu vs best** | 0.76× | 1.1× | 2.1× | **6.2×** | **11×** | **~21–25×** |

The baselines re-decode the containing chunk per sample, so they waste `(sample_chunk−1)/
sample_chunk` of their bytes; insitu reads once. The advantage therefore grows linearly
with chunk size and shrinks to a slight *loss* at the GRIB end (`c1`, one sample/chunk —
nothing to amortize).

<iframe src="../figures/g1_throughput_vs_chunk.html" width="100%" height="480" frameborder="0"></iframe>

### Throughput by engine, and the baseline tuning

<iframe src="../figures/g2_ablation.html" width="100%" height="420" frameborder="0"></iframe>

The DataLoader baselines are reported at their best `num_workers` — they keep scaling
toward 32 and still top out well below insitu:

<iframe src="../figures/g7_worker_tuning.html" width="100%" height="420" frameborder="0"></iframe>

For reference, the in-memory ceiling (whole array in RAM, zero IO) runs ~7.6–7.9 GB/s;
insitu at `c8` is ~0.7 GB/s — the gap is IO, not loader overhead (see story 4).

---

## Story 2 — the V2 decoupling

Read concurrency (`max_inflight`) is decoupled from residency/shuffle (`block_chunks`):
throughput climbs to the network knee and **stays flat**, while residency is **pinned**,
as `max_inflight` rises. V1's nested caps produced a sawtooth; V2's single semaphore
smooths it — so the result is a clean rise-to-plateau, the same on every spatial grid
(`era5_fat_g4/g16/g36`):

| max_inflight | 1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MB/s (`fat_g16`) | 23 | 108 | 276 | 883 | 1078 | 1071 | 1102 | 1129 |
| resident chunks | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |

You dial throughput from 23 → ~1130 MB/s purely via `max_inflight`, at **constant memory**
— concurrency and residency are independent knobs.

---

## Story 3 — the cache (decode-once across epochs)

With a budget that holds the split (`--caches resident`, spilled to NVMe), epoch 2 reads
come from the pool — no S3, no decode — so warm ≫ cold:

| sample_chunk | cold MB/s | warm MB/s | speedup |
|---|--:|--:|--:|
| c1 | 430 | **3977** | **9.2×** |
| c8 | 777 | 4526 | 5.8× |
| c32 | 750 | 4557 | 6.1× |

The cross-epoch probe on `fat_g16` confirms it independently (1006 → 4509, 4.5×). The
worker stacks have **no shared cross-epoch cache** — they re-read S3 every epoch — so this
is the result that flips the GRIB end back to insitu for *multi-epoch training*.

<iframe src="../figures/g4_cache_epochs.html" width="100%" height="420" frameborder="0"></iframe>

---

## Story 4 — efficiency vs the raw-GET ceiling

How much of the NIC do we keep? insitu's decoded MB/s as a **% of the raw-GET ceiling**
(obstore reading the same bytes, no decode/gather), on `fat_g16`:

| storage | insitu (decoded) | raw-GET ceiling | % kept |
|---|--:|--:|--:|
| S3 | 1187 | 1467 | **81%** |
| S3 Express One Zone | 1261 | 1501 | **84%** |

So ~80% of the network ceiling survives decode + gather. S3 Express saturates the ceiling
at **concurrency 16** (single-digit-ms GETs) where regular S3 needs 32–64, and it rescues
the GRIB end: insitu `c1` runs **820 MB/s on Express vs 283 on S3 (2.9×)**.

---

## Memory + cold-start by engine (G5/G6)

The suite's per-row RSS can't compare engines (single-process high-water; the 32 worker
children of `workers`/`xbatcher` aren't counted). `probe_memory` runs each engine in its
**own subprocess** and samples peak RSS over the whole **process tree**. Read-once (anon
working set), so it's apples-to-apples:

**c1 (GRIB):**

| engine | peak RSS | procs | TTFB cold | warm MB/s |
|---|--:|--:|--:|--:|
| **insitu** | **0.9 GB** | **1** | **0.2 s** | 333 |
| workers | 19.9 GB | 35 | 3.3 s | 534 |
| xbatcher | 22.6 GB | 34 | 1.3 s | 588 |

**c16 (fat) — insitu wins every axis:**

| engine | peak RSS | procs | TTFB cold | warm MB/s |
|---|--:|--:|--:|--:|
| **insitu** | **3.1 GB** | **1** | **0.7 s** | **908** |
| workers | 23.4 GB | 34 | 15.3 s | 47.9 |
| xbatcher | 27.1 GB | 34 | 15.6 s | 48.9 |

→ **~8× memory, ~19× throughput, ~22× TTFB.** insitu's footprint is one Python+obstore
process (paid once); the baselines pay the interpreter floor **32×** plus a 208 MB field
re-decoded per sample at fat chunks. The independent **WeatherBench2** run on the real
public store (`examples/wb2_xbatcher.py` vs `wb2_dataloader.py`, 8 workers) makes it
concrete: xbatcher manages **~40–60 samples/s** at **~2–3 s** to first batch, insitu
**~2250 samples/s** at **~320 ms** — **~40×** throughput, because WB2's fat time-chunks
punish the per-sample decode while insitu reads each chunk once.

---

## Free-threading readiness

insitubatch's throughput is **GIL-independent by design**: the heavy work already runs
outside the GIL — fetch (obstore/Rust), decode (numcodecs zstd, C), scatter/gather
(vectorized numpy) — and scheduling is a single asyncio loop, so there is no GIL-held
hot path for free-threading to accelerate. (Decode even parallelizes *under* the GIL:
zstd releases it, so the `decode_threads` sweep scales 1→2 on the GIL build too.) On the
3.13 free-threaded build the engine is **correct** — the scatter is disjoint and readiness
is published under the lock, so the lock, not the GIL, is the happens-before edge — and it
runs at the **same speed** as the GIL build.

So free-threading here is **correctness + future-proofing, not a speedup** — and *not
depending* on the GIL is a stronger position than needing it. The
[flamegraph](https://github.com/emfdavid/insitubatch/blob/main/bench/results/profile_fat_g16.svg)
(`py-spy --native`) makes it visual: time is in Rust IO + C decode + numpy, with only a
thin Python sliver.

## Deferred

- **Prefetch overlap vs per-batch compute** (`g3`) — needs a `compute_ms` sweep; insitu
  stays GPU-fed while baselines stall once IO-bound.
- **GPU-native path** (M2) — `device_transform` after DLPack; GPU-utilization graphs.

## Reproduce

Full dataset matrix and per-story commands are in the
[benchmark plan](https://github.com/emfdavid/insitubatch/blob/main/bench/benchmark_plan.md).
The story-1 spectrum on a pre-generated S3 family, then rebuild the figures:

```bash
uv run python -m bench --url-prefix "s3://$BUCKET/era5" --storage s3 \
  --out bench/results/story1_spectrum.jsonl \
  --engines naive,workers,xbatcher,insitu --chunk-sizes 1,2,4,8,16,32 \
  --num-workers 32 --max-batches 64 --repeats 3 --warmup-batches 32

uv run python -m bench.plot --in bench/results/story1_spectrum.jsonl --out docs/figures --cdn
```
