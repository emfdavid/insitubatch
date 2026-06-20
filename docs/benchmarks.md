# Benchmarks

Live Plotly figures (hover, zoom, toggle traces) from the benchmark suite. The
suite isolates each optimization against good-faith, **tuned** baselines; full
methodology and the win-claim gate are in the
[benchmark plan](https://github.com/emfdavid/insitubatch/blob/main/bench/benchmark_plan.md).

!!! note "Real run — first results (`exp_b`)"
    These are from a **real S3 run** on the
    [`c6id.8xlarge`](https://github.com/emfdavid/insitubatch/blob/main/bench/ops_aws.md)
    (32 vCPU, in-region S3), not a laptop: ERA5-shaped data, `721×1440` fields
    (4.15 MB/sample), `sample_chunk=8`, a bounded slice (128 batches/config) with a
    warm-up burst first to clear S3 cold-start. Both stacks are **tuned** — insitu
    swept over `block_chunks`, the worker baselines over `num_workers` up to 32 (=
    vCPUs) — so this is not a strawman. Scope: a single chunk size so far; the
    chunk-size spectrum (G1), prefetch-overlap (G3), cache cold/warm (G4), and a
    clean memory measurement (G5) are pending later experiments.

## Headline

At `sample_chunk=8`, insitubatch delivers **~8× the throughput** of the best-tuned
baseline and reaches its first batch **~10× sooner**:

| stack (tuned) | throughput | time-to-first-batch |
|---|---:|---:|
| **insitubatch** (`bc=64`) | **1172 MB/s** | ~1 s |
| xbatcher + DataLoader (`nw=32`) | 146 MB/s | ~10 s |
| map-style workers (`nw=32`) | 125 MB/s | ~10–13 s |

The 8× is exactly `sample_chunk`: both stacks move ~the same *raw* bytes/s (~1.1 GB/s,
the network ceiling), but a map-style `__getitem__` decodes the whole 8-sample chunk
to return **one** sample — so 7/8 of its bytes are wasted re-decoding. insitu reads
each chunk **once**. The gap therefore grows linearly with chunk size (and shrinks
to ~1× at GRIB/chunk=1). "Chunks, not samples," in one number.

## The comparison set

| Engine | What it is | Role |
|---|---|---|
| `insitu` | insitubatch: one async event loop, prefetch, shared cache | the system under test |
| `naive` | sequential synchronous reads, one sample at a time | the floor |
| `workers` | map-style `Dataset` + `DataLoader(num_workers=N)` | the realistic baseline |
| `xbatcher` | `xbatcher.BatchGenerator` + `DataLoader` (the Earthmover stack) | the credibility bar (B2) |
| `memory` | data preloaded into RAM, compute only | the in-memory ceiling |

Each engine is reported at its **tuned** optimum (insitu over `block_chunks`, the
DataLoader baselines over `num_workers`).

## G2 — Throughput by engine (tuned)

<iframe src="figures/g2_ablation.html" width="100%" height="480" frameborder="0"></iframe>

## G7 — Baseline tuning curve

Throughput vs `num_workers` for the DataLoader baselines — this is what "tuned to
their best" means, and where the best-of points in G2 come from. They keep scaling
toward ~32 workers and still top out ~8× below insitu.

<iframe src="figures/g7_worker_tuning.html" width="100%" height="480" frameborder="0"></iframe>

## G6 — Time-to-first-batch

Worker spin-up (32 processes + cold reads) vs the event loop's first read.

<iframe src="figures/g6_ttfb.html" width="100%" height="480" frameborder="0"></iframe>

## Pending

- **G1 — throughput vs chunk-size spectrum.** Needs the `c1..c32` family; the 8×
  is predicted to grow with chunk size.
- **G3 — prefetch overlap vs per-batch compute.** Needs a `compute_ms` sweep.
- **G4 — cache cold vs warm epochs.** Needs `epochs≥2` with same-chunk reuse.
- **G5 — resident memory by engine.** Deferred until per-config process isolation —
  in a single-process run `ru_maxrss` is a monotonic high-water and heap isn't
  returned between configs, so the measurement is confounded.
- **Fat regime + inner (spatial) chunking** (`exp_c`): where the gap is largest and
  insitu needs spatial chunking for concurrency — see
  [Architecture](architecture.md).

## Reproduce

```bash
# the exp_b slice (tuned, bounded, warm) on a pre-generated S3 family
uv run python -m bench --url-prefix "s3://$BUCKET/era5" --cache-dir /mnt/nvme/cache \
  --chunk-sizes 8 --engines insitu,workers,xbatcher --caches none \
  --block-chunks 16,32,64 --num-workers 8,16,32 --epochs 1 --max-batches 128 \
  --out bench/results/exp_b.jsonl

# (re)build the embeddable, CDN-loaded figures used on this page
uv run python -m bench.plot --in bench/results/exp_b.jsonl --out docs/figures --cdn
```
