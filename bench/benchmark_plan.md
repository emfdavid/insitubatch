# Benchmark plan — proving the optimizations

Goal: a benchmark set that **isolates the contribution of each optimization
against good-faith baselines**, across the chunk-size spectrum, for both training
and inference. A weak baseline makes the whole result dismissible — so the
baselines below are load-bearing.

> **Win-claim gate:** no public/headline "we beat the standard loader" claim until
> **B2 (xbatcher + DataLoader)** is in the comparison and tuned. B0/B1 are enough
> to develop against; B2 is the credibility bar for a claim.

## Comparison set

Baselines (`bench/baselines.py`, to build):
- **B0 — naive `IterableDataset`, `num_workers=0`.** Synchronous per-chunk zarr
  reads, single thread. The floor / "default base case".
- **B1 — map-style `Dataset` + `DataLoader(num_workers=N, prefetch_factor=k)`.**
  How practitioners actually do it. **Must be tuned** (sweep `N`/`k`, report the
  best) or the comparison is a strawman.
- **B2 — xbatcher + `DataLoader`** (the Earthmover/domain-standard stack).
  **Required before claiming a win** — it is the loader the community will
  compare us to.
- **B3 — fully in-memory (the ceiling).** Whole dataset preloaded into RAM (host
  array; GPU-resident in M2), then iterate with the same compute step — zero IO.
  This is the **compute-bound ceiling**: no out-of-core loader can beat having all
  data resident. Used as a reference line / normalization ("% of in-memory"), not
  a loader to beat. Two consequences:
  - It only fits on a **small reference dataset**, not the 200 GB grid — so it is
    measured separately at the same chunk/compute config and drawn as a ceiling.
  - insitubatch's **warm cache (E3)** should *approach* B3 when the data fits in
    `cache_chunks` — an internal consistency check (warm cache ≈ in-memory).

insitubatch ablation (one optimization at a time, to show marginal value):
- **E1** async fan-out only — `prefetch_depth=1`, `cache_chunks=0`
- **E2** + prefetch overlap — `prefetch_depth=k`
- **E3** + chunk cache — report epoch-2 (warm)

(`naive_sync`, the existing no-torch sequential reader, stays as a sanity floor.)

## Axes and metrics

Independent (graph axes):

| Axis | Values | Why |
|---|---|---|
| sample-chunk size | 1, 2, 4, 8, 16, 32 | the GRIB→fat spectrum; choose inner dims so chunk bytes stay ~1–64 MB (obstore's flat band) |
| storage | local `file://`, **S3** | the async win is latency-bound — only real on S3 |
| engine | B0, B1, B2, E1–E3 | the ablation + baselines |
| compute step | `compute_ms` ∈ {0, realistic} | 0 = pure IO; >0 exposes prefetch overlap |

Dependent (metrics): samples/s, MB/s (decoded), time-to-first-batch, peak host
RSS, `buffer_peak`, and (M2) GPU utilization. **Headline normalization:**
throughput as **% of the B3 in-memory ceiling** at matched chunk/compute config —
the most legible way to state "near-memory speed from cloud, with bounded memory."
The `Result` row records every axis + provenance so each JSONL line is
self-describing.

## Graphs (deliverables)
- **G1 — throughput vs sample-chunk size** (lines per engine; local panel + S3
  panel), with **B3's in-memory ceiling as a dashed line**. Story: the async
  advantage grows as chunks shrink toward GRIB, and how close streaming gets to
  the ceiling.
- **G2 — ablation bars** per regime: B0 → B1 → B2 → E1 → E2 → E3, with the **B3
  ceiling marked**. Marginal value of each optimization; E3-warm should sit near B3.
- **G3 — throughput vs `compute_ms`** (training), **B3 ceiling overlaid**:
  insitubatch stays GPU-fed via prefetch while baselines stall once IO-bound; all
  curves asymptote to the ceiling as compute dominates.
- **G4 — epoch-over-epoch with cache**: epoch-1 cold vs epoch-2 warm. Cache value.
- **G5 — peak memory vs `batch_size` / `block_chunks`**: insitubatch flat
  (bounded by block) vs B1 growing with `workers × prefetch_factor`.
- **G6 — time-to-first-batch**: worker-fork startup vs insitubatch.

## Training vs inference

Both run the same grid; they differ in config and which graphs matter:

| | Training | Inference / scoring (Earth2Studio-shaped) |
|---|---|---|
| shuffle | `True` | `False` (sequential) |
| epochs | ≥2 (cold + warm) | 1 pass |
| compute step | forward+backward (higher `compute_ms`) | forward only / none (lower) |
| headline graphs | G2, G3, G4, G5 | G1, G3, G6 |
| proves | sustained GPU-fed throughput, cache reuse, bounded memory | hides fetch behind rollout; the IO-bound regime where async wins biggest |

The inference + **GRIB-per-timestep + S3** combination is where the local 2.8×
signal should become a large win: every timestep is a separate high-latency read
that the baselines serialize per worker.

## Harness changes
1. `bench/baselines.py` — B0, B1 (tuned), B3 (in-memory ceiling), **B2 (xbatcher)**;
   plus a small fits-in-RAM reference dataset config for B3.
2. `make_dataset` family generator — same logical data at each sample-chunk size
   (`era5_c1`, `era5_c4`, …) so the chunk-size axis is comparable (same bytes,
   different chunking).
3. Runner — sweep `{chunk size × engine × storage × compute_ms × epoch}`, append
   JSONL; extend `Result`; `--repeats` (3–5) for medians + spread.
4. `compute_ms` knob — simulate the step (CPU: GIL-releasing op; GPU later: real
   matmul) so prefetch overlap is observable.
5. `bench/plot.py` — JSONL → G1–G6 (matplotlib, a `bench` extra). 
6. Fix rough edges: per-dataset `--url` runs (not both regime configs on one URL);
   exclude a warmup batch.

## Rigor (what makes it convincing)
- **Tune the baselines** (`num_workers`, `prefetch_factor`, xbatcher batch dims) to
  their best; report the tuning.
- **Cold vs warm** controlled (fresh process / drop page cache for cold S3;
  cache-off for read-once comparisons).
- **≥3 repeats**, report median + min/max; exclude warmup batches.
- **Provenance** in every row: instance type, region, NIC, vCPU, codec, date.
- **Show where we lose** (fat chunks on local disk) — honesty earns the wins.

## Phasing
- **1a (local, free):** build baselines + runner + plotting + dataset-family gen;
  validate the whole grid on small data. Proves the harness and gives correct
  *shapes* for G2/G5/G6. (Absolute async wins won't show locally — expected.)
- **1b (EC2 / S3):** run the grid on the box (see [ops_aws.md](ops_aws.md)) → the
  real numbers; G1/G3/G4 come alive on S3.
- **M2 (GPU):** real compute step + GPU-native path; GPU-utilization graphs.

## Build order
1. `baselines.py` B0 + B1 + B3 (in-memory ceiling), extended `Result`/runner,
   dataset-family gen — test-first.
2. `compute_ms` knob + `plot.py`; validate 1a locally.
3. **B2 (xbatcher)** — before any win claim.
4. Run 1b on S3; then M2.
