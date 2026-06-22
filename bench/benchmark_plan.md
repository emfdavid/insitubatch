# Benchmark plan — proving the optimizations

Goal: a benchmark set that **isolates the contribution of each optimization
against good-faith baselines**, across the chunk-size spectrum. A weak baseline
makes the whole result dismissible — so the baselines below are load-bearing.

> **Win-claim gate:** no public/headline "we beat the standard loader" claim until
> **B2 (xbatcher + DataLoader)** is in the comparison and tuned. The naive/worker
> baselines are enough to develop against; B2 is the credibility bar for a claim.

## The four stories

The post-V2 architecture tells four clean results. Everything below maps to one
of these:

1. **Chunks, not samples.** insitubatch reads each stored chunk **once** and
   vector-gathers every sample inside it; the map-style `DataLoader` decodes the
   containing chunk **per sample**. → the large win vs baselines, growing as
   chunks fatten (more samples amortized per decode). *(suite, story-1 run)*
2. **The V2 decoupling.** Read concurrency (`max_inflight`) is decoupled from
   residency/shuffle (`block_chunks`). Throughput climbs to the knee then stays
   flat, and **memory stays flat**, as `max_inflight` rises at fixed
   `block_chunks`. The zero-compute case shows the **sawtooth**: peaks where
   `max_inflight` evenly divides the tiles-per-batch. *(probe_decode 1b)*
3. **The cache.** Decode-once across epochs: with a large `cache_budget_bytes`,
   epoch-2 reads come from the pool (heap or mmap'd `.npy`), so warm ≫ cold.
   *(probe_decode 1c, suite epochs≥2)*
4. **Efficiency vs the ceiling.** insitu decoded MB/s as a **% of the raw-GET
   ceiling** (obstore reading the same bytes, no decode/gather). This is the
   honest "how much of the NIC are we keeping" number. *(probe_decode 2)*

Plus two framing sections, not headline claims:
- **Free-threading readiness** — we're correct and faster-at-peak under 3.13t, the
  ecosystem (numcodecs re-enables the GIL) isn't. Run the FT panels to show the
  ceiling and the upstream gate. *(probe_decode under `PYTHON_GIL=0`)*
- **S3 Express One Zone** — the IO-ceiling stress test: single-digit-ms GETs push
  the loader, not the network. Re-run story 4 on a directory bucket. *(`--s3-express`)*

## Comparison set

Engines (`bench/engines.py`):
- **naive** — `IterableDataset`, `num_workers=0`. Synchronous per-chunk zarr reads,
  single thread. The floor / default base case.
- **workers** — map-style `Dataset` + `DataLoader(num_workers=N, prefetch_factor=k)`.
  The DIY pattern: `__getitem__` returns one sample, so the containing chunk is
  decoded once **per sample** with no shared cache. **Must be tuned** (sweep `N`,
  report best) or it's a strawman. This is *not* the Earthmover stack.
- **xbatcher** — xbatcher + `DataLoader` (the **Earthmover** / domain-standard
  stack). **Required before claiming a win.** Match *their* tuning, not our
  defaults: the post used `num_workers=32`, `prefetch_factor=3` — so sweep
  `num_workers` and report the best. Our xbatcher feeds the **obstore** store
  (storage held constant vs insitu → a stronger, conservative baseline than their
  gcsfs path). Their headline ~15× is **internal** (tuned vs untuned xbatcher),
  not xbatcher-vs-another-loader — so "beating Earthmover" means insitu vs a
  well-tuned xbatcher.
- **insitu** — the engine under test.
- **memory** — whole dataset preloaded into RAM, then iterate with the same compute
  step. The **compute-bound ceiling** (zero IO), drawn as a reference line, not a
  loader to beat. It **preloads the whole array (ignores `--max-batches`)**, so run
  it **on its own** on a moderate set — keep it out of the spectrum sweep.

> **Tune the baselines.** Find the best `num_workers` once (`--num-workers 8,16,32`
> on a single `--chunk-sizes`), then run the spectrum at that single tuned value —
> sweeping low worker counts across the whole grid just times the baseline being
> slow.

## Axes and metrics

| Axis | Values | Why |
|---|---|---|
| sample-chunk size | 1, 2, 4, 8, 16, 32 | the GRIB→fat spectrum; inner dims chosen so chunk bytes stay ~1–64 MB (obstore's flat band) |
| storage | local `file://`, **S3**, **S3 Express** | the async win is latency-bound — only real on S3; Express stresses it hardest |
| engine | naive, workers, xbatcher, insitu, memory | the comparison set |
| `max_inflight` | 1 … 256 | story 2 (probe only) |
| `block_chunks` | residency/shuffle window | story 2 memory-flatness (suite `--block-chunks`) |
| compute step | `compute_ms` ∈ {0, realistic} | 0 = pure IO; >0 exposes prefetch overlap |

Metrics (per JSONL row, self-describing): **samples/s**, **MB/s (decoded)**,
**time-to-first-batch (ms)**, **peak RSS** with anon/file split, plus `% of the
raw-GET ceiling` (story 4) and `% of the in-memory ceiling` (vs the `memory`
engine). Provenance in every row: instance type, region, vCPU, codec, date.

---

## Full run — datasets + commands

Run from the bench VM (see [ops_aws.md](ops_aws.md)). Set once:

```bash
export BUCKET=insitubatch-bench                       # regular S3, same region as the VM
export XBUCKET=insitubatch-bench--use1-az4--x-s3      # S3 Express directory bucket, AZ-matched
export AWS_REGION=us-east-1
```

All datasets are a coarsened ERA5 single-level field, **inner = (361, 720)**
(~1.04 MB/sample, f4, zstd `auto`). That keeps a single-inner stored chunk in
obstore's flat band across the whole `sample_chunk` sweep (c1 ≈ 1 MB … c32 ≈ 33 MB).

Sizing: N≈3000 at `batch_size=32` is ~96 batches/epoch, so timed runs cap at
**`--max-batches 64`** — comfortably inside one epoch (no wrap) and plenty for a
stable median. The heaviest store is `era5_c32` at ~3.2 GB compressed; the fat
grids are ~3.3 GB each.

### Make the datasets

**1 — chunk-size family** (story 1; single var, single inner chunk, N≈3000):

```bash
for spc in 1 2 4 8 16 32; do
  uv run python bench/make_dataset.py \
    --url s3://$BUCKET/era5_c${spc}.zarr \
    --n-samples 3072 --inner 361,720 --sample-chunk $spc --variables t2m
done
```

**2 — fat, single inner chunk** (story 2 contrast: fat sample chunk, no spatial
fan-out → concurrency collapses to one giant chunk/batch):

```bash
uv run python bench/make_dataset.py \
  --url s3://$BUCKET/era5_fat.zarr \
  --n-samples 3200 --inner 361,720 --sample-chunk 200 --variables t2m
```

**3 — fat + spatial grid** (story 2 main: same fat sample chunk, inner-chunked so
one sample fans out into a *grid* of tiles → restores read concurrency). Three
tile densities:

```bash
# ~4 tiles/sample  (2x2),  tile ≈ 52 MB
uv run python bench/make_dataset.py --url s3://$BUCKET/era5_fat_g4.zarr \
  --n-samples 3200 --inner 361,720 --sample-chunk 200 --inner-chunks 181,360 --variables t2m
# ~16 tiles/sample (4x4),  tile ≈ 13 MB
uv run python bench/make_dataset.py --url s3://$BUCKET/era5_fat_g16.zarr \
  --n-samples 3200 --inner 361,720 --sample-chunk 200 --inner-chunks 91,180 --variables t2m
# ~36 tiles/sample (6x6),  tile ≈ 5.8 MB
uv run python bench/make_dataset.py --url s3://$BUCKET/era5_fat_g36.zarr \
  --n-samples 3200 --inner 361,720 --sample-chunk 200 --inner-chunks 61,120 --variables t2m
```

**4 — multi-variable** (cross-variable gather; 6 surface vars, sc=8, spatial,
N≈2000):

```bash
uv run python bench/make_dataset.py \
  --url s3://$BUCKET/era5_multi.zarr \
  --n-samples 2048 --inner 361,720 --sample-chunk 8 --inner-chunks 181,360 \
  --variables t2m,u10,v10,msl,tp,d2m
```

**5 — S3 Express copies** (story 4 stress; mirror the c1 GRIB end + the g16 grid
onto the directory bucket — add `--s3-express`):

```bash
uv run python bench/make_dataset.py --s3-express \
  --url s3://$XBUCKET/era5_c1.zarr \
  --n-samples 3072 --inner 361,720 --sample-chunk 1 --variables t2m
uv run python bench/make_dataset.py --s3-express \
  --url s3://$XBUCKET/era5_fat_g16.zarr \
  --n-samples 3200 --inner 361,720 --sample-chunk 200 --inner-chunks 91,180 --variables t2m
```

### Story 1 — chunks-not-samples (the suite, vs baselines)

Tune workers once, then run the spectrum at the best value:

Each suite run gets its own `--out` — rows are **appended**, so sharing a file
across stories piles duplicates into one JSONL and muddles `--plot`.

```bash
# (a) find best num_workers on one chunk size
uv run python -m bench --url-prefix s3://$BUCKET/era5 --storage s3 \
  --out bench/results/story1_tune.jsonl \
  --engines workers,xbatcher --chunk-sizes 8 \
  --num-workers 8,16,32 --max-batches 64 --repeats 3 --warmup-batches 32

# (b) the spectrum at the tuned best (drop `memory` here; it ignores --max-batches)
uv run python -m bench --url-prefix s3://$BUCKET/era5 --storage s3 \
  --out bench/results/story1_spectrum.jsonl --fig-dir bench/figures/story1 \
  --engines naive,workers,xbatcher,insitu \
  --chunk-sizes 1,2,4,8,16,32 --num-workers 32 \
  --max-batches 64 --repeats 3 --warmup-batches 32 --plot

# (c) the in-memory ceiling, on its own, on a moderate set (whole array preloaded)
uv run python -m bench --url-prefix s3://$BUCKET/era5 --storage s3 \
  --out bench/results/story1_ceiling.jsonl \
  --engines memory,insitu --chunk-sizes 8 --repeats 3
```

### Story 2 — the V2 decoupling (max_inflight sweep / sawtooth)

The probe sweeps `max_inflight` at fixed residency; run it on the GRIB end and on
each spatial grid (different tiles-per-batch → different sawtooth period). Zero
compute = pure IO so the sawtooth is visible:

```bash
for ds in era5_c1 era5_fat era5_fat_g4 era5_fat_g16 era5_fat_g36; do
  uv run python bench/probe_decode.py --url s3://$BUCKET/$ds.zarr \
    --max-inflight 1,2,4,8,16,32,64,128,256 --block-chunks 32 \
    --max-chunks 256 --repeats 5 --no-raw \
    | tee bench/results/story2_sawtooth_$ds.log
done
```

Memory-flatness (story 2, the `block_chunks` axis stays flat in RSS while
`max_inflight` rises) comes from the suite with an explicit `--block-chunks` sweep:

```bash
uv run python -m bench --url-prefix s3://$BUCKET/era5 --storage s3 \
  --out bench/results/story2_mem.jsonl \
  --engines insitu --chunk-sizes 8 --block-chunks 8,32,128 \
  --max-batches 64 --repeats 3
```

### Story 3 — the cache (decode-once across epochs)

Point the cache dir at instance-store NVMe (`c6id`/`i4i`/`c7gd`):

```bash
uv run python bench/probe_decode.py --url s3://$BUCKET/era5_fat_g16.zarr \
  --max-inflight 64 --block-chunks 32 \
  --max-chunks 256 --repeats 3 --cache-dir /mnt/nvme/insitu-cache \
  | tee bench/results/story3_cache_probe.log
```

Epoch-over-epoch via the suite (cold epoch-1 vs warm epoch-2):

```bash
uv run python -m bench --url-prefix s3://$BUCKET/era5 --storage s3 \
  --out bench/results/story3_cache.jsonl --fig-dir bench/figures/story3 \
  --engines insitu --chunk-sizes 1,8,32 --epochs 2 \
  --max-batches 64 --repeats 3 --cache-dir /mnt/nvme/insitu-cache --plot
```

### Story 4 — efficiency vs the raw-GET ceiling

The probe's sec-2 raw-GET section (on by default) reads the same bytes with no
decode/gather; compare insitu's decoded MB/s (sec 1b) against it:

```bash
uv run python bench/probe_decode.py --url s3://$BUCKET/era5_fat_g16.zarr \
  --max-inflight 64 --concurrency 8,16,32,64 --max-chunks 256 --repeats 5 \
  | tee bench/results/story4_ceiling.log
```

### Free-threading readiness panels

3.13t build, GIL forced off (numcodecs re-enables it on import — see the
[numcodecs GIL gate] note), `decode_threads` controlled (≤ n_cores) and swept:

```bash
PYTHON_GIL=0 uv run python bench/probe_decode.py --url s3://$BUCKET/era5_c1.zarr \
  --decode-threads 1,2,4,8,16,32 --max-inflight 64 --max-chunks 256 --repeats 5 \
  | tee bench/results/ft_decode_threads.log
PYTHON_GIL=0 uv run python bench/probe_decode.py --url s3://$BUCKET/era5_fat_g16.zarr \
  --max-inflight 1,2,4,8,16,32,64,128 --max-chunks 256 --repeats 5 \
  --profile bench/results/ft_profile.svg | tee bench/results/ft_inflight.log
```

### S3 Express One Zone stress (story 4 on a directory bucket)

```bash
# raw-GET ceiling + insitu on Express
uv run python bench/probe_decode.py --s3-express \
  --url s3://$XBUCKET/era5_fat_g16.zarr \
  --max-inflight 64 --concurrency 8,16,32,64 --max-chunks 256 --repeats 5 \
  | tee bench/results/express_ceiling.log
# the GRIB-end suite on Express (vs the same engines on regular S3)
uv run python -m bench --s3-express \
  --url-prefix s3://$XBUCKET/era5 --storage s3 \
  --out bench/results/express_suite.jsonl \
  --engines naive,workers,xbatcher,insitu --chunk-sizes 1 \
  --num-workers 32 --max-batches 64 --repeats 3 --warmup-batches 32
```

---

## Graphs (deliverables)

`bench/plot.py` renders JSONL → interactive Plotly HTML; `--plot` on the runner
renders after a run.

- **G1 — throughput vs sample-chunk size** (lines per engine; local + S3 panels),
  with the `memory` ceiling dashed. Story 1: the async advantage grows as chunks
  shrink toward GRIB, and how close streaming gets to the ceiling.
- **G2 — throughput vs `max_inflight`** at fixed `block_chunks`, per grid density
  (the sawtooth). Story 2.
- **G3 — peak RSS (anon) vs `block_chunks`** while `max_inflight` rises: insitu's
  working set flat (bounded by block + inflight) vs `workers` growing with
  `num_workers × prefetch_factor`. Story 2 memory.
- **G4 — epoch-1 cold vs epoch-2 warm** with a large cache budget. Story 3.
- **G5 — decoded MB/s as % of raw-GET ceiling** per chunk size, S3 vs Express.
  Story 4.
- **G6 — time-to-first-batch**: worker-fork startup vs insitu.

## Rigor (what makes it convincing)

- **Tune the baselines** (`num_workers`, xbatcher batch dims) to their best; report
  the tuning.
- **Cold vs warm** controlled (fresh process / cache-off for read-once; the warmup
  burst in the runner only primes the HTTP/TLS pool, not the data cache).
- **≥3 repeats** (5 for the noisy probe sweeps), report median + min/max; exclude
  warmup batches (`--warmup-batches`).
- **Provenance** in every row: instance type, region, NIC, vCPU, codec, date.
- **Show where we lose** (fat chunks on local disk; the FT median swing) — honesty
  earns the wins.

## Phasing

- **Local (free):** validate the whole grid on small data (`uv run python -m bench`
  / `--full`) — correct *shapes* for G1/G2/G3/G6. Absolute async wins won't show
  locally; expected.
- **EC2 / S3:** run the matrix above → the real numbers; G1/G4/G5 come alive.
  Cache runs use a local-NVMe instance, cache dir on the instance store.
- **GPU (M2):** real compute step + GPU-native path; GPU-utilization graphs.

[numcodecs GIL gate]: ../../.claude/projects/-Users-davidstuebe-projects-insitubatch/memory/numcodecs-gil-gate.md
