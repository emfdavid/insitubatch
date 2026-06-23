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
- **Free-threading readiness** — throughput is **GIL-independent by design**: the heavy
  work already runs outside the GIL (Rust IO, C zstd, vectorized numpy), so 3.13t
  performs the same as the GIL build. The FT work bought **correctness** (disjoint
  scatter, publish under the lock) and future-proofing, **not** a speedup. Run the FT
  panels as a **no-regression** check, not to chase a win. *(probe_decode under `PYTHON_GIL=0`)*
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
stable median. Each family member is ~3.2 GB; the fat grids ~3.3 GB each.

> **What the `make_dataset` print reports — and what we measure.** The
> `~N MB uncompressed` line is the *logical* size (`n_samples × inner × 4`),
> printed for a quick sanity check — not a `stat` of the written store. The values
> are `standard_normal` f32, which is **incompressible**, so `compress=auto` (zstd)
> shrinks them ~0% and the **on-disk footprint ≈ this number**. That is deliberate:
> bytes-moved ≈ uncompressed keeps the MB/s and "% of raw-GET ceiling" math clean,
> and the codec still runs on the decode path (zstd decompress per chunk) so the
> decode-cost story is real. Caveat: real ERA5 compresses ~2–4×, so this synthetic
> data moves *more* bytes/sample than production (conservative for the network
> ceiling) — see the `make_dataset` docstring ("synthetic-but-realistic").

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
  uv run python -m bench.probe_decode --url s3://$BUCKET/$ds.zarr \
    --max-inflight 1,2,4,8,16,32,64,128,256 --block-chunks 32 \
    --max-chunks 256 --repeats 5 --no-raw --no-decode-sweep \
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
uv run python -m bench.probe_decode --url s3://$BUCKET/era5_fat_g16.zarr \
  --max-inflight 64 --block-chunks 32 \
  --max-chunks 256 --repeats 3 --cache-dir /mnt/nvme/insitu-cache \
  | tee bench/results/story3_cache_probe.log
```

Epoch-over-epoch via the suite (cold epoch-1 vs warm epoch-2). Two essentials, or the
warm epoch shows no benefit: **`--caches resident`** sizes the pool to hold the whole
train split (read-once "none" re-reads each epoch), and **full epochs** (no
`--max-batches`) so epoch 0 caches the entire split and any epoch-1 draw order hits:

```bash
uv run python -m bench --url-prefix s3://$BUCKET/era5 --storage s3 \
  --out bench/results/story3_cache.jsonl --fig-dir bench/figures/story3 \
  --engines insitu --caches resident --chunk-sizes 1,8,32 --epochs 2 \
  --repeats 3 --cache-dir /mnt/nvme/insitu-cache --plot
```

### Story 4 — efficiency vs the raw-GET ceiling

The probe's sec-2 raw-GET section (on by default) reads the same bytes with no
decode/gather; compare insitu's decoded MB/s (sec 1b) against it:

```bash
uv run python -m bench.probe_decode --url s3://$BUCKET/era5_fat_g16.zarr \
  --max-inflight 64 --concurrency 8,16,32,64 --max-chunks 256 --repeats 5 \
  | tee bench/results/story4_ceiling.log
```

**Flamegraph — the native hot path (supports story 4 + the FT framing).** `--profile`
wraps the sec-1b sweep in `py-spy --native`, which sees obstore (Rust) and numcodecs (C);
the picture is time in **Rust IO + C decode + numpy**, with only a thin Python sliver —
*why* we're IO/decode-bound and *why* free-threading has no GIL-held work to accelerate.
Run it on the regular env (py-spy is in `--extra bench`) at one steady-state concurrency:

```bash
sudo sysctl -w kernel.yama.ptrace_scope=0   # once; py-spy --native needs ptrace
uv run python -m bench.probe_decode --url s3://$BUCKET/era5_fat_g16.zarr \
  --max-inflight 64 --max-chunks 256 --repeats 3 --no-decode-sweep --no-raw \
  --profile bench/results/profile_fat_g16.svg
```

### Memory by engine (the G5 rebuild)

`probe_memory` runs **each engine in its own subprocess** and samples peak RSS over the
whole **process tree** (so the 32 DataLoader workers of `workers`/`xbatcher` are counted
— the suite's `peak_rss_mb` saw only the main process). It also runs **two epochs** and
reports **TTFB cold (ep0, worker-spawn cost) and warm (ep1)** plus warm MB/s. Read-once
(no `--cache-dir`), so RSS is the anon working set, apples to apples. Prewarm is automatic
(stable run-over-run). Two runs, each a *different* memory mechanism:

**GRIB end (c1) — the fan-out.** The baselines pay the per-worker interpreter floor 32×;
insitu is one process. This is where cold-start TTFB also wins biggest.

```bash
uv run python -m bench.probe_memory --url s3://$BUCKET/era5_c1.zarr --storage s3 \
  --engines insitu,workers,xbatcher --sample-chunk 1 \
  --num-workers 32 --batch-size 32 --max-batches 64 | tee bench/results/g5_memory_c1.log
```

**Fat chunk (c16) — insitu wins *both* memory and throughput.** The baselines re-decode
the 16-sample chunk per sample across 32 workers and pay the interpreter floor 32× → still
~20 GB; insitu reads each chunk once with bounded residency (`2 × 16 × 16.6 MB ≈ 0.5 GB`,
default `block_chunks` is fine) → ~1 GB. Unlike c1, at c16 insitu **also wins throughput**
(story 1: ~11×), so this is the clean "wins on both axes" point.

```bash
uv run python -m bench.probe_memory --url s3://$BUCKET/era5_c16.zarr --storage s3 \
  --engines insitu,workers,xbatcher --sample-chunk 16 \
  --num-workers 32 --batch-size 32 --max-batches 64 | tee bench/results/g5_memory_c16.log
```

(The genuinely fat `era5_fat`, sc=200, is the most dramatic *memory* contrast — each of 32
workers materializes the full 208 MB field per sample — but the per-sample decode of a
208 MB chunk makes the baselines impractically slow to run; c16 proves the same memory
win and pairs it with a throughput win, fast.)

This folds in **G6 (TTFB)**: insitu's TTFB is clean (~210 ms, flat cold/warm — no worker
pool to spawn) **and** a bounded single-process footprint vs the 33-process fan-out.

> **Reading the columns — two caveats.**
> - **`warmMB/s` is read-once (no insitu cache), deliberately.** Caching here would size
>   the pool to the whole split and balloon insitu's RSS, defeating the memory
>   measurement. So at c1 insitu *loses* throughput (read-once, GRIB end — consistent
>   with story 1); the cache-warm throughput win is **story 3** (4.5×), measured
>   separately. Don't mix them.
> - **Cross-engine TTFB from this probe is confounded** — engines run sequentially over
>   one S3 prefix (cumulative warming) and each subprocess has its own cold TLS pool, so
>   `workers` (2nd) vs `xbatcher` (3rd) cold TTFB isn't a clean spawn-cost comparison, and
>   xbatcher's per-batch xarray cost can make its warm ≥ cold. Use insitu's TTFB from
>   here, but take the **clean cold-start comparison from `examples/wb2_xbatcher.py`**
>   (fresh loader per epoch, isolated). The memory + procs columns are the solid deliverable.

### Cold-start on WeatherBench2 (the Earthmover demo, head-to-head)

The bench engines run synthetic ERA5-shaped data; these two examples run the **public
WeatherBench2 ARCO store** (gs://, anonymous) through the *actual* Earthmover stack and
the insitu equivalent — a recognizable, reproducible cold-start comparison. This is the
clean TTFB story the G5 probe's cross-engine column can't give. Needs the `bench` extra
(xbatcher, gcsfs) + torch; drop `--wb2` for a network-free synthetic sanity run.

```bash
# Earthmover stack (xbatcher + torch DataLoader): worker-process brute force. --compare
# sweeps the worker start methods -- spawn re-imports per worker (slow), forkserver/preload
# amortize it -- and prints the mp-mode TTFB table.
uv run python -m examples.wb2_xbatcher --wb2 --compare --subregion 48,48 --max-batches 100 \
  | tee bench/results/wb2_xbatcher.log

# insitu equivalent: one in-process event loop, num_workers=0 -- no spawn, first batch in ms.
uv run python -m examples.wb2_dataloader --wb2 --subregion 48,48 --max-batches 100 \
  | tee bench/results/wb2_dataloader.log
```

The contrast (synthetic preview): xbatcher pays **~2.5–2.8 s** to first batch (and spawn
costs ~43 s wall to re-import per worker; forkserver-preload trims it to ~2.7 s), while
insitu's event loop is **~11 ms**. That is the framing — xbatcher is the domain-standard
*batch definition*, but its **engine is worker-process brute force** (many procs, heavy
memory, slow cold start); insitu keeps the ndim batch semantics with one async loop,
**batteries-included across the chunk spectrum** — its edge grows with samples-per-chunk
and needs enough chunks in flight to saturate IO (it only gives ground in the pathological
few-giant-single-inner-chunk case, which spatial chunking fixes).

### Free-threading readiness panels

**What these show (and don't).** insitubatch's throughput is **GIL-independent**: fetch
(obstore/Rust), decode (numcodecs zstd, C), and scatter/gather (vectorized numpy) all
release the GIL, and scheduling is a single asyncio loop — so there is no GIL-held hot
path for free-threading to speed up. Empirically, decode already parallelizes *under
the GIL* (the `fat_g16` `decode_threads` sweep scales 1→2 even on the GIL build, because
zstd releases it). So these panels are a **no-regression / correctness** check on 3.13t,
not a speedup demo; do not expect (or report) an FT win. The remaining GIL-held slice is
thin per-chunk Python bookkeeping, kept small by the O(chunks)-not-O(samples) rule.

Operationally: free-threading is **3.13t only** — there is no free-threaded 3.12, so
`PYTHON_GIL=0` on the default env is a silent no-op (you get GIL numbers). Use a separate 3.13t env
and **pin the interpreter on `uv run`**, exactly as the README's "Free-threaded
(3.13t)" section (numcodecs re-enables the GIL on import — see the
[numcodecs GIL gate] note — so the GIL must be forced off):

```bash
# once: build the FT env on the free-threaded interpreter
uv python install 3.13t
UV_PROJECT_ENVIRONMENT=.venv-ft uv sync --python 3.13t --extra bench
# always assert the GIL is actually off before trusting any FT number
PYTHON_GIL=0 UV_PROJECT_ENVIRONMENT=.venv-ft uv run --python 3.13t \
  python -c "import sys, zarr, numcodecs; assert not sys._is_gil_enabled(); print('GIL-free OK')"
```

`decode_threads` controlled (≤ n_cores) and swept. The env vars must be **inline on
each command** — a `VAR=val …` prefix stashed in a shell variable and run as `$VAR …`
is parsed as a command name, not an assignment (`PYTHON_GIL=0: command not found`):

```bash
# decode_threads panel (this is the sec-1 sweep -- keep it here)
PYTHON_GIL=0 UV_PROJECT_ENVIRONMENT=.venv-ft uv run --python 3.13t \
  python -m bench.probe_decode --url s3://$BUCKET/era5_c1.zarr \
  --decode-threads 1,2,4,8,16,32 --max-inflight 64 --max-chunks 256 --repeats 5 \
  | tee bench/results/ft_decode_threads.log

# max_inflight sweep (decode_threads panel already covered above). No --profile here:
# the native-hot-path flamegraph is the regular-env story-4 artifact, env-agnostic.
PYTHON_GIL=0 UV_PROJECT_ENVIRONMENT=.venv-ft uv run --python 3.13t \
  python -m bench.probe_decode --url s3://$BUCKET/era5_fat_g16.zarr \
  --max-inflight 1,2,4,8,16,32,64,128 --max-chunks 256 --repeats 5 --no-decode-sweep \
  | tee bench/results/ft_inflight.log
```

> The output banner must read a **3.13 free-threaded** interpreter — if it says
> `CPython 3.12.x`, the pin didn't take and you're measuring the GIL. Expect the FT
> numbers to **match** the GIL build (that is the result: GIL-independence), not beat
> it. numcodecs re-enabling the GIL on import only matters for Python-level
> parallelism, which our GIL-releasing hot path doesn't rely on.

### S3 Express One Zone stress (story 4 on a directory bucket)

```bash
# raw-GET ceiling + insitu on Express
uv run python -m bench.probe_decode --s3-express \
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
  > **Blocked on a valid measurement (the G5/memory rebuild — do first).** Today
  > `peak_rss_mb` reads `/proc/self/status` of the **single suite process**, so (a) it's
  > a monotonic high-water shared by every engine run in that process (all engines
  > report the same number) and (b) it **excludes the 32 worker child processes** — so
  > `workers`/`xbatcher`, whose whole memory cost is the fan-out, look *lighter* than
  > insitu. The fix: run **each engine config in its own subprocess** (clean per-engine
  > high-water) and sample **peak RSS over the whole process tree** (main + children,
  > e.g. `psutil` polled on a timer, since workers hold chunks transiently). Report
  > total peak + the anon/file split. Until then, **do not cite RSS** in any comparison.
- **G4 — epoch-1 cold vs epoch-2 warm** with a large cache budget. Story 3.
- **G5 — decoded MB/s as % of raw-GET ceiling** per chunk size, S3 vs Express.
  Story 4.
- **G6 — time-to-first-batch**: worker-fork startup vs insitu.

## Rigor (what makes it convincing)

- **Tune the baselines** (`num_workers`, xbatcher batch dims) to their best; report
  the tuning.
- **Cold vs warm** controlled. The runner warms **every** prefix before timing
  (each `_c<spc>.zarr` is its own S3 key prefix with a separate request-rate ramp);
  the probe does one throwaway prewarm burst at the top (`--no-warm` to disable).
  The prewarm primes obstore's TLS pool **and** the per-prefix ramp — not the data
  cache, so read-once/cold-vs-warm comparisons stay honest (the cache test's epoch-0
  becomes S3-warm/pool-cold, isolating decode+pool reuse from the S3 ramp).
- **≥3 repeats** (5 for the noisy probe sweeps), report median + min/max; exclude
  warmup batches (`--warmup-batches`).
- **Provenance** in every row: instance type, region, NIC, vCPU, codec, date.
- **Show where we lose / draw** (fat chunks on local disk; insitu below the baselines
  at the GRIB end `c1`; FT matching — not beating — the GIL build) — honesty earns the
  wins. The FT median swing is oversubscription noise (`decode_threads ≈ cpu+4 > cores`),
  not signal; control `decode_threads ≤ cores` and report p50/p95.

## Phasing

- **Local (free):** validate the whole grid on small data (`uv run python -m bench`
  / `--full`) — correct *shapes* for G1/G2/G3/G6. Absolute async wins won't show
  locally; expected.
- **EC2 / S3:** run the matrix above → the real numbers; G1/G4/G5 come alive.
  Cache runs use a local-NVMe instance, cache dir on the instance store.
- **GPU (M2):** real compute step + GPU-native path; GPU-utilization graphs.

[numcodecs GIL gate]: ../../.claude/projects/-Users-davidstuebe-projects-insitubatch/memory/numcodecs-gil-gate.md
