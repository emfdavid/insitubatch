# Benchmarks

These are the graphs from the benchmark suite, embedded live (Plotly — hover,
zoom, toggle traces). The suite isolates the contribution of each optimization
against good-faith baselines across the chunk-size spectrum. The full
methodology, baseline definitions, and the win-claim gate are in the
[benchmark plan](https://github.com/emfdavid/insitubatch/blob/main/bench/benchmark_plan.md).

!!! warning "These figures are an illustrative *local synthetic* run"
    The figures below come from a small synthetic dataset on a laptop
    (`file://` store, no network), produced purely to demonstrate the suite and
    the embedding. They are **not** a performance claim. The headline comparison
    requires the realistic regime — cloud S3 reads, real chunk sizes, and a tuned
    `xbatcher + DataLoader` baseline (B2) on the
    [EC2 NVMe instance](https://github.com/emfdavid/insitubatch/blob/main/bench/ops_aws.md).
    On tiny local chunks the IO that insitubatch overlaps is nearly free, so the
    advantage doesn't show — that's expected, and exactly why the real run matters.

## The comparison set

| Engine | What it is | Role |
|---|---|---|
| `insitu` | insitubatch: one async event loop, shared chunk cache, prefetch | the system under test |
| `naive` | sequential synchronous reads, one sample at a time | the floor |
| `workers` | map-style `Dataset` + `DataLoader(num_workers=N)` | the realistic baseline |
| `xbatcher` | `xbatcher.BatchGenerator` + `DataLoader` (the Earthmover stack) | the credibility bar (B2) |
| `memory` | data preloaded into RAM, compute only | the in-memory ceiling |

The DataLoader baselines (`workers`, `xbatcher`) are swept over `num_workers` and
reported at their **best** setting, so the comparison isn't a strawman.

## G1 — Throughput vs sample-chunk size

The chunk-size spectrum, from GRIB-per-timestep (1 sample/chunk) to fat chunks.
`memory` is the in-memory ceiling.

<iframe src="figures/g1_throughput_vs_chunk.html" width="100%" height="480" frameborder="0"></iframe>

## G2 — Ablation by engine

Throughput per engine at the tuned baseline settings.

<iframe src="figures/g2_ablation.html" width="100%" height="480" frameborder="0"></iframe>

## G3 — Prefetch overlap vs per-batch compute

As the simulated training step grows, prefetch hides more of the read latency —
the insitu curve should flatten relative to the baselines.

<iframe src="figures/g3_throughput_vs_compute.html" width="100%" height="480" frameborder="0"></iframe>

## G4 — Cache: cold vs warm epochs

Cross-epoch reuse of prepped chunks (epoch 0 cold vs epoch 1+ warm), insitu only.

<iframe src="figures/g4_cache_epochs.html" width="100%" height="480" frameborder="0"></iframe>

## G5 — Resident memory by engine

Resident **heap** (RssAnon) per engine — worker processes each hold their own copy;
insitu shares one cache. The heap is the real memory bound: DiskCache's mmap'd
`.npy` is file-backed and reclaimable, so it shows as bounded heap even when total
RSS looks large (and `ru_maxrss` is a misleading monotonic high-water in the
single-process suite).

<iframe src="figures/g5_peak_memory.html" width="100%" height="480" frameborder="0"></iframe>

## G6 — Time-to-first-batch

How long until the loop gets its first batch (worker spin-up vs event-loop start).

<iframe src="figures/g6_ttfb.html" width="100%" height="480" frameborder="0"></iframe>

## G7 — Baseline tuning curve

Throughput vs `num_workers` for the DataLoader baselines — this is what "tuned to
their best" means, and where the best-of points in G1/G2 come from.

<iframe src="figures/g7_worker_tuning.html" width="100%" height="480" frameborder="0"></iframe>

## Reproduce

```bash
# 1. run the suite (writes one JSONL row per engine/config/epoch)
uv run python -m bench --full --plot                       # local synthetic
uv run python -m bench --full --url-prefix s3://bucket/era5 \
    --cache-dir /mnt/nvme/cache --request-payer --plot      # cloud S3

# 2. (re)build the embeddable, CDN-loaded figures used on this page
uv run python -m bench.plot --in bench/results/suite.jsonl --out docs/figures --cdn
```

The `--cdn` flag loads `plotly.js` from a CDN instead of inlining it, keeping each
figure small enough to commit and embed here.
