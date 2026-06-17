# WeatherBench2 walkthrough

A **runnable, real-cloud** comparison — distinct from the controlled
[benchmark suite](benchmarks.md) (G1–G7 on a synthetic chunk-size family). Here
the same public dataset is fed through two stacks so the contrast is reproducible
end-to-end:

- [`examples/wb2_dataloader.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/wb2_dataloader.py)
  — insitubatch (one async event loop, `num_workers=0`).
- [`examples/wb2_xbatcher.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/wb2_xbatcher.py)
  — the Earthmover stack (xarray + xbatcher + torch `DataLoader` workers), with a
  `--mp` knob to compare worker start regimes.

Both crop a spatial subregion of `2m_temperature` from the public WeatherBench2
ERA5 zarr (zarr v2 on GCS) and report time-to-first-batch and throughput.

## Environment

Run on the EC2 box from the [AWS ops runbook](https://github.com/emfdavid/insitubatch/blob/main/bench/ops_aws.md)
(`c6id.8xlarge`, us-east-1). Note this reads **GCS from AWS** — cross-cloud, so
the network ceiling is lower than an in-region S3 store would give; treat the
absolute throughput as a floor, and the *shape* of the comparison as the point.

!!! warning "Compare MB/s, not samples/s, against full-resolution runs"
    WeatherBench2 `128x64` is a **downsampled** ERA5: each field is `128·64·4` ≈
    **32 KB**, about **130× smaller** than a full-resolution `721×1440` field
    (~4.15 MB). So the `samples/s` here look fast mainly because the samples are
    tiny — in **bytes/s** this run is ~96 MB/s, the same ~100 MB/s ceiling the
    [benchmark suite](benchmarks.md) sees on full-resolution data. Don't read the
    walkthrough's `samples/s` as comparable to the suite's; convert to MB/s.

## insitubatch

```bash
uv run python -m examples.wb2_dataloader --wb2 --subregion 48,32 --max-batches 1000
```

Median of 5 runs (16,000 samples each; cross-cloud GCS→AWS, so the spread is mostly
network variance):

| metric | median | range (n=5) |
|---|---:|---:|
| samples/s | 2915 | 2247 – 2985 |
| TTFB (ms) | 338 | 291 – 546 |
| mean wait (ms) | 5.3 | 5.1 – 6.8 |

This is the **zero-compute** case (`--train-step-ms 0`): the loader is purely
IO-bound, so the mean per-batch wait (6.6 ms) is dominated by the per-shuffle-block
refill — the **sawtooth** explained in
[Architecture → Prefetch](architecture.md#prefetch) and
[Startup latency](architecture.md#startup-latency-the-inference-angle). One-block
read-ahead overlaps that refill with any real per-batch compute; add
`--train-step-ms` to watch the boundary stalls disappear. Cold-start TTFB (~455 ms)
is the single chunk read before the first batch.

## xbatcher (worker stack)

```bash
uv run python -m examples.wb2_xbatcher --wb2 --compare --max-batches 500 --num-workers 16
```

```
regime                       workers   ttfb_ms   samples/s   wall_s
xbatcher spawn                    16     913.5         210    76.07
xbatcher forkserver               16    1510.6         238    67.21
xbatcher forkserver-preload       16     811.3         280    57.06
```

**Why is plain `forkserver` worse than `spawn` on TTFB?** Without
`set_forkserver_preload` the fork-server process starts nearly empty, so each
forked worker still re-imports torch/xarray/xbatcher itself — you pay spawn's
per-worker import cost **plus** the one-time fork-server bootstrap on the critical
path to the first batch. `spawn` skips that server step. forkserver only pays off
once you **preload** the heavy modules into the server: the fork is then cheap and
import-free, which is why `forkserver-preload` has the lowest TTFB and the best
throughput. (Throughput is much closer across the three — at 500 batches startup is
amortized and they converge toward the worker stack's steady rate; the spread is
mostly TTFB.)

## Side by side

| Stack | regime | TTFB (ms) | samples/s |
|---|---|---:|---:|
| **insitubatch** | event loop (`num_workers=0`) | 338 | **2915** |
| xbatcher | spawn (16 workers) | 913 | 210 |
| xbatcher | forkserver (16 workers) | 1511 | 238 |
| xbatcher | forkserver-preload (16 workers) | 811 | **280** |

Both stacks deliver the same 16,000 samples from the same WeatherBench2 store; the
configs differ in the ways each is run idiomatically (insitubatch: `num_workers=0`,
parallelism in the event loop, median of 5 runs; xbatcher: tuned to 16 workers,
single run). The ~10× throughput gap is the structural one: the xbatcher map-style
dataset reads + decodes a 40-timestep chunk **once per sample**, while insitubatch
reads each chunk **once** and slices every sample from it
([the read plan](architecture.md#prefetch)). The worker stack's *other* cost is
cold-start latency (TTFB) — see
[the fork-safety tax and inference-startup note](architecture.md#startup-latency-the-inference-angle)
for why `forkserver-preload` is its best case and why insitubatch pays none of it.
