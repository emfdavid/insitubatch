# WeatherBench2 walkthrough

A **runnable, real-cloud** comparison — distinct from the controlled
[benchmark suite](benchmarks.md), which sweeps synthetic chunk sizes. Here the same
public dataset is fed through two stacks so the contrast is reproducible
end-to-end:

- [`examples/wb2_dataloader.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/wb2_dataloader.py)
  — insitubatch (one async event loop, `num_workers=0`).
- [`examples/wb2_xbatcher.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/wb2_xbatcher.py)
  — a [xarray](https://xarray.dev) + [xbatcher](https://github.com/xarray-contrib/xbatcher)
  + torch `DataLoader` worker stack, with a `--mp` knob to compare worker start
  regimes. This is the worker-based pattern Earthmover wrote up in
  [Build a cloud-native data loader for ML training](https://earthmover.io/blog/cloud-native-dataloader);
  xbatcher itself is a community [xarray-contrib](https://github.com/xarray-contrib) project.

Both crop a spatial subregion of `2m_temperature` from the public WeatherBench2
ERA5 zarr (zarr v2 on GCS) and report time-to-first-batch and throughput.

## Environment

Run on the EC2 box from the [AWS ops runbook](https://github.com/emfdavid/insitubatch/blob/main/bench/ops_aws.md)
(`c6id.16xlarge`, us-east-1, 25 Gb/s). Note this reads **GCS from AWS** — cross-cloud,
so the network ceiling is lower than an in-region S3 store would give; treat the
absolute throughput as a floor, and the *shape* of the comparison as the point.

Both stacks are measured on **identical 48×32 samples** (16,000 each, same store),
each preceded by a discarded warmup run to absorb the GCS per-prefix ramp; xbatcher
is reported at its best worker count (swept 16/32/64).

!!! warning "Compare MB/s, not samples/s, against full-resolution runs"
    WeatherBench2 `128x64` is a **downsampled** ERA5: each field is `128·64·4` ≈
    **32 KB**, about **130× smaller** than a full-resolution `721×1440` field
    (~4.15 MB). So the `samples/s` here look fast mainly because the samples are
    tiny — in **bytes/s** insitubatch's run below is only ~140 MB/s. Don't read the
    walkthrough's `samples/s` as comparable to the [benchmark suite](benchmarks.md)'s
    full-resolution numbers; convert to MB/s first.

## insitubatch

```bash
uv run python -m examples.wb2_dataloader --wb2 --subregion 48,32 --max-batches 1000
```

Median of 5 timed runs (a discarded warmup precedes them; 16,000 samples each;
cross-cloud GCS→AWS, so the spread is mostly network variance):

| metric | median | range (n=5) |
|---|---:|---:|
| samples/s | 4447 | 4014 – 4922 |
| TTFB (ms) | 317 | 295 – 335 |
| mean wait (ms) | 3.3 | 3.2 – 3.9 |

This is the **zero-compute** case (`--train-step-ms 0`): the loader is purely
IO-bound, so the mean per-batch wait (table above) is dominated by the
per-shuffle-block refill — the **sawtooth** explained in
[Architecture → Prefetch](architecture.md#prefetch) and
[Startup latency](architecture.md#startup-latency-the-inference-angle). One-block
read-ahead overlaps that refill with any real per-batch compute; add
`--train-step-ms` to watch the boundary stalls disappear. The TTFB above is the
single cold chunk read before the first batch.

## xbatcher (worker stack)

```bash
uv run python -m examples.wb2_xbatcher --wb2 --subregion 48,32 --compare --max-batches 500 --num-workers 16
```

Median of 3 runs at 16 workers (xbatcher's best — see below):

```
regime                       workers   ttfb_ms   samples/s
xbatcher spawn                    16       874         230
xbatcher forkserver               16       818         249
xbatcher forkserver-preload       16       883         289
```

**16 workers is xbatcher's best here, and adding workers *hurts*.** Each worker
reads and decodes the full 40-timestep chunk *once per sample*, so more workers
multiply that redundant decode rather than adding useful concurrency: throughput
falls to ≈186 samples/s at 32 workers and ≈110 at 64. This is the **inverse** of
the GRIB / one-sample-per-chunk regime (see [benchmarks](benchmarks.md)), where the
chunk holds a single sample and more worker processes *do* add useful read
concurrency — there xbatcher scales up to 64 workers. The right worker count is a
property of the chunk layout, not a constant.

**On worker start regimes (`--mp`):** at this scale the three are within run-to-run
noise on TTFB (~820–880 ms on a 64-core box with warm GCS); `forkserver-preload`
keeps a modest *throughput* edge. The structural cost is that *every* regime pays
~850 ms of worker-stack startup before the first batch — versus insitubatch's
317 ms — see
[the fork-safety tax and inference-startup note](architecture.md#startup-latency-the-inference-angle).

## Side by side

| Stack | regime | TTFB (ms) | samples/s |
|---|---|---:|---:|
| **insitubatch** | event loop (`num_workers=0`) | **317** | **4447** |
| xbatcher | spawn (16 workers) | 874 | 230 |
| xbatcher | forkserver (16 workers) | 818 | 249 |
| xbatcher | forkserver-preload (16 workers) | 883 | **289** |

Both stacks deliver the **same 16,000 samples at the same 48×32 shape** from the
same WeatherBench2 store; the configs differ only in the ways each is run
idiomatically (insitubatch: `num_workers=0`, parallelism in the event loop;
xbatcher: tuned to its best worker count). The ~15× throughput gap is the
structural one: the xbatcher map-style dataset reads + decodes a 40-timestep chunk
**once per sample**, while insitubatch reads each chunk **once** and slices every
sample from it ([the read plan](architecture.md#prefetch)). The worker stack's
*other* cost is cold-start latency (TTFB, ~2.7× here) — see
[the fork-safety tax and inference-startup note](architecture.md#startup-latency-the-inference-angle)
for why insitubatch pays none of it. This is the **fat-chunk** regime, which favors
insitubatch most; at the GRIB / one-sample-per-chunk end the gap narrows and the
worker fan-out's warm throughput can lead — see [benchmarks](benchmarks.md).
