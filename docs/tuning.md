# Tuning: batch, shuffle window, concurrency, memory

This page is practical guidance for setting up `InSituDataset` on your own store. For *why*
the engine behaves this way — the read plan, the pool, the prefetch pipeline — see
[Architecture](architecture.md).

## The mental model

Hold these two sentences and the rest follows:

- A **sample** is one slice of the **sample axis** — a timestep, an observation, a model
  state, a microscopy `Z`-plane, whatever your rows are (it can be *any* single physical axis,
  not just axis 0) — spanning the whole inner extent of its chunk.
- A **batch** draws `batch_size` shuffled samples from a **window** of `block_chunks`
  sample-axis chunks that the loader keeps decoded in memory at once.

So the loader reads each chunk once, holds a rolling window of them, and serves shuffled
batches out of that window — concurrency fills the window, the window bounds memory.

## The knobs you set

All of these are `InSituDataset(...)` arguments except the last, which is fixed when the
store is written.

| plain name | argument | what it does | default |
|---|---|---|---|
| batch size | `batch_size` | samples per batch | 32 |
| shuffle window | `block_chunks` | outer chunks held resident + shuffled across at once | 16 |
| reads in flight | `max_inflight` | concurrent stored-chunk GETs (the network dial) | 32 |
| batch queue | `prefetch_depth` | assembled batches queued ahead of your training step | 2 |
| cache | `cache_budget_bytes`, `cache_dir` | decoded data retained across epochs (decode-once) | off |
| stored-chunk size | `inner_chunks` (write time) | the fetch unit — how each chunk is split for IO | — |

`batch_size` is the ordinary ML knob and barely touches IO. `block_chunks` trades shuffle
quality against RAM. `max_inflight` is what you turn to saturate the network. The cache is
how repeated epochs (or repeated scoring passes) skip re-reading. `inner_chunks` is the one
decision made when the data is written, and it sets how cheap concurrency can be.

## The memory model

Peak memory is the sum of three independently-bounded pieces — none grows with epoch length
or dataset size:

- **Shuffle window:** `block_chunks × outer_chunk_bytes` — the decoded chunks held resident.
- **Reads in flight:** `max_inflight × stored_chunk_bytes` — the fetch pipeline.
- **Batch queue:** `prefetch_depth × batch_bytes` — assembled batches awaiting the consumer.

where an outer chunk is `sample_chunk × ∏inner_shape × itemsize` and a stored chunk is
`sample_chunk × ∏inner_chunk × itemsize`.

The point to internalize: **raising concurrency costs *stored-chunk*-sized memory, not
*outer-chunk*-sized** — but only when the data is inner (spatially) chunked. If each outer
chunk is a single stored chunk, those two sizes are equal and concurrency gets expensive
(the "fat, single inner" regime below).

If you set `cache_budget_bytes` above the working set, residency rises to that budget on
purpose — that extra memory *is* the cross-epoch cache. Point `cache_dir` at local NVMe to
spill it to disk instead of RAM.

## Shuffle quality

`block_chunks` is also the shuffle-quality knob. Each batch is drawn from the samples in the
current window — `block_chunks × samples-per-chunk` of them — so set the window so that pool
is comfortably larger than `batch_size`; otherwise a batch is just one or two chunks' worth
of correlated samples. The chunks are re-permuted every epoch, so even a modest window
converges toward a full-dataset shuffle over many epochs — the regime training actually runs
in. [`shuffle_quality`](api.md) scores an emitted order 0–1 (1 ≈ global) if you want to
measure it; [Architecture](architecture.md) explains why the block-local shuffle converges.

## The recipe

1. **At write time, pick `inner_chunks`** so a stored chunk is ~10–50 MB: small enough that
   many reads in flight stay cheap, large enough that per-request overhead doesn't dominate.
2. **Start with the defaults** (`max_inflight=32`, `block_chunks=16`). 32 reads in flight
   saturates in-region S3 in most cases.
3. **Raise `block_chunks`** for better shuffle quality, as far as your RAM allows
   (`block_chunks × outer_chunk_bytes` ≤ your budget).
4. **Tune `max_inflight`** if the network isn't saturated — raise it until decoded MB/s
   stops climbing. From the repo you can measure the knee directly:

   ```bash
   python -m bench.probe_decode --url <store> --concurrency 1,4,8,16,32
   ```

5. **Sanity-check concurrency cost** (`max_inflight × stored_chunk_bytes`). If it's large,
   your stored chunks are too big — chunk the inner dims (step 1).
6. **For multi-epoch training**, set `cache_budget_bytes` to hold the split (and `cache_dir`
   on NVMe to spill); epoch 0 warms it and later epochs read decode-once.

## Regimes

| regime | shape | guidance |
|---|---|---|
| **GRIB** (chunk=1) | 1 sample/chunk, single inner | concurrency follows `block_chunks`; a worker loader is competitive here single-pass (nothing to amortize), but the cross-epoch cache wins repeated passes |
| **moderate** | ~8–40 samples/chunk, single inner | the common case; `max_inflight ≈ 32`. insitu's edge grows with `sample_chunk` (each chunk is read once, not re-decoded per sample) |
| **fat, single inner** | huge outer chunk, single inner | the stored chunk *is* the outer chunk, so concurrency costs full-chunk memory. **Rechunk the inner dims**, or shrink `sample_chunk` |
| **fat, spatial** | huge outer chunk, inner grid | the sweet spot: small stored chunks make high `max_inflight` cheap; keep `block_chunks` small for low residency |

## Advanced: decode threads

`decode_threads` (on `SchedulerConfig`) sizes the pool that runs codec decode and the
scatter memcpy. It defaults to auto (`min(32, cpu+4)`) and rarely needs changing; on a busy
box ~8 can beat auto by avoiding oversubscription. It is only reachable when you drive a
`Scheduler` directly — `InSituDataset` uses the auto default. There is no separate
inner-fan-out cap: the inner grid is dialed by `max_inflight`, which fetches at stored-chunk
granularity.
