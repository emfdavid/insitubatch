# Tuning: chunks, concurrency, memory

Three knobs decide insitubatch's throughput and memory. They're easy to set once
you hold the mental model: **a sample is one outer-axis slice spanning all the inner
chunks of its outer chunk; a batch draws `batch_size` shuffled samples from a window
of `block_chunks` outer chunks.** From that, everything follows.

## The knobs

| knob | controls | set it for |
|---|---|---|
| `block_chunks` | shuffle window + decoded chunks held resident | shuffle quality and the RAM you can spend |
| `max_inflight` | reads in flight (network saturation) | the raw-GET knee of your store |
| `inner_chunks` (at write time) | stored-chunk size = the fetch unit | cheap concurrency without tiny requests |

## The memory model

```
peak ≈ block_chunks × (sample_chunk · ∏inner_shape · itemsize)   # residency (shuffle window)
     + max_inflight × (sample_chunk · ∏inner_chunk  · itemsize)   # fetch pipeline (in flight)
     + prefetch_depth × batch_bytes                               # assembled-batch queue
```

The one invariant to internalize: **raising concurrency costs *stored-chunk*-sized
memory, not *outer-chunk*-sized — but only if your data is inner (spatially)
chunked.** If each outer chunk is a single stored chunk, the two terms collapse and
concurrency gets expensive (see the fat-single-inner row below).

!!! note "v1 vs v2"
    In v1, read concurrency follows `block_chunks` (a block fetches its chunks), so
    these terms aren't yet fully independent — to get more concurrency you raise
    `block_chunks` and pay residency for it. The [V2 decoupled fetch
    scheduler](https://github.com/emfdavid/insitubatch/blob/main/DESIGN.md) makes
    `max_inflight` an independent budget. The guidance below is written for the V2
    model; under v1, read `max_inflight` as "set `block_chunks` to your desired
    concurrency, memory permitting."

## The recipe

1. **Pick `inner_chunks` (at write time)** so a stored chunk is ~10–50 MB — small
   enough that `max_inflight` is cheap, large enough that per-request overhead
   doesn't dominate.
2. **Set `max_inflight`** to your store's raw-GET knee. Measure it:
   `python -m bench.probe_decode --url <store> --concurrency 1,4,8,16,32` — pick
   where MB/s flattens (~16–32 for in-region S3).
3. **Set `block_chunks`** for shuffle quality and the residency you can afford:
   `block_chunks × outer_chunk_bytes ≤ your RAM budget`.
4. **Sanity-check concurrency cost**: `max_inflight × stored_chunk_bytes`. If that's
   large, your stored chunks are too big — chunk the inner dims.

## Regimes

| regime | shape | guidance |
|---|---|---|
| **GRIB** (chunk=1) | 1 sample/chunk, single inner | concurrency = `block_chunks`; the standard worker loader is competitive here (no per-chunk redundancy) |
| **moderate** | ~8–40 samples/chunk, single inner | the headline case; `max_inflight ≈ 32`. insitu's edge ≈ `sample_chunk` (the worker loader re-decodes the whole chunk per sample) |
| **fat, single inner** | huge outer chunk, single inner | pathological: the stored chunk *is* the outer chunk, so concurrency = memory. **Rechunk spatially**, or shrink `sample_chunk` |
| **fat, spatial** | huge outer chunk, inner grid | the sweet spot: small stored chunks → high `max_inflight` is cheap; keep `block_chunks` small for low residency |

## Decode and inner-fan-out knobs

- `decode_threads` (`IOConfig`) — the loop's executor for codec decode. `0` = auto
  (`min(32, cpu+4)`). On a busy box, ~8 often beats auto (oversubscription).
- `read_concurrency` (`IOConfig`) — zarr's inner fan-out per `getitem` (v1 only).
  Set it **≥ the inner-grid count** so a spatial field doesn't take an extra partial
  wave (e.g. 15 inner tiles at the default cap of 10 = 2 waves ≈ half rate). V2
  folds this into the single `max_inflight` budget.
