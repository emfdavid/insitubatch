# Tuning: batch, shuffle window, concurrency, memory

This page is practical guidance for setting up `InSituDataset` on your own store. For *why*
the engine behaves this way — the read plan, the pool, the prefetch pipeline — see
[Architecture](architecture.md); for the measurements behind the numbers here, see
[Benchmarks](benchmarks.md).

Read [Two operating points](#two-operating-points) first if you are serving inference rather
than training — several defaults below are chosen for training and are the wrong call for a
latency-sensitive single pass.

## The mental model

Hold these two sentences and the rest follows:

- A **sample** is one slice of the **sample axis** — a timestep, an observation, a model
  state, a microscopy `Z`-plane, whatever your rows are (it can be *any* single physical axis,
  not just axis 0) — spanning the whole inner extent of its chunk.
- A **batch** draws `batch_size` shuffled samples from a **window** of `block_chunks`
  sample-axis chunks that the loader keeps decoded in memory at once.

So the loader reads each chunk once, holds a rolling window of them, and serves shuffled
batches out of that window — concurrency fills the window, the window bounds memory.

## Two operating points

The same knobs point in different directions depending on what you are optimizing, and the
difference is large enough that guidance for one is actively wrong for the other.

- **Training** optimizes **steady-state throughput per GB of RAM**. It runs for hours, so the
  first batch costs nothing, and it re-reads the data every epoch, so caching and shuffle
  quality matter.
- **Inference / scoring** optimizes **time to first batch**. It is single-pass (no shuffle to
  protect, nothing to cache across epochs) and often short-lived, so a cold start you could
  ignore in training is the whole latency budget.

`max_inflight` is where they diverge, and not in the way its name suggests. Sweeping it over
real WeatherBench2 ERA5 on a GPU loop:

| `max_inflight` | samples/s | stall | peak RSS | **cold TTFB** | warm TTFB |
|---|--:|--:|--:|--:|--:|
| 1 | 1332.1 | 3.7% | 1095 MB | **3462 ms** | 87 ms |
| 4 | 1330.3 | 3.6% | 1121 MB | 656 ms | 85 ms |
| 16 | 1326.9 | 3.7% | 1198 MB | 347 ms | 88 ms |
| 32 (default) | 1332.1 | 3.4% | 1246 MB | **317 ms** | 80 ms |

**Steady-state throughput does not move at all** — 1332.1 at either end, stall flat — while
cold TTFB moves **11×**. On a loop with enough compute per byte, read-ahead is a *cold-start*
dial, not a throughput dial. Dropping it to 1 buys ~150 MB (12%) for free in training, and
costs an inference service more than three seconds on its first request.

The corollary for the other two axes: **`block_chunks` is a training knob** (it buys shuffle
quality, and only `.train` shuffles — eval views are deterministic), and the **cross-epoch
cache is a training knob** (single-pass scoring never reads a chunk twice).

## The knobs you set

All of these are `InSituDataset(...)` arguments except the last, which is fixed when the
store is written.

| plain name | argument | what it does | default |
|---|---|---|---|
| batch size | `batch_size` | samples per batch | 32 |
| shuffle window | `block_chunks` | outer chunks held resident + shuffled across at once | 16, raised if needed to hold one batch |
| reads in flight | `max_inflight` | concurrent stored-chunk GETs — sets **cold start**; sets throughput only while IO-bound | 32 |
| batch queue | `prefetch_depth` | assembled batches queued ahead of your training step | 2 |
| cache | `cache_budget_bytes`, `cache_dir` | decoded data retained across epochs (decode-once) | off |
| stored-chunk size | `inner_chunks` (write time) | the fetch unit — how each chunk is split for IO | — |

`batch_size` is the ordinary ML knob and barely touches IO. `block_chunks` trades shuffle
quality against RAM. `max_inflight` saturates the network *while you are IO-bound* — once
compute is the constraint it stops buying throughput and buys cold start instead, which is
why the table above matters more than its name suggests. The cache is how repeated epochs (or
repeated scoring passes) skip re-reading. `inner_chunks` is the one decision made when the
data is written, and it sets how cheap concurrency can be.

## The memory model

Peak memory is the sum of three independently-bounded pieces — none grows with epoch length
or dataset size:

- **Shuffle window:** `block_chunks × resident_chunk_bytes` — the decoded chunks held resident.
- **Reads in flight:** `max_inflight × stored_chunk_bytes` — the fetch pipeline.
- **Batch queue:** `prefetch_depth × batch_bytes` — assembled batches awaiting the consumer.

`InSituDataset.print_summary()` evaluates all of this for *your* geometry and configuration,
before any read happens — including the ragged multiplier below, the run length `gather` will
get, and an estimated peak. Reach for it rather than doing the arithmetic by hand:

```python
ds = InSituDataset(store, manifest, batch_size=32, block_chunks=16)
ds.print_summary()              # ds.print_summary(iterations=2) if you will zip two views
```

where a stored chunk is `sample_chunk × ∏inner_chunk × itemsize`, and a resident chunk is
the **stored chunks that compose it**, held whole:
`n_stored_chunks × stored_chunk_bytes`. That equals the logical
`sample_chunk × ∏inner_shape × itemsize` exactly when the chunk grid divides the array
evenly, and exceeds it when it does not — see [Ragged chunk grids](#ragged-chunk-grids-cost-more-than-their-arrays)
below.

The point to internalize: **raising concurrency costs *stored-chunk*-sized memory, not
*outer-chunk*-sized** — but only when the data is inner (spatially) chunked. If each outer
chunk is a single stored chunk, those two sizes are equal and concurrency gets expensive
(the "fat, single inner" regime below).

If you set `cache_budget_bytes` above the working set, residency rises to that budget on
purpose — that extra memory *is* the cross-epoch cache. Point `cache_dir` at local NVMe to
spill it to disk instead of RAM.

### Several iterations at once multiply the budget

The three bounds above describe **one** iteration. One `InSituDataset` owns one chunk pool,
and every active iteration shares it — `zip(ds.train, ds.val)`, or two `DataLoader`s over the
same dataset. That is supported, and chunks a windowed read pulls across a split boundary are
decoded once and reused by both. But each iteration holds its **own** references to the chunks
it is working on, so residency is the sum, not the maximum:

```
cache_budget_bytes  >=  n_concurrent_iterations x (block_chunks x resident_chunk_bytes)
```

**The auto-sized default is deliberately computed for one iteration, and stays that way.**
The engine cannot know how many iterations you intend to run, so any automatic multiplier
would be a guess that silently costs memory for the single-iteration case — which is almost
every case. Running several is the explicit choice, so sizing for it is yours too.

Run two without raising the budget and the loader stops with `residency budget exhausted: ...`,
naming how many iterations are sharing the pool. It cannot free a slot, because every resident
chunk is legitimately referenced by one of them. Raise `cache_budget_bytes` (or lower
`block_chunks`, which shrinks each iteration's share) — not `max_inflight`, which is a
concurrency dial and is not what is binding here.

!!! note "This got stricter once pin accounting became owner-scoped"

    Before that fix, starting a second iteration silently released the first one's
    references. That freed budget by accident and hid the requirement — and the cost was
    that the first iteration's in-use chunks could be evicted mid-gather, delivering
    plausible wrong data. A budget that appeared to work for `zip(ds.train, ds.val)` was
    relying on that bug. Sizing for the sum is the honest requirement.

### The batch queue is a high-water mark

Batch buffers are pooled and held for the dataset's lifetime, so that third bound is the
largest number of batches ever simultaneously live — not the steady state. A transient burst
that holds many at once (`list(ds.val)` to materialize a split, gradient accumulation over N
steps, an exported tensor kept past its batch) permanently raises the resident floor to that
burst's size.

It cannot raise **peak** memory. The floor is exactly what was simultaneously live at the
peak, which fresh-allocation-per-batch also held at that instant; pooling only declines to
give it back afterwards. If the burst fit, the floor fits.

`close()` is the only release, and it drops the cross-epoch chunk cache and store session with
it — so size the budget for your burst rather than planning to reclaim between phases. The
per-epoch log line reports the pool directly, and `allocated` staying nonzero after the first
epoch is the signal that batches are not coming back:

```
epoch 3 (train): chunks 151/151 hit (100%), peak resident 51;
                 batch buffers 17 x pinned = 544.0 MiB, 100 lent, 0 allocated
```

Under `pin_host_buffers` / `as_torch(..., device=...)` the floor is page-locked, which the
kernel cannot reclaim — so it is bounded separately at RAM/8 by default. Past that the loader
hands out ordinary pageable memory and warns once, rather than raising or stalling; raise
`pin_budget_bytes` if you meant to exceed it.

## Sharing a `cache_dir` between processes

**One process at a time may write a given `cache_dir`.** The loader enforces it: the pool
takes an advisory lock on the directory for its lifetime, and a second writer fails fast
with an error naming the holder rather than corrupting the first one's chunks. This applies
whenever `cache_dir` is set — with or without `persist=True` — because the two write the
same filenames.

The failure it replaces was silent. A cached chunk is an mmap'd `.npy`; a second process
re-admitting a chunk the first has mapped used to truncate the file underneath it, and the
reader carried on with right-shape, right-dtype, wrong numbers. Throughput, shapes and
smoke tests all pass. Chunk files are now replaced rather than truncated in place, so a
held mapping keeps reading real data; the lock is what keeps two writers from disagreeing
about *what* should be there in the first place.

The workload this is shaped around — one job warms a cache, several score against it — is
spelled `readonly_cache=True`:

```python
# Run once: warm the cache.
warm = InSituDataset(store, manifest, cache_dir="/data/era5-cache", persist=True)
for batch in warm.all: ...
warm.close()

# Then, concurrently, as many readers as you like.
ds = InSituDataset(store, manifest, cache_dir="/data/era5-cache", readonly_cache=True)
```

A read-only opener takes the lock *shared*: any number coexist with each other, none with a
writer. It writes nothing — no chunk files, no log entries — and **a cache miss raises**.
That is deliberate: the flag asserts *this cache is complete for what I am about to read*,
and a silent fall-back to fetching would make it a performance hint instead of a contract.
The error names the array and chunk, and the usual cause is a different split,
`sample_range` or transform set than the run that warmed it. `reset_stale_cache=True` is
rejected in this mode — a reader may not delete files another process is using.

### What happens to active readers when a writer invalidates the cache

Nothing, because the writer never starts. Invalidation — `reset_stale_cache=True` deleting
every entry, or a run re-admitting chunks it evicted — is work a *writer* does, and a writer
cannot open the directory while any reader holds it. The exclusion is at construction, before
the writer can allocate, evict or delete anything.

The layer underneath does not depend on that, though, which matters on the configurations
where the lock is not available. A reader that already holds a chunk's mapping keeps reading
real data even if the file is replaced (the new content goes to a new inode) *or* deleted
(POSIX keeps the inode alive until the last reference goes). So the worst a bypassed lock
can do to an active reader is cost it a **future** open — a miss, which `readonly_cache`
turns into a loud error — never wrong numbers in a batch that looks fine.

Both properties are pinned by tests rather than left as reasoning:
`test_a_writer_cannot_start_while_another_process_is_reading` and
`test_deleting_a_cached_file_does_not_disturb_a_held_mapping`.

### If you hit the lock

```
insitubatch: cache_dir '/data/era5-cache' is already open for writing by another
process (PID 44315 on host gpu-07, since 14:22:10).
```

To find the holder:

```bash
fuser -v /data/era5-cache/.insitu.lock
lsof /data/era5-cache/.insitu.lock
```

**Do not delete the lockfile.** It is the first thing people try and it is actively
harmful. The lock is held by the kernel against an open file description, not by the file:
deleting it releases nothing, and it makes the next two processes lock *different inodes* —
reintroducing exactly the corruption the check prevents. The PID/host/time in the file are
a **hint** for the message; the lock itself is authoritative.

There is also no such thing as a stale lock, so there is no cleanup procedure to run. The
kernel releases it when the process dies — `SIGKILL`, OOM and spot preemption included. If
you are seeing the error, that process is alive.

### Put `cache_dir` on local NVMe, not NFS

This is the mmap tier used as designed, not a new restriction. Over a network filesystem it
is slow, and — more to the point — **unarbitrated**: `flock` may be emulated per client, so
two processes on different hosts can each believe they hold the write lock. The loader
warns when it can detect one (Linux, via the mount table), but detection is not a fix. A
network `cache_dir` and a platform with no POSIX locking (Windows) are the two
configurations where two writers can still corrupt each other, and both warn saying so.

## Shuffle quality

`block_chunks` is also the shuffle-quality knob. Each batch is drawn from the samples in the
current window — `block_chunks × samples-per-chunk` of them — so set the window comfortably
larger than `batch_size`; otherwise a batch is just one or two chunks' worth of correlated
samples. A window *smaller* than a batch is not merely poor shuffle, it is unschedulable (the
batch would need more blocks resident than the residency floor holds), so the loader raises
`block_chunks` to `ceil(batch_size / samples-per-chunk)` when you ask for less and logs that
it did. On finely chunked stores — one sample per chunk, like per-frame or per-spectrum
archives — that floor is what the default 16 becomes. The chunks are re-permuted every epoch,
so even a modest window
converges toward a full-dataset shuffle over many epochs — the regime training actually runs
in. [`shuffle_quality`](api.md) scores an emitted order 0–1 (1 ≈ global) if you want to
measure it; [Architecture](architecture.md) explains why the block-local shuffle converges.

This section is training-only: only `.train` shuffles — `.val`, `.test` and `.all` are
deterministic — so a scoring pass has no shuffle quality to protect.

## Which stage is actually the bottleneck?

The knobs above only help if you turn the right one, and the symptom that sends people to
the wrong one is always the same: **the batch queue is empty**. Slow storage, a saturated
decode pool, and a residency budget too small to admit the next chunk all present that way,
and they want opposite fixes. `ds.last_pass` separates them.

```python
for batch in ds.train:
    ...

st = ds.last_pass
print(st.limiting_stage)                    # e.g. "residency"
print(st.times.admission_parked_s)          # the evidence behind it
```

Or read it from the per-epoch log line
(`logging.getLogger("insitubatch").setLevel(logging.INFO)`), which ends in
`limited by: <stage> -- <what to do>`.

### Try it: a misconfigured loader that diagnoses itself

This runs as written — the store is one of the public benchmark stores, no credentials and
no build step. `era5_c1` is the chunk-per-sample (GRIB-like) end of the family: 6000 full
`721x1440` fields, one per chunk.

**Keep the `sleep`.** It stands in for a training step, and without one the question is
degenerate: a `for batch in ds.train: pass` loop takes batches as fast as they appear, so
the queue is empty on every sample however fast the loader is. The report says so if you
drop it, but then all it can tell you is throughput.

```python
import logging
import time

from insitubatch import InSituDataset, obstore_store, open_geometries, split_by_chunk

logging.basicConfig(level=logging.INFO, format="%(message)s")

store = obstore_store(
    "gs://insitubatch-bench-insitubatch/era5_c1.zarr", skip_signature=True
)
geoms = open_geometries(store)
manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))

# max_inflight=1 is the mistake. One read at a time against an object store.
ds = InSituDataset(store, manifest, batch_size=16, block_chunks=8, max_inflight=1)

for i, batch in enumerate(ds.train):
    time.sleep(0.15)  # stand in for a training step
    if i == 9:
        break
```

```
epoch 0 (train): 10 batches in 15.3s | queue 0/2 (empty 100%, fed 0%)
  | inflight peak 1/1 | resident 32 chunks, 127 MiB of 127 MiB | consumer 1.35s
  | wait fetch 14.06s, parked 15.18s | cpu decode 1.05s, gather 0.32s
  | limited by: store -- ... the loader waited on the store. Raise max_inflight ...
    Admission also parked on the residency budget (50%), but that is most likely
    backpressure from this stage rather than a budget that is too small.
```

Two things to read. The verdict is `store`, and the parked total — nearly as large as the
fetch total — is called out as **backpressure rather than a cause**: with one read in
flight, chunks are fetched slowly and therefore sit resident longer, so the budget stays
full (`127 MiB of 127 MiB`). Raising `cache_budget_bytes` would fix none of it.

Now raise `max_inflight`, and keep raising it:

| `max_inflight` | wall for 10 batches | verdict |
|---:|---:|---|
| 1 | 15.3 s | `store` (+ residency backpressure) |
| 8 | 1.8 s | `store` |
| **16** | **1.1 s** | `store`, permits saturated |
| 32 | 1.6 s | `store`, permits saturated |

**16 is the knee, and 32 is worse.** Once every permit is in use the advice changes to say
so: *"Raise max_inflight and re-measure — but if throughput stops improving, the store or
the network is the floor and this is the honest answer rather than a knob left unturned."*
That is the truth for this run: it reads GCS cross-cloud over the public internet. Run it
in-region and this is where it stops.

Note `wait fetch` **rises** as the pass gets faster (14.06s at `max_inflight=1`, 15.86s at
16). That is the summing rule, not a contradiction: 16 tiles now wait concurrently, so the
total is task-seconds against a 1.1s wall. Read the ratios between stages, never any stage
against the clock.

| what the report shows | limiting stage | what to do |
|---|---|---|
| batch queue **fed** (`fed_frac` high) | `consumer` | nothing — the loader kept up and your training step is the constraint. This is the goal |
| queue empty, `fetch_wait_s` dominant | `store` | raise `max_inflight`; check the store is in-region and on the fast backend |
| as above, **and `inflight_peak == max_inflight`** | `store` | same, but re-measure: if throughput stops improving the network is the floor, not a knob left unturned |
| queue empty but `consumer_s` is a few % of wall | *reported, not diagnosed* | your loop has no real training step, so the queue cannot stay fed; read throughput instead |
| queue empty, `decode_s` / `assemble_s` dominant | `decode` | raise `decode_threads`; check the `chunk_transform` is vectorized numpy that releases the GIL |
| queue empty, `admission_parked_s` dominant | `residency` | raise `cache_budget_bytes`, or lower `batch_size` / `block_chunks` / concurrent iterations |
| queue empty, `gather_s` / `batch_transform_s` dominant | `gather` | check the gather run length in `describe()`, and any `batch_transform` |

`residency` is only named when *nothing else* explains the pressure. Parking is
backpressure: whatever is slow downstream holds chunks pinned until the budget fills, so a
parked total that merely tracks another stage is a symptom. When a runner-up is large enough
to explain it, the report names that stage instead and says why.

The **residency** row is the one worth knowing exists. A budget-starved loader is otherwise
indistinguishable from slow storage — it is exactly how the pre-#39 deadlock presented — and
`admission_parked_s` is the only counter that tells them apart. It is the non-terminal
neighbour of the state the loader raises `residency budget exhausted` on: same cause, caught
before it becomes provably fatal.

`limiting_stage` returns `"unknown"` rather than guessing when the evidence does not
separate the candidates — no samples, starvation too rare to matter, or no stage owning
enough of the accounted time. A confident wrong verdict costs more than an admission.

!!! note "Two clocks, on purpose"

    Waiting is measured with `perf_counter` (waiting *is* the quantity); in-thread cost is
    measured with `thread_time` (CPU actually burned). A wall clock around a thread hop
    measures GIL wait, not work — that is how the scatter memcpy once read as 51% of the hot
    path when its real share is 7.7–10.1%.

    So `fetch_wait_s` is summed across the tiles in flight and **will exceed wall time**.
    Read the stages against each other, never against the clock.

## The recipe

1. **At write time, pick `inner_chunks`** so a stored chunk is ~10–50 MB: small enough that
   many reads in flight stay cheap, large enough that per-request overhead doesn't dominate.
2. **Start with the defaults** (`max_inflight=32`, `block_chunks=16`). 32 reads in flight
   saturates in-region S3 in most cases.
3. **Size `block_chunks` to your RAM budget** (`block_chunks × resident_chunk_bytes` ≤ what
   you have — on a ragged grid that is *more* than the outer chunk's logical size). Which
   direction to push it from there is the branch below.
4. **Tune `max_inflight`** by the metric your operating point cares about — raise it until
   decoded MB/s stops climbing, *and* check TTFB, because past the IO-bound region only TTFB
   keeps responding. From the repo you can measure the knee directly:

   ```bash
   python -m bench.probe_decode --url <store> --concurrency 1,4,8,16,32
   ```

   If throughput is flat across that range you are compute-bound, and the knob is now buying
   cold start alone — see the table above before turning it down to reclaim memory.

5. **Sanity-check concurrency cost** (`max_inflight × stored_chunk_bytes`). If it's large,
   your stored chunks are too big — chunk the inner dims (step 1).

Then branch on what you are running:

**For multi-epoch training**

- Set `cache_budget_bytes` to hold the split (and `cache_dir` on NVMe to spill); epoch 0 warms
  it and later epochs read decode-once.
- Raise `block_chunks` as far as RAM allows — it is your shuffle quality.
- If RAM is tight, **`max_inflight` is the cheapest thing to give up**: measured, dropping it
  to 1 cost nothing in steady-state throughput and returned ~12% of peak RSS. You pay the cold
  start once, at the start of a run that lasts hours.

**For inference / single-pass scoring**

- **Leave `max_inflight` high.** It is the cold-start knob, worth ~11× on time to first batch,
  and the ~150 MB it costs is the cheapest latency you will ever buy. This is the one place
  not to economize.
- Keep `block_chunks` small — there is no shuffle to protect (eval views are deterministic),
  so the window only needs to hold the read plan.
- Skip the cross-epoch cache unless you score the same data more than once; a single pass
  never reads a chunk twice, so the budget is pure overhead.
- If the first batch still dominates, the remaining lever is at write time: smaller
  `inner_chunks` make the first window's reads finish sooner (step 1).

## Regimes

| regime | shape | guidance |
|---|---|---|
| **GRIB** (chunk=1) | 1 sample/chunk, single inner | concurrency follows `block_chunks`; a worker loader is competitive here single-pass (nothing to amortize), but the cross-epoch cache wins repeated passes |
| **moderate** | ~8–40 samples/chunk, single inner | the common case; `max_inflight ≈ 32`. insitu's edge grows with `sample_chunk` (each chunk is read once, not re-decoded per sample) |
| **fat, single inner** | huge outer chunk, single inner | the stored chunk *is* the outer chunk, so concurrency costs full-chunk memory. **Rechunk the inner dims**, or shrink `sample_chunk` |
| **fat, spatial** | huge outer chunk, inner grid | the sweet spot: small stored chunks make high `max_inflight` cheap; keep `block_chunks` small for low residency |

## Advanced: decode threads

`decode_threads` (on `SchedulerConfig`) sizes the pool that runs codec decode. It defaults
to auto (`min(32, cpu+4)`) and rarely needs changing; on a busy box ~8 can beat auto by
avoiding oversubscription. It is only reachable when you drive a `Scheduler` directly —
`InSituDataset` uses the auto default. There is no separate inner-fan-out cap: the inner
grid is dialed by `max_inflight`, which fetches at stored-chunk granularity.

**The decode pool is process-wide.** It is created once and outlives every scheduler, so
`decode_threads` takes effect on the **first** dataset built in the process; a later,
different value is ignored and logs a warning. That is deliberate rather than a limitation:
how many threads can usefully decode at once is a property of the *machine* — cores, memory
bandwidth — not of the dataset being read, and two datasets in one process should not run
two pools competing for the same cores. If you want a non-default size, set it on the first
dataset you create.

## Ragged chunk grids cost more than their arrays

A zarr chunk grid that does not divide the array evenly still stores **full-size chunks at
the edges**. An ERA5-shaped array of 721 latitudes chunked at 180 occupies five stored
chunks — 900 rows of storage for 721 rows of data.

The pool holds stored chunks **whole**, padding included, because the buffer unit is the
stored chunk (one unit, one shape). So resident bytes per chunk are
`n_tiles × prod(chunk_shape) × itemsize`, which can exceed the logical chunk:

| geometry | logical chunk | actually resident | ratio |
|---|---|---|---|
| 720×1440 @ 45×90 (divides evenly) | 63.3 MiB | 63.3 MiB | 1.000× |
| 2048×2048 @ 256×256 (divides evenly) | 16.0 MiB | 16.0 MiB | 1.000× |
| **721×1440 @ 180×360** | 3.96 MiB | 4.94 MiB | **1.248×** |
| **as above, short final outer chunk** | 1.98 MiB | 3.96 MiB | **1.997×** |

The automatic budget accounts for this, and so does `resident_bytes` — the number you are
told is the number you are using. What it means for you is that **a ragged grid needs a
proportionally larger `cache_budget_bytes`** if you set one by hand. If your grid divides
evenly, this section costs you nothing.

Your `chunk_transform` is unaffected: it receives the **logical** chunk, clipped, as a real
contiguous array. It never sees padding, never masks an edge, and never needs to know the
chunk grid.
