# insitubatch

[![PyPI](https://img.shields.io/pypi/v/insitubatch.svg)](https://pypi.org/project/insitubatch/)
[![CI](https://github.com/emfdavid/insitubatch/actions/workflows/ci.yml/badge.svg)](https://github.com/emfdavid/insitubatch/actions/workflows/ci.yml)
[![docs](https://github.com/emfdavid/insitubatch/actions/workflows/docs.yml/badge.svg)](https://emfdavid.github.io/insitubatch/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Train in place on n-dimensional cloud tensors.**

`insitubatch` is a data-loader orchestration layer for PyTorch, JAX and TensorFlow, built on
top of *already-solved* async cloud IO (obstore / zarr v3 / icechunk).

Between a dataset small enough to hold in memory and one worth building a purpose-built ETL
pipeline for, there is a wide middle: archives too large to load, not yours to rewrite, or
read in ways that keep changing.

`insitubatch` is a loader for that middle: it turns a Zarr archive — in the layout it already
has, wherever it already lives — into a shuffled, split-aware stream of ready-to-train
tensors, without copying or rewriting the archive.

The design rests on two invariants:

- **A stored chunk is fetched and decoded exactly once**, however many samples, batches or
  epochs reference it — one async event loop streams chunks under a single concurrency budget
  into a bounded pool that is both the residency tier and the cache.
- **Python work scales with the chunks a batch touches, not the samples it contains** —
  planning and assembly are vectorized numpy, and the torch surface runs `num_workers=0`.

The sample axis is a **role**, not a fixed dimension, so one engine covers ERA5 forecasting,
OME-NGFF microscopy segmentation over `Z`, and Hubble and SDSS archives read straight out of
FITS as virtual byte-range references — see
[Examples](https://emfdavid.github.io/insitubatch/examples/).

What it asks of a store is **many chunks along the sample axis** — splits and shuffling are
both chunk-granular, so chunks are the unit of each. A sample must also sit inside one chunk
on that axis; it may span every other axis freely.

**Status: Beta** — feature-complete for the documented scope and validated on real cloud IO
(S3, GCS, and an L4 GPU). Pre-1.0, so breaking changes are still allowed. Open work is tracked
in [issues](https://github.com/emfdavid/insitubatch/issues), and the
[contributing guide](https://emfdavid.github.io/insitubatch/contributing/) covers scope,
governance, and what a change has to carry to land.

📖 **Docs:** <https://emfdavid.github.io/insitubatch/> —
[Tuning](https://emfdavid.github.io/insitubatch/tuning/) for the chunks↔concurrency↔memory
model, and [Architecture](https://emfdavid.github.io/insitubatch/architecture/) for how the
pieces fit together.

## Install

```bash
pip install insitubatch                  # core engine (numpy Batch; no framework)
pip install "insitubatch[torch]"         # + torch DLPack adapter
pip install "insitubatch[jax]"           # + JAX adapter (CPU wheel)
pip install "insitubatch[jax-cuda]"      # + JAX adapter on CUDA (jax[cuda12])
pip install "insitubatch[tf]"            # + TensorFlow adapter
pip install "insitubatch[cache]"         # stronger cross-run cache invalidation -- see Caching
pip install "insitubatch[icechunk]"      # icechunk_store()
pip install "insitubatch[arraylake]"     # arraylake_store()
pip install "insitubatch[gpu]"           # CUDA box only: cupy + kvikio
```

Install **one** framework adapter per environment: torch, JAX and TensorFlow load duplicate
OpenMP/XLA/protobuf runtimes and crash when co-installed (TF pulls JAX in via Keras 3). The
core engine imports no framework, and importing `insitubatch` pulls none in.

## Quickstart

`InSituDataset` is a **framework-neutral source of numpy `Batch` objects**. You iterate its
split *views* — `ds.train` shuffled, `ds.val` / `ds.test` / `ds.all` deterministic — which
all share **one** pool, so a chunk two splits both read is decoded once.

**This runs as written** — the store is public, and the window is deliberately small, about
2 GB of reads:

```python
import logging
from insitubatch import InSituDataset, obstore_store, open_geometries, split_by_chunk

logging.basicConfig(level=logging.INFO)  # the per-epoch line is the main diagnostic
n_epochs = 2

# One Store per backend, all read by the same InSituDataset: obstore_store (file://, s3://,
# gs://, az://), icechunk_store (an Icechunk repo by URL), fsspec_store (GCS Rapid,
# requester-pays), arraylake_store (Arraylake-hosted). This one is a public benchmark store:
# full-resolution 721x1440 ERA5-shaped t2m, where era5_c{1,2,4,8,16,32} differ only in how
# many samples one chunk holds (values are random numbers - not real data!).
store = obstore_store("gs://insitubatch-bench-insitubatch/era5_c16.zarr", skip_signature=True)

# `variables=` SELECTS. Omit it and you open every array in the store -- on a 25-variable
# archive that is a 25-variable run and a residency budget to match.
geoms = open_geometries(store, variables=["t2m"])

# The manifest carries the SPLIT, not the selection. Contiguous chunk blocks by default
# (no time-series leakage); pass contiguous=False for exchangeable samples. `sample_range`
# restricts to a slice of the sample axis -- the way to try a large store cheaply. Widen it
# (or drop it) for real work; 640 samples is 40 chunks, of which train gets 32.
manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1), sample_range=(0, 640))

# Passing `geometries=` is what scopes the run to those variables.
ds = InSituDataset(store, manifest, geometries=geoms, batch_size=32, block_chunks=16)

for epoch in range(n_epochs):
    ds.set_epoch(epoch)
    for batch in ds.train:  # numpy Batch: {var: np.ndarray} + sample_indices
        ...
    for batch in ds.val:    # deterministic; shares the pool with train
        ...
```

The second epoch's log line reports a **cache hit rate** against the first: that is
decode-once across epochs, from the pool alone, with no `cache_dir` configured.

Two behaviours of that loop are worth knowing before you widen it. Splits are
**chunk-granular** — you subset whole chunks, never individual samples — so a `sample_range`
landing mid-chunk pulls that edge chunk in whole, and a split that rounds to zero chunks yields
nothing rather than raising (`print_summary()` will show it, and it is worth checking if a
validation curve comes out suspiciously flat). And shuffling is **block-local** rather than
global: samples are drawn from a window of `block_chunks` chunks, which is what keeps residency
bounded, so widening the window costs memory while `describe()["shuffle_quality"]` scores how
close the result gets to a global shuffle.

### Handing off to a framework

The `Batch` above is numpy, and the adapter to each framework is thin: **zero-copy on CPU via
DLPack** for torch and JAX, while TF takes one CPU copy (its experimental DLPack is
unreliable).

Device placement differs by framework, and the difference is one sentence each. **torch**
takes a `device=` on `as_torch` / `to_torch`, which also page-locks the batch buffers and
issues the copy itself. **JAX** lands on `jax.devices()[0]`, like any other jax array, with
`device=` to override. **TF** places by its own policy — on a GPU box, `/GPU:0`. Page-locking
is torch-only: JAX cannot consume pinned host memory and exposes no event to gate buffer
recycling on, so `to_torch(device=)` is the only path that owns its transfer.

Pick the one line for your framework — these are **alternatives, not a script**, since torch,
JAX and TensorFlow cannot share a process:

```python
# torch -- parallelism is in our event loop, so num_workers=0, batch_size=None
from insitubatch import as_torch
from torch.utils.data import DataLoader

loader = DataLoader(as_torch(ds.train, device="cuda"), batch_size=None, num_workers=0)
for batch in loader:  # {var: torch.Tensor} on the GPU; omit device= to stay on CPU
    ...
```

```python
# JAX -- iterate a view and convert each batch
from insitubatch import to_jax

for batch in ds.train:
    jbatch = to_jax(batch)  # {var: jax.Array} on jax.devices()[0]
```

```python
# TensorFlow -- wraps a view via from_generator
from insitubatch import as_tf_dataset

tfds = as_tf_dataset(ds.val)  # a tf.data.Dataset
```

## What it will cost — before, and after

`ds.print_summary()` answers from **geometry and configuration alone** — it opens no store,
fetches nothing and runs no pass. This is the real output for the `ds` built above. That is
the point: a report that had to touch the store
would be useless in exactly the situation you want it.

```console
>>> ds.print_summary()
variables
  t2m
    6000x721x1440 float32   chunks 16x721x1440   sample axis 0
    375 chunks of 16 sample(s)   field 721x1440 in 1 stored chunk(s) of 721x1440
    stored chunk 63.4 MiB   resident per chunk 63.4 MiB   gather run 4.0 MiB

configuration (resolved)
  batch_size 32   block_chunks 16   max_inflight 32   prefetch_depth 2
  shuffle on (seed 0, quality 0.99)   window no   shuffle pool 256 samples for a 32-sample batch
  splits (chunks)  train 32  val 4  test 4
  cache backing heap

memory, accounted, for 1 concurrent iteration(s)
  residency       1.98 GiB   budget (automatic: the working-set floor)
  in flight       1.98 GiB   max_inflight x stored chunk
  batch queue    380.2 MiB   (prefetch_depth + 1) x batch
  accounted       4.33 GiB   sum of the rows above
  ESTIMATED       5.41 GiB   accounted x 1.25 -- plan for this
  [... two paragraphs on allocator retention, and on sizing concurrent iterations ...]

notes
  [t2m] one stored chunk per outer chunk at 63.4 MiB: concurrency and memory are coupled here,
      so each of max_inflight=32 slots costs a whole chunk. Inner-chunk the field to separate
      them.
```

The **notes** earn the call: here every one of the 32 reads in flight costs a full 63.4 MiB,
so `max_inflight` and memory are welded together by the layout, not by the config. The unit
throughout is the **stored chunk** — what one `store.get` returns, which on a **sharded**
array is the *shard*, not zarr's inner `chunks`; `ArrayGeometry.chunk_bytes` reports what one
really costs. The [memory model](https://emfdavid.github.io/insitubatch/tuning/) has the
arithmetic, including what an allocating `chunk_transform` adds.

`describe()` returns the same report as data; `describe(iterations=N)` sizes the budget for
`N` passes sharing the pool, since the auto budget covers **one** — so `zip(ds.train,
ds.val)` has to say so.

Where `describe()` predicts, **`ds.last_pass` reports**. Every producer-side problem presents
identically, as an empty batch queue, and slow storage / a saturated decode pool / too small a
residency budget want opposite fixes — so `ds.last_pass.limiting_stage` applies a rule to the
sampled depths and per-stage times and names *one* stage, or `"unknown"` rather than guessing.
With logging on, the per-epoch INFO line ends in `limited by: <stage> -- <what to do>` and
reports re-reads and evictions whenever they are non-zero: rising re-reads at a steady hit rate
mean a budget that is churning rather than holding. For bug reports,
`insitubatch.print_debug_info()` prints the storage stack, the installed framework adapter and
the free-threading state.

## Transforms

Two hooks, placed by cost. A **`chunk_transform`** `(DecodedChunk) -> DecodedChunk` runs per
decoded chunk of one variable *before* the cache boundary, so its output is **cached** — the
home for scaling, unit conversion, dtype casts and regrids. A **`batch_transform`**
`(Batch) -> Batch` runs per assembled batch with all variables aligned, **uncached** — for
cross-variable derived fields and per-sample random augmentation. Both take a **sequence**,
and both must be vectorized numpy that releases the GIL (a per-element Python loop serializes
the decode pool).

```python
from insitubatch import applies

ds = InSituDataset(
    store, manifest, geoms,
    chunk_transforms=[applies(["2m_temperature"], kelvin_to_celsius)],
    batch_transforms=[wind_speed],
)
```

**Scope a chunk transform with `applies(...)`, not with an `if` inside its body.** An in-body
test is invisible to the engine, which folds a reshaping transform's declared `output_inner`
into *every* array's geometry; the unaffected arrays are then gathered as truncated prefixes
of themselves and can never revive from cache — **with no exception raised**. A bare
transform applies to everything, and a name matching no array raises at construction.

A **reshaping** transform (regrid) must declare `output_inner(geom) -> (inner_shape, dtype)`
so the cache can size its slot; returning a shape that disagrees with it now raises. Check
both against **one chunk of your real store** before training:

```bash
insitubatch-check-transform \
    "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr" \
    --var 2m_temperature \
    --transform examples/transforms.py:kelvin_to_celsius --skip-signature
```

It reports the chunk geometry, validates a declared `output_inner` against the real output,
and gives a GIL-release verdict. It exits non-zero on failure, so you can gate your own
pre-commit hook or CI step on it. Worked examples:
[`examples/transforms.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/transforms.py).

## Caching

The pool is already a cache **within** a run and **across epochs**. `cache_dir` spills chunks
to disk, and `persist=True` keeps them for **later runs** — on a windowed shuffled pass it
also releases each block's chunks as it drains and re-reads them from the cache, taking
residency from the whole split to a three-block floor.

```python
ds = InSituDataset(store, manifest, geoms, cache_dir="/mnt/nvme/cache", persist=True)
```

What a reader has to know before pointing two jobs at one cache:

- **One writer per `cache_dir`.** The pool holds an advisory `flock` for its lifetime, so a
  second writer **fails fast** naming the holder's PID and host. There is no such thing as a
  stale lock — the kernel releases it when the process dies, `SIGKILL` included — so there is
  no cleanup step, and **deleting the lockfile is actively harmful**: it releases nothing and
  makes the next two processes lock different inodes.
- **`readonly_cache=True` for readers.** Any number coexist with the writer, they write
  nothing, and **a cache miss raises** — which is what makes it a contract ("this cache is
  complete for what I am about to read") rather than a slow path that silently refetches.
- **Editing a chunk transform invalidates only the arrays it is scoped to.** The error names
  them and `reset_stale_cache=True` rebuilds just those; an array with no entries is *cold*,
  not stale, so adding a variable is not a cache reset.
- **Install `insitubatch[cache]` if a transform is parameterized by data.** Without
  cloudpickle the fingerprint hashes a transform's source plus its `repr`, and numpy
  summarizes any array over 1000 elements — so a re-fitted `StandardScaler` can reopen a cache
  as a **hit** and serve the previous statistics, with no error. It also takes a `cache_key`
  you bump per fit.
- A network `cache_dir`, or a platform without POSIX locking, is **unarbitrated** — both warn
  at construction.

## How it works, and where it wins

The classic PyTorch `DataLoader` spreads work across worker **processes**, each running a
synchronous `__getitem__` — so against cloud zarr there is no shared chunk cache (every worker
re-reads the same chunk), read concurrency reaches no further than one sample, and dask thread
pools end up nested inside forked workers. `insitubatch` inverts it: one async event loop, one
concurrency budget, one bounded pool that doubles as the cache.

The payoff is **two-regime**. On a well-chunked store it *matches* a hand-tuned
xbatcher/worker pool (swept to 32 workers) while running in one process at bounded memory.
Where the layout is not sample-optimized — fat time chunks, overlapping windows, verification
grids — it pulls far ahead, because each shared chunk decodes once where a per-sample
`__getitem__` re-reads it, and the advantage grows with samples per chunk.

**Time to first batch is the exception to the two regimes.** There is no worker pool to
start, no fork, and no store to reopen per worker, so insitubatch is serving batches while a
worker stack is still coming up — and it wins cold start at *every* chunk size measured,
including the one where it loses on throughput. That is what makes it usable for inference,
where a service cannot keep a hot 32-worker pool alive between requests.

The **honest boundary** is three things. Shuffling is block-local, not global. At one sample
per chunk there is nothing to amortize, so a tuned worker pool edges ahead on single-pass
throughput. And **if the array fits in memory, load it** — reading it whole and indexing it in
RAM beats any streaming loader, this one included; the benchmark suite measures exactly that
and calls it the ceiling. What insitubatch buys is bounded memory, not raw speed. Numbers,
hardware and methodology: [Benchmarks](https://emfdavid.github.io/insitubatch/benchmarks/).

### Platform support

| platform | status |
|---|---|
| **Linux** | supported, and the only thing CI proves — every job is `ubuntu-latest` |
| **macOS** | expected to work; **untested** — no CI job, no regular local runs |
| **Windows** | **untested**, and unarbitrated: no POSIX advisory locking, so two processes sharing a `cache_dir` cannot be stopped from corrupting each other. The loader warns at construction |

*Untested* is not *unsupported*: there is no evidence either way. Reports — or a CI job — are
welcome.

The engine is validated **free-threading-correct** on 3.13t, which is a correctness and
future-proofing claim, not a speedup: fetch, decode and gather all release the GIL already,
so 3.13t runs at the same speed as the GIL build.

## Contributing

insitubatch is maintained by one person today, and that is a transitional state rather than
the intent — the governance is written down *before* it is strictly needed. Bug reports,
performance reports, docs and new-domain examples are all real contributions, and the path
into the core developer group is merit-based and public.

Start with the [contributing guide](https://emfdavid.github.io/insitubatch/contributing/):
dev setup, the five commands CI enforces, the scope limits that decide whether a change can
land, and the policy on AI-assisted contributions. Then
[GOVERNANCE.md](https://github.com/emfdavid/insitubatch/blob/main/GOVERNANCE.md) and
[CODE_OF_CONDUCT.md](https://github.com/emfdavid/insitubatch/blob/main/CODE_OF_CONDUCT.md).
For anything larger than a bug fix, please
[open an issue](https://github.com/emfdavid/insitubatch/issues/new/choose) first.

## License

MIT — see [LICENSE](https://github.com/emfdavid/insitubatch/blob/main/LICENSE).
