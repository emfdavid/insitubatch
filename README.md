# insitubatch

[![PyPI](https://img.shields.io/pypi/v/insitubatch.svg)](https://pypi.org/project/insitubatch/)
[![CI](https://github.com/emfdavid/insitubatch/actions/workflows/ci.yml/badge.svg)](https://github.com/emfdavid/insitubatch/actions/workflows/ci.yml)
[![docs](https://github.com/emfdavid/insitubatch/actions/workflows/docs.yml/badge.svg)](https://emfdavid.github.io/insitubatch/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Train in place on n-dimensional cloud tensors.**

`insitubatch` is the data-loader orchestration layer that sits on top of
*already-solved* async cloud IO (obstore / zarr v3 / icechunk) for PyTorch,
Jax and TensorFlow. It turns an existing Zarr archive into a shuffled,
split-aware data source built to **keep the GPU fed** — **with no reshard**
— and a Python hot path that scales with **chunks, not samples**.

It is **domain-general**: the sample axis is a *role*, not a fixed dimension. The same engine
trains on ERA5/weather over time, segments **OME-NGFF microscopy** volumes over `Z`
([example](examples/microscopy/) — raw image + label mask co-batched with no reshard), and
trains on **astronomy straight out of FITS** — Hubble frames ([example](examples/hubble/)) and
SDSS galaxy spectra ([example](examples/sdss/)) indexed as virtual byte-range references, no
pixels moved. One contract, *any* single sample axis, variables that chunk it differently.
(It also maps cleanly onto **radio-astronomy** MSv4 visibilities — a
[documented mapping](docs/architecture.md), not yet a built example.)

The SDSS reference stores are **published and readable by anyone**, so you can stream real
spectra without building anything:
[`gs://insitubatch-bench-insitubatch/astronomy/`](https://storage.googleapis.com/insitubatch-bench-insitubatch/astronomy/README.md)
(`python -m examples.sdss.train_torch --source published`).

> The IO race is over (obstore/icechunk saturate the NIC). The *loader* race is
> open. `insitubatch` builds the layer that projects like light-speed-io and
> hypergrib stopped one step short of. See [DESIGN.md](DESIGN.md).

## Why

The classic PyTorch `DataLoader` spreads work across worker **processes**, each
running a *synchronous* `__getitem__`. Against cloud Zarr that means no shared
chunk cache (every worker re-reads the same chunk), read concurrency that reaches no
further than one sample, and dask thread pools nested inside forked workers.
`insitubatch` **inverts** it: one async event loop streams stored chunks under a single
concurrency budget into a bounded pool that holds them and assembles batches on
demand — the pool doubles as the cache; torch runs `num_workers=0`.

The payoff is a **two-regime** story against the worker-process `DataLoader`. On a
**well-chunked** store it **matches a hand-tuned worker/xbatcher pool** (swept to 32 workers)
while running in **one process at bounded memory**, reaching first batch in a fraction of a
second rather than the seconds a worker pool spends starting. When the chunk layout **isn't
sample-optimized** — fat time-chunks, overlapping windows, verification grids — it pulls
**far ahead of even a tuned pool**: read planning decodes each shared chunk **once** where a
per-sample `__getitem__` re-reads it, so the win **grows with samples-per-chunk** (to ~25× at the fat end of the ERA5 sweep) and cross-epoch
caching compounds it. The **honest boundary**: at the one-sample-per-chunk (GRIB) end there is
nothing to amortize, so a tuned pool edges ahead on single-pass throughput, and against an
*unbounded* concurrent gather on large fields bounded-inflight streaming trails per byte — the
sweet spot is streaming with bounded memory, not a universal speed win. Full comparison:
[Benchmarks](https://emfdavid.github.io/insitubatch/benchmarks/).

## Status

🚧 **alpha, but validated on real cloud IO.** On an in-region S3 run
(`c6id.8xlarge`, 32 vCPU, coarsened ERA5 `361×720` fields), at fat chunks
(`sample_chunk=16`) insitubatch delivers **~19× the throughput** of a *tuned*
`xbatcher`/worker `DataLoader` baseline (swept to 32 workers), in **~8× less
memory**, and reaches its first batch **~22× sooner** (0.7 s vs 15.6 s) — the
map-style baseline re-decodes a whole chunk per sample; insitubatch reads each
chunk once. That is the *fat* end of the sweep, where the advantage is largest;
at one sample per chunk there is nothing to amortize and a tuned pool edges ahead
on single-pass throughput. Full numbers + methodology:
[the benchmarks page](https://emfdavid.github.io/insitubatch/benchmarks/).

The engine is the **decoupled fetch scheduler**: reads flatten to *stored chunks*
under one `max_inflight` budget (no nested inner/outer concurrency caps), decoded
stored chunks are adopted **by reference** into a **`ChunkPool`** that is the
residency tier *and* the cache (byte budget + pin/LRU, heap or mmap-on-NVMe); `gather`
places each one straight into its rectangle of the batch. **Read concurrency and
residency/shuffle span are independent dials** — the decoupling reaches ~1 GB/s at
flat, low memory (validated on S3; see below). Built: planner + chunk-aligned
splits, async obstore reads, the scheduler + pool (with **decode-once caching**,
cross-epoch and **cross-run** via `persist=True`), chunk/batch **transforms** (incl. a
fitted `StandardScaler`), **prefetch**, the torch / JAX / TF surfaces, and runnable
[examples](examples/); validated free-threading-correct on 3.13t. Not yet built:
`Regrid` + the **GPU/device** transform stage — see the roadmap in [DESIGN.md](DESIGN.md).

📖 **Docs:** <https://emfdavid.github.io/insitubatch/>
(see [Tuning](https://emfdavid.github.io/insitubatch/tuning/) for the
chunks↔concurrency↔memory model).

## Install

```bash
pip install insitubatch              # core engine (numpy Batch; no framework)
pip install "insitubatch[torch]"     # + torch DLPack adapter (insitubatch.frameworks)
pip install "insitubatch[jax]"       # + JAX adapter
pip install "insitubatch[tf]"        # + TensorFlow adapter
```

For development:

```bash
uv sync                  # core engine + dev tools
uv sync --extra torch    # add the torch handoff (frameworks.as_torch)
uv sync --extra jax      # add the JAX handoff (frameworks.to_jax)
uv sync --extra tf       # add the TF handoff (frameworks.as_tf_dataset)
uv sync --extra gpu      # CUDA box only: cupy + kvikio zero-copy path
```

## Tests

```bash
uv run pytest -q                              # the suite
uv run ruff check src tests bench             # lint
uv run mypy src                               # types
```

The torch-handoff tests skip unless torch is installed (`uv sync --extra torch`);
the same is enforced in CI.

> **One framework per environment.** torch, JAX and TensorFlow cannot coexist in one
> Python process — together they load duplicate OpenMP/XLA/protobuf runtimes and the
> process crashes (`SIGSEGV` / abort). Separate pytest *processes* in one env are not
> enough: TF (via its bundled Keras 3) transitively imports JAX whenever JAX is
> *installed*, so the two collide even if only the TF tests are selected. So install
> just one adapter at a time when running the framework tests:
> ```bash
> uv sync --extra torch && uv run pytest -q   # torch adapter + core
> uv sync --extra jax   && uv run pytest -q   # JAX adapter (others importorskip-skip)
> uv sync --extra tf    && uv run pytest -q   # TF adapter
> ```
> CI does exactly this — one job per framework, each with a single adapter installed,
> plus a separate lint/types job that installs every extra but runs no pytest (mypy
> doesn't import the frameworks, so co-installation is harmless there). This is a
> framework-coexistence limitation, not an insitubatch one — the core engine and each
> adapter are independent.

### Free-threaded (3.13t)

The `ChunkPool` is free-threading-correct **by construction**: a delivering thread does its
write **before** the lock — each tile is its own key, so writers never collide — and
publishes readiness **under** it, so the lock, not the GIL, is the happens-before edge
to the consuming gather. The race probe is
`test_pool_concurrent_scatter_is_race_free` (64 tiles, 32 threads).

The batch-buffer pool is *enforced* rather than structural, and the distinction is worth
knowing: it decides a buffer is free by reading `sys.getrefcount` on the buffer's owner
against a calibrated idle baseline, and a refcount read off-GIL can be stale. Reading
*high* only wastes a buffer (the pool allocates another). Reading *low* would hand live
memory to a second writer, so the pool treats a below-baseline count as impossible and
**raises** rather than lending — see `buffers.py`. That guard is what makes the mechanism
safe on 3.13t; it is not the same claim as the pool scatter's.

Run the suite GIL-free on a free-threaded interpreter:

```bash
uv python install 3.13t
# Separate env so the default .venv stays put. numcodecs has no free-threaded
# wheel yet, so it compiles from sdist (needs a C/C++ compiler: Xcode CLT on
# macOS, gcc/gcc-c++ on Linux). torch/bench have no FT wheels -> core deps only.
UV_PROJECT_ENVIRONMENT=.venv-ft uv sync --python 3.13t

# numcodecs re-enables the GIL on import (not yet declared GIL-safe), so force it
# off and confirm it took before trusting the run:
PYTHON_GIL=0 UV_PROJECT_ENVIRONMENT=.venv-ft uv run --python 3.13t \
  python -c "import sys, zarr, numcodecs; assert not sys._is_gil_enabled(); print('GIL-free OK')"
PYTHON_GIL=0 UV_PROJECT_ENVIRONMENT=.venv-ft uv run --python 3.13t pytest -q
```

CI mirrors this: a `{3.12, 3.13}` matrix plus a `3.13t` job that asserts the GIL is
actually off before testing. Throughput is **GIL-independent by design** — fetch
(obstore/Rust), decode (numcodecs zstd, C), and gather (vectorized numpy) all
release the GIL — so 3.13t runs at the **same speed** as the GIL build, not faster. The
free-threading work is **correctness + future-proofing, not a speedup**; *not depending*
on the GIL is the point (see [DESIGN.md](DESIGN.md)).

## Shape of the API

The core `InSituDataset` is a **framework-neutral source of numpy `Batch` objects** — it
inherits nothing framework-specific. You iterate its split *views* (`ds.train` shuffled,
`ds.val` / `ds.test` / `ds.all` deterministic), which all share **one** pool, so a chunk
two splits both read decodes once. Handoff to torch / JAX / TF is a thin, optional DLPack
adapter (re-exported from the package root; defined in `insitubatch.frameworks`) — the core
imports no framework, and importing `insitubatch` pulls none in.

```python
from insitubatch import InSituDataset, obstore_store, open_geometries, split_by_chunk

# The engine reads a zarr Store; build one per backend. obstore_store covers
# file://, s3://, gs://, az://. (fsspec_store reaches GCS Rapid/requester-pays;
# arraylake_store opens an Icechunk session — same InSituDataset below.)
store = obstore_store("gs://insitubatch-bench-insitubatch/era5_c16.zarr", skip_signature=True)
geoms = open_geometries(store)  # {var: ArrayGeometry} from zarr metadata
# contiguous chunk blocks by default (no time-series leakage);
# pass contiguous=False for exchangeable samples (independent scenes)
manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))

ds = InSituDataset(store, manifest, batch_size=32, block_chunks=16)

for epoch in range(n_epochs):
    ds.set_epoch(epoch)
    for batch in ds.train:  # numpy Batch: {var: np.ndarray} + sample_indices
        ...
    for batch in ds.val:  # deterministic; shares the pool with train
        ...
```

That store is one of the public benchmark stores, readable by anyone
(`gs://insitubatch-bench-insitubatch/era5_c{1,2,4,8,16,32}.zarr` — the same
`era5_c*` chunk-size family the [benchmarks](https://emfdavid.github.io/insitubatch/benchmarks/)
sweep, full-resolution `721×1440` ERA5-shaped `t2m`, differing only in samples per chunk).

### What it will cost, before it costs it

`ds.print_summary()` answers from **geometry and configuration alone** — it opens no store,
fetches nothing, and runs no pass. That is what makes it usable in the situation it exists
for: finding out that a configuration needs 5 GiB, or that concurrency and memory are welded
together by the chunk layout, *before* waiting an hour to discover it.

```console
>>> ds.print_summary()
insitubatch dataset

variables
  t2m
    6000x721x1440 float32   chunks 16x721x1440   sample axis 0
    375 chunks of 16 sample(s)   field 721x1440 in 1 stored chunk(s) of 721x1440
    stored chunk 63.4 MiB   resident per chunk 63.4 MiB   gather run 4.0 MiB

configuration (resolved)
  batch_size 32   block_chunks 16   max_inflight 32   prefetch_depth 2
  shuffle on (seed 0, quality 0.96)   window no   shuffle pool 256 samples for a 32-sample batch
  splits (chunks)  train 300  val 38  test 37
  cache backing heap

memory, accounted, for 1 concurrent iteration(s)
  residency       1.98 GiB   budget (automatic: the working-set floor)
  in flight       1.98 GiB   max_inflight x stored chunk
  batch queue    380.2 MiB   (prefetch_depth + 1) x batch
  accounted       4.33 GiB   sum of the rows above
  ESTIMATED       5.41 GiB   accounted x 1.25 -- plan for this
  ...

notes
  [t2m] one stored chunk per outer chunk at 63.4 MiB: concurrency and memory are coupled here,
      so each of max_inflight=32 slots costs a whole chunk. Inner-chunk the field to separate
      them.
```

The **notes** section is the part that earns the call. Here it caught the *fat, single-inner*
regime: because the whole `721×1440` field is one stored chunk, every one of the 32 reads in
flight costs a full 63.4 MiB — so `max_inflight` and memory are welded together, and the fix
is at write time (inner-chunk the field), not in the loader config. `describe()` returns the
same report as data if you would rather assert on it in CI than read it, and
`describe(iterations=N)` sizes the budget for `N` passes sharing the pool.

Hand off to a framework — **zero-copy on CPU via DLPack for torch and JAX**; TF takes one
CPU copy (its experimental DLPack is unreliable — see `frameworks.to_tf`). The ecosystems
differ — torch needs a `Dataset` subclass, JAX iterates directly, TF wraps via
`from_generator`:

```python
from insitubatch import as_tf_dataset, as_torch, to_jax
from torch.utils.data import DataLoader

# torch: parallelism is in our event loop, so num_workers=0, batch_size=None
loader = DataLoader(as_torch(ds.train), batch_size=None, num_workers=0)  # {var: torch.Tensor}

for batch in ds.train:      # JAX: iterate a view, convert each batch
    jbatch = to_jax(batch)  # {var: jax.Array}

tfds = as_tf_dataset(ds.val)  # a tf.data.Dataset
```

## Transforms — and checking one before you train

Two hooks, placed by cost: a **`chunk_transform`** `(DecodedChunk) -> DecodedChunk` runs per
decoded chunk (one variable), before the cache boundary, so its output is **cached** — the home
for scaling, unit conversion, dtype cast, regrid; and a **`batch_transform`** `(Batch) -> Batch`
runs per assembled batch (all variables aligned), **uncached** — for cross-variable derived
fields and per-sample random augmentation. Both are pure numpy; see
[`examples/transforms.py`](examples/transforms.py) (K→C chunk stage + windspeed batch stage).

A `chunk_transform` must be **vectorized numpy that releases the GIL** (a per-element Python
loop serializes the decode pool), and a **reshaping** one (regrid) must declare
`output_inner(geom) -> (inner_shape, dtype)` so the cache can size its slot. Check both against
**one chunk of your real store** before training:

```console
$ insitubatch-check-transform \
    gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr \
    --var 2m_temperature --transform examples/transforms.py:kelvin_to_celsius --skip-signature

  sample axis : 92040 samples, 40/chunk, 2301 chunks
  chunk 0    : 40 samples -> source shape (40, 128, 64) = 1.3 MB decoded
transform output:
  (40, 128, 64) float32  ->  (40, 128, 64) float32   shape- and dtype-preserving
cacheability: shape/dtype-preserving, no output_inner needed -> cacheable as-is.
GIL-release probe (thread-scaling, 4 threads):
  speedup 3.50x (>= 2.40) -> releases the GIL (vectorized).
PASS: chunk_transform checks all passed.
```

The target is `module:attr` or `path/to/file.py:attr` (a transform class is instantiated).
It reports the chunk geometry, validates a declared `output_inner` against the real output
(catching the mismatch the cache would later reject), and gives a GIL-release verdict — a
non-zero exit gates a pre-commit hook. Pass `--no-gil-probe` for a fast structural-only check;
the GIL probe needs a realistically-sized chunk (a toy array is dominated by call overhead). For
the **reshaping** path, try `--transform examples/transforms.py:Coarsen` — a chunk-local regrid
that halves the grid and declares `output_inner`, so the report shows the validated shape change.

## Contributing

insitubatch is maintained by one person today, and that is a transitional state rather than the
intent — the project is being built to be worked on by several people, with the governance
written down *before* it is strictly needed so that the next maintainers have a framework to
build on. Bug reports, performance reports, docs and new-domain examples are all
real contributions, and the path into the core developer group is merit-based and public.

Start with the [contributing guide](https://emfdavid.github.io/insitubatch/contributing/): it
covers the dev setup, the scope limits that decide whether a change can land, the
one-framework-per-environment test caveat, what a performance claim has to carry, and the
policy on AI-assisted contributions. Then [GOVERNANCE.md](GOVERNANCE.md) for how decisions get
made, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), which applies to maintainers first.

For anything larger than a bug fix, please
[open an issue](https://github.com/emfdavid/insitubatch/issues/new/choose) first.

## License

MIT — see [LICENSE](LICENSE).
