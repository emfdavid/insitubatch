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

> The IO race is over (obstore/icechunk saturate the NIC). The *loader* race is
> open. `insitubatch` builds the layer that projects like light-speed-io and
> hypergrib stopped one step short of. See [DESIGN.md](DESIGN.md).

## Why

The classic PyTorch `DataLoader` spreads work across worker **processes**, each
running a *synchronous* `__getitem__`. Against cloud Zarr that means no shared
chunk cache (every worker re-reads the same chunk), no way to drive async
obstore, and dask thread pools nested inside forked workers. `insitubatch`
**inverts** it: one async event loop streams stored chunks under a single
concurrency budget and scatters them into a bounded pool that assembles batches —
the pool doubles as the cache; torch runs `num_workers=0`.

The payoff is being a **batteries-included** choice at both operating points: for
**inference** it pays no worker-pool cold start (first batch in ~ms, not seconds — keeping a
production hot pool alive is its own challenge); for **training** it uses far less memory
(one process, not 32), reads each chunk once, and caches across epochs. Across the chunk
spectrum it leads a *tuned* worker/xbatcher baseline; xbatcher stays ahead on single-pass
throughput at the GRIB end (one sample per chunk, where there is nothing to amortize), and
there the cross-epoch cache flips multi-epoch training back to insitu. Full comparison:
[Benchmarks](https://emfdavid.github.io/insitubatch/benchmarks/).

## Status

🚧 **alpha, but validated on real cloud IO.** On an in-region S3 run
(`c6id.8xlarge`, ERA5-shaped `721×1440` fields, `sample_chunk=8`), insitubatch
delivers **~8× the throughput** of a *tuned* `xbatcher`/worker `DataLoader`
baseline (swept to 32 workers) and reaches its first batch **~10× sooner** — the
map-style baseline re-decodes a whole chunk per sample; insitubatch reads each
chunk once. Full numbers + methodology:
[the benchmarks page](https://emfdavid.github.io/insitubatch/benchmarks/).

The engine is the **decoupled fetch scheduler**: reads flatten to *stored chunks*
under one `max_inflight` budget (no nested inner/outer concurrency caps), decoded
tiles scatter into a **`ChunkPool`** that is the assembly buffer *and* the cache
(byte budget + pin/LRU, heap or mmap-on-NVMe). **Read concurrency and
residency/shuffle span are independent dials** — the decoupling reaches ~1 GB/s at
flat, low memory (validated on S3; see below). Built: planner + chunk-aligned
splits, async obstore reads, the scheduler + pool (with **cross-epoch decode-once
caching**), chunk/batch **transforms** (incl. a fitted `StandardScaler`),
**prefetch**, the torch / JAX / TF surfaces, and runnable [examples](examples/);
validated free-threading-correct on 3.13t. Not yet built: `Regrid` + the **GPU/device**
transform stage, and cross-*run* cache persistence — see the roadmap in [DESIGN.md](DESIGN.md).

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

The engine is free-threading-correct by construction: the `ChunkPool`'s scatter
does its disjoint copy **before** the lock and publishes readiness **under** it, so
the lock — not the GIL — is the happens-before edge to the consuming gather. The
race probe is `test_pool_concurrent_scatter_is_race_free` (64 tiles, 32 threads).

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
(obstore/Rust), decode (numcodecs zstd, C), and scatter/gather (vectorized numpy) all
release the GIL — so 3.13t runs at the **same speed** as the GIL build, not faster. The
free-threading work is **correctness + future-proofing, not a speedup**; *not depending*
on the GIL is the point (see [DESIGN.md](DESIGN.md)).

## Shape of the API

The core `InSituDataset` is a **framework-neutral source of numpy `Batch` objects** — it
inherits nothing framework-specific. You iterate its split *views* (`ds.train` shuffled,
`ds.val` / `ds.test` / `ds.all` deterministic), which all share **one** pool, so a chunk
two splits both read decodes once. Handoff to torch / JAX / TF is a thin, optional DLPack
adapter layer in `insitubatch.frameworks`; the core imports no framework.

```python
from insitubatch import open_geometries, split_by_chunk
from insitubatch.source import InSituDataset

url = "file:///data/era5.zarr"           # or "s3://bucket/era5.zarr" — same code
geoms = open_geometries(url)             # {var: ArrayGeometry} from zarr metadata
manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))

ds = InSituDataset(url, manifest, batch_size=32, block_chunks=16)

for epoch in range(n_epochs):
    ds.set_epoch(epoch)
    for batch in ds.train:               # numpy Batch: {var: np.ndarray} + sample_indices
        ...
    for batch in ds.val:                 # deterministic; shares the pool with train
        ...
```

Hand off to a framework — **zero-copy on CPU via DLPack for torch and JAX**; TF takes one
CPU copy (its experimental DLPack is unreliable — see `frameworks.to_tf`). The ecosystems
differ — torch needs a `Dataset` subclass, JAX iterates directly, TF wraps via
`from_generator`:

```python
from insitubatch.frameworks import as_torch, to_jax, as_tf_dataset
from torch.utils.data import DataLoader

# torch: parallelism is in our event loop, so num_workers=0, batch_size=None
loader = DataLoader(as_torch(ds.train), batch_size=None, num_workers=0)  # {var: torch.Tensor}

for batch in ds.train:      # JAX: iterate a view, convert each batch
    jbatch = to_jax(batch)  # {var: jax.Array}

tfds = as_tf_dataset(ds.val)  # a tf.data.Dataset
```

## License

MIT — see [LICENSE](LICENSE).
