# insitubatch

**Train in place on n-dimensional cloud tensors.**

`insitubatch` is the data-loader orchestration layer that sits on top of
*already-solved* async cloud IO (obstore / zarr v3 / icechunk) for PyTorch, Jax
and TensorFlow. It turns an existing Zarr archive into a shuffled, split-aware
data source built to **keep the GPU fed** — **with no reshard** — and a Python
hot path that scales with **chunks, not samples**.

It is **domain-general**: the sample axis is a *role*, not a fixed dimension. The same engine
trains on ERA5/weather over time, segments **OME-NGFF microscopy** volumes over `Z`
([runnable example](https://github.com/emfdavid/insitubatch/blob/main/examples/microscopy) —
raw image + label mask co-batched with no reshard), and maps cleanly onto **radio-astronomy**
visibilities. See the [use-case tables](architecture.md#use-case-support).

!!! quote
    The IO race is over (obstore/icechunk saturate the NIC). The *loader* race is
    open. `insitubatch` builds the layer that projects like light-speed-io and
    hypergrib stopped one step short of.

**Where it wins.** On a well-chunked store it **matches a hand-tuned worker `DataLoader`**
(swept to its best worker count) **at a fraction of the memory** — one process, bounded
residency, ~ms to first batch instead of seconds of pool cold-start. When the chunk layout
**isn't sample-optimized** — fat time-chunks, overlapping windows, verification grids — it pulls
**far ahead of even a tuned worker pool**, because read planning decodes each shared chunk once
where per-sample workers re-read it (the win grows with samples-per-chunk). It is **not** a
universal speed win: at the one-sample-per-chunk (GRIB) end, or against an unbounded gather on
large fields, a tuned pool can edge ahead per byte. Numbers: [Benchmarks](benchmarks.md).

## The problem, and the inversion

The classic PyTorch `DataLoader` puts parallelism in worker **processes**, each running a
*synchronous* `__getitem__`. Against cloud Zarr that fights itself: no shared chunk cache
(every worker re-reads the same chunk), no way to drive async obstore, and dask thread
pools nested inside forked workers. The usual escape — **resharding** to one-sample-per-file
— is a second copy of the dataset that throws away the chunk locality the store already has.

`insitubatch` keeps the data in place and **inverts the loader**:

> Classic `DataLoader`: parallelism lives in **`num_workers` OS processes**, each running a
> synchronous `__getitem__`. insitubatch: parallelism lives in **one async event loop**;
> batch assembly is the consumer.

That single move unlocks async obstore, a **shared chunk cache**, bounded memory, and
**prefetch overlap** with the training step; torch runs `num_workers=0`.
[Architecture](architecture.md) has the full frictions breakdown, the loader/prefetch
diagrams, and the read-plan abstraction;
[DESIGN.md](https://github.com/emfdavid/insitubatch/blob/main/DESIGN.md) has the why.

## Shape of the API

The core `InSituDataset` is a framework-neutral iterable of numpy `Batch` objects;
torch / JAX / TF handoff is a thin optional DLPack adapter in `insitubatch.frameworks`.

```python
from insitubatch import (
    InSituDataset,
    as_tf_dataset,
    as_torch,
    obstore_store,
    open_geometries,
    split_by_chunk,
    to_jax,
)
from torch.utils.data import DataLoader

# The engine reads a zarr Store; obstore_store builds one for file://, s3://, gs://.
# (fsspec_store for GCS Rapid/requester-pays; arraylake_store for Icechunk sessions.)
store = obstore_store("file:///data/era5.zarr")  # or "s3://bucket/era5.zarr"
geoms = open_geometries(store)           # {var: ArrayGeometry} from zarr metadata
manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))

ds = InSituDataset(store, manifest, batch_size=32, block_chunks=16)

for epoch in range(n_epochs):
    ds.set_epoch(epoch)
    for batch in ds.train:               # numpy Batch: {var: np.ndarray} + sample_indices
        ...
    for batch in ds.val:                 # deterministic; shares the pool with train
        ...

# Framework handoff (DLPack, zero-copy on CPU for torch/JAX; TF copies once):
loader = DataLoader(as_torch(ds.train), batch_size=None, num_workers=0)  # torch
jbatch = to_jax(next(iter(ds.train)))                                    # JAX:   {var: jax.Array}
tfds = as_tf_dataset(ds.val)                                             # TF:    tf.data.Dataset
```

See [`examples/advection`](https://github.com/emfdavid/insitubatch/blob/main/examples/advection) for
working CNN forecast models using insituBatch implemented with Torch, Jax and Tensorflow with
real ERA5 data.

A runnable, network-free version of this — paralleling the Earthmover
`dataloader-demo`, with a spatial subregion pulled out by a `batch_transform` —
lives in
[`examples/wb2_dataloader.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/wb2_dataloader.py):

```bash
uv run python -m examples.wb2_dataloader            # tiny synthetic data, no network
uv run python -m examples.wb2_dataloader \
    --url s3://bucket/era5.zarr --var 2m_temperature --subregion 48,48 --request-payer
```

## Install (dev)

```bash
uv sync                  # core engine + dev tools
uv sync --extra torch    # torch handoff (frameworks.as_torch)
uv sync --extra jax      # JAX handoff (frameworks.to_jax)
uv sync --extra tf       # TF handoff (frameworks.as_tf_dataset)
uv sync --extra bench    # benchmark suite (xbatcher baseline + plotly)
uv sync --extra gpu      # CUDA box only: cupy + kvikio zero-copy path
```

## Status

**Alpha — validated on real cloud IO.** Built: planner + chunk-aligned splits, async
obstore reads, the decoupled fetch **`Scheduler`** + **`ChunkPool`** (assembly buffer
*and* cache — byte budget + pin/LRU, heap or mmap-on-NVMe, with **cross-run
persistence** via `persist=True`), approximate (shuffle-block) shuffle, chunk/batch
**transforms** (incl. a fitted `StandardScaler`), **prefetch**, and the **torch / JAX /
TF** surfaces. Not yet built: `Regrid` + the **GPU/device** transform stage, and
multi-timestep windows that cross chunk boundaries.

[DESIGN.md](https://github.com/emfdavid/insitubatch/blob/main/DESIGN.md) is the single
source of truth for status, the roadmap, and the scope limits.
