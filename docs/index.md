# insitubatch

**Train in place on n-dimensional cloud tensors.**

`insitubatch` is the data-loader orchestration layer that sits on top of
*already-solved* async cloud IO (obstore / zarr v3 / icechunk). It turns an
existing Zarr archive into a shuffled, split-aware PyTorch source built to **keep
the GPU fed** — **with no reshard** — and a Python hot path that scales with
**chunks, not samples**.

!!! quote
    The IO race is over (obstore/icechunk saturate the NIC). The *loader* race is
    open. `insitubatch` builds the layer that projects like light-speed-io and
    hypergrib stopped one step short of.

## The problem

The classic PyTorch `DataLoader` spreads work across worker **processes**, each
running a *synchronous* `__getitem__`. Against cloud Zarr that means:

- **No shared chunk cache** — every worker re-reads (and re-decodes) the same
  chunk, because a sample is a slice of a chunk and neighbouring samples land in
  the same chunk.
- **No way to drive async obstore** — a synchronous `__getitem__` can't fan out
  concurrent range requests, so the NIC sits idle between blocking reads.
- **Dask thread pools nested inside forked workers** — the usual xarray path
  fights itself for cores.

The fix people reach for is to **reshard**: rewrite the archive into
one-sample-per-file shards. That is a second copy of the dataset, a pipeline to
maintain, and it throws away the chunk locality the store already has.

## The inversion

`insitubatch` keeps the data where it is and **inverts the loader**:

> Classic `DataLoader`: parallelism lives in **`num_workers` OS processes**, each
> running a synchronous `__getitem__`. insitubatch: parallelism lives in **one
> async event loop**; batch assembly is the consumer.

That single move is what unlocks async obstore, a **shared chunk cache**, bounded
memory, and **prefetch overlap** with the training step. Torch runs with
`num_workers=0` — the concurrency is ours, not the worker pool's.

See [Architecture](architecture.md) for the loader/prefetch diagrams and the
read-plan abstraction, and the
[design rationale](https://github.com/emfdavid/insitubatch/blob/main/DESIGN.md)
for the why.

## Shape of the API

```python
from insitubatch import open_geometries, split_by_chunk, SplitName
from insitubatch.source import InSituDataset
from torch.utils.data import DataLoader

url = "file:///data/era5.zarr"           # or "s3://bucket/era5.zarr" — same code
geoms = open_geometries(url)             # {var: ArrayGeometry} from zarr metadata
manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))

ds = InSituDataset(url, manifest, split=SplitName.TRAIN,
                   batch_size=32, block_chunks=16)

# parallelism lives in insitubatch's event loop, not in workers:
loader = DataLoader(ds, batch_size=None, num_workers=0)
for epoch in range(n_epochs):
    ds.set_epoch(epoch)
    for batch in loader:                 # {var: torch.Tensor}
        ...
```

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
uv sync --extra torch    # add the torch IterableDataset surface
uv sync --extra bench    # benchmark suite (xbatcher baseline + plotly)
uv sync --extra gpu      # CUDA box only: cupy + kvikio zero-copy path
```

## Status

🚧 **Pre-alpha.** Real obstore-backed zarr v3 async reads work end-to-end;
chunk-aligned splits, approximate (shuffle-block) shuffle, a bounded buffer,
chunk/batch **transforms** (incl. a fitted `StandardScaler`), **prefetch**, and a
pluggable **chunk cache** (in-memory or mmap-on-NVMe, byte-LRU) are implemented
and tested.

Not yet built: `Regrid` and the **GPU/device** transform stage; multi-timestep
windows that cross chunk boundaries; JAX/TF surfaces. See the roadmap and scope
limits in the
[design rationale](https://github.com/emfdavid/insitubatch/blob/main/DESIGN.md).
