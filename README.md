# insitubatch

**Train in place on n-dimensional cloud tensors.**

`insitubatch` is the data-loader orchestration layer that sits on top of
*already-solved* async cloud IO (obstore / zarr v3 / icechunk). It turns an
existing Zarr archive into a shuffled, split-aware, GPU-saturating PyTorch
source — **with no reshard** — and a Python hot path that scales with **chunks,
not samples**.

> The IO race is over (obstore/icechunk saturate the NIC). The *loader* race is
> open. `insitubatch` builds the layer that projects like light-speed-io and
> hypergrib stopped one step short of. See [DESIGN.md](DESIGN.md).

## Why

The classic PyTorch `DataLoader` spreads work across worker **processes**, each
running a *synchronous* `__getitem__`. Against cloud Zarr that means no shared
chunk cache (every worker re-reads the same chunk), no way to drive async
obstore, and dask thread pools nested inside forked workers. `insitubatch`
**inverts** it: one async event loop drives concurrent reads; a bounded
shuffle-block buffer assembles batches; torch runs `num_workers=0`.

## Status

🚧 **Pre-alpha skeleton.** Abstractions and control flow are in place; the live
store read in `io.py` and the GPU path are stubbed. Not yet usable for real
training — this is the design substrate.

## Install (dev)

```bash
uv sync                  # core engine + dev tools
uv sync --extra torch    # add the torch IterableDataset surface
uv sync --extra gpu      # CUDA box only: cupy + kvikio zero-copy path
```

## Shape of the API (target)

```python
from insitubatch import split_by_chunk, ArrayGeometry, SplitName
from insitubatch.source import InSituDataset
from torch.utils.data import DataLoader

geom = ArrayGeometry("t2m", shape=(8760, 721, 1440), chunks=(24, 721, 1440), dtype=...)
manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))

ds = InSituDataset(store, {"t2m": geom}, manifest, SplitName.TRAIN,
                   batch_size=32, block_chunks=16)

# parallelism lives in insitubatch's event loop, not in workers:
loader = DataLoader(ds, batch_size=None, num_workers=0)
for epoch in range(n_epochs):
    ds.set_epoch(epoch)
    for batch in loader:
        ...
```

## License

MIT — see [LICENSE](LICENSE).
