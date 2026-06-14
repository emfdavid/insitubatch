# insitubatch

**Train in place on n-dimensional cloud tensors.**

`insitubatch` is the data-loader orchestration layer that sits on top of
*already-solved* async cloud IO (obstore / zarr v3 / icechunk). It turns an
existing Zarr archive into a shuffled, split-aware PyTorch source built to **keep
the GPU fed** — **with no reshard** — and a Python hot path that scales with
**chunks, not samples**.

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

🚧 **Pre-alpha — Phase 0 complete (local).** Real obstore-backed zarr v3 async
reads work end-to-end against a local `file://` store; chunk-aligned splits,
approximate (shuffle-block) shuffle, a bounded buffer, chunk/batch **transforms**
(incl. a fitted `StandardScaler`), and **prefetch** (a background producer that
runs ahead of the consumer) are implemented and tested (23 tests). Early signal:
the GRIB-per-timestep regime is ~2.8× over a naive sync baseline locally.

Not yet built: the **chunk cache**, `Regrid` + the **GPU/device** transform stage.
Torch is the only framework surface so far (JAX/TF planned). See the roadmap and
scope limits in [DESIGN.md](DESIGN.md) and [docs/architecture.md](docs/architecture.md).

## Install (dev)

```bash
uv sync                  # core engine + dev tools
uv sync --extra torch    # add the torch IterableDataset surface
uv sync --extra gpu      # CUDA box only: cupy + kvikio zero-copy path
```

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

## License

MIT — see [LICENSE](LICENSE).
