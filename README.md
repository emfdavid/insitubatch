# insitubatch

[![PyPI](https://img.shields.io/pypi/v/insitubatch.svg)](https://pypi.org/project/insitubatch/)
[![CI](https://github.com/emfdavid/insitubatch/actions/workflows/ci.yml/badge.svg)](https://github.com/emfdavid/insitubatch/actions/workflows/ci.yml)
[![docs](https://github.com/emfdavid/insitubatch/actions/workflows/docs.yml/badge.svg)](https://emfdavid.github.io/insitubatch/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

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

🚧 **Pre-alpha, but validated on real cloud IO.** On an in-region S3 run
(`c6id.8xlarge`, ERA5-shaped `721×1440` fields, `sample_chunk=8`), insitubatch
delivers **~8× the throughput** of a *tuned* `xbatcher`/worker `DataLoader`
baseline (swept to 32 workers) and reaches its first batch **~10× sooner** — the
map-style baseline re-decodes a whole chunk per sample; insitubatch reads each
chunk once. Full numbers + methodology:
[the benchmarks page](https://emfdavid.github.io/insitubatch/benchmarks/).

Built: planner + chunk-aligned splits, async obstore reads with bounded fan-out, a
shuffle-block buffer with one-block read-ahead, chunk/batch **transforms** (incl. a
fitted `StandardScaler`), **prefetch**, a pluggable **chunk cache** (heap or
mmap-on-NVMe, byte-LRU), the torch surface, and runnable [examples](examples/). Not
yet built: `Regrid` + the **GPU/device** transform stage; JAX/TF surfaces. The
**V2 decoupled fetch scheduler** (one concurrency budget over inner+outer chunks,
buffer-as-cache) is designed and de-risked — see the roadmap in
[DESIGN.md](DESIGN.md).

📖 **Docs:** <https://emfdavid.github.io/insitubatch/>
(see [Tuning](https://emfdavid.github.io/insitubatch/tuning/) for the
chunks↔concurrency↔memory model).

## Install

```bash
pip install insitubatch              # core engine
pip install "insitubatch[torch]"     # + the torch IterableDataset surface
```

For development:

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
