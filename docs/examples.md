# Examples

The sample axis is a **role, not a fixed dimension** — so the same engine trains on weather
over time, segments microscopy volumes over `Z`, and streams telescope frames out of an
archival format that was never zarr. Each example below is a *different geometry* on a real
public store, with **no reshard** anywhere.

All of them are runnable from a checkout (they are not shipped in the wheel), and every one
has an offline synthetic `--source` so you can run it with no network or cloud credentials.

| Example | Domain & store | What is different about the data | What it proves |
|---|---|---|---|
| [`advection/`](#advection-a-24-hour-forecast-in-three-frameworks) | Weather — WeatherBench2 ERA5 (`gs://`, anonymous), Arraylake/Icechunk, or synthetic | Input at *t*, target at *t+24 h* as **offset views of one array** | Windowed multi-offset sampling; one dataset → **torch, JAX and TF** |
| [`microscopy/`](#microscopy-cell-segmentation-over-z) | Bio-imaging — IDR OME-NGFF `(T,C,Z,Y,X)` on `s3://idr` | Samples a **middle** axis (`sample_axis=2`); two co-registered variables chunked **1 vs 30 planes** deep | Arbitrary sample axis + per-variable chunk size — the engine is not weather-specific |
| [`hubble/`](#hubble-denoising-real-telescope-frames-from-fits) | Astronomy — Hubble WFC3/IR frames of M16 on MAST's `s3://stpubdata` | **FITS, not zarr** — indexed as virtual byte-range references; one frame *is* one chunk | Training in place over an archival format; streaming value without decode amortization |
| [WB2 pair](#the-weatherbench2-cold-start-pair) | Weather — WeatherBench2 ERA5 | The same task on **two engines** | The cold-start / memory trade-off vs an xbatcher worker stack |
| [`transforms.py`](#transforms-and-normalization) · [`fit_scaler.py`](#transforms-and-normalization) | Any — tiny offline store | — | Why there are *two* transform stages, and how to fit a scaler over the loader |

Full flags, data-source tables and design notes for each live in
[`examples/README.md`](https://github.com/emfdavid/insitubatch/blob/main/examples/README.md).

## advection — a 24-hour forecast in three frameworks

One [`InSituDataset`](https://github.com/emfdavid/insitubatch/blob/main/examples/advection/data.py)
reads three fields at time *t* (temperature `t2m` and the 10 m wind `u10`, `v10`) and the
target `t2m` 24 hours later via `g.shift(horizon)`. Input and target are **offset views of
the same in-place array** — the windowing unlock — and nothing is resharded. The resulting
numpy `Batch` then trains the **same tiny CNN** in three frameworks through the DLPack
adapters; the three files differ only in framework calls.

```bash
uv sync --extra bench --extra torch               # PyTorch   (torch.nn)
uv run python -m examples.advection.train_torch

uv sync --extra bench --extra jax                 # JAX       (flax + optax)
uv run python -m examples.advection.train_jax

uv sync --extra bench --extra tf                  # TensorFlow (Keras)
uv run python -m examples.advection.train_tf
```

Each run prints 24-hour forecast skill on held-out data — a model that *reads the wind* to
predict advection, versus the persistence baseline. `--source wb2` runs the same code
against real ERA5 in the cloud; there the claim is "same pipeline, real data", **not** SOTA
skill (24 h temperature persistence is a strong baseline).

!!! warning "Install one framework at a time"
    Having torch, JAX and TensorFlow present in the same uv venv can segfault. Sync the
    extra for the one you are running.

## microscopy — cell segmentation over Z

The cross-domain showcase: same engine, different geometry. Where advection samples the
*outer* time axis, [`microscopy/`](https://github.com/emfdavid/insitubatch/blob/main/examples/microscopy/data.py)
samples a *middle* axis — one Z-plane of an OME-NGFF `(T,C,Z,Y,X)` confocal stack — and
gathers two co-registered variables per anchor: a 2-channel `raw` image chunked **one plane
deep** on Z, and its `mask` label chunked **30 planes deep** and tiled in Y/X. Different
physical chunking, different channel counts, one sample grid, no reshard.

```bash
uv sync --extra torch
uv run python -m examples.microscopy.train_torch                 # synthetic cells (offline)
uv run python -m examples.microscopy.train_torch --source idr    # the real IDR image (streamed)
```

The task is per-plane foreground segmentation and the baseline is a global **Otsu**
threshold — the segmentation analogue of persistence. Otsu reads each pixel's intensity
alone, so a smooth autofluorescence haze gradient defeats it; a tiny CNN that reads the
neighbourhood beats it. Each run prints held-out foreground IoU, model vs Otsu.

## hubble — denoising real telescope frames from FITS

The **archival-format** showcase: this data never was zarr.
[`hubble/`](https://github.com/emfdavid/insitubatch/blob/main/examples/hubble/data.py)
indexes real Hubble WFC3/IR frames of M16 (the Eagle Nebula) on MAST's public AWS bucket as
**virtual references** — [VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr)
parses each `_flt.fits` header and commits byte-range references to a local Icechunk repo.
No pixels are copied: the store is a few kB pointing at the original FITS objects, and
insitubatch streams frames straight from S3 with `sample_axis=0`. The indexing libraries are
build-time only — the training hot path is `icechunk` + numpy.

```bash
uv sync --extra torch
uv run python -m examples.hubble.train_torch      # offline synthetic frames (default)

# real Hubble frames on S3 — build the virtual-reference store, then stream and train:
uv run --with virtualizarr --with kerchunk --with astropy --with icechunk --with s3fs \
  python -m examples.hubble.train_torch --source hubble --build
```

Because a FITS image is one chunk, this demonstrates the **streaming-in-place** value (no
reshard over a giant archive) rather than the many-samples-per-chunk decode amortization the
chunked-zarr examples show — the honest boundary of the thesis, kept visible on purpose.

## The WeatherBench2 cold-start pair

The same task two ways, so you can see the trade-off and pick per workload:
[`wb2_dataloader.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/wb2_dataloader.py)
is the insitu single-event-loop loader (with `--backend fsspec` for the gcsfs A/B), and
[`wb2_xbatcher.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/wb2_xbatcher.py)
is the xbatcher + torch `DataLoader` worker stack, following Earthmover's `dataloader-demo`,
focused on cold-start latency and how `forkserver-preload` cuts it.

```bash
uv run python -m examples.wb2_dataloader          # tiny synthetic data, no network
```

The [WeatherBench2 walkthrough](walkthrough.md) narrates this pair end to end, and
[Benchmarks](benchmarks.md) has the measured numbers.

## Transforms and normalization

[`transforms.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/transforms.py)
puts the two user transform stages side by side on a tiny offline store — a Kelvin→Celsius
`chunk_transform` (per chunk, one variable, cached) and a cross-variable windspeed
`batch_transform` (needs the assembled batch, uncached). It is the clearest illustration of
*why there are two*; see [Transforms](architecture.md#transforms-three-stages-placed-by-cost)
for the placement model.

[`fit_scaler.py`](https://github.com/emfdavid/insitubatch/blob/main/examples/fit_scaler.py)
fits a `StandardScaler` over the loader with sklearn `partial_fit` — the recommended pattern
(it warms the cache) versus caching scaled chunks.

```bash
uv run python -m examples.transforms              # no network
```
