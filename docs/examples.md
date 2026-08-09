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
| [`sdss/`](#sdss-reconstructing-galaxy-spectra-streamed-in-place) | Astronomy — SDSS DR17 `spPlate` spectra, or our **public reference stores** | **FITS binary-image tables** re-chunked by byte arithmetic; two layouts from the same bytes | Both regimes from one dataset: many-fibers-per-chunk *decode amortization* vs one-fiber-per-chunk *archive-scale streaming* |
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
uv sync --extra torch --extra astronomy
uv run python -m examples.hubble.train_torch --source hubble --build
```

Because a FITS image is one chunk, this demonstrates the **streaming-in-place** value (no
reshard over a giant archive) rather than the many-samples-per-chunk decode amortization the
chunked-zarr examples show — the honest boundary of the thesis, kept visible on purpose.

## sdss — reconstructing galaxy spectra streamed in place

The companion to Hubble, and an [astroML](https://github.com/astroML/astroML) mirror.
astroML's spectral-PCA workflow downloads the raw archive and resamples every spectrum onto a
common grid into one `spec4000.npz` — the download-and-reshard step this project argues
against. [`sdss/`](https://github.com/emfdavid/insitubatch/blob/main/examples/sdss/data.py)
instead indexes SDSS DR17 **`spPlate`** frames as virtual references and trains on them where
they lie.

The trick is that a plate already holds ~640 fibers on one common log-wavelength grid, stored
fiber-major — so a block of fibers is a contiguous byte range, and **re-chunking is arithmetic
on the manifest, not a copy.** The same bytes give two layouts:

- **one plate → 64 fibers per chunk.** One read decodes 64 spectra: the *decode-amortization*
  regime.
- **many plates → one fiber per chunk.** Plates cover slightly different wavelength ranges, so
  joining them means cropping each to the shared window, which breaks fiber contiguity. You get
  archive-scale concatenation (extending to the ~2800-plate archive) and pay one read per
  spectrum: the *streaming* regime.

The crop is exact, not interpolated: every plate uses the same `dloglam` step and their start
wavelengths differ by whole bins, so the windows land on one grid. The two regimes are mutually
exclusive here — an honest property of the FITS byte layout, not a tuning choice.

### Streaming without building anything

The stores are published read-anonymous, so the scan is already done. This needs `icechunk`
only — no FITS stack, no credentials, no build step:

```bash
uv sync --extra torch --extra astronomy
uv run python -m examples.sdss.train_torch --source published                      # 6 plates
uv run python -m examples.sdss.train_torch --source published --published 1plate   # 1 plate
```

`--published` picks the layout: `1plate` (640 spectra, 64 per chunk), `6plate` (3840 spectra,
one per chunk), or `6plate-mirror` (same as `6plate`, references pointing at our mirror of the
FITS files — the default, and the one to prefer). Bucket layout and a plain-zarr recipe are in
the [dataset README](https://storage.googleapis.com/insitubatch-bench-insitubatch/astronomy/README.md).

### Or index the archive yourself

```bash
uv run python -m examples.sdss.train_torch                                   # offline synthetic
uv run python -m examples.sdss.train_torch --source sdss --build             # 1 plate
uv run python -m examples.sdss.train_torch --source sdss --build --plates 8  # 8 plates
```

`build_store` reads the `spPlate` headers with VirtualiZarr and writes an Icechunk repo of
byte-range references — kilobytes, whatever the archive size. Point it at more plate URLs to
index a larger slice.

The task mirrors astroML's **spectral reconstruction**: recover the spectrum through a
low-dimensional bottleneck. The baseline is **PCA** at the same latent dim — the optimal
*linear* reconstruction — and a small 1-D **convolutional autoencoder** trained over the
streamed batches beats it, because varying redshift shifts the lines and a fixed linear basis
reconstructs a shifted spectrum poorly. `--source synthetic` (the default) needs no network and
carries that claim; the real run is the same pipeline on the real archive.

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
