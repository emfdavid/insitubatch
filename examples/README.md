# insitubatch examples

Runnable examples (not shipped in the wheel). Run from the repo root, e.g.
`python -m examples.advection.train_torch`.

The InsituDataset is a general purpose, batteries included, tool for batching zarr compatible 
data into the python ML ecosystem. Most examples are forecasting focused because there are large
public zarr datasets available and the time shifting on insitu data is an added challenge that
shows the generalization; [`microscopy/`](#microscopy--ome-ngff-cell-segmentation-over-z) is the
cross-domain companion — a different geometry (sample over Z, two variables chunked differently)
on a real bio-imaging store, to show the engine is not weather-specific.

## advection/ — a 24-hour forecast, one dataset, three frameworks

The M-W showcase: **multi-variable, windowed, train-in-place** sampling driving a real
forecast — and the framework-neutral payoff. One [`InSituDataset`](advection/data.py)
reads three fields at time *t* (temperature `t2m` and the 10 m wind `u10`, `v10`) and the
target `t2m` 24 h later (`g.shift(horizon)`) — input and target are **offset views of the
same in-place array**, the inputs are arrays gathered at the **same anchor**, and *nothing
is resharded*. The same numpy `Batch` then trains the **same tiny CNN** in three frameworks
via the zero-copy DLPack adapters — the files differ only in framework calls:
```bash
uv sync --extra bench --extra torch               # PyTorch   (torch.nn)
uv run python -m examples.advection.train_torch   #

uv sync --extra bench --extra jax                 # JAX       (flax + optax)
uv run python -m examples.advection.train_jax     #

uv sync --extra bench --extra tf                  # TensorFlow (Keras)
uv run python -m examples.advection.train_tf      #
```

Each prints the 24-hour forecast skill on held-out data — the model (which *reads the wind*
to predict the advection) vs the persistence baseline. All three share one CLI
([`data.py`](advection/data.py)); the files differ only in framework calls.

Make sure only one ML toolkit is installed at a time when running the models. Having multiple
present in the uv venv can cause segfaults.

### Data sources (`--source`)

| `--source` | store | extra | notes |
| --- | --- | --- | --- |
| `synthetic` *(default)* | a fast offline **advected field** written to a temp zarr | — | beats persistence by construction; `--n-steps` sets the trajectory length |
| `wb2` | public **WeatherBench2** ERA5, `gs://` (anonymous) | `--extra bench` (gcsfs) | the *same code* on real cloud data, no reshard |
| `arraylake` | the same ERA5 as an **Arraylake/Icechunk** repo | `--extra arraylake` | needs `al auth login` / `ARRAYLAKE_TOKEN`; `--repo` / `--group` select it |

On real ERA5 the claim is "same pipeline, real data", **not** SOTA skill — 24 h persistence
of temperature is a strong baseline.

### Flags

- `--device cpu|cuda` — where the **training loop** moves each batch. The dataset stays
  numpy (placement is the loop's job, not the loader's); torch does an explicit `.to(device)`,
  JAX/TF auto-place on the accelerator and `--device cpu` forces/hides it.
- `--sample-range START,STOP` — a **finite training window** on the time axis, so you can
  subset a multi-decade real store (e.g. `0,2920` ≈ 2 years at 6-hourly).
- `--cache-dir DIR` — spill the decoded-chunk cache to fast local disk (NVMe) for
  **cross-epoch reuse** over a high-latency network.
- `--max-inflight N` — throttle read-ahead depth (lower ⇒ lower cold time-to-first-batch
  when the network is the bottleneck).
- `--epochs`, `--batch-size`, `--repo`, `--group`, `--url` round out the CLI (`--help`).

Running the examples on a GPU box against real ERA5 (G6 + NVMe, the Deep Learning Base AMI,
CUDA torch via `--torch-backend`, Arraylake auth) is documented in
[`bench/ops_aws.md`](../bench/ops_aws.md) §10.

**The pattern to take away:** point insitu at an existing cloud zarr, declare your inputs
and a shifted target as `(label, path, offset)` views, and train in your framework — the
windowing (`Batch.offsets`, `Batch.stack`, `Batch.read_indices`) and the no-reshard,
chunk-once IO are the engine's job.

## microscopy/ — OME-NGFF cell segmentation over Z

The cross-domain showcase: **the same engine, a different geometry.** Where advection samples
the *outer* time axis and windows it, [`microscopy/`](microscopy/data.py) samples a *middle*
axis — one Z-plane of an OME-NGFF `(T,C,Z,Y,X)` confocal stack (`sample_axis=2`) — and gathers
two co-registered variables at each anchor: the 2-channel `raw` image (chunked **one plane
deep** on Z) and its `mask` label (chunked **30 planes deep**, tiled in Y/X). Different physical
chunking, different channel count, one sample grid, **no reshard** — the arbitrary-sample-axis +
per-variable-chunking unlock, on a real store.

```bash
uv sync --extra torch
uv run python -m examples.microscopy.train_torch                    # synthetic cells (offline)
uv run python -m examples.microscopy.train_torch --source idr       # the real IDR image (streamed)
```

The task is per-plane **foreground segmentation**, and the baseline is a global intensity
threshold (**Otsu**) — the segmentation analog of persistence. Otsu reads each pixel's intensity
alone, so a smooth autofluorescence *haze* gradient defeats it (a bright-background pixel
outshines a dim cell elsewhere; no single threshold separates them). A tiny CNN that reads the
neighborhood — sharp cells vs low-frequency haze — beats it: on the synthetic store by
construction, on the real IDR image the claim is "same pipeline, real data, no reshard" (not
SOTA Dice — the label is expert instance annotation and the model is four conv layers). Each run
prints the held-out foreground IoU, model vs Otsu.

### Data sources (`--source`)

| `--source` | store | notes |
| --- | --- | --- |
| `synthetic` *(default)* | offline **cell stack** written to a temp zarr | Gaussian cells + a haze gradient noise Otsu can't beat; `--n-planes` / `--size` / `--mask-chunk` size it |
| `idr` | the public **IDR** OME-NGFF image `s3://idr/...` (EMBL-EBI) | streamed anonymously; `--sample-range` subsets the 236-plane stack |

Only `train_torch.py` ships here — framework-neutrality is the advection example's job; this
example's job is to prove the *geometry* generalizes.

## hubble/ — denoising real telescope frames from FITS (no reshard)

The **archival-format** showcase: the data never was zarr. [`hubble/`](hubble/data.py) indexes
real Hubble WFC3/IR frames of **M16 (the Eagle Nebula)** on MAST's public AWS bucket
(`s3://stpubdata`, anonymous) as **virtual references** — [VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr)
parses each `_flt.fits` header (`kerchunk.fits`) and commits byte-range references to a local
Icechunk repo. No pixels are copied or resharded; the store is a few kB pointing at the original
FITS objects. insitubatch then streams the frames straight from S3, `sample_axis=0` making each
frame one sample. The build-time index libraries (`virtualizarr`/`kerchunk`/`astropy`) are needed
*only* to build the store — the training hot path is `icechunk` + numpy, never kerchunk.

```bash
uv sync --extra torch
uv run --with scipy python -m examples.hubble.train_torch          # offline synthetic frames (default)

# real Hubble frames on S3 -- indexes them into a virtual-reference store first (needs the
# build-time stack), then streams and trains:
uv run --with virtualizarr --with kerchunk --with astropy --with icechunk --with s3fs --with scipy \
  python -m examples.hubble.train_torch --source hubble --build
```

The task is per-frame **Gaussian-noise removal** (a deliberately didactic stand-in — the point is
training on the real archive, not a SOTA denoiser), and the baseline is a **median filter** (the
no-training reference). Two chunk stages run vectorized on the decode pool — a robust
per-frame normalization (`clean_normalize`) then a block-mean `Coarsen` (1014→253, keeping the CPU
demo light while still reading whole frames) — and the per-sample noise lives in the `AddNoise`
batch stage, per the transform-cost contract. A short run beats the baseline (real ≈26.4 vs ≈24.7 dB
PSNR; synthetic sharp-star frames similar). Because a FITS image is one chunk, this is the
*streaming-in-place* value (no reshard over a giant archive), not the many-samples-per-chunk
decode-amortization of the chunked-zarr examples. `--source synthetic` (the default) needs no
network or FITS stack — it also backs the drift test in `tests/test_hubble.py`.

> Note: MAST's *anonymous* bucket throttles (HTTP 503) under heavy concurrent read-ahead, so
> `max_inflight` is capped low by default; an authenticated/retrying path on AWS would raise it.

## The WeatherBench2 cold-start pair (with xbatcher)

The same task two ways — complementary engines for the same ndim-batch problem — so you
can see the cold-start trade-off and pick what fits your workload:

- [`wb2_dataloader.py`](wb2_dataloader.py) — the insitu single-event-loop loader.
  `--backend fsspec` (needs `--extra gcsfs`) reads the same store through `fsspec_store`
  (gcsfs) instead of obstore — the A/B for GCS Rapid/zonal + Requester-Pays.
- [`wb2_xbatcher.py`](wb2_xbatcher.py) — the xbatcher + torch `DataLoader` worker stack
  (xbatcher defines the batches, the `DataLoader` runs them), the pattern from Earthmover's
  `dataloader-demo`, with a focus on cold-start latency and how `forkserver-preload` cuts it
  (useful whichever loader you ship).

## Transforms

- [`transforms.py`](transforms.py) — the two user transform stages side by side on a tiny
  offline store: a Kelvin→Celsius `chunk_transform` (per chunk, one variable, cached) and a
  cross-variable windspeed `batch_transform` (needs the assembled batch, uncached). The
  clearest illustration of *why there are two*; see docs/architecture.md "Transforms" for
  the placement model. Runs with no network: `uv run python -m examples.transforms`.

## Normalization

- [`fit_scaler.py`](fit_scaler.py) — fit a `StandardScaler` over the loader with sklearn
  `partial_fit` (warms the cache), the recommended pattern vs caching scaled chunks.
