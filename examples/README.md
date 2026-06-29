# insitubatch examples

Runnable examples (not shipped in the wheel). Run from the repo root, e.g.
`python -m examples.advection.train_torch`.

The InsituDataset is a general purpose, batteries included, tool for batching zarr compatible 
data into the python ML ecosystem. The examples are forecasting focused because there are large
public zarr datasets available and the time shifting on insitu data is an added challenge that
shows the generalization.

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

## The WeatherBench2 cold-start pair (with Earthmover's xbatcher)

The same task two ways — complementary engines for the same ndim-batch problem — so you
can see the cold-start trade-off and pick what fits your workload:

- [`wb2_dataloader.py`](wb2_dataloader.py) — the insitu single-event-loop loader.
- [`wb2_xbatcher.py`](wb2_xbatcher.py) — Earthmover's `dataloader-demo` stack (xbatcher
  defines the batches, torch `DataLoader` runs them), with a focus on cold-start latency
  and how `forkserver-preload` cuts it (useful whichever loader you ship).

## Transforms

- [`transforms.py`](transforms.py) — the two user transform stages side by side on a tiny
  offline store: a Kelvin→Celsius `chunk_transform` (per chunk, one variable, cached) and a
  cross-variable windspeed `batch_transform` (needs the assembled batch, uncached). The
  clearest illustration of *why there are two*; see docs/architecture.md "Transforms" for
  the placement model. Runs with no network: `uv run python -m examples.transforms`.

## Normalization

- [`fit_scaler.py`](fit_scaler.py) — fit a `StandardScaler` over the loader with sklearn
  `partial_fit` (warms the cache), the recommended pattern vs caching scaled chunks.
