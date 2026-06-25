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
python -m examples.advection.train_torch   # PyTorch   (torch.nn)
python -m examples.advection.train_jax     # JAX       (flax + optax)   [uv sync --extra jax]
python -m examples.advection.train_tf      # TensorFlow (Keras)         [uv sync --extra tf]
```

Each prints the 24-hour forecast skill on held-out data — the model (which *reads the wind*
to predict the advection) vs the persistence baseline. The default is a fast offline
**synthetic advected field**; `--wb2` runs the *same code* on the public **WeatherBench2**
ERA5 store (`gs://`, anonymous — `uv sync --extra bench` for gcsfs) — real cloud data, no
reshard. (On real ERA5 the claim is "same pipeline, real data", not SOTA skill: 24 h
persistence of temperature is a strong baseline.)

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

## Normalization

- [`fit_scaler.py`](fit_scaler.py) — fit a `StandardScaler` over the loader with sklearn
  `partial_fit` (warms the cache), the recommended pattern vs caching scaled chunks.
