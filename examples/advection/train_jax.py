"""Forecast t2m 24 h ahead with a tiny CNN, in **JAX** (flax + optax), on one insitu dataset.

Same shared, framework-neutral data layer as ``train_torch.py`` -- the numpy ``Batch`` is
handed to JAX zero-copy via ``to_jax`` (DLPack). Only this file is JAX; JAX auto-places on
the accelerator it finds (``--device cpu`` forces CPU). JAX/TF use channels-last
(``B, H, W, C``). Usage, sources and flags: ``examples/README.md``.
"""

from __future__ import annotations

from typing import Any

import flax.linen as fnn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from insitubatch import Batch, InSituDataset, to_jax

from .data import build_datasets, cli, evaluate


class AdvectionCNN(fnn.Module):
    """3 input channels (t2m, u10, v10) -> 1 channel tendency. Inputs standardized per
    channel; four 3x3 circular convs (receptive field 9) cover the ~5-cell displacement."""

    hidden: int = 32

    @fnn.compact
    def __call__(self, x: jax.Array) -> jax.Array:  # x: (B, H, W, 3) -> tendency (B, H, W, 1)
        xn = (x - x.mean((0, 1, 2), keepdims=True)) / (x.std((0, 1, 2), keepdims=True) + 1e-6)
        for _ in range(3):
            xn = fnn.relu(fnn.Conv(self.hidden, (3, 3), padding="CIRCULAR")(xn))
        return fnn.Conv(1, (3, 3), padding="CIRCULAR")(xn)


def _channels(batch: Batch) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(x, persistence, target)`` as channels-last JAX arrays, zero-copy via DLPack."""
    d = to_jax(batch)  # {label: (B, H, W)}
    x = jnp.stack([d["t2m"], d["u10"], d["v10"]], axis=-1)  # (B, H, W, 3)
    return x, d["t2m"][..., None], d["target"][..., None]  # (B, H, W, 1) each


def train(ds: InSituDataset, *, epochs: int, device: str = "cpu") -> tuple[float, float]:
    """Train the CNN; return ``(model_rmse, persistence_rmse)`` -- 24 h forecast skill on val.

    JAX auto-places on the accelerator it finds (install ``jax[cuda12]`` for GPU); ``device
    == "cpu"`` forces CPU. Either way the dataset stays numpy and ``to_jax`` is a per-batch
    DLPack hand-off."""
    if device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    model = AdvectionCNN()
    opt = optax.adam(1e-3)
    ds.set_epoch(0)
    sample_x, _, _ = _channels(next(iter(ds.train)))
    params = model.init(jax.random.key(0), sample_x)
    opt_state = opt.init(params)

    @jax.jit
    def step(
        params: Any, opt_state: Any, x: jax.Array, persistence: jax.Array, target: jax.Array
    ) -> tuple[Any, Any, jax.Array]:
        def loss_fn(p: Any) -> jax.Array:
            pred = persistence + jnp.asarray(model.apply(p, x))
            return jnp.mean((pred - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    for epoch in range(epochs):
        ds.set_epoch(epoch)
        last = 0.0
        for batch in ds.train:
            params, opt_state, loss = step(params, opt_state, *_channels(batch))
            last = float(loss)
        print(f"epoch {epoch}  train mse {last:.4f}")

    def predict(batch: Batch) -> np.ndarray:  # (B, 1, H, W) to match the shared eval
        x, persistence, _ = _channels(batch)
        forecast = persistence + jnp.asarray(model.apply(params, x))  # (B, H, W, 1)
        return np.asarray(forecast).transpose(0, 3, 1, 2)

    return evaluate(ds.val, predict)


def main() -> None:
    args = cli()
    ds = build_datasets(args)
    model_rmse, persistence_rmse = train(ds, epochs=args.epochs, device=args.device)
    print(
        f"\n24 h forecast RMSE on held-out data: model {model_rmse:.3f}  vs  "
        f"persistence {persistence_rmse:.3f}  ({persistence_rmse / model_rmse:.1f}x better)"
    )


if __name__ == "__main__":
    main()
