"""Forecast t2m 24 h ahead with a tiny CNN, in **TensorFlow** (Keras), on one insitu dataset.

    python -m examples.advection.train_tf          # synthetic advected field (offline)
    python -m examples.advection.train_tf --wb2    # WeatherBench2 (real ERA5, gs://)

Same shared, framework-neutral data layer as ``train_torch.py`` / ``train_jax.py`` -- the
numpy ``Batch`` is handed to TF zero-copy via ``to_tf`` (DLPack). Only this file is TF. The
model learns the **tendency** (the change on persistence); beating persistence means it
learned the wind-driven advection. JAX/TF use channels-last (``B, H, W, C``).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from insitubatch.frameworks import to_tf
from insitubatch.source import InSituDataset
from insitubatch.types import Batch

from .data import build_datasets, cli, evaluate


class CircularConv(keras.layers.Layer):
    """3x3 convolution with circular (periodic) padding -- the synthetic field wraps, and
    it matches the torch/jax ``padding_mode="circular"`` so the three models are the same."""

    def __init__(self, filters: int, **kw: object) -> None:
        super().__init__(**kw)
        self.conv = keras.layers.Conv2D(filters, 3, padding="valid")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.concat([x[:, -1:], x, x[:, :1]], axis=1)  # wrap rows
        x = tf.concat([x[:, :, -1:], x, x[:, :, :1]], axis=2)  # wrap cols
        return self.conv(x)


def _standardize(z: tf.Tensor) -> tf.Tensor:
    mean = tf.reduce_mean(z, axis=[0, 1, 2], keepdims=True)
    std = tf.math.reduce_std(z, axis=[0, 1, 2], keepdims=True)
    return (z - mean) / (std + 1e-6)


def build_model(hidden: int = 32) -> keras.Model:
    """3 input channels -> 1 channel tendency; four circular convs (receptive field 9)."""
    return keras.Sequential(
        [
            keras.layers.Lambda(_standardize),
            CircularConv(hidden),
            keras.layers.ReLU(),
            CircularConv(hidden),
            keras.layers.ReLU(),
            CircularConv(hidden),
            keras.layers.ReLU(),
            CircularConv(1),
        ]
    )


def _channels(batch: Batch) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """``(x, persistence, target)`` as channels-last TF tensors, zero-copy via DLPack."""
    d = to_tf(batch)  # {label: (B, H, W)}
    x = tf.stack([d["t2m"], d["u10"], d["v10"]], axis=-1)  # (B, H, W, 3)
    return x, d["t2m"][..., None], d["target"][..., None]  # (B, H, W, 1) each


def train(ds: InSituDataset, *, epochs: int) -> tuple[float, float]:
    """Train the CNN; return ``(model_rmse, persistence_rmse)`` -- 24 h forecast skill on val."""
    model = build_model()
    opt = keras.optimizers.Adam(1e-3)
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        last = 0.0
        for batch in ds.train:
            x, persistence, target = _channels(batch)
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean((persistence + model(x, training=True) - target) ** 2)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables, strict=True))
            last = float(loss)
        print(f"epoch {epoch}  train mse {last:.4f}")

    def predict(batch: Batch) -> np.ndarray:  # (B, 1, H, W) to match the shared eval
        x, persistence, _ = _channels(batch)
        return (persistence + model(x)).numpy().transpose(0, 3, 1, 2)

    return evaluate(ds.val, predict)


def main() -> None:
    args = cli()
    ds = build_datasets(args)
    model_rmse, persistence_rmse = train(ds, epochs=args.epochs)
    print(
        f"\n24 h forecast RMSE on held-out data: model {model_rmse:.3f}  vs  "
        f"persistence {persistence_rmse:.3f}  ({persistence_rmse / model_rmse:.1f}x better)"
    )


if __name__ == "__main__":
    main()
