"""Forecast t2m 24 h ahead with a tiny CNN, in **PyTorch**, on one insitu dataset.

    python -m examples.advection.train_torch          # synthetic advected field (offline)
    python -m examples.advection.train_torch --wb2    # WeatherBench2 (real ERA5, gs://)

The data layer (``examples/advection/data.py``) is framework-neutral -- a numpy ``Batch``
read in place from the store, inputs ``{t2m, u10, v10}@t`` and target ``t2m@(t+24h)`` as
offset views of the same arrays, no reshard. Only this file is torch: ``to_torch`` is a
zero-copy DLPack hand-off. ``train_jax.py`` / ``train_tf.py`` train the *same* model on
the *same* dataset. The model learns the **tendency** (the change on top of persistence),
so beating the persistence baseline means it learned the wind-driven advection.
"""

from __future__ import annotations

import torch
from torch import nn

from insitubatch.frameworks import to_torch
from insitubatch.source import InSituDataset
from insitubatch.types import Batch

from .data import build_datasets, cli, evaluate


class AdvectionCNN(nn.Module):
    """3 input channels (t2m, u10, v10) -> 1 channel tendency. Inputs are standardized per
    channel; the forecast is ``persistence + tendency`` (predict the change, not the field).
    Four 3x3 circular convolutions -- receptive field 9 -- cover the ~5-cell displacement."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()

        def conv(i: int, o: int) -> nn.Conv2d:
            return nn.Conv2d(i, o, 3, padding=1, padding_mode="circular")

        self.net = nn.Sequential(
            conv(3, hidden),
            nn.ReLU(),
            conv(hidden, hidden),
            nn.ReLU(),
            conv(hidden, hidden),
            nn.ReLU(),
            conv(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 3, H, W) -> tendency (B, 1, H, W)
        xn = (x - x.mean((0, 2, 3), keepdim=True)) / (x.std((0, 2, 3), keepdim=True) + 1e-6)
        return self.net(xn)


def _forecast(model: AdvectionCNN, batch: Batch) -> torch.Tensor:
    """t2m(t+24h) forecast for one batch: persistence + the model's predicted tendency."""
    d = to_torch(batch)  # {label: (B, H, W) tensor}, zero-copy via DLPack
    x = torch.stack([d["t2m"], d["u10"], d["v10"]], dim=1)  # (B, 3, H, W)
    return d["t2m"][:, None] + model(x)  # (B, 1, H, W)


def train(train_ds: InSituDataset, val_ds: InSituDataset, *, epochs: int) -> tuple[float, float]:
    """Train the CNN; return ``(model_rmse, persistence_rmse)`` -- 24 h forecast skill on val."""
    model = AdvectionCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        train_ds.set_epoch(epoch)
        model.train()
        last = 0.0
        for batch in train_ds:
            target = to_torch(batch)["target"][:, None]
            loss = nn.functional.mse_loss(_forecast(model, batch), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        print(f"epoch {epoch}  train mse {last:.4f}")
    model.eval()
    with torch.no_grad():
        return evaluate(val_ds, lambda b: _forecast(model, b).detach().numpy())


def main() -> None:
    args = cli()
    train_ds, val_ds = build_datasets(args)
    model_rmse, persistence_rmse = train(train_ds, val_ds, epochs=args.epochs)
    print(
        f"\n24 h forecast RMSE on held-out data: model {model_rmse:.3f}  vs  "
        f"persistence {persistence_rmse:.3f}  ({persistence_rmse / model_rmse:.1f}x better)"
    )


if __name__ == "__main__":
    main()
