"""Forecast t2m 24 h ahead with a tiny CNN, in **PyTorch**, on one insitu dataset.

Only this file is torch: ``to_torch`` is a zero-copy DLPack hand-off and the train loop
moves each batch to ``--device``. ``train_jax.py`` / ``train_tf.py`` train the *same* model
on the *same* numpy ``Batch``. The model learns the **tendency** (the change on top of
persistence), so beating persistence means it read the wind-driven advection.

Sources (``--source``), the finite training window (``--sample-range``), GPU placement and
the NVMe cache flags are documented in ``examples/README.md``; the framework-neutral data
layer is ``examples/advection/data.py``. For the loader stall / in-memory-ceiling benchmark
built on top of this loop, see ``train_torch_metrics.py`` -- kept separate so this file
stays a clean usage example.
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


def _forecast(model: AdvectionCNN, batch: Batch, device: torch.device) -> torch.Tensor:
    """t2m(t+24h) forecast for one batch: persistence + the model's predicted tendency.

    ``to_torch`` is a zero-copy DLPack hand-off on CPU; moving to ``device`` is the H2D
    copy -- placement is the training loop's job, not the dataset's (it speaks numpy)."""
    d = to_torch(batch)  # {label: (B, H, W) tensor}, CPU via DLPack
    x = torch.stack([d["t2m"], d["u10"], d["v10"]], dim=1).to(device)  # (B, 3, H, W)
    return d["t2m"][:, None].to(device) + model(x)  # (B, 1, H, W)


def train(ds: InSituDataset, *, epochs: int, device: str = "cpu") -> tuple[float, float]:
    """Train the CNN; return ``(model_rmse, persistence_rmse)`` -- 24 h forecast skill on val."""
    dev = torch.device(device)
    model = AdvectionCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        model.train()
        last = 0.0
        for batch in ds.train:
            target = to_torch(batch)["target"][:, None].to(dev)
            loss = nn.functional.mse_loss(_forecast(model, batch, dev), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        print(f"epoch {epoch}  train mse {last:.4f}")
    model.eval()
    with torch.no_grad():
        return evaluate(ds.val, lambda b: _forecast(model, b, dev).detach().cpu().numpy())


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
