"""Downscale solar irradiance with a tiny CNN, in **PyTorch**, on one insitu dataset.

Only this file is torch: ``to_torch`` is a zero-copy DLPack hand-off and the train loop moves
each batch to ``--device``. The model is handed the bilinearly upsampled coarse field and
predicts a *residual*, so beating the bilinear baseline means it recovered sub-grid structure
the interpolation cannot -- the cloud edges the block average washed out.

Sources (``--source synthetic|gfs``), the finite time window (``--sample-range``), the
residency knobs and GPU placement are documented in ``examples/README.md``; the framework-
neutral data layer is ``examples/dynamical/data.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from insitubatch import Batch, InSituDataset

from .data import build_datasets, cli, evaluate, inputs_and_targets


class ResidualCNN(nn.Module):
    """1 input channel -> 1 residual channel. The input is standardized, four 3x3 convolutions
    (circular padding in longitude, replicate in latitude -- the globe wraps east-west but not
    north-south) give a receptive field of 9, and the output is added back to the interpolated
    field, so the model only has to learn what bilinear upsampling left out."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 3, padding=1, padding_mode="replicate"),
        )
        self.scale = 300.0  # W/m^2: keeps the standardized residual near unit magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W) -> (B, 1, H, W) W/m^2
        xn = (x - x.mean((0, 2, 3), keepdim=True)) / (x.std((0, 2, 3), keepdim=True) + 1e-6)
        return x + self.scale * self.net(xn)


def _predict(model: ResidualCNN, batch: Batch, device: torch.device) -> torch.Tensor:
    """The model's field for one batch, in W/m^2.

    The crop and the block average happen in numpy (``inputs_and_targets``) because they define
    the task, not the model; placement on ``device`` is the loop's job, not the dataset's.
    """
    x, _target = inputs_and_targets(batch)
    return model(torch.from_numpy(x).unsqueeze(1).to(device))


def train(ds: InSituDataset, *, epochs: int, device: str = "cpu") -> tuple[float, float]:
    """Train the CNN; return ``(model_rmse, bilinear_rmse)`` in W/m^2 on val."""
    dev = torch.device(device)
    model = ResidualCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        model.train()
        last = 0.0
        for batch in ds.train:
            _x, target = inputs_and_targets(batch)
            y = torch.from_numpy(target).unsqueeze(1).to(dev)  # (B, 1, H, W) W/m^2
            loss = nn.functional.mse_loss(_predict(model, batch, dev), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        print(f"epoch {epoch}  train mse {last:9.1f}  (rmse {np.sqrt(last):6.1f} W/m2)")
    model.eval()
    with torch.no_grad():
        return evaluate(ds.val, lambda b: _predict(model, b, dev).squeeze(1).cpu().numpy())


def main() -> None:
    args = cli()
    ds = build_datasets(args)
    model_rmse, bilinear_rmse = train(ds, epochs=args.epochs, device=args.device)
    print(
        f"\ndownscaling RMSE on held-out data: model {model_rmse:.2f}  vs  "
        f"bilinear {bilinear_rmse:.2f} W/m2  ({model_rmse - bilinear_rmse:+.2f})"
    )


if __name__ == "__main__":
    main()
