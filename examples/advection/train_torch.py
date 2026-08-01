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

from insitubatch import InSituDataset, to_torch

from .data import build_datasets, cli, evaluate, format_skill


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


def _forecast(model: AdvectionCNN, d: dict[str, torch.Tensor]) -> torch.Tensor:
    """t2m(t+24h) forecast for one already-transferred batch: persistence + predicted tendency.

    Takes **device tensors, not a** ``Batch``: the H2D copy belongs to the caller, once per
    step. Passing the batch in and converting here would move every variable a second time,
    since the loop also needs ``target`` -- twice the traffic the step actually requires, which
    is exactly the kind of overhead a pinning measurement would then misattribute to pinning.

    Stacking happens **after** the transfer, on the device: stacking first would build a fresh
    (B, 3, H, W) CPU tensor -- a copy of the whole batch into memory the loader does not own,
    which is both wasted work and unpinnable, so it would defeat page-locked buffers for the
    one tensor that matters most.

    That trade is not free, and the cost is device memory. Counting a (B, H, W) variable as one
    unit: stacking on the host holds 5 units on the GPU (the stack, plus t2m and target moved
    separately) and copies 5 across PCIe; stacking here holds **7** (four transferred variables
    plus the stack) and copies 4. So it is +2 units of GPU memory for one fewer variable-copy
    per step -- 224 MiB vs 160 MiB at a 32 MiB payload, against conv activations that are
    larger still. A loop tight on device memory would instead transfer straight into the slices
    of a preallocated ``x``, which needs an ``out=``-style device target this adapter does not
    have yet (it would also have to hold the source until the copy lands, which is exactly what
    ``to_torch(..., device=...)`` does for us here).
    """
    x = torch.stack([d["t2m"], d["u10"], d["v10"]], dim=1)  # (B, 3, H, W), on device
    return d["t2m"][:, None] + model(x)  # (B, 1, H, W)


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
            d = to_torch(batch, device=dev)  # the step's one H2D transfer, all variables
            loss = nn.functional.mse_loss(_forecast(model, d), d["target"][:, None])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        print(f"epoch {epoch}  train mse {last:.4f}")
    model.eval()
    with torch.no_grad():
        return evaluate(
            ds.val, lambda b: _forecast(model, to_torch(b, device=dev)).detach().cpu().numpy()
        )


def main() -> None:
    args = cli()
    ds = build_datasets(args)
    model_rmse, persistence_rmse = train(ds, epochs=args.epochs, device=args.device)
    print(format_skill(model_rmse, persistence_rmse))


if __name__ == "__main__":
    main()
