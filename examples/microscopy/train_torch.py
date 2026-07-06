"""Segment cells with a tiny CNN, in **PyTorch**, on one insitu dataset.

Only this file is torch: ``to_torch`` is a zero-copy DLPack hand-off and the train loop moves
each batch to ``--device``. The model reads the 2-channel image plus its spatial neighborhood,
so beating the Otsu (global-threshold) baseline means it used context the threshold cannot see
-- the haze that overlaps cell and background intensity ranges. The dataset stays numpy; the
two variables (``raw`` Z-chunk 1, ``mask`` Z-chunk 30) are co-batched over Z with no reshard.

Sources (``--source synthetic|idr``), the finite Z window (``--sample-range``), GPU placement
and the NVMe cache flags are documented in ``examples/README.md``; the framework-neutral data
layer is ``examples/microscopy/data.py``.
"""

from __future__ import annotations

import torch
from torch import nn

from insitubatch import Batch, InSituDataset, to_torch

from .data import build_datasets, cli, evaluate, inputs_and_targets


class SegCNN(nn.Module):
    """2 input channels -> 1 foreground logit. Inputs are standardized per channel; four 3x3
    convolutions (reflect-padded -- microscopy is not periodic) give a receptive field of 9,
    enough to tell a sharp cell from the smooth haze a per-pixel threshold cannot."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()

        def conv(i: int, o: int) -> nn.Conv2d:
            return nn.Conv2d(i, o, 3, padding=1, padding_mode="reflect")

        self.net = nn.Sequential(
            conv(2, hidden),
            nn.ReLU(),
            conv(hidden, hidden),
            nn.ReLU(),
            conv(hidden, hidden),
            nn.ReLU(),
            conv(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 2, H, W) -> logits (B, 1, H, W)
        xn = (x - x.mean((0, 2, 3), keepdim=True)) / (x.std((0, 2, 3), keepdim=True) + 1e-6)
        return self.net(xn)


def _logits(model: SegCNN, batch: Batch, device: torch.device) -> torch.Tensor:
    """Foreground logits for one batch. ``to_torch`` is a zero-copy DLPack hand-off on CPU;
    moving to ``device`` is the H2D copy -- placement is the loop's job, not the dataset's."""
    d = to_torch(batch)  # {label: (B, T, C, H, W) tensor}, CPU via DLPack
    x = d["raw"][:, 0].float().to(device)  # (B, 2, H, W); float (real raw is uint16), drop T
    return model(x)


def train(ds: InSituDataset, *, epochs: int, device: str = "cpu") -> tuple[float, float]:
    """Train the CNN; return ``(model_iou, otsu_iou)`` -- foreground IoU on val."""
    dev = torch.device(device)
    model = SegCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        model.train()
        last = 0.0
        for batch in ds.train:
            _x, target = inputs_and_targets(batch)
            y = torch.from_numpy(target).to(dev)  # (B, 1, H, W) in {0, 1}
            loss = nn.functional.binary_cross_entropy_with_logits(_logits(model, batch, dev), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        print(f"epoch {epoch}  train bce {last:.4f}")
    model.eval()
    with torch.no_grad():
        return evaluate(ds.val, lambda b: torch.sigmoid(_logits(model, b, dev)).cpu().numpy())


def main() -> None:
    args = cli()
    ds = build_datasets(args)
    model_iou, otsu_iou = train(ds, epochs=args.epochs, device=args.device)
    print(
        f"\nforeground IoU on held-out data: model {model_iou:.3f}  vs  "
        f"Otsu threshold {otsu_iou:.3f}  ({model_iou - otsu_iou:+.3f})"
    )


if __name__ == "__main__":
    main()
