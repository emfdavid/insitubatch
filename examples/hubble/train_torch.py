"""Train a small denoiser on real Hubble frames, streamed in place by insitubatch.

Only this file touches torch: ``to_torch`` is a zero-copy DLPack hand-off, and the loop moves
tensors to the device itself. The data layer (``data.py``) is framework-free numpy.

Run (builds the reference store on first use)::

    python -m examples.hubble.train_torch --build --epochs 5

The store indexes real WFC3/IR frames of M16 (the Eagle Nebula) on MAST's public S3 bucket
as virtual references -- no pixels are resharded or downloaded ahead of time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from insitubatch import InSituDataset, to_torch

from .data import build_datasets, cli, median_baseline, psnr


class Denoiser(nn.Module):
    """A tiny residual CNN (DnCNN-lite): predict the noise, subtract it from the input."""

    def __init__(self, channels: int = 32, depth: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.net(x)  # residual denoising


def evaluate(model: nn.Module, ds: InSituDataset, device: str = "cpu") -> float:
    """Mean PSNR (dB) of the model's denoised output vs the clean frame, over the val split."""
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in ds.val:
            d = to_torch(batch)
            pred = model(d["noisy"].to(device)).cpu().numpy()
            scores.append(psnr(pred, batch.arrays["clean"]))
    return float(np.mean(scores)) if scores else float("nan")


def baseline_psnr(ds: InSituDataset) -> float:
    """Mean PSNR (dB) of the no-training median-filter baseline over the val split."""
    scores = [psnr(median_baseline(b.arrays["noisy"]), b.arrays["clean"]) for b in ds.val]
    return float(np.mean(scores)) if scores else float("nan")


def train(ds: InSituDataset, *, epochs: int, device: str = "cpu") -> nn.Module:
    model = Denoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        model.train()
        total, n = 0.0, 0
        for batch in ds.train:
            d = to_torch(batch)
            noisy, clean = d["noisy"].to(device), d["clean"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(noisy), clean)
            loss.backward()
            opt.step()
            total += loss.item() * noisy.shape[0]
            n += noisy.shape[0]
        val = evaluate(model, ds, device)
        print(f"epoch {epoch}: train_mse {total / max(n, 1):.4f}  val_psnr {val:.2f} dB")
    return model


def main(argv: list[str] | None = None) -> None:
    args = cli(argv)
    ds = build_datasets(args)

    base = baseline_psnr(ds)
    print(f"median-filter baseline val PSNR: {base:.2f} dB (no training)\n")
    model = train(ds, epochs=args.epochs)
    print(f"\ntrained denoiser   val PSNR: {evaluate(model, ds):.2f} dB")
    print(f"median baseline    val PSNR: {base:.2f} dB")
    ds.close()


if __name__ == "__main__":
    main()
