"""Train a small autoencoder on real SDSS spectra, streamed in place by insitubatch.

Only this file touches torch: ``to_torch`` is a zero-copy DLPack hand-off, and the loop moves
tensors to the device itself. The data layer (``data.py``) is framework-free numpy.

Run (builds the reference store on first use)::

    python -m examples.sdss.train_torch --source sdss --build --epochs 15

The store indexes real SDSS ``spPlate`` frames on ``data.sdss.org`` as virtual references -- no
flux is resampled or downloaded ahead of time. The task mirrors astroML's spectral-PCA
reconstruction; the streamed autoencoder is compared against a PCA baseline at the same latent dim.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from insitubatch import InSituDataset, to_torch

from .data import (
    LATENT_DIM,
    build_datasets,
    cli,
    collect,
    fit_pca,
    pca_reconstruct,
    recon_mse,
)


class AutoEncoder(nn.Module):
    """A small 1-D convolutional autoencoder: compress a spectrum to ``latent_dim``, reconstruct it.

    Convolutional on purpose: galaxy spectra are translation-structured (varying redshift shifts the
    lines), which a *linear* PCA of a fixed latent dim reconstructs poorly -- a moving line is not a
    low-rank combination of fixed templates. The conv encoder learns the shift, so at the same
    bottleneck it beats the PCA baseline (as on real spectra).
    """

    def __init__(self, n_wave: int, latent_dim: int = LATENT_DIM, channels: int = 16) -> None:
        super().__init__()
        self.n_wave = n_wave
        self.padded = ((n_wave + 3) // 4) * 4  # conv path halves length twice
        bottleneck_len = self.padded // 4
        c = channels
        self.enc_conv = nn.Sequential(
            nn.Conv1d(1, c, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(c, 2 * c, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.enc_lin = nn.Linear(2 * c * bottleneck_len, latent_dim)
        self.dec_lin = nn.Linear(latent_dim, 2 * c * bottleneck_len)
        self.unflatten = nn.Unflatten(1, (2 * c, bottleneck_len))
        self.dec_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(2 * c, c, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(c, 1, 7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = F.pad(x, (0, self.padded - self.n_wave)).unsqueeze(1)  # (B, 1, padded)
        z = self.enc_lin(self.enc_conv(xp).flatten(1))
        out = self.dec_conv(self.unflatten(self.dec_lin(z))).squeeze(1)  # (B, padded)
        return out[:, : self.n_wave]


def evaluate(model: nn.Module, ds: InSituDataset, device: str = "cpu") -> float:
    """Mean reconstruction MSE of the model over the val split (lower is better)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in ds.val:
            d = to_torch(batch)
            pred = model(d["noisy"].to(device))
            clean = d["clean"].to(device)
            total += float(((pred - clean) ** 2).mean()) * clean.shape[0]
            n += clean.shape[0]
    return total / max(n, 1)


def pca_baseline_mse(ds: InSituDataset, latent_dim: int = LATENT_DIM) -> float:
    """Mean reconstruction MSE of the PCA baseline (fit on train, scored on val).

    Like astroML's ``spec4000`` workflow, PCA needs every training spectrum resident to fit -- the
    opposite of the streamed autoencoder. That contrast is the point of the example.
    """
    _, train_clean = collect(ds, "train")
    val_noisy, val_clean = collect(ds, "val")
    mean, components = fit_pca(train_clean, k=latent_dim)
    return recon_mse(pca_reconstruct(val_noisy, mean, components), val_clean)


def train(ds: InSituDataset, *, n_wave: int, epochs: int, device: str = "cpu") -> nn.Module:
    model = AutoEncoder(n_wave).to(device)
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
        print(f"epoch {epoch}: train_mse {total / max(n, 1):.4f}  val_mse {val:.4f}")
    return model


def spectrum_width(ds: InSituDataset) -> int:
    """Wavelength-bin count, read from the first val batch (the clean spectrum width)."""
    for batch in ds.val:
        return int(batch.arrays["clean"].shape[-1])
    raise RuntimeError("empty dataset: cannot infer spectrum width")


def main(argv: list[str] | None = None) -> None:
    args = cli(argv)
    ds = build_datasets(args)
    n_wave = spectrum_width(ds)

    base = pca_baseline_mse(ds, latent_dim=args.latent_dim)
    print(f"PCA baseline val MSE: {base:.4f} (latent dim {args.latent_dim}, needs all spectra)\n")
    model = train(ds, n_wave=n_wave, epochs=args.epochs)
    print(f"\nautoencoder  val MSE: {evaluate(model, ds):.4f}")
    print(f"PCA baseline val MSE: {base:.4f}")
    ds.close()


if __name__ == "__main__":
    main()
