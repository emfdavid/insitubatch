"""SDSS spectral-reconstruction example: offline synthetic spectra guard the stream+train path.

The real example indexes an SDSS ``spPlate`` FITS over HTTPS as virtual references (build-time
VirtualiZarr/kerchunk/astropy + network). These tests instead drive a synthetic ``flux
(fiber, wave)`` zarr with the *same* geometry (many fibers per chunk over ``sample_axis=0``), so
the insitubatch-facing wiring -- the NaN-scrubbing chunk stage, the per-sample-noise batch stage,
the ``(noisy, clean)`` labels, the PCA baseline, and the torch loop -- can't drift silently when
insitubatch changes. No network, no FITS stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.sdss.data import (
    FLUX_VAR,
    PUBLIC_STORES,
    fit_pca,
    make_synthetic_store,
    open_published,
    pca_reconstruct,
    recon_mse,
    reconstruct_dataset,
)
from insitubatch import obstore_store, open_geometries


@pytest.fixture
def synth_store(tmp_path) -> str:
    """A small synthetic spPlate-like store: 8 plates x 96 fibers, NaN pixels, redshift lines."""
    url = f"file://{tmp_path}/sdss.zarr"
    make_synthetic_store(url, n_plates=8, fibers_per_plate=96, n_wave=256, seed=0)
    return url


@pytest.fixture
def one_fiber_per_chunk_store(tmp_path) -> str:
    """The *multi-plate* geometry: 128 chunks of a single fiber each.

    ``_many_plates_common_grid`` crops every plate to the shared wavelength window, which
    breaks multi-fiber contiguity, so a fiber is its own virtual chunk. Sized past the
    residency floor on purpose: the smaller fixtures in ``test_sdss_build`` hold fewer
    chunks than the budget, so the budget never fills and cannot be exercised at all.
    """
    url = f"file://{tmp_path}/sdss_1fiber.zarr"
    make_synthetic_store(url, n_plates=128, fibers_per_plate=1, n_wave=64, seed=0)
    return url


def test_multi_plate_geometry_streams_a_full_epoch(one_fiber_per_chunk_store, run_by) -> None:
    # The example's own default batch_size (64) over one-fiber chunks: this is the shipped
    # multi-plate configuration, and it used to wedge -- 64 samples span four 16-chunk
    # shuffle-blocks against a two-block residency floor, so no batch was ever delivered.
    ds = reconstruct_dataset(obstore_store(one_fiber_per_chunk_store), batch_size=64)
    ds.set_epoch(0)
    geom = open_geometries(obstore_store(one_fiber_per_chunk_store), variables=[FLUX_VAR])[FLUX_VAR]
    assert geom.sample_chunk_size == 1  # one fiber per chunk (the streaming regime)

    def epoch() -> list[np.ndarray]:
        return [b.sample_indices.copy() for b in ds.train]

    seen = np.concatenate(run_by(30.0, epoch))
    n_train = len(ds.manifest.chunks["train"])
    assert len(seen) == n_train and len(set(seen.tolist())) == n_train  # each fiber once
    ds.close()


def test_published_store_table_is_well_formed() -> None:
    # Offline guard on the public-bucket coordinates (the suite never hits the network): every
    # entry must name a repo prefix and the URL prefix its virtual chunks resolve against, and
    # an unknown name must fail with the choices rather than a KeyError from inside icechunk.
    assert set(PUBLIC_STORES) == {"1plate", "6plate", "6plate-mirror"}
    for prefix, url_prefix in PUBLIC_STORES.values():
        assert prefix.startswith("astronomy/")
        assert url_prefix.startswith("https://") and url_prefix.endswith("/")
    with pytest.raises(ValueError, match="unknown store"):
        open_published("nope")


def test_many_fibers_per_chunk(synth_store) -> None:
    # The spPlate-derived geometry: one plate is one chunk, and a chunk holds many fiber samples
    # (the decode-amortization regime, unlike Hubble's one image per chunk).
    geoms = open_geometries(obstore_store(synth_store), variables=[FLUX_VAR], sample_axis=0)
    geom = geoms[FLUX_VAR]
    assert geom.n_samples == 8 * 96
    assert geom.sample_chunk_size == 96
    assert geom.n_chunks == 8


def test_reconstruct_batch_shapes_and_scrubbing(synth_store) -> None:
    ds = reconstruct_dataset(obstore_store(synth_store), batch_size=32, shuffle=False)
    ds.set_epoch(0)
    batch = next(iter(ds.train))

    assert set(batch.arrays) == {"noisy", "clean"}
    assert batch.arrays["clean"].shape == (32, 256)
    assert batch.arrays["noisy"].shape == batch.arrays["clean"].shape
    # normalize scrubs the NaN bad-pixels and clips to the robust-standardized range.
    assert np.isfinite(batch.arrays["clean"]).all()
    assert np.isfinite(batch.arrays["noisy"]).all()
    assert batch.arrays["clean"].min() >= -5.0 and batch.arrays["clean"].max() <= 5.0
    # Corrupt actually perturbs the spectrum (the batch stage ran).
    assert not np.allclose(batch.arrays["noisy"], batch.arrays["clean"])


def test_split_is_by_chunk(synth_store) -> None:
    # No fiber leaks across train/val/test: splits partition whole plates (chunks).
    ds = reconstruct_dataset(obstore_store(synth_store), batch_size=64, shuffle=False)
    ds.set_epoch(0)
    seen = {s: set() for s in ("train", "val", "test")}
    for split in seen:
        for b in getattr(ds, split):
            seen[split].update(int(i) for i in b.sample_indices)
    assert seen["train"] and seen["val"] and seen["test"]
    assert not (seen["train"] & seen["val"])
    assert not (seen["train"] & seen["test"])
    assert not (seen["val"] & seen["test"])


def test_pca_baseline_reconstructs(synth_store) -> None:
    # The no-training baseline must beat the trivial mean-spectrum reconstruction.
    ds = reconstruct_dataset(obstore_store(synth_store), batch_size=64, shuffle=False)
    ds.set_epoch(0)
    clean = np.concatenate([b.arrays["clean"] for b in ds.train])
    noisy = np.concatenate([b.arrays["noisy"] for b in ds.train])
    mean, comps = fit_pca(clean, k=16)
    trivial = np.broadcast_to(mean, clean.shape)
    assert recon_mse(pca_reconstruct(noisy, mean, comps), clean) < recon_mse(trivial, clean)


def test_torch_beats_baseline(synth_store) -> None:
    pytest.importorskip("torch")
    from examples.sdss.train_torch import evaluate, pca_baseline_mse, spectrum_width, train

    ds = reconstruct_dataset(obstore_store(synth_store), batch_size=32)
    base = pca_baseline_mse(ds, latent_dim=16)
    model = train(ds, n_wave=spectrum_width(ds), epochs=25)
    # A nonlinear conv autoencoder beats linear PCA at the same latent dim on the redshift manifold.
    assert evaluate(model, ds) < base
