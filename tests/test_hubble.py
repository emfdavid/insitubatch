"""Hubble FITS denoising example: offline synthetic frames guard the stream+train path.

The real example indexes Hubble ``_flt.fits`` on S3 as virtual references, which needs the
build-time stack (VirtualiZarr/kerchunk/astropy) and the network. These tests instead drive a
synthetic ``SCI (frame, y, x)`` zarr with the *same* geometry (one frame per chunk over
``sample_axis=0``), so the insitubatch-facing wiring -- the NaN-scrubbing chunk stage, the
per-sample-noise batch stage, the ``(noisy, clean)`` labels, and the torch loop -- can't drift
silently when insitubatch changes. No network, no FITS stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.hubble.data import (
    SCI_VAR,
    denoise_dataset,
    make_synthetic_store,
    median_baseline,
    psnr,
)
from insitubatch import obstore_store, open_geometries


@pytest.fixture
def synth_store(tmp_path) -> str:
    """A small synthetic Hubble-like store: 16 frames, NaN bad-pixels, wide dynamic range."""
    url = f"file://{tmp_path}/hubble.zarr"
    make_synthetic_store(url, n_frames=16, size=128, seed=0)
    return url


def test_one_frame_per_chunk(synth_store) -> None:
    # The FITS-derived geometry: each frame is one chunk along the sample axis.
    geom = open_geometries(obstore_store(synth_store), variables=[SCI_VAR], sample_axis=0)[SCI_VAR]
    assert geom.n_samples == 16
    assert geom.sample_chunk_size == 1


def test_denoise_batch_shapes_and_scrubbing(synth_store) -> None:
    ds = denoise_dataset(obstore_store(synth_store), batch_size=4, shuffle=False, coarsen=4)
    ds.set_epoch(0)
    batch = next(iter(ds.train))

    assert set(batch.arrays) == {"noisy", "clean"}
    assert batch.arrays["clean"].shape[1:] == (1, 32, 32)  # 128 // coarsen=4, channel axis added
    assert batch.arrays["noisy"].shape == batch.arrays["clean"].shape
    # clean_normalize scrubs the NaN bad-pixels and clips to the robust-standardized range.
    assert np.isfinite(batch.arrays["clean"]).all()
    assert np.isfinite(batch.arrays["noisy"]).all()
    assert batch.arrays["clean"].min() >= -5.0 and batch.arrays["clean"].max() <= 5.0
    # AddNoise actually perturbs the frame (the batch stage ran).
    assert not np.allclose(batch.arrays["noisy"], batch.arrays["clean"])


def test_median_baseline_denoises(synth_store) -> None:
    # The no-training reference must at least remove noise: higher PSNR than the noisy input.
    ds = denoise_dataset(obstore_store(synth_store), batch_size=8, shuffle=False)
    ds.set_epoch(0)
    batch = next(iter(ds.train))
    noisy, clean = batch.arrays["noisy"], batch.arrays["clean"]
    assert psnr(median_baseline(noisy), clean) > psnr(noisy, clean)


def test_torch_beats_baseline(synth_store) -> None:
    pytest.importorskip("torch")
    from examples.hubble.train_torch import baseline_psnr, evaluate, train

    ds = denoise_dataset(obstore_store(synth_store), batch_size=4)
    base = baseline_psnr(ds)
    model = train(ds, epochs=20)
    assert evaluate(model, ds) > base
