"""Shape / dtype / in-range-ness tests for the vectorized `_subregion_crop`.

The batched RNG draw consumes the stream in a different order than the old
per-sample loop, so output for a given seed is not bit-identical to the
previous implementation — these tests assert the properties that must hold
instead of recorded values (#29).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from examples.wb2_dataloader import _subregion_crop


def _apply(a: np.ndarray, subregion: tuple[int, int], seed: int) -> np.ndarray:
    batch = SimpleNamespace(arrays={"t2m": a})
    return _subregion_crop("t2m", subregion, seed)(batch).arrays["t2m"]


def _find_window(sample: np.ndarray, crop: np.ndarray) -> tuple[int, int] | None:
    """Locate the (i, j) whose window equals the crop, or None if none does."""
    lat, lon = sample.shape[-2], sample.shape[-1]
    h, w = crop.shape[-2], crop.shape[-1]
    for i in range(lat - h + 1):
        for j in range(lon - w + 1):
            if np.array_equal(sample[..., i : i + h, j : j + w], crop):
                return (i, j)
    return None


def test_full_size_crop_is_identity_3d():
    a = np.random.default_rng(1).standard_normal((4, 6, 8))
    out = _apply(a, (6, 8), seed=0)
    assert out.shape == (4, 6, 8)
    assert np.array_equal(out, a)


def test_shape_and_dtype_preserved_with_middle_dims():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((5, 3, 7, 9)).astype("float32")  # (B, T, LAT, LON)
    out = _apply(a, (4, 5), seed=3)
    assert out.shape == (5, 3, 4, 5)
    assert out.dtype == a.dtype


def test_each_crop_is_a_real_contiguous_window_of_its_own_sample():
    # Distinct values per sample expose any cross-sample bleed: sample b is
    # filled with the constant b, so a crop that mixed samples would show
    # more than one value in a slice.
    rng = np.random.default_rng(4)
    a = rng.standard_normal((6, 2, 10, 12))
    out = _apply(a, (5, 7), seed=5)
    for b in range(a.shape[0]):
        window = _find_window(a[b], out[b])
        assert window is not None, f"sample {b}: crop is not a window of its own sample"


def test_per_sample_offsets_are_independent():
    # Constant-fill samples: every value in out[b] equals b, and at least two
    # samples land on different windows when the domain is large enough for
    # the seeds to differ.
    a = np.zeros((8, 10, 10))
    for b in range(a.shape[0]):
        a[b] = b
    out = _apply(a, (3, 3), seed=6)
    for b in range(a.shape[0]):
        assert np.all(out[b] == b), f"sample {b} contains values from another sample"


def test_same_seed_is_reproducible_and_different_seed_differs():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((4, 9, 9))
    out_a1 = _apply(a, (4, 4), seed=42)
    out_a2 = _apply(a, (4, 4), seed=42)
    out_b = _apply(a, (4, 4), seed=43)
    assert np.array_equal(out_a1, out_a2)
    assert not np.array_equal(out_a1, out_b)
