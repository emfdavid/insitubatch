"""OME-NGFF segmentation example: per-variable-chunked, over-Z sampling + the model learns.

One fixture builds a small synthetic cell stack whose two variables are chunked *differently*
along the Z (sample) axis -- ``raw`` one plane deep, ``mask`` many planes deep. The tests
assert the engine co-batches them off one sample grid with no reshard, and that the torch
model beats the Otsu (global-threshold) baseline it can only beat by reading spatial context.
"""

from __future__ import annotations

import pytest

from examples.microscopy.data import (
    IDR_MASK,
    IDR_RAW,
    SAMPLE_AXIS,
    inputs_and_targets,
    iou,
    make_cells_store,
    otsu_foreground,
    segmentation_dataset,
)
from insitubatch import obstore_store, open_geometries

MASK_CHUNK = 12


@pytest.fixture
def cells_store(tmp_path) -> str:
    """A small two-channel cell stack (fast: shallow, small planes)."""
    url = f"file://{tmp_path}/cells.zarr"
    make_cells_store(url, n_planes=48, size=48, mask_chunk=MASK_CHUNK, seed=0)
    return url


def test_dataset_is_per_variable_chunked_over_z(cells_store) -> None:
    geoms = open_geometries(
        obstore_store(cells_store), variables=[IDR_RAW, IDR_MASK], sample_axis=SAMPLE_AXIS
    )
    # Same Z (sample) length, different chunking along it -- the per-variable-chunking case.
    assert geoms[IDR_RAW].n_samples == geoms[IDR_MASK].n_samples == 48
    assert geoms[IDR_RAW].sample_chunk_size == 1
    assert geoms[IDR_MASK].sample_chunk_size == MASK_CHUNK

    ds = segmentation_dataset(obstore_store(cells_store), batch_size=8, shuffle=False)
    ds.set_epoch(0)
    batch = next(iter(ds.train))

    assert set(batch.arrays) == {"raw", "mask"}
    assert batch.arrays["raw"].shape[1:] == (1, 2, 48, 48)  # (T=1, C=2, Y, X) carried whole
    assert batch.arrays["mask"].shape[1:] == (1, 1, 48, 48)
    x, target = inputs_and_targets(batch)
    assert x.shape[1] == 2  # two input channels (T squeezed)
    assert target.shape[1] == 1
    assert set(target.ravel().tolist()) <= {0.0, 1.0}  # binarized foreground


def test_otsu_baseline_is_beatable(cells_store) -> None:
    # The synthetic haze defeats a global threshold, so Otsu leaves real headroom to beat.
    ds = segmentation_dataset(obstore_store(cells_store), batch_size=8, shuffle=False)
    ds.set_epoch(0)
    batch = next(iter(ds.train))
    _x, target = inputs_and_targets(batch)
    assert iou(otsu_foreground(batch), target) < 0.9


def test_torch_beats_otsu(cells_store) -> None:
    pytest.importorskip("torch")
    from examples.microscopy.train_torch import train

    model_iou, otsu_iou = train(segmentation_dataset(obstore_store(cells_store)), epochs=12)
    assert model_iou > otsu_iou
