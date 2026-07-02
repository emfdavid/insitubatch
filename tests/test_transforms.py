"""Transforms: StandardScaler and the two CPU stages (chunk + batch)."""

from __future__ import annotations

import numpy as np
import zarr

from insitubatch import (
    SplitName,
    StandardScaler,
    ensure_local_dir,
    obstore_store,
    open_geometries,
    split_by_chunk,
)
from insitubatch.source import InSituDataset
from insitubatch.types import ChunkRead, DecodedChunk


def _write(tmp_path, *, n=80, spc=8, inner=(3, 4), seed=0) -> tuple[str, np.ndarray]:
    url = f"file://{tmp_path}/d.zarr"
    ensure_local_dir(url)
    store = obstore_store(url, read_only=False)
    group = zarr.open_group(store=store, mode="w")
    arr = group.create_array("t2m", shape=(n, *inner), chunks=(spc, *inner), dtype="f4")
    src = (np.random.default_rng(seed).standard_normal((n, *inner)) * 5.0 + 10.0).astype("f4")
    arr[:] = src
    return url, src


# -- StandardScaler ---------------------------------------------------------


def test_standard_scaler_applies_known_stats() -> None:
    data = np.arange(2 * 3 * 4, dtype="f4").reshape(2, 3, 4)
    chunk = DecodedChunk(read=ChunkRead("t2m", 0), data=data.copy(), sample_offset=0)
    sc = StandardScaler(mean={"t2m": np.float64(10.0)}, std={"t2m": np.float64(2.0)})
    out = sc(chunk)
    np.testing.assert_allclose(out.data, (data - 10.0) / (2.0 + 1e-8), rtol=1e-5)


def test_standard_scaler_save_load(tmp_path) -> None:
    sc = StandardScaler(mean={"t2m": np.zeros((3, 1, 1))}, std={"t2m": np.ones((3, 1, 1))})
    p = tmp_path / "sc.npz"
    sc.save(p)
    back = StandardScaler.load(p)
    np.testing.assert_array_equal(back.mean["t2m"], sc.mean["t2m"])
    np.testing.assert_array_equal(back.std["t2m"], sc.std["t2m"])


# -- end-to-end through InSituDataset --------------------------------------


def test_chunk_transform_normalizes_batches(tmp_path) -> None:
    url, src = _write(tmp_path)
    geoms = open_geometries(obstore_store(url))
    manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
    # Pre-fit global stats over the train split (as a user would, or via the
    # partial_fit-over-loader pattern), then apply at the chunk stage.
    train = src[manifest.sample_indices(SplitName.TRAIN, geoms["t2m"])].astype("f8")
    m, s = train.mean(), train.std()
    sc = StandardScaler(mean={"t2m": np.float64(m)}, std={"t2m": np.float64(s)})

    ds = InSituDataset(
        obstore_store(url),
        manifest,
        batch_size=10,
        block_chunks=4,
        chunk_transforms=[sc],
    )
    ds.set_epoch(0)
    for batch in ds.train:
        idx = batch.sample_indices
        np.testing.assert_allclose(
            batch.arrays["t2m"], (src[idx] - m) / (s + 1e-8), rtol=1e-4, atol=1e-4
        )


def test_reshaping_chunk_transform_through_dataset(tmp_path) -> None:
    """A reshaping chunk_transform (mean over the last inner axis, (3,4)->(3,)) flows
    end-to-end through InSituDataset: every batch carries the post-transform geometry."""
    url, src = _write(tmp_path, inner=(3, 4))
    geoms = open_geometries(obstore_store(url))
    manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))

    class MeanLastAxis:
        def __call__(self, chunk):
            chunk.data = chunk.data.mean(axis=-1)  # (n,3,4) -> (n,3)
            return chunk

        def output_inner(self, geom):
            return geom.inner_shape[:-1], geom.dtype

    ds = InSituDataset(
        obstore_store(url),
        manifest,
        batch_size=10,
        block_chunks=4,
        chunk_transforms=[MeanLastAxis()],
    )
    ds.set_epoch(0)
    for batch in ds.train:
        idx = batch.sample_indices
        assert batch.arrays["t2m"].shape == (len(idx), 3)  # reshaped inner
        np.testing.assert_allclose(batch.arrays["t2m"], src[idx].mean(axis=-1), rtol=1e-5)


def test_cross_variable_windspeed_is_a_batch_transform(tmp_path) -> None:
    # Validates the documented capability: U10,V10 -> windspeed lives at the BATCH
    # stage (the chunk stage sees one variable at a time).
    url = f"file://{tmp_path}/uv.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    rng = np.random.default_rng(0)
    su = rng.standard_normal((40, 2, 2)).astype("f4")
    sv = rng.standard_normal((40, 2, 2)).astype("f4")
    group.create_array("u10", shape=(40, 2, 2), chunks=(8, 2, 2), dtype="f4")[:] = su
    group.create_array("v10", shape=(40, 2, 2), chunks=(8, 2, 2), dtype="f4")[:] = sv

    geoms = open_geometries(obstore_store(url))
    manifest = split_by_chunk(geoms["u10"], fractions=(0.8, 0.1, 0.1))

    def windspeed(batch):
        batch.arrays["wspd"] = np.sqrt(batch.arrays["u10"] ** 2 + batch.arrays["v10"] ** 2)
        return batch

    ds = InSituDataset(
        obstore_store(url),
        manifest,
        batch_size=8,
        block_chunks=4,
        batch_transforms=[windspeed],
    )
    ds.set_epoch(0)
    for batch in ds.train:
        idx = batch.sample_indices
        expected = np.sqrt(su[idx] ** 2 + sv[idx] ** 2)
        np.testing.assert_allclose(batch.arrays["wspd"], expected, rtol=1e-5)
