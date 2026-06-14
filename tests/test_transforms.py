"""Transforms: StandardScaler, fit-with-our-own-infra, and the two CPU stages."""

from __future__ import annotations

import numpy as np
import zarr

from insitubatch import (
    SplitName,
    StandardScaler,
    ensure_local_dir,
    fit_standard_scaler,
    open_geometries,
    split_by_chunk,
    store_from_url,
)
from insitubatch.source import InSituDataset
from insitubatch.types import ChunkRead, DecodedChunk


def _write(tmp_path, *, n=80, spc=8, inner=(3, 4), seed=0) -> tuple[str, np.ndarray]:
    url = f"file://{tmp_path}/d.zarr"
    ensure_local_dir(url)
    store = store_from_url(url, read_only=False)
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


# -- fit_standard_scaler (with our own infra) -------------------------------


def test_fit_recovers_scalar_stats(tmp_path) -> None:
    url, src = _write(tmp_path)
    geoms = open_geometries(url)
    manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
    sc = fit_standard_scaler(url, manifest, geoms)  # keep_axes=() -> scalar per var

    x = src[manifest.sample_indices(SplitName.TRAIN, geoms["t2m"])].astype("f8")
    np.testing.assert_allclose(np.squeeze(sc.mean["t2m"]), x.mean(), rtol=1e-4)
    np.testing.assert_allclose(np.squeeze(sc.std["t2m"]), x.std(), rtol=1e-4)


def test_fit_per_level_stats(tmp_path) -> None:
    url, src = _write(tmp_path, inner=(5, 3, 4))  # (n, level, lat, lon)
    geoms = open_geometries(url)
    manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
    sc = fit_standard_scaler(url, manifest, geoms, keep_axes=(0,))  # keep the level axis

    assert sc.mean["t2m"].shape == (5, 1, 1)
    x = src[manifest.sample_indices(SplitName.TRAIN, geoms["t2m"])].astype("f8")
    np.testing.assert_allclose(np.squeeze(sc.mean["t2m"]), x.mean(axis=(0, 2, 3)), rtol=1e-4)


# -- end-to-end through InSituDataset --------------------------------------


def test_chunk_transform_normalizes_batches(tmp_path) -> None:
    url, src = _write(tmp_path)
    geoms = open_geometries(url)
    manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
    sc = fit_standard_scaler(url, manifest, geoms)
    m, s = np.squeeze(sc.mean["t2m"]), np.squeeze(sc.std["t2m"])

    ds = InSituDataset(url, manifest, split=SplitName.TRAIN, batch_size=10,
                       block_chunks=4, to_tensor=False, chunk_transforms=[sc])
    ds.set_epoch(0)
    for batch in ds:
        idx = batch.sample_indices
        np.testing.assert_allclose(
            batch.arrays["t2m"], (src[idx] - m) / (s + 1e-8), rtol=1e-4, atol=1e-4
        )


def test_cross_variable_windspeed_is_a_batch_transform(tmp_path) -> None:
    # Validates the documented capability: U10,V10 -> windspeed lives at the BATCH
    # stage (the chunk stage sees one variable at a time).
    url = f"file://{tmp_path}/uv.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=store_from_url(url, read_only=False), mode="w")
    rng = np.random.default_rng(0)
    su = rng.standard_normal((40, 2, 2)).astype("f4")
    sv = rng.standard_normal((40, 2, 2)).astype("f4")
    group.create_array("u10", shape=(40, 2, 2), chunks=(8, 2, 2), dtype="f4")[:] = su
    group.create_array("v10", shape=(40, 2, 2), chunks=(8, 2, 2), dtype="f4")[:] = sv

    geoms = open_geometries(url)
    manifest = split_by_chunk(geoms["u10"], fractions=(0.8, 0.1, 0.1))

    def windspeed(batch):
        batch.arrays["wspd"] = np.sqrt(batch.arrays["u10"] ** 2 + batch.arrays["v10"] ** 2)
        return batch

    ds = InSituDataset(url, manifest, split=SplitName.TRAIN, batch_size=8,
                       block_chunks=4, to_tensor=False, batch_transforms=[windspeed])
    ds.set_epoch(0)
    for batch in ds:
        idx = batch.sample_indices
        expected = np.sqrt(su[idx] ** 2 + sv[idx] ** 2)
        np.testing.assert_allclose(batch.arrays["wspd"], expected, rtol=1e-5)
