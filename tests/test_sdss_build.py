"""Guard the SDSS spPlate virtual-reference build modes -- offline, with synthetic FITS images.

The real example indexes SDSS ``spPlate`` frames over HTTPS; :func:`build_store` re-chunks them by
rewriting the *chunk manifest* (byte arithmetic), never the pixels. Two modes:

* :func:`_single_plate_fiber_chunks` -- one plate -> contiguous fiber-block chunks (full width).
* :func:`_many_plates_common_grid` -- N plates cropped to a shared wavelength window -> one flat
  fiber axis, one fiber per chunk.

These tests write tiny big-endian FITS images with SDSS-style ``COEFF0``/``COEFF1`` grid headers
(no network), run each mode through Icechunk + insitubatch, and check the delivered spectra are
byte-identical to the source -- so the manifest byte-offset math can't drift. Need the ``astronomy``
extra.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("virtualizarr")
pytest.importorskip("kerchunk")
pytest.importorskip("astropy")
pytest.importorskip("icechunk")

from examples.sdss.data import (  # noqa: E402
    FLUX_VAR,
    _many_plates_common_grid,
    _single_plate_fiber_chunks,
)
from insitubatch import InSituDataset, open_geometries, split_by_chunk  # noqa: E402

DLOGLAM = 1e-4


def _write_plate(
    tmp_dir: str, name: str, *, n_fiber: int, n_wave: int, coeff0: float, seed: int
) -> np.ndarray:
    """Write a synthetic spPlate-like FITS: a big-endian ``(fiber, wave)`` image on a log grid."""
    from astropy.io import fits

    rng = np.random.default_rng(seed)
    data = (10.0 + rng.normal(0.0, 1.0, size=(n_fiber, n_wave))).astype(">f4")
    hdu = fits.PrimaryHDU(data)
    hdu.header["COEFF0"] = coeff0
    hdu.header["COEFF1"] = DLOGLAM
    hdu.writeto(f"{tmp_dir}/{name}")
    return data.astype(np.float32)


def _open_plate(tmp_dir: str, name: str):  # -> xr.Dataset
    from obstore.store import LocalStore
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import FITSParser
    from virtualizarr.registry import ObjectStoreRegistry

    prefix = f"file://{tmp_dir}"
    registry = ObjectStoreRegistry({prefix: LocalStore(prefix=tmp_dir)})
    vds = open_virtual_dataset(url=f"{prefix}/{name}", registry=registry, parser=FITSParser())
    return vds.rename({"PRIMARY": FLUX_VAR})


def _commit_and_open(virtual, tmp_dir: str):  # -> zarr Store
    import icechunk

    prefix = f"file://{tmp_dir}/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(prefix, icechunk.local_filesystem_store(tmp_dir))
    )
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(f"{tmp_dir}/repo"),
        config=config,
        authorize_virtual_chunk_access=icechunk.containers_credentials({prefix: None}),
    )
    session = repo.writable_session("main")
    virtual.virtualize.to_icechunk(session.store)
    session.commit("build")
    return repo.readonly_session("main").store


def _read_all(store) -> dict[int, np.ndarray]:
    geoms = open_geometries(store, variables=[FLUX_VAR], sample_axis=0)
    ds = InSituDataset(
        store,
        split_by_chunk(geoms[FLUX_VAR], fractions=(1.0, 0.0, 0.0)),
        geometries=geoms,
        batch_size=256,
        shuffle=False,
    )
    got: dict[int, np.ndarray] = {}
    for batch in ds.train:
        for k, idx in enumerate(batch.sample_indices):
            got[int(idx)] = batch.arrays[FLUX_VAR][k]
    ds.close()
    return got


def test_single_plate_fiber_chunks(tmp_path) -> None:
    truth = _write_plate(str(tmp_path), name="p.fits", n_fiber=12, n_wave=20, coeff0=3.58, seed=0)
    plate = _open_plate(str(tmp_path), "p.fits")

    store = _commit_and_open(_single_plate_fiber_chunks(plate, fibers_per_chunk=4), str(tmp_path))
    geom = open_geometries(store, sample_axis=0)[FLUX_VAR]
    assert geom.n_samples == 12
    assert geom.sample_chunk_size == 4  # many fibers per chunk (decode-amortization)
    assert geom.n_chunks == 3

    got = _read_all(store)
    for i in range(12):
        np.testing.assert_array_equal(got[i], truth[i])


def test_many_plates_common_grid_aligns_and_concats(tmp_path) -> None:
    # Two plates offset by whole bins with different widths: cropping to the common window must
    # align them onto one grid exactly and concat into a flat fiber axis.
    a = _write_plate(str(tmp_path), name="a.fits", n_fiber=6, n_wave=24, coeff0=3.5800, seed=1)
    b = _write_plate(str(tmp_path), name="b.fits", n_fiber=6, n_wave=22, coeff0=3.5803, seed=2)
    plates = [_open_plate(str(tmp_path), "a.fits"), _open_plate(str(tmp_path), "b.fits")]

    virtual = _many_plates_common_grid(plates)
    store = _commit_and_open(virtual, str(tmp_path))
    geom = open_geometries(store, sample_axis=0)[FLUX_VAR]
    assert geom.n_samples == 12  # 6 + 6 fibers, folded into one axis
    assert geom.sample_chunk_size == 1  # one fiber per chunk (streaming)

    # common window: lo = max start = 3.5803; a starts 3 bins earlier, b at bin 0.
    off_a, off_b = 3, 0
    width = geom.inner_shape[0]
    got = _read_all(store)
    for f in range(6):
        np.testing.assert_array_equal(got[f], a[f, off_a : off_a + width])  # plate a fibers
        np.testing.assert_array_equal(got[6 + f], b[f, off_b : off_b + width])  # plate b fibers
    # both plates now share bin 0 == loglam 3.5803, so the grids are aligned, not merely truncated.
    assert width == min(24 - off_a, 22 - off_b)


def test_many_plates_rejects_unaligned_grids(tmp_path) -> None:
    # A half-bin COEFF0 offset cannot be cropped losslessly -> fail fast, do not silently misalign.
    _write_plate(str(tmp_path), name="a.fits", n_fiber=4, n_wave=16, coeff0=3.5800, seed=3)
    _write_plate(str(tmp_path), name="b.fits", n_fiber=4, n_wave=16, coeff0=3.58005, seed=4)
    plates = [_open_plate(str(tmp_path), "a.fits"), _open_plate(str(tmp_path), "b.fits")]
    with pytest.raises(ValueError, match="not bin-aligned"):
        _many_plates_common_grid(plates)
