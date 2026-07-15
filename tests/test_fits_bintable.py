"""Validate FITS *binary-table* (structured big-endian dtype) support end-to-end via insitubatch.

The SDSS spectra example streams spPlate *images*, but the archival format also carries binary
tables (per-object spectra, catalogs) as big-endian structured dtypes. Reading those as virtual
references exercises a fragile path: ``kerchunk.fits`` maps the table to one structured-dtype zarr
array, and VirtualiZarr must translate the kerchunk refs into a zarr-v3 structured dtype. That
translation needs the fix in ``zarr-developers/VirtualiZarr#1037`` (pinned in ``[tool.uv.sources]``
via the ``astronomy`` extra): without it, ``from_kerchunk_refs`` raises on the list-valued field
specs and base64 fill value.

This offline test synthesizes a big-endian ``BinTable`` (no network, no SDSS), runs it through the
whole chain -- kerchunk -> VirtualiZarr -> Icechunk -> insitubatch -- and checks the delivered field
values against the astropy ground truth. It is the drift guard that keeps FITS binary-table support
(and the pinned VZ fix) exercised.

Endianness note: VirtualiZarr/icechunk store the struct fields with native (little-endian) *labels*
while the referenced FITS bytes stay big-endian, so a consumer reinterprets each field with
``.view('>f4')`` at the numpy boundary before handing native-order data to a framework (DLPack
rejects both structured dtypes and non-native byte order). That projection is the documented,
vectorized boundary step -- it is required independently of the zarr codec, so this test does not
depend on the zarr byte-order pin.
"""

from __future__ import annotations

import numpy as np
import pytest

# The FITS binary-table chain needs the whole build-time stack (the `astronomy` extra).
pytest.importorskip("virtualizarr")
pytest.importorskip("kerchunk")
pytest.importorskip("astropy")
pytest.importorskip("icechunk")

from insitubatch import InSituDataset, open_geometries, split_by_chunk  # noqa: E402


def _build_bintable_store(tmp_dir: str) -> tuple[str, np.ndarray]:
    """Write a big-endian FITS BinTable and index it into a local Icechunk repo of virtual refs.

    Returns ``(store_path, ground_truth_flux)``.
    """
    import icechunk
    from astropy.io import fits
    from obstore.store import LocalStore
    from virtualizarr import open_virtual_dataset
    from virtualizarr.parsers import FITSParser
    from virtualizarr.registry import ObjectStoreRegistry

    n = 64
    flux = (np.sin(np.linspace(0.0, 6.0, n)) * 10.0).astype(">f4")  # big-endian, as in real FITS
    ivar = np.linspace(1.0, 2.0, n).astype(">f4")
    cols = fits.ColDefs(
        [
            fits.Column(name="flux", format="E", array=flux),
            fits.Column(name="ivar", format="E", array=ivar),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name="COADD")]).writeto(
        f"{tmp_dir}/spec.fits"
    )

    prefix = f"file://{tmp_dir}"
    registry = ObjectStoreRegistry({prefix: LocalStore(prefix=tmp_dir)})
    vds = open_virtual_dataset(url=f"{prefix}/spec.fits", registry=registry, parser=FITSParser())

    ice_prefix = prefix + "/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(ice_prefix, icechunk.local_filesystem_store(tmp_dir))
    )
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(f"{tmp_dir}/repo"),
        config=config,
        authorize_virtual_chunk_access=icechunk.containers_credentials({ice_prefix: None}),
    )
    session = repo.writable_session("main")
    vds.virtualize.to_icechunk(session.store)
    session.commit("index bintable")
    return f"{tmp_dir}/repo", flux


def test_fits_bintable_roundtrips_through_insitubatch(tmp_path) -> None:
    import icechunk

    store_path, flux_truth = _build_bintable_store(str(tmp_path))

    ice_prefix = f"file://{tmp_path}/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(ice_prefix, icechunk.local_filesystem_store(str(tmp_path)))
    )
    repo = icechunk.Repository.open(
        icechunk.local_filesystem_storage(store_path),
        config=config,
        authorize_virtual_chunk_access=icechunk.containers_credentials({ice_prefix: None}),
    )
    store = repo.readonly_session("main").store

    geoms = open_geometries(store, sample_axis=0)
    (var,) = geoms  # kerchunk names the HDU variable (here "COADD")
    geom = geoms[var]
    assert geom.n_samples == flux_truth.size
    assert geom.dtype.names == ("flux", "ivar")  # a structured (record) dtype survived the chain

    ds = InSituDataset(
        store,
        split_by_chunk(geom, fractions=(1.0, 0.0, 0.0)),
        geometries=geoms,
        batch_size=64,
        shuffle=False,
    )
    got: dict[int, np.ndarray] = {}
    for batch in ds.train:
        rows = batch.arrays[var]
        for k, idx in enumerate(batch.sample_indices):
            got[int(idx)] = rows[k]
    ds.close()

    assert len(got) == flux_truth.size
    delivered = np.array([got[i] for i in range(flux_truth.size)])
    # The referenced bytes are big-endian; reinterpret the field at the numpy boundary.
    flux = delivered["flux"].view(">f4").astype(np.float32)
    np.testing.assert_allclose(flux, flux_truth.astype(np.float32))


def test_fits_bintable_endianness_note_is_load_bearing(tmp_path) -> None:
    # Guard the documented endianness behaviour: the native-label field reads as garbage, and the
    # ``.view('>f4')`` reinterpret is what recovers the values. If a future zarr/VZ release delivers
    # correctly-swapped native data, THIS test flips -- the signal to drop the workaround.
    import icechunk

    store_path, flux_truth = _build_bintable_store(str(tmp_path))
    ice_prefix = f"file://{tmp_path}/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(ice_prefix, icechunk.local_filesystem_store(str(tmp_path)))
    )
    repo = icechunk.Repository.open(
        icechunk.local_filesystem_storage(store_path),
        config=config,
        authorize_virtual_chunk_access=icechunk.containers_credentials({ice_prefix: None}),
    )
    import zarr

    (var,) = open_geometries(repo.readonly_session("main").store, sample_axis=0)
    raw = zarr.open_array(repo.readonly_session("main").store, path=var, mode="r")[:]
    assert not np.allclose(raw["flux"].astype("f8"), flux_truth.astype("f8"))  # native label: wrong
    assert np.allclose(raw["flux"].view(">f4").astype("f8"), flux_truth.astype("f8"))  # reinterpret
