"""Variables may chunk the sample axis differently (the OME-NGFF raw+labels pairing:
raw Z-chunk 1, label mask Z-chunk 30). They must still share the sample-axis *length*;
the anchor grid is the manifest's chunking, and each variable maps global anchors onto
its own chunk grid.
"""

from __future__ import annotations

import numpy as np
import zarr

from insitubatch import ensure_local_dir, obstore_store, open_geometries, split_by_chunk
from insitubatch.plan import _read_chunks
from insitubatch.source import InSituDataset
from insitubatch.types import ArrayGeometry


def _write_two(url: str, n: int, spc_a: int, spc_b: int, inner=(3, 3)):
    ensure_local_dir(url)
    group = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    srcs = {}
    for name, spc in (("raw", spc_a), ("labels", spc_b)):
        arr = group.create_array(name, shape=(n, *inner), chunks=(spc, *inner), dtype="f4")
        data = np.arange(n * int(np.prod(inner)), dtype="f4").reshape(n, *inner)
        data = data + (0.0 if name == "raw" else 1000.0)  # distinguishable per variable
        arr[:] = data
        srcs[name] = data
    return srcs


def _check_values(ds: InSituDataset, srcs: dict[str, np.ndarray]) -> int:
    seen = 0
    for b in ds.all:
        for i, anchor in enumerate(b.sample_indices):
            for name, src in srcs.items():
                np.testing.assert_array_equal(b.arrays[name][i], src[anchor])
            seen += 1
    return seen


def test_coarse_variable_maps_onto_fine_reference(tmp_path) -> None:
    # reference = fine (raw spc=1); the coarse label chunks (spc=8) map onto the anchor grid.
    url = f"file://{tmp_path}/a.zarr"
    srcs = _write_two(url, n=40, spc_a=1, spc_b=8)
    geoms = open_geometries(obstore_store(url))
    manifest = split_by_chunk(geoms["raw"], fractions=(1.0, 0.0, 0.0))  # anchor grid spc=1
    ds = InSituDataset(obstore_store(url), manifest, geometries=geoms, batch_size=6, shuffle=False)
    assert _check_values(ds, srcs) == 40
    ds.close()


def test_fine_variable_maps_onto_coarse_reference(tmp_path) -> None:
    # reference = coarse (labels spc=8); the fine raw chunks (spc=1) map on -- the budget
    # must hold enough of the fine variable's many chunks per block (else admission deadlocks).
    url = f"file://{tmp_path}/b.zarr"
    srcs = _write_two(url, n=40, spc_a=1, spc_b=8)
    geoms = open_geometries(obstore_store(url))
    manifest = split_by_chunk(geoms["labels"], fractions=(1.0, 0.0, 0.0))  # anchor grid spc=8
    ds = InSituDataset(
        obstore_store(url), manifest, geometries=geoms, batch_size=6, block_chunks=2, shuffle=False
    )
    assert _check_values(ds, srcs) == 40
    ds.close()


def test_windowed_target_on_coarser_variable(tmp_path) -> None:
    # shift + per-variable chunks together: a forecast whose target is a *differently-chunked*
    # variable read one step ahead. input = raw (spc=1); target = labels (spc=8) shifted +1.
    url = f"file://{tmp_path}/w.zarr"
    srcs = _write_two(url, n=40, spc_a=1, spc_b=8)
    opened = open_geometries(obstore_store(url))
    geoms = {"x": opened["raw"], "y": opened["labels"].shift(1)}
    manifest = split_by_chunk(opened["raw"], fractions=(1.0, 0.0, 0.0))  # anchor grid spc=1
    ds = InSituDataset(
        obstore_store(url), manifest, geometries=geoms, batch_size=6, block_chunks=3, shuffle=False
    )
    seen = 0
    for b in ds.all:
        for i, anchor in enumerate(b.sample_indices):
            np.testing.assert_array_equal(b.arrays["x"][i], srcs["raw"][anchor])
            np.testing.assert_array_equal(b.arrays["y"][i], srcs["labels"][anchor + 1])
            seen += 1
    assert seen == 39  # shift(1) drops the last anchor (no sample at n)
    ds.close()


def test_read_chunks_partial_tail_and_empty_span_guard() -> None:
    # A coarse variable spc=4 over N=10: chunks [0-3], [4-7], [8-9] -- the last runs out at 2.
    g = ArrayGeometry(path="c", shape=(10, 2), chunks=(4, 2), dtype=np.dtype("f4"))
    assert g.n_chunks == 3  # ceil(10 / 4) -- the tail is a partial chunk, not dropped
    assert list(_read_chunks(g, anchor_chunk=8, ref_spc=1)) == [2]  # sample 8 -> partial chunk 2
    assert list(_read_chunks(g, anchor_chunk=9, ref_spc=1)) == [2]  # sample 9 -> partial chunk 2
    assert list(_read_chunks(g.shift(1), anchor_chunk=8, ref_spc=1)) == [2]  # sample 9 -> chunk 2
    # The empty-span guard (lo > hi -> range(0)): reachable only if an off-array anchor were
    # NOT dropped upstream, which the pipeline always does -- so we call it directly. Anchor
    # chunk 9 shifted +2 reads sample 11, off the end of the 10-sample axis.
    assert list(_read_chunks(g.shift(2), anchor_chunk=9, ref_spc=1)) == []


def test_uneven_tail_partial_coarse_chunk_with_shift(tmp_path) -> None:
    # N=10 is not divisible by the coarse chunk (spc=8): the last labels chunk holds only
    # samples [8, 9]. The shifted target reads into that partial tail chunk; verify values.
    url = f"file://{tmp_path}/tail.zarr"
    srcs = _write_two(url, n=10, spc_a=1, spc_b=8)
    opened = open_geometries(obstore_store(url))
    geoms = {"x": opened["raw"], "y": opened["labels"].shift(1)}
    manifest = split_by_chunk(opened["raw"], fractions=(1.0, 0.0, 0.0))  # ref grid spc=1
    ds = InSituDataset(
        obstore_store(url), manifest, geometries=geoms, batch_size=4, block_chunks=3, shuffle=False
    )
    seen = 0
    for b in ds.all:
        for i, anchor in enumerate(b.sample_indices):
            np.testing.assert_array_equal(b.arrays["x"][i], srcs["raw"][anchor])
            np.testing.assert_array_equal(b.arrays["y"][i], srcs["labels"][anchor + 1])
            seen += 1
    assert seen == 9  # anchor 9 dropped by shift(1); anchor 8 reads labels[9] from the tail chunk
    ds.close()


def test_uneven_tail_coarse_reference_partial_anchor_chunk(tmp_path) -> None:
    # The *reference* grid is coarse and uneven: labels spc=4 over N=10 gives 3 anchor chunks,
    # the last (anchors 8, 9) partial. Verify those tail anchors are still drawn and every
    # variable -- including the fine raw -- fetches them correctly (the larger chunk runs out).
    url = f"file://{tmp_path}/tailref.zarr"
    srcs = _write_two(url, n=10, spc_a=1, spc_b=4)
    opened = open_geometries(obstore_store(url))
    manifest = split_by_chunk(opened["labels"], fractions=(1.0, 0.0, 0.0))  # ref grid spc=4
    assert manifest.n_chunks == 3  # last anchor chunk is partial
    ds = InSituDataset(
        obstore_store(url), manifest, geometries=opened, batch_size=3, block_chunks=2, shuffle=False
    )
    anchors: list[int] = []
    for b in ds.all:
        for i, anchor in enumerate(b.sample_indices):
            for name, src in srcs.items():
                np.testing.assert_array_equal(b.arrays[name][i], src[anchor])
            anchors.append(int(anchor))
    assert sorted(anchors) == list(range(10))  # the partial tail anchor chunk yields 8 and 9
    ds.close()


def test_uneven_chunking_shuffled(tmp_path) -> None:
    url = f"file://{tmp_path}/c.zarr"
    srcs = _write_two(url, n=48, spc_a=2, spc_b=12)
    geoms = open_geometries(obstore_store(url))
    manifest = split_by_chunk(geoms["raw"], fractions=(1.0, 0.0, 0.0))
    ds = InSituDataset(
        obstore_store(url), manifest, geometries=geoms, batch_size=5, block_chunks=3, shuffle=True
    )
    # shuffle still pairs each anchor's raw+labels correctly and covers every sample once.
    anchors: list[int] = []
    for b in ds.all if False else ds.train:
        for i, anchor in enumerate(b.sample_indices):
            for name, src in srcs.items():
                np.testing.assert_array_equal(b.arrays[name][i], src[anchor])
            anchors.append(int(anchor))
    assert sorted(anchors) == list(range(48))
    ds.close()
