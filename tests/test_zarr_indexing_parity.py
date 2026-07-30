"""Dogfood parity: our read planner vs zarr's new ``zarr-indexing`` package.

`zarr-indexing` (zarr-developers/zarr-python#4196, the leaf split of the #3906
IndexTransform work) is a standalone, NumPy-only package implementing
TensorStore-style index transforms + chunk resolution against a
``DimensionGridLike`` protocol. This test dogfoods it as a *correctness oracle*
for :func:`insitubatch.plan.build_stored_chunk_reads`: it asserts that
``zarr_indexing.iter_chunk_transforms`` resolves the exact same set of stored
chunks our hand-rolled planner produces, once :class:`ArrayGeometry` supplies a
``DimensionGridLike`` per (logical, sample-first) axis.

It is deliberately *not* a hot-path swap. Our planner resolves the UNION of many
disjoint sample windows in draw order, deduped across variables, into a
scatter-into-slots plan with no gather map -- none of which ``iter_chunk_transforms``
(single selection, mandatory ``out_indices`` scatter map) covers. What the parity
proves is narrower and real: our per-selection chunk math is reproducible at the
package's contract boundary, so ``ArrayGeometry`` is a valid ``DimensionGridLike``
consumer.

The parity tests answer "does it match what we do today?". The second test
(:func:`test_scattered_sample_read_amplification`) is a *thesis guard*, not an
adoption pitch: it demonstrates why we do **not** expose arbitrary per-sample
random access. ``zarr_indexing`` can resolve a scattered, cross-chunk sample
selection into a per-chunk scatter map -- but resolving the coordinates does
nothing about **read amplification**: "one sample from every chunk" still fetches
and decodes every touched chunk to use one row each, which is the reshard/
random-access cost we traded away for the block-shuffle approximation. The index
math was never the bottleneck; making it free changes nothing on the axis that
hurts. The test drives such a gather through the package and puts a hard number on
the waste (5 samples wanted -> all 12 in the touched chunks read), so the boundary
stays visible if anyone is later tempted to read the fancy resolver as a "future
engine" for insitubatch. It is not one -- see DESIGN.md, "Upstream capabilities".

The package is not a dependency; the test skips unless it is importable. To run it::

    uv run --with path/to/packages/zarr-indexing pytest tests/test_zarr_indexing_parity.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

zi = pytest.importorskip("zarr_indexing")

from insitubatch.plan import _read_chunks, build_stored_chunk_reads  # noqa: E402
from insitubatch.types import ArrayGeometry  # noqa: E402


@dataclass(frozen=True)
class AxisGrid:
    """A regular chunk grid on one axis; satisfies ``zarr_indexing.DimensionGridLike``."""

    size: int
    chunk: int

    def index_to_chunk(self, idx: int) -> int:
        return idx // self.chunk

    def chunk_offset(self, chunk_ix: int) -> int:
        return chunk_ix * self.chunk

    def chunk_size(self, chunk_ix: int) -> int:
        return min(self.chunk, self.size - chunk_ix * self.chunk)

    def indices_to_chunks(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        return indices // self.chunk


def _logical_shape(geom: ArrayGeometry) -> tuple[int, ...]:
    return (geom.n_samples, *geom.inner_shape)


def _logical_grids(geom: ArrayGeometry) -> list[AxisGrid]:
    """One ``AxisGrid`` per logical (sample-first) axis -- ``StoredChunkRead.coords`` order."""
    shape = _logical_shape(geom)
    chunks = (geom.sample_chunk_size, *geom.inner_chunks)
    return [AxisGrid(s, c) for s, c in zip(shape, chunks, strict=True)]


def _read_window(geom: ArrayGeometry, anchor: int, ref_spc: int) -> tuple[int, int]:
    """Inclusive ``[lo, hi]`` sample window an anchor reads (mirrors ``plan._read_chunks``)."""
    start = anchor * ref_spc
    stop = min(start + ref_spc, geom.n_samples)
    lo = max(0, start + geom.offset)
    hi = min(geom.n_samples - 1, stop - 1 + geom.offset)
    return lo, hi


def _zi_chunk_coords(geom: ArrayGeometry, lo: int, hi: int) -> set[tuple[int, ...]]:
    """Chunk set ``zarr_indexing`` resolves for the logical box ``[lo, hi] x (all inner)``."""
    shape = _logical_shape(geom)
    mins = (lo, *([0] * (len(shape) - 1)))
    maxs = (hi + 1, *shape[1:])
    transform = zi.IndexTransform.identity(zi.IndexDomain(inclusive_min=mins, exclusive_max=maxs))
    resolved = zi.iter_chunk_transforms(transform, _logical_grids(geom))
    return {coords for coords, _sub, _out in resolved}


def _planner_chunk_coords(geom: ArrayGeometry, anchor: int, ref_spc: int) -> set[tuple[int, ...]]:
    """Chunk set our planner produces for one anchor (logical, sample-first coords)."""
    return {
        (cid, *inner)
        for cid in _read_chunks(geom, anchor, ref_spc)
        for inner in geom.inner_coords()
    }


def _n_ref_chunks(geom: ArrayGeometry, ref_spc: int) -> int:
    return -(-geom.n_samples // ref_spc)


f4 = np.dtype("f4")

# (id, geometry, ref_spc) -- exercises offset windows (incl. running off both ends),
# coarser/finer-than-ref variable grids, short final chunks, inner grids, sample_axis != 0.
CASES: list[tuple[str, ArrayGeometry, int]] = [
    ("baseline-1d", ArrayGeometry("a", (100,), (10,), f4), 10),
    ("inner-grid-3d", ArrayGeometry("a", (100, 90, 45), (10, 32, 32), f4), 10),
    ("offset-plus1", ArrayGeometry("a", (100, 90, 45), (10, 32, 32), f4, offset=1), 10),
    ("offset-plus5", ArrayGeometry("a", (100, 8), (10, 8), f4, offset=5), 10),
    ("offset-minus3", ArrayGeometry("a", (100, 8), (10, 8), f4, offset=-3), 10),
    ("coarser-than-ref", ArrayGeometry("a", (100, 8), (20, 8), f4), 10),
    ("finer-than-ref", ArrayGeometry("a", (100, 8), (5, 8), f4), 10),
    ("short-final-chunk", ArrayGeometry("a", (95, 45), (10, 32), f4), 10),
    (
        "sample-axis-2",
        ArrayGeometry("a", (4, 3, 100, 90, 45), (4, 3, 10, 32, 32), f4, sample_axis=2),
        10,
    ),
    ("offset-short-inner", ArrayGeometry("a", (95, 90, 45), (10, 32, 32), f4, offset=3), 10),
    ("offset-off-tail", ArrayGeometry("a", (100, 8), (10, 8), f4, offset=10), 10),
    ("offset-off-head", ArrayGeometry("a", (100, 8), (10, 8), f4, offset=-10), 10),
]


@pytest.mark.parametrize("geom,ref_spc", [(g, r) for _, g, r in CASES], ids=[c[0] for c in CASES])
def test_planner_matches_zarr_indexing_per_anchor(geom: ArrayGeometry, ref_spc: int) -> None:
    """Every anchor's resolved chunk set matches ``iter_chunk_transforms`` exactly."""
    saw_nonempty = False
    for anchor in range(_n_ref_chunks(geom, ref_spc)):
        lo, hi = _read_window(geom, anchor, ref_spc)
        planner = _planner_chunk_coords(geom, anchor, ref_spc)
        if lo > hi:  # window reads entirely off the array (edge) -> both empty
            assert not planner
            assert len(_read_chunks(geom, anchor, ref_spc)) == 0
            continue
        assert _zi_chunk_coords(geom, lo, hi) == planner
        saw_nonempty = True
    assert saw_nonempty


@pytest.mark.parametrize("geom,ref_spc", [(g, r) for _, g, r in CASES], ids=[c[0] for c in CASES])
def test_build_stored_chunk_reads_matches_zarr_indexing(geom: ArrayGeometry, ref_spc: int) -> None:
    """The full planner entry point resolves the same stored-chunk set over all anchors."""
    anchors = list(range(_n_ref_chunks(geom, ref_spc)))
    got = {r.coords for r in build_stored_chunk_reads(anchors, {geom.path: geom}, ref_spc)}
    expected: set[tuple[int, ...]] = set()
    for anchor in anchors:
        expected |= _planner_chunk_coords(geom, anchor, ref_spc)
    assert got == expected


# --- thesis guard: why we do NOT expose arbitrary per-sample random access -----


def test_scattered_sample_read_amplification() -> None:
    """Resolving a scattered selection is cheap; *serving* it is the cost we reject.

    ``zarr_indexing`` happily resolves an arbitrary, cross-chunk sample selection
    (``oindex``) into a per-chunk scatter map -- the gather ``out[out_sel] =
    tile[chunk_sel]`` reconstructs ``data[picks]`` exactly, including a cross-chunk
    spread and a repeated index. But resolving the coordinates does nothing about
    **read amplification**: every touched chunk is fetched and decoded in full to
    yield a handful of rows. Here 5 distinct wanted samples force reading all 12 in
    the touched chunks. Scale that to "one sample per chunk" and TTFB collapses --
    which is exactly the reshard/random-access pattern insitubatch trades away for
    the block-shuffle approximation (see DESIGN.md). The index math was never the
    bottleneck, so a free resolver does not make this pattern viable. This test
    stands as the boundary marker, not a feature request.
    """
    n, spc, inner = 12, 4, 3
    data = np.arange(n * inner).reshape(n, inner).astype(np.float32)
    picks = np.array([0, 3, 4, 4, 9, 11], dtype=np.intp)  # scattered, cross-chunk, with a repeat
    grids = [AxisGrid(n, spc), AxisGrid(inner, inner)]

    # The package resolves + scatters the selection correctly -- the math is not the issue.
    transform = zi.IndexTransform.from_shape((n, inner)).oindex[picks]
    out = np.zeros((picks.size, inner), dtype=np.float32)
    touched: set[int] = set()
    for chunk_coords, sub_t, out_idx in zi.iter_chunk_transforms(transform, grids):
        chunk_sel, out_sel, _drop = zi.sub_transform_to_selections(sub_t, out_idx)
        tile = data[chunk_coords[0] * spc : (chunk_coords[0] + 1) * spc]  # one stored chunk
        out[out_sel] = tile[chunk_sel]
        touched.add(chunk_coords[0])
    np.testing.assert_array_equal(out, data[picks])

    # ...but the cost is IO, not indices: serving those picks reads every touched
    # chunk whole -- a strict superset of the samples actually wanted.
    geom = ArrayGeometry("a", (n, inner), (spc, inner), f4)
    reads = build_stored_chunk_reads(sorted(touched), {geom.path: geom}, spc)
    read_samples: set[int] = set()
    for r in reads:
        read_samples |= set(geom.samples_in_chunk(r.chunk_index))
    distinct_picks = {int(p) for p in picks}
    assert distinct_picks < read_samples  # strict subset: the archetypal over-read
    assert len(distinct_picks) == 5  # wanted
    assert len(read_samples) == 12  # actually read (all of the 3 touched chunks)
