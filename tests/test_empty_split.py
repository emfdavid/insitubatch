"""An empty split yields no batches. It does not raise from inside numpy.

`fractions=(1.0, 0.0, 0.0)` is an ordinary request -- train on everything, hold nothing
back -- and a small store produces the same shape by accident whenever 10% of the chunks
rounds to zero. Iterating that split went through `np.concatenate([])` and surfaced as
``ValueError: need at least one array to concatenate``, which names neither the split nor
the dataset. It is how the advection and SDSS examples fail the moment their synthetic
store is shrunk, which is the first thing anyone does when trying the library out.

Empty is not an error here: the manifest recorded the split as empty because it was asked
to. Fail-fast covers contracts that cannot be honoured, not collections that are legitimately
empty -- an empty split is `[]`, and iterating `[]` yields nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from insitubatch import obstore_store, open_geometries, split_by_chunk
from insitubatch.shuffle import block_shuffled_order, sequential_order
from insitubatch.source import InSituDataset


@pytest.fixture
def dataset(write_zarr):
    """A store split so that val and test hold no chunks at all."""
    url, _ = write_zarr(n=64, spc=4, inner=(4, 4))
    geometries = open_geometries(obstore_store(url))
    manifest = split_by_chunk(geometries["t2m"], fractions=(1.0, 0.0, 0.0))
    assert len(manifest.chunks["val"]) == 0, "fixture must actually produce an empty split"
    ds = InSituDataset(obstore_store(url), manifest, geometries=geometries, batch_size=4)
    ds.set_epoch(0)
    return ds


@pytest.mark.parametrize("split", ["val", "test"])
def test_an_empty_split_yields_nothing(dataset, split: str) -> None:
    assert list(getattr(dataset, split)) == []


def test_an_empty_split_does_not_disturb_a_populated_one(dataset) -> None:
    """The control: draining the empty split leaves the real one intact.

    They share a pool and a scheduler owner is minted per pass, so an early return has to
    leave that bookkeeping as it found it.
    """
    assert list(dataset.val) == []
    got = sum(int(b.arrays["t2m"].shape[0]) for b in dataset.train)
    assert got == 64


def test_an_empty_split_is_repeatable(dataset) -> None:
    """Twice, and across an epoch boundary -- an early return still has teardown to do."""
    assert list(dataset.val) == []
    dataset.set_epoch(1)
    assert list(dataset.val) == []
    assert list(dataset.test) == []


@pytest.mark.parametrize("order_fn", [sequential_order, block_shuffled_order])
def test_the_order_builders_accept_no_chunks(order_fn) -> None:
    """Both draw orders, since `shuffle` picks between them and either can meet an empty split."""
    empty = np.array([], dtype=np.int64)
    kwargs = {"seed": 0, "epoch": 0} if order_fn is block_shuffled_order else {}

    order = order_fn(empty, 4, 64, block_chunks=4, **kwargs)

    assert len(order) == 0
    assert order.n_blocks == 0, "no chunks is no blocks, not one empty one"
    rows = order.rows
    assert rows.ndim == 2 and rows.shape[1] == 2, "shape must stay (N, 2) so callers can index"
