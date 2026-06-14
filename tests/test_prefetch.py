"""Prefetch: the producer assembles batches ahead of the consumer.

The pre-M1.5 loop was demand-driven (a batch was assembled only when pulled), so
the event loop sat idle during the compute step. Prefetch decouples production
from consumption via a bounded queue.
"""

from __future__ import annotations

import time

import numpy as np
import zarr

from insitubatch import (
    SplitName,
    ensure_local_dir,
    open_geometries,
    split_by_chunk,
    store_from_url,
)
from insitubatch.source import InSituDataset


def _write(tmp_path, *, n=160, spc=8, inner=(2, 2)) -> str:
    url = f"file://{tmp_path}/d.zarr"
    ensure_local_dir(url)
    group = zarr.open_group(store=store_from_url(url, read_only=False), mode="w")
    arr = group.create_array("t2m", shape=(n, *inner), chunks=(spc, *inner), dtype="f4")
    arr[:] = np.arange(n * int(np.prod(inner)), dtype="f4").reshape(n, *inner)
    return url


def test_producer_runs_ahead_of_consumer(tmp_path) -> None:
    url = _write(tmp_path)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))

    produced: list[int] = []

    def record(batch):  # runs in the producer thread
        produced.append(int(batch.sample_indices[0]))
        return batch

    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        batch_size=4,
        block_chunks=2,
        prefetch_depth=2,
        to_tensor=False,
        batch_transforms=[record],
    )
    ds.set_epoch(0)

    it = iter(ds)
    first = next(it)  # pulling one batch starts the producer, which runs ahead
    time.sleep(0.2)  # let the background producer fill the bounded queue

    # With prefetch_depth=2 the producer assembles the consumed batch + fills the
    # queue + parks on the next put => >= depth + 1 produced. Demand-driven would
    # have produced exactly 1.
    assert len(produced) >= 3, f"producer did not run ahead: produced={produced}"

    rest = list(it)  # draining the rest still works
    assert first is not None
    assert len(rest) >= 1


def test_prefetch_preserves_values_and_coverage(tmp_path) -> None:
    url = _write(tmp_path)
    geom = open_geometries(url)["t2m"]
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))
    src = zarr.open_group(store=store_from_url(url), mode="r")["t2m"][:]

    ds = InSituDataset(
        url,
        manifest,
        split=SplitName.TRAIN,
        batch_size=4,
        block_chunks=2,
        prefetch_depth=3,
        to_tensor=False,
    )
    ds.set_epoch(0)

    seen: list[np.ndarray] = []
    for batch in ds:
        idx = batch.sample_indices
        np.testing.assert_array_equal(batch.arrays["t2m"], src[idx])
        seen.append(idx)

    all_seen = np.concatenate(seen)
    train = manifest.sample_indices(SplitName.TRAIN, geom)
    assert sorted(all_seen.tolist()) == sorted(train.tolist())
