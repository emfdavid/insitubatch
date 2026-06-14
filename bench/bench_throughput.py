"""Throughput harness: insitubatch vs a classic DataLoader baseline.

The headline number we care about: **samples/s delivered to the training step**
while holding peak host memory bounded -- across the fat-chunk <-> degenerate
(GRIB-per-timestep) spectrum.

This is a skeleton. The baseline and the live-store path are marked TODO; the
synthetic-store path already exercises the full insitubatch control flow so the
orchestration overhead (planning + buffer + gather, minus real IO) is
measurable today.

Run:  uv run python bench/bench_throughput.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from insitubatch import ArrayGeometry, SplitName, split_by_chunk
from insitubatch.source import InSituDataset


@dataclass
class Case:
    name: str
    n_samples: int
    sample_chunk: int  # 1 == degenerate GRIB-per-timestep; large == fat chunks
    inner: tuple[int, ...]
    batch_size: int
    block_chunks: int


CASES = [
    Case("fat_chunks", 4096, 64, (32, 32), batch_size=32, block_chunks=16),
    Case("grib_per_timestep", 4096, 1, (721, 1440), batch_size=32, block_chunks=256),
]


def run_case(case: Case) -> None:
    geom = ArrayGeometry(
        "t2m",
        shape=(case.n_samples, *case.inner),
        chunks=(case.sample_chunk, *case.inner),
        dtype=np.dtype("f4"),
    )
    manifest = split_by_chunk(geom, fractions=(0.8, 0.1, 0.1))
    ds = InSituDataset(
        store=None,  # synthetic store; io.py returns zeros until wired
        geometries={"t2m": geom},
        manifest=manifest,
        split=SplitName.TRAIN,
        batch_size=case.batch_size,
        block_chunks=case.block_chunks,
        to_tensor=False,
    )
    ds.set_epoch(0)

    t0 = time.perf_counter()
    n = 0
    for batch in ds:
        n += batch.arrays["t2m"].shape[0]
    dt = time.perf_counter() - t0
    print(f"{case.name:20s}  {n:6d} samples  {dt:7.3f}s  {n / dt:9.1f} samples/s")


def main() -> None:
    print("insitubatch orchestration throughput (synthetic store, zeros payload)\n")
    for case in CASES:
        run_case(case)
    print("\nTODO: add classic DataLoader baseline + live obstore-backed store.")


if __name__ == "__main__":
    main()
