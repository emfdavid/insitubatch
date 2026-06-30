"""Develop a ``chunk_transform`` against your real dataset, before training.

A ``chunk_transform`` carries two contracts that otherwise only fail at training time:

* it must be **vectorized numpy that releases the GIL** -- a pure-Python per-element
  transform silently serializes the decode thread pool and kills IO overlap;
* a **reshaping** transform must declare ``output_inner(geom) -> (inner_shape, dtype)``
  that *agrees* with what ``__call__`` produces -- the cross-run cache sizes its slot from
  it (a disagreement raises deep in :meth:`ChunkPool._persist`).

This CLI runs the transform on **one chunk of the real store** and reports the chunk
geometry, the input->output shape/dtype, whether the declared output geometry matches, and
an empirical GIL-release verdict (a thread-scaling probe that models the decode-pool overlap)::

    python -m insitubatch.check_transform s3://bucket/era5.zarr --var t2m \\
        --transform my_pkg.transforms:Regrid

    insitubatch-check-transform ./era5.zarr --var t2m --transform ./prep.py:scale

The target is ``module.path:attr`` or ``./file.py:attr``. Exit code is non-zero when a check
fails (declared-output mismatch, an undeclared reshape, or a GIL-held verdict), so it can
gate a pre-commit / CI step.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import statistics
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from .pool import output_geometry
from .store import as_store, open_geometries
from .types import ArrayGeometry, ChunkRead, DecodedChunk

ChunkTransform = Callable[[DecodedChunk], DecodedChunk]


def load_transform(target: str) -> ChunkTransform:
    """Resolve ``module.path:attr`` or ``./file.py:attr`` to a callable transform."""
    if ":" not in target:
        raise ValueError(f"--transform must be 'module:attr' or 'file.py:attr', got {target!r}")
    mod_part, _, attr = target.partition(":")
    if not attr:
        raise ValueError(f"--transform is missing the ':attr' part, got {target!r}")
    if mod_part.endswith(".py") or os.sep in mod_part or "/" in mod_part:
        path = Path(mod_part).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"transform file not found: {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load a module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)
    if not hasattr(module, attr):
        raise AttributeError(f"{attr!r} not found in {mod_part}")
    fn = getattr(module, attr)
    if isinstance(fn, type):  # a transform *class* -> instantiate it (no-arg ctor)
        try:
            fn = fn()
        except TypeError as exc:
            raise TypeError(
                f"{target} is a class whose constructor needs arguments ({exc}); expose a "
                "configured instance at module scope instead (e.g. `my_transform = Regrid(...)`) "
                "and point --transform at that."
            ) from exc
    if not callable(fn):
        raise TypeError(f"{target} resolved to {type(fn).__name__}, which is not callable")
    return fn


def read_chunk(url: str, var: str, chunk_index: int, geom: ArrayGeometry) -> DecodedChunk:
    """Decode exactly one outer chunk of ``var`` into a :class:`DecodedChunk`.

    Uses a plain zarr slice -- this is a one-off development read, not the training hot path
    (the no-getitem stance is about throughput), so zarr stitching the inner grid is fine."""
    group = zarr.open_group(store=as_store(url), mode="r")
    arr = group[var]
    samples = geom.samples_in_chunk(chunk_index)
    data = np.asarray(arr[samples.start : samples.stop])  # type: ignore[index]
    return DecodedChunk(read=ChunkRead(var, chunk_index), data=data, sample_offset=samples.start)


def _fresh(base: DecodedChunk) -> DecodedChunk:
    """A pristine source-shaped copy of the input chunk (transforms may mutate in place)."""
    return DecodedChunk(read=base.read, data=base.data.copy(), sample_offset=base.sample_offset)


def _run_iters(
    fn: ChunkTransform, pristine: np.ndarray, read: ChunkRead, offset: int, iters: int
) -> None:
    """Run ``fn`` ``iters`` times, each on a fresh source-shaped copy of ``pristine``.

    A fresh copy per call is required for correctness: transforms may mutate ``chunk.data``
    in place, and a reshaping transform cannot be re-run on its own (changed-shape) output."""
    for _ in range(iters):
        fn(DecodedChunk(read=read, data=pristine.copy(), sample_offset=offset))


def gil_probe(
    fn: ChunkTransform, base: DecodedChunk, *, threads: int, min_seconds: float, repeats: int
) -> dict[str, float]:
    """Thread-scaling speedup of ``fn``: vectorized (GIL-releasing) work scales with threads;
    a pure-Python per-element transform stays ~1x because it holds the GIL.

    Each worker runs ``iters`` calls on its *own* pristine source buffer (memory is bounded by
    ``threads``, not the call count, so even a sub-millisecond transform gets a full
    ``min_seconds`` window). serial = 1 thread doing ``iters``; parallel = ``threads`` threads
    each doing ``iters`` (so ``threads x`` the work). speedup = threads * t_serial / t_parallel
    -> ~threads when the GIL is released, ~1 when it is held. The per-call source copy is
    included symmetrically; it releases the GIL too, so it never *masks* a GIL-holding
    transform (which dominates the timing) -- it only avoids false alarms on fast ones."""
    read, pristine, offset = base.read, base.data, base.sample_offset
    _run_iters(fn, pristine, read, offset, 1)  # warm up: imports / lazy alloc / page faults

    t0 = time.perf_counter()
    _run_iters(fn, pristine, read, offset, 1)
    one = max(time.perf_counter() - t0, 1e-6)
    iters = max(1, int(min_seconds / one) + 1)

    def serial() -> float:
        t = time.perf_counter()
        _run_iters(fn, pristine, read, offset, iters)
        return time.perf_counter() - t

    def parallel() -> float:
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [
                ex.submit(_run_iters, fn, pristine, read, offset, iters) for _ in range(threads)
            ]
            for f in futs:
                f.result()
        return time.perf_counter() - t

    s = statistics.median(serial() for _ in range(repeats))
    p = statistics.median(parallel() for _ in range(repeats))
    return {
        "iters": float(iters),
        "serial_s": s,
        "parallel_s": p,
        "speedup": threads * s / p if p > 0 else float("inf"),
        "per_call_ms": s / iters * 1e3,
        "mb_s": pristine.nbytes / 1e6 / (s / iters) if s > 0 else float("inf"),
    }


def _gil_enabled() -> bool | None:
    """True/False on 3.13+ (free-threaded build => False); None when undeterminable (<=3.12)."""
    probe = getattr(sys, "_is_gil_enabled", None)
    return bool(probe()) if probe is not None else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="insitubatch-check-transform",
        description="Develop a chunk_transform against one chunk of your real dataset.",
    )
    p.add_argument("url", help="zarr store URL (file://, s3://, gs://, ...)")
    p.add_argument("--var", required=True, help="variable (array) name in the store")
    p.add_argument("--transform", required=True, help="'module.path:attr' or './file.py:attr'")
    p.add_argument("--chunk", type=int, default=0, help="outer (sample-axis) chunk index to probe")
    p.add_argument("--threads", type=int, default=0, help="GIL-probe threads (0 = min(4, cpus))")
    p.add_argument("--repeats", type=int, default=3, help="timing runs per point; report median")
    p.add_argument("--min-seconds", type=float, default=0.3, help="min serial timing window")
    p.add_argument(
        "--no-gil-probe",
        action="store_true",
        help="skip the GIL/vectorization probe "
        "(geometry + output_inner checks only -- fast, deterministic)",
    )
    a = p.parse_args(argv)

    threads = a.threads or min(4, os.cpu_count() or 1)

    try:
        fn = load_transform(a.transform)
    except (ValueError, FileNotFoundError, ImportError, AttributeError, TypeError) as exc:
        print(f"error: could not load --transform {a.transform!r}: {exc}", file=sys.stderr)
        return 2

    geoms = open_geometries(a.url, variables=[a.var])
    geom = geoms[a.var]
    if not 0 <= a.chunk < geom.n_chunks:
        print(f"error: --chunk {a.chunk} out of range [0, {geom.n_chunks})", file=sys.stderr)
        return 2

    n_samples = len(geom.samples_in_chunk(a.chunk))
    src_mb = int(np.prod(geom.slot_shape(a.chunk))) * geom.dtype.itemsize / 1e6
    tiles = geom.n_inner_chunks(a.chunk)
    print(f"dataset: {a.url}  var={a.var}")
    print(
        f"  sample axis : {geom.n_samples} samples, {geom.sample_chunk_size}/chunk, "
        f"{geom.n_chunks} chunks"
    )
    print(f"  inner shape : {geom.inner_shape}  dtype={geom.dtype}  inner tiles/chunk={tiles}")
    print(
        f"  chunk {a.chunk:<5}: {n_samples} samples -> source shape "
        f"{geom.slot_shape(a.chunk)} = {src_mb:.1f} MB decoded"
    )

    base = read_chunk(a.url, a.var, a.chunk, geom)
    in_shape, in_dtype = base.data.shape, base.data.dtype
    out = fn(_fresh(base))
    out_data = out.data
    print("\ntransform output:")
    print(f"  {in_shape} {in_dtype}  ->  {out_data.shape} {out_data.dtype}")
    reshaped = out_data.shape[1:] != in_shape[1:]
    recast = out_data.dtype != in_dtype
    tags = [t for t, on in (("reshaped", reshaped), ("recast", recast)) if on]
    print(f"  {'  '.join(tags) if tags else 'shape- and dtype-preserving'}")
    if np.issubdtype(out_data.dtype, np.floating):
        bad = int(np.count_nonzero(~np.isfinite(out_data)))
        if bad:
            print(f"  WARNING: output has {bad} non-finite (NaN/inf) values")

    # -- declared-output validation -----------------------------------------
    ok = True
    declares = hasattr(fn, "output_inner")
    print("\ncacheability (declared output geometry):")
    if declares:
        declared = output_geometry(geom, [fn])
        match = declared.inner_shape == out_data.shape[1:] and declared.dtype == out_data.dtype
        ok = ok and match
        verdict = "OK" if match else "MISMATCH"
        print(f"  declares output_inner -> {declared.inner_shape} {declared.dtype}  [{verdict}]")
        if not match:
            print(
                f"  FAIL: declared {declared.inner_shape}/{declared.dtype} != actual "
                f"{out_data.shape[1:]}/{out_data.dtype} -- ChunkPool._persist would raise."
            )
    elif reshaped or recast:
        ok = False
        print(
            "  FAIL: this transform reshapes/recasts but does not declare output_inner. The "
            "cache cannot size its slot -> add output_inner(geom) -> (inner_shape, dtype) "
            "(see insitubatch.transforms.ReshapingChunkTransform)."
        )
    else:
        print("  shape/dtype-preserving, no output_inner needed -> cacheable as-is.")

    # -- GIL-release / vectorization probe ----------------------------------
    if a.no_gil_probe:
        print("\nGIL-release probe: skipped (--no-gil-probe).")
        print(
            f"\n{'PASS' if ok else 'FAIL'}: chunk_transform checks "
            f"{'all passed' if ok else 'found problems (see above)'}."
        )
        return 0 if ok else 1

    print(f"\nGIL-release probe (thread-scaling, {threads} threads):")
    gil = _gil_enabled()
    probe = gil_probe(fn, base, threads=threads, min_seconds=a.min_seconds, repeats=a.repeats)
    print(
        f"  per-call {probe['per_call_ms']:.2f} ms  (~{probe['mb_s']:.0f} MB/s)  "
        f"iters={int(probe['iters'])}  "
        f"serial {probe['serial_s']:.3f}s  parallel {probe['parallel_s']:.3f}s"
    )
    if gil is False:
        print(
            "  speedup "
            f"{probe['speedup']:.2f}x -- GIL-release: N/A on a free-threaded build "
            "(Python loops also scale; check per-call latency for vectorization)."
        )
    else:
        target = 0.6 * min(threads, os.cpu_count() or 1)
        passed = probe["speedup"] >= target
        ok = ok and passed
        if passed:
            print(
                f"  speedup {probe['speedup']:.2f}x (>= {target:.2f}) "
                "-> releases the GIL (vectorized)."
            )
        else:
            print(
                f"  FAIL: speedup {probe['speedup']:.2f}x (< {target:.2f}) -> appears to HOLD the "
                "GIL. A pure-Python per-element transform serializes the decode pool and kills "
                "IO overlap; rewrite as vectorized numpy."
            )

    print(
        f"\n{'PASS' if ok else 'FAIL'}: chunk_transform checks "
        f"{'all passed' if ok else 'found problems (see above)'}."
    )
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
