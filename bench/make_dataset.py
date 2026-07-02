"""Generate a synthetic-but-realistic zarr dataset for benchmarking.

Writes a zarr group with one or more variables, in either regime of the
fat-chunk <-> GRIB-per-timestep spectrum (DESIGN.md, "the spectrum"):

  fat  : many samples per sample-axis chunk (ARCO-style, time-chunked)
  grib : one sample per chunk (the degenerate end; each timestep its own chunk)

Same code targets local or cloud -- only the ``--url`` scheme changes::

    uv run python bench/make_dataset.py --url file:///tmp/era5.zarr --regime fat
    uv run python bench/make_dataset.py --url s3://bucket/era5.zarr --regime grib
"""

from __future__ import annotations

import argparse
from typing import Literal

import numpy as np
import zarr

from insitubatch.store import ensure_local_dir, obstore_store


def make_dataset(
    url: str,
    *,
    n_samples: int,
    inner: tuple[int, ...],
    sample_chunk: int,
    variables: list[str],
    compress: bool = True,
    seed: int = 0,
    inner_chunks: tuple[int, ...] | None = None,
    write_batch_mb: int = 1024,
    write_concurrency: int = 32,
    s3_express: bool = False,
) -> None:
    """Write a ``(n_samples, *inner)`` array per variable, chunked along axis 0.

    ``inner_chunks`` chunks the inner (spatial) dims too: one sample-axis chunk then
    maps to a *grid* of stored chunks, so a single sample's field fans out into many
    concurrent reads. That restores read concurrency in the fat-sample regime (few
    sample-axis chunks) — how ARCO/ERA5 is typically chunked. Default: single inner
    chunk (the v1 simplification).

    Speed: data is written a *slab* of many chunks at a time rather than
    chunk-by-chunk. A multi-chunk ``arr[start:stop] = ...`` lets zarr fan the
    chunk writes out concurrently on its async loop (so cloud PUTs overlap instead
    of being one serial round-trip each); ``write_batch_mb`` bounds the slab — and
    thus the writer's RAM — and ``write_concurrency`` raises zarr's in-flight write
    limit (default 10). Values are generated as float32 directly (no float64 copy).
    """
    inner_chunks = inner_chunks or inner
    if len(inner_chunks) != len(inner):
        raise ValueError(f"inner_chunks {inner_chunks} must match inner dims {inner}")
    ensure_local_dir(url)
    # S3 Express One Zone (directory buckets, --x-s3): obstore needs s3_express=True;
    # it is NOT inferred from the bucket name. (env equivalent: AWS_S3_EXPRESS=true)
    store_kwargs = {"s3_express": True} if s3_express else {}
    store = obstore_store(url, read_only=False, **store_kwargs)
    group = zarr.open_group(store=store, mode="w")
    rng = np.random.default_rng(seed)
    chunks = (sample_chunk, *inner_chunks)
    compressors: Literal["auto"] | None = "auto" if compress else None

    # Slab size = a whole number of sample-axis chunks fitting in write_batch_mb.
    bytes_per_row = int(np.prod(inner)) * 4
    rows_per_batch = max(1, (write_batch_mb * 1_000_000) // bytes_per_row // sample_chunk)
    slab_rows = rows_per_batch * sample_chunk

    # Named dims so xarray (and thus the xbatcher baseline) can open the store;
    # our own engine reads by index and ignores names.
    dim_names = ("time", *(f"dim{i}" for i in range(len(inner))))
    with zarr.config.set({"async.concurrency": write_concurrency}):
        for var in variables:
            arr = group.create_array(
                var,
                shape=(n_samples, *inner),
                chunks=chunks,
                dtype="f4",
                compressors=compressors,
                dimension_names=dim_names,
            )
            for start in range(0, n_samples, slab_rows):
                stop = min(start + slab_rows, n_samples)
                arr[start:stop] = rng.standard_normal((stop - start, *inner), dtype=np.float32)

    nbytes = n_samples * int(np.prod(inner)) * 4 * len(variables)
    print(
        f"wrote {url}  vars={variables}  shape=({n_samples},{','.join(map(str, inner))})  "
        f"chunk0={sample_chunk}  ~{nbytes / 1e6:.1f} MB uncompressed  "
        f"compress={'auto' if compress else 'none'}"
    )


def make_family(
    url_prefix: str,
    chunk_sizes: tuple[int, ...],
    *,
    n_samples: int,
    inner: tuple[int, ...],
    variables: tuple[str, ...] = ("t2m",),
    compress: bool = True,
    seed: int = 0,
    write_batch_mb: int = 1024,
    write_concurrency: int = 32,
) -> dict[int, str]:
    """Write the same logical dataset at several sample-chunk sizes.

    Produces ``{url_prefix}_c{spc}.zarr`` per ``spc``. Random values differ across
    chunkings (slab fill), but byte count, dtype, inner shape, and codec match — so
    throughput is comparable along the chunk-size axis.
    """
    paths: dict[int, str] = {}
    for spc in chunk_sizes:
        url = f"{url_prefix}_c{spc}.zarr"
        make_dataset(
            url,
            n_samples=n_samples,
            inner=inner,
            sample_chunk=spc,
            variables=list(variables),
            compress=compress,
            seed=seed,
            write_batch_mb=write_batch_mb,
            write_concurrency=write_concurrency,
        )
        paths[spc] = url
    return paths


def _inner(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="file:// or s3:// target")
    p.add_argument("--regime", choices=["fat", "grib"], default="fat")
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--inner", type=_inner, default=(64, 64), help="comma-separated, e.g. 721,1440")
    p.add_argument(
        "--inner-chunks", type=_inner, default=None, help="chunk inner dims too, e.g. 256,256"
    )
    p.add_argument("--sample-chunk", type=int, default=None, help="override regime default")
    p.add_argument("--variables", default="t2m", help="comma-separated names")
    p.add_argument("--uncompressed", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--write-batch-mb", type=int, default=1024, help="slab size (RAM bound) per write"
    )
    p.add_argument("--write-concurrency", type=int, default=32, help="zarr in-flight chunk writes")
    p.add_argument(
        "--s3-express", action="store_true", help="target an S3 Express One Zone directory bucket"
    )
    a = p.parse_args()

    sample_chunk = (
        a.sample_chunk if a.sample_chunk is not None else (64 if a.regime == "fat" else 1)
    )
    make_dataset(
        a.url,
        n_samples=a.n_samples,
        inner=a.inner,
        sample_chunk=sample_chunk,
        variables=a.variables.split(","),
        compress=not a.uncompressed,
        seed=a.seed,
        inner_chunks=a.inner_chunks,
        write_batch_mb=a.write_batch_mb,
        write_concurrency=a.write_concurrency,
        s3_express=a.s3_express,
    )


if __name__ == "__main__":
    main()
