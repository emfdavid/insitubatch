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

import numpy as np
import zarr

from insitubatch.store import ensure_local_dir, store_from_url


def make_dataset(
    url: str,
    *,
    n_samples: int,
    inner: tuple[int, ...],
    sample_chunk: int,
    variables: list[str],
    compress: bool = True,
    seed: int = 0,
) -> None:
    """Write a ``(n_samples, *inner)`` array per variable, chunked along axis 0.

    Inner dims are single-chunk (the v1 contract), so one sample-axis chunk maps
    to exactly one stored chunk. Data is written chunk-by-chunk to keep the
    writer's memory bounded regardless of total size.
    """
    ensure_local_dir(url)
    store = store_from_url(url, read_only=False)
    group = zarr.open_group(store=store, mode="w")
    rng = np.random.default_rng(seed)
    chunks = (sample_chunk, *inner)
    compressors = "auto" if compress else None

    # Named dims so xarray (and thus the xbatcher baseline) can open the store;
    # our own engine reads by index and ignores names.
    dim_names = ("time", *(f"dim{i}" for i in range(len(inner))))
    for var in variables:
        arr = group.create_array(
            var,
            shape=(n_samples, *inner),
            chunks=chunks,
            dtype="f4",
            compressors=compressors,
            dimension_names=dim_names,
        )
        for start in range(0, n_samples, sample_chunk):
            stop = min(start + sample_chunk, n_samples)
            arr[start:stop] = rng.standard_normal((stop - start, *inner)).astype("f4")

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
) -> dict[int, str]:
    """Write the same logical dataset at several sample-chunk sizes.

    Produces ``{url_prefix}_c{spc}.zarr`` per ``spc``. Random values differ across
    chunkings (chunk-by-chunk fill), but byte count, dtype, inner shape, and codec
    match — so throughput is comparable along the chunk-size axis.
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
    p.add_argument("--sample-chunk", type=int, default=None, help="override regime default")
    p.add_argument("--variables", default="t2m", help="comma-separated names")
    p.add_argument("--uncompressed", action="store_true")
    p.add_argument("--seed", type=int, default=0)
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
    )


if __name__ == "__main__":
    main()
