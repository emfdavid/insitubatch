"""Storage shim: one URL, any backend.

The whole local-now / cloud-later story is a single function. ``obstore`` already
dispatches on URL scheme (``file://``, ``s3://``, ``gs://``, ``az://``,
``memory://``) and ``zarr.storage.ObjectStore`` wraps it for the async zarr path.
So Phase 0 (local ``file://``) and Phase 1 (``s3://...``) differ only in the URL --
no hot-path code change, and the read path stays pure Rust (no fsspec layer).

We deliberately do *not* route through fsspec / universal_pathlib on the read
hot path: the entire thesis is that obstore wins by bypassing the fsspec/s3fs
Python layer. (obstore.fsspec exists if path-style ergonomics are ever wanted
off the hot path -- but not here.)
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import numpy as np
import obstore
import zarr
import zarr.storage

from .types import ArrayGeometry


def store_from_url(url: str, *, read_only: bool = True, **kwargs: Any) -> zarr.storage.ObjectStore:
    """Return a zarr ObjectStore for ``url`` (any obstore-supported scheme).

    ``file:///abs/path.zarr`` for local; ``s3://bucket/path.zarr`` for cloud.
    Extra ``kwargs`` pass through to ``obstore.store.from_url`` (region,
    credentials, client options, ...).
    """
    obs = obstore.store.from_url(url, **kwargs)
    return zarr.storage.ObjectStore(obs, read_only=read_only)


def ensure_local_dir(url: str) -> str:
    """For a ``file://`` URL, create the target directory so writes can land.

    obstore's LocalStore will not create the prefix for you. No-op for non-file
    schemes. Returns the URL unchanged for chaining.
    """
    parsed = urlparse(url)
    if parsed.scheme in ("", "file"):
        os.makedirs(parsed.path, exist_ok=True)
    return url


def open_geometries(
    url: str,
    variables: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, ArrayGeometry]:
    """Introspect a zarr group at ``url`` into ``{name: ArrayGeometry}``.

    Lets ``InSituDataset`` be built from a URL alone -- geometry (shape, chunks,
    dtype) is read from the array metadata rather than hand-specified.
    """
    store = store_from_url(url, **kwargs)
    group = zarr.open_group(store=store, mode="r")
    names = variables if variables is not None else [k for k, _ in group.arrays()]
    out: dict[str, ArrayGeometry] = {}
    for name in names:
        arr = group[name]  # raises KeyError if the name is absent
        if not isinstance(arr, zarr.Array):
            raise TypeError(
                f"{name!r} in {url} is a {type(arr).__name__}, not an array; "
                "open_geometries handles arrays (variables), not subgroups."
            )
        out[name] = ArrayGeometry(
            name=name,
            shape=tuple(arr.shape),
            chunks=tuple(arr.chunks),
            dtype=np.dtype(arr.dtype),
        )
    return out
