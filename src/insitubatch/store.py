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

A URL is one way to name a store, not the only one. The engine's actual contract
is "a zarr-v3 ``Store``", so callers may hand in a prebuilt store instead of a
URL (:data:`StoreLike`, normalized by :func:`as_store`). This is required for
Icechunk: a session store is bound to a repository snapshot/branch and has no URL
that round-trips to it -- but it is a zarr Store, so the hot path is unchanged.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import numpy as np
import obstore
import zarr
import zarr.storage
from zarr.abc.store import Store

from .types import ArrayGeometry

# What the engine reads from: a URL (we build an obstore-backed store) or an
# already-built zarr-v3 Store. The hot path only ever speaks the zarr Store
# interface, so any backend works -- Icechunk session stores in particular have
# no URL that round-trips to them, so they must be passed as objects.
StoreLike = str | Store
"""A store argument: either a URL string (resolved by :func:`as_store` /
:func:`store_from_url`) or an already-constructed obstore ``Store``."""


def store_from_url(url: str, *, read_only: bool = True, **kwargs: Any) -> zarr.storage.ObjectStore:
    """Return a zarr ObjectStore for ``url`` (any obstore-supported scheme).

    ``file:///abs/path.zarr`` for local; ``s3://bucket/path.zarr`` for cloud.
    Extra ``kwargs`` pass through to ``obstore.store.from_url`` (region,
    credentials, client options, ...).
    """
    obs = obstore.store.from_url(url, **kwargs)
    return zarr.storage.ObjectStore(obs, read_only=read_only)


def as_store(store: StoreLike, *, read_only: bool = True, **kwargs: Any) -> Store:
    """Normalize a URL *or* an already-built zarr Store into a zarr Store.

    A ``str`` is opened via :func:`store_from_url` (obstore-backed); an existing
    Store (e.g. an Icechunk session store) is returned unchanged. ``kwargs`` and
    ``read_only`` configure URL construction only -- passing them alongside a
    prebuilt store is a usage error (the store already carries that state).
    """
    if isinstance(store, str):
        return store_from_url(store, read_only=read_only, **kwargs)
    if kwargs:
        raise TypeError(
            f"store_kwargs {sorted(kwargs)} apply only to URL stores; a prebuilt "
            f"{type(store).__name__} was passed and already carries its configuration."
        )
    return store


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
    store: StoreLike,
    variables: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, ArrayGeometry]:
    """Introspect a zarr group (URL or Store) into ``{name: ArrayGeometry}``.

    Lets ``InSituDataset`` be built from a store spec alone -- geometry (shape,
    chunks, dtype) is read from the array metadata rather than hand-specified.
    """
    group = zarr.open_group(store=as_store(store, **kwargs), mode="r")
    names = variables if variables is not None else [k for k, _ in group.arrays()]
    out: dict[str, ArrayGeometry] = {}
    for name in names:
        arr = group[name]  # raises KeyError if the name is absent
        if not isinstance(arr, zarr.Array):
            raise TypeError(
                f"{name!r} in {store!r} is a {type(arr).__name__}, not an array; "
                "open_geometries handles arrays (variables), not subgroups."
            )
        out[name] = ArrayGeometry(
            path=name,
            shape=tuple(arr.shape),
            chunks=tuple(arr.chunks),
            dtype=np.dtype(arr.dtype),
        )
    return out
