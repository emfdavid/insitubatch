"""Storage: the engine reads a zarr ``Store``; constructors build one per backend.

The engine's whole contract is "a zarr-v3 ``Store``" -- the hot path only ever
speaks that interface, so any backend works. There is no URL-vs-object dispatch:
callers pick a constructor for their backend and hand the resulting ``Store`` to
:class:`~insitubatch.source.InSituDataset` and :func:`open_geometries`.

- :func:`obstore_store` -- URL-addressable stores via ``obstore`` (``file://``,
  ``s3://``, ``gs://``, ``az://``, ``memory://``). Pure-Rust read path, no fsspec
  layer; the local-now / cloud-later story is just a different URL.
- :func:`fsspec_store` -- fsspec-backed, for what obstore does not reach (GCS
  Rapid/zonal over gRPC, requester-pays).
- :func:`arraylake_store` -- an Arraylake/Icechunk session store, bound to a
  repository snapshot/branch (no URL round-trips to it, so it must be an object).

Anything that is already a zarr ``Store`` (a custom store, an Icechunk session)
is passed straight to the engine -- no constructor needed.
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


def obstore_store(url: str, *, read_only: bool = True, **kwargs: Any) -> Store:
    """Return an obstore-backed zarr ``Store`` for ``url`` (any obstore scheme).

    ``file:///abs/path.zarr`` for local; ``s3://bucket/path.zarr`` for cloud.
    Extra ``kwargs`` pass through to ``obstore.store.from_url`` (region,
    credentials, client options, ...). The read path stays pure Rust -- no fsspec
    Python layer.
    """
    obs = obstore.store.from_url(url, **kwargs)
    return zarr.storage.ObjectStore(obs, read_only=read_only)


def fsspec_store(url: str, *, read_only: bool = True, **storage_options: Any) -> Store:
    """Return a zarr ``FsspecStore`` for ``url`` (any fsspec-supported backend).

    Reaches stores via a backend fsspec filesystem -- notably GCS Rapid/zonal
    buckets (gRPC) and GCS requester-pays, which obstore does not currently
    support. ``**storage_options`` pass straight through to
    ``FsspecStore.from_url`` (credentials, project, endpoint, Rapid config, ...).

    Requires an fsspec backend for the URL scheme: ``insitubatch[gcsfs]`` for
    ``gs://``, or bring your own (``s3fs``, ...). A sync backend (e.g. local
    ``file://``) is auto-wrapped as async by zarr; ``gs://`` via gcsfs is
    natively async. See :func:`obstore_store` for the obstore-backed constructor.
    """
    return zarr.storage.FsspecStore.from_url(
        url, storage_options=storage_options or None, read_only=read_only
    )


def arraylake_store(repo: str, *, branch: str = "main") -> Store:
    """Open an Arraylake repo and return its read-only Icechunk session store.

    Auth comes from a cached ``al auth login`` or ``ARRAYLAKE_TOKEN``; the client
    vends the bucket credentials for the repo. The returned object is a zarr-v3
    ``Store`` bound to the branch snapshot -- exactly what the engine accepts.
    Requires ``insitubatch[arraylake]``.
    """
    from arraylake import Client

    return Client().get_repo(repo).readonly_session(branch).store


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
    store: Store,
    variables: list[str] | None = None,
) -> dict[str, ArrayGeometry]:
    """Introspect a zarr group ``Store`` into ``{name: ArrayGeometry}``.

    Lets ``InSituDataset`` be built from a store alone -- geometry (shape,
    chunks, dtype) is read from the array metadata rather than hand-specified.
    Build the ``store`` with :func:`obstore_store` / :func:`fsspec_store` /
    :func:`arraylake_store`, or pass any prebuilt zarr ``Store``.
    """
    group = zarr.open_group(store=store, mode="r")
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
