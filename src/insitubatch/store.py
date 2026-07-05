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

import asyncio
import contextlib
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
    # LocalFileSystem does not create parent dirs on write (unlike obstore's LocalStore
    # and every object store, where prefixes are implicit), so writing a zarr's nested
    # chunk paths 404s. Default auto_mkdir for file:// so local writes behave like the
    # other backends; harmless on reads, and never sent to object stores.
    if urlparse(url).scheme in ("", "file"):
        storage_options.setdefault("auto_mkdir", True)
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


def close_store(store: Store) -> None:
    """Best-effort teardown for a store that holds an async fsspec session (gcsfs, s3fs).

    Such a backend creates an aiohttp session on the first event loop that awaits it --
    for a zarr store, that is zarr's loop, not fsspec's -- but gcsfs's finalizer captures
    ``fs.loop`` (which is ``None`` here) and closes the session on the *wrong* loop at GC,
    spewing a harmless-looking "Task was destroyed / attached to a different loop"
    traceback and leaking the connection. Closing the session here on the loop it actually
    lives on makes that finalizer a no-op.

    A no-op for stores with no such session (obstore's ``ObjectStore`` has no ``.fs``) and
    for already-closed or not-running loops. gcsfs recreates the session lazily, so a
    store closed here still works if reused -- but call this only when done with it.
    """
    fs: Any = getattr(store, "fs", None)
    session = getattr(fs, "_session", None)
    loop = getattr(session, "_loop", None)
    if session is None or loop is None or session.closed or not loop.is_running():
        return
    with contextlib.suppress(Exception):  # teardown is best-effort; never raise from close
        asyncio.run_coroutine_threadsafe(session.close(), loop).result(timeout=5)
        fs._session = None


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
    *,
    sample_axis: int = 0,
) -> dict[str, ArrayGeometry]:
    """Introspect a zarr group ``Store`` into ``{name: ArrayGeometry}``.

    Lets ``InSituDataset`` be built from a store alone -- geometry (shape,
    chunks, dtype) is read from the array metadata rather than hand-specified.
    Build the ``store`` with :func:`obstore_store` / :func:`fsspec_store` /
    :func:`arraylake_store`, or pass any prebuilt zarr ``Store``.

    ``sample_axis`` names which *physical* axis is the outer (sample) axis for
    **every** returned variable -- ``0`` (default: time for ERA5/HRRR) or, e.g., the
    ``Z`` of an OME-NGFF ``(T,C,Z,Y,X)`` stack sampled slice-by-slice (``sample_axis=2``).
    Variables that need *different* sample axes are built individually (construct
    :class:`ArrayGeometry` per array); the shape/chunks stay in physical order.
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
            sample_axis=sample_axis,
        )
    return out
