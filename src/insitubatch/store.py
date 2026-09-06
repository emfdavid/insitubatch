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
- :func:`icechunk_store` -- an Icechunk repository you address by URL, public or with
  your own cloud credentials.
- :func:`arraylake_store` -- an Icechunk repository *hosted by Arraylake*, addressed by
  catalog name and opened with an Arraylake login rather than cloud credentials.

The last two both end in an Icechunk session store, and which one you want is decided by
where the repository lives, not by preference: a bucket you can name in a URL is
:func:`icechunk_store`; a repo in an Arraylake catalog has no such URL and is
:func:`arraylake_store`. They are not alternatives for the same repository.

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


def icechunk_store(
    url: str,
    *,
    branch: str = "main",
    anonymous: bool = False,
    region: str | None = None,
    **kwargs: Any,
) -> Store:
    """Return the read-only session ``Store`` of an Icechunk repository at ``url``.

    ``s3://bucket/prefix`` (the common case for public archives), ``gs://``, or a local
    ``file:///path``. Public buckets need ``anonymous=True``; otherwise credentials come
    from the environment the way every other AWS/GCS tool finds them (profile, instance
    role, ``AWS_*`` variables, an SSO login). Extra ``kwargs`` pass through to Icechunk's
    storage constructor.

    Icechunk is a *versioned* format, so a store is a snapshot: ``branch`` picks which one,
    and the returned session is read-only, which is what the engine wants -- training reads
    a fixed view of the data even while a writer appends to the repo.

    Requires ``insitubatch[icechunk]``::

        store = icechunk_store(
            "s3://dynamical-noaa-gfs/noaa-gfs-analysis/v0.1.0.icechunk",
            anonymous=True, region="us-west-2",
        )

    For an Arraylake-hosted repo use :func:`arraylake_store` instead: it is addressed by
    catalog name rather than URL and authenticates with an Arraylake login.
    """
    import icechunk

    parsed = urlparse(url)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in ("s3", "s3a"):
        storage = icechunk.s3_storage(
            bucket=bucket, prefix=prefix, region=region, anonymous=anonymous, **kwargs
        )
    elif parsed.scheme in ("gs", "gcs"):
        storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix, **kwargs)
    elif parsed.scheme in ("", "file"):
        storage = icechunk.local_filesystem_storage(parsed.path or url, **kwargs)
    else:
        raise ValueError(
            f"icechunk_store does not know the scheme {parsed.scheme!r} in {url!r}; it "
            "handles s3://, gs:// and file://. For an Arraylake-hosted repository use "
            "arraylake_store(name), which is addressed by catalog name rather than URL."
        )
    return icechunk.Repository.open(storage).readonly_session(branch).store


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


def _storage_chunks(arr: object) -> tuple[int, ...]:
    """The shape of one **stored object** -- what a chunk key addresses.

    On a **sharded** array these differ and only one is the storage unit:
    ``metadata.chunks`` is the *inner* chunk (read granularity inside a shard), while
    ``chunk_grid.chunk_shape`` is the shard, which is what ``store.get(key)`` returns and
    what the codec chain (whose head is the ``ShardingCodec``) decodes. Planning reads or
    building an ``ArraySpec`` from ``chunks`` on such an array asks for a key that holds a
    shard and then decodes it as if it were one inner chunk -- which fails the shard index's
    CRC rather than saying anything useful.

    Identical on unsharded arrays, and defined for zarr-v2 metadata as well, so this is the
    single format-agnostic spelling of "one stored chunk". Read through
    ``ChunkGrid.from_metadata`` rather than ``metadata.chunk_grid``: the latter is deprecated
    on ``ArrayV2Metadata``, and v2 is not optional here (WeatherBench2 ARCO is v2).
    """
    from zarr.core.chunk_grids import ChunkGrid

    return tuple(ChunkGrid.from_metadata(arr.metadata).chunk_shape)  # type: ignore[attr-defined]


def _dimension_names(arr: object) -> tuple[str, ...] | None:
    """The array's dimension names, in either format's spelling, or ``None`` if it has none.

    zarr-v3 carries ``dimension_names`` in metadata; zarr-v2 carries xarray's
    ``_ARRAY_DIMENSIONS`` attribute. A store written by anything CF-aware has one or the
    other, and neither is guaranteed (a hand-built or OME-NGFF store may have neither).
    """
    names = getattr(arr.metadata, "dimension_names", None)  # type: ignore[attr-defined]
    if names is None:
        names = arr.attrs.get("_ARRAY_DIMENSIONS")  # type: ignore[attr-defined]
    return tuple(str(n) for n in names) if names else None


def _is_coordinate(name: str, arr: object) -> bool:
    """True for an array that describes the grid rather than being data on it.

    Two shapes, both of which a CF store puts in the same group as its variables:

    * a **grid mapping** -- 0-D, all of its content in attributes (``spatial_ref``'s
      ``crs_wkt``). It has no sample axis, so it is not merely useless to batch, it cannot
      be turned into an :class:`ArrayGeometry` at all.
    * a **coordinate** -- 1-D and named after its own dimension (``latitude`` over
      ``('latitude',)``). Batching it would deliver axis labels as if they were fields, and
      its sample-axis length is the axis length, which will not agree with any variable's.

    Deliberately narrow: an array is only a coordinate if it is *self-named*, so a 1-D data
    variable (a station series over ``('time',)``) is kept. Naming it in ``variables=``
    overrides this in either direction.
    """
    if getattr(arr, "ndim", None) == 0:
        return True
    dims = _dimension_names(arr)
    return dims is not None and len(dims) == 1 and dims[0] == name


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
    if variables is not None:
        names = variables
    else:
        # Infer: a CF group holds coordinates and a grid mapping alongside its variables,
        # and taking every array makes the 0-D one raise and the 1-D ones batchable.
        skipped = {k: a for k, a in group.arrays() if _is_coordinate(k, a)}
        names = [k for k, _ in group.arrays() if k not in skipped]
        if not names:
            raise ValueError(
                f"no batchable arrays found in {store!r}: every array in the group looks "
                f"like a coordinate or grid mapping ({sorted(skipped)}). A 0-D array has no "
                "sample axis, and a 1-D array named after its own dimension is an axis "
                "label rather than data on the grid. If one of these really is the data you "
                "mean to batch, name it explicitly -- open_geometries(store, "
                'variables=["<name>"]) -- which bypasses this inference entirely.'
            )
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
            chunks=_storage_chunks(arr),
            dtype=np.dtype(arr.dtype),
            sample_axis=sample_axis,
        )
    return out
