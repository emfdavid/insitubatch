"""Store-backend switch for the benchmark A/B: obstore (Rust) vs fsspec (gcsfs).

The library keeps per-backend constructors with no str-vs-Store dispatch (PEP 20,
DESIGN.md "One contract, any backend"). The benchmark is the *one* place that
deliberately builds *any* of them from a ``--backend`` flag: on the same GCS data,
obstore-over-HTTP vs fsspec/gcsfs measures how much overhead the Python fsspec
layer adds -- the deciding experiment for whether fsspec earns a co-equal fast
path (DESIGN.md M-GCS). fsspec is also the only path to GCS Rapid/zonal (gRPC),
which obstore cannot reach at all. ``arraylake`` reads an Icechunk repo (its ``url``
is the ``org/repo`` name; ``branch`` selects the ref).
"""

from __future__ import annotations

from typing import Any

from zarr.abc.store import Store

from insitubatch import arraylake_store, fsspec_store, obstore_store

BACKENDS = ("obstore", "fsspec", "arraylake")


def build_store(backend: str, url: str, *, read_only: bool = True, **kwargs: Any) -> Store:
    """Build a zarr ``Store`` for ``url`` via the named backend.

    ``kwargs`` pass straight through to the backend constructor, so callers own the
    (backend-specific) knobs -- obstore's ``request_payer`` / ``s3_express``, gcsfs's
    ``requester_pays`` / ``token``, or arraylake's ``branch``. ``arraylake`` opens a
    readonly Icechunk session, so it has no ``read_only`` toggle and ``url`` is the
    ``org/repo`` name rather than a URL.
    """
    if backend == "arraylake":
        return arraylake_store(url, **kwargs)
    if backend == "fsspec":
        return fsspec_store(url, read_only=read_only, **kwargs)
    if backend == "obstore":
        return obstore_store(url, read_only=read_only, **kwargs)
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
