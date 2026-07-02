"""Store constructors that need no live backend to test the wiring.

``arraylake_store`` is a thin promotion of the Arraylake/Icechunk session-store
recipe into the library; it needs a client + auth + network to run for real, so
here we mock the ``arraylake`` module and assert only the call chain + argument
threading -- the part that would break silently on a rename.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from unittest.mock import MagicMock

from insitubatch import arraylake_store, close_store, obstore_store


def test_arraylake_store_threads_repo_and_branch(monkeypatch) -> None:
    sentinel_store = object()  # stands in for the zarr session Store
    session = MagicMock()
    session.store = sentinel_store
    repo = MagicMock()
    repo.readonly_session.return_value = session
    client = MagicMock()
    client.get_repo.return_value = repo

    fake_arraylake = MagicMock()
    fake_arraylake.Client = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "arraylake", fake_arraylake)

    got = arraylake_store("org/repo", branch="dev")

    assert got is sentinel_store
    fake_arraylake.Client.assert_called_once_with()
    client.get_repo.assert_called_once_with("org/repo")
    repo.readonly_session.assert_called_once_with("dev")


def test_arraylake_store_defaults_to_main_branch(monkeypatch) -> None:
    repo = MagicMock()
    client = MagicMock()
    client.get_repo.return_value = repo
    fake_arraylake = MagicMock()
    fake_arraylake.Client = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "arraylake", fake_arraylake)

    arraylake_store("org/repo")

    repo.readonly_session.assert_called_once_with("main")


def test_close_store_is_noop_for_obstore(tmp_path) -> None:
    # obstore's ObjectStore has no async session (no .fs) -> close_store must be a
    # harmless no-op, not raise. This is the common path (default backend).
    close_store(obstore_store(f"file://{tmp_path}"))  # no exception = pass


def test_close_store_closes_async_session_on_its_loop() -> None:
    # An fsspec/gcsfs store's aiohttp session lives on some loop; close_store must close
    # it *on that loop* and drop the handle so gcsfs's finalizer is a no-op. Mocked so it
    # runs without a cloud backend: a live loop in a thread + a fake session.
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    class FakeSession:
        def __init__(self) -> None:
            self.closed = False
            self._loop = loop

        async def close(self) -> None:
            self.closed = True

    session = FakeSession()
    fs = type("FakeFS", (), {"_session": session})()
    store = type("FakeStore", (), {"fs": fs})()

    close_store(store)  # type: ignore[arg-type]  # duck-typed store stand-in

    assert session.closed is True  # closed on its own loop
    assert fs._session is None  # handle dropped so the finalizer won't re-close
    loop.call_soon_threadsafe(loop.stop)
