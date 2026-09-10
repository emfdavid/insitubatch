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


def test_close_store_is_a_no_op_on_a_loop_that_is_not_running() -> None:
    """The branch that makes the test above need its wait.

    `close_store` schedules the close on the session's own loop, which it can only do while
    that loop is running. A caller closing a store whose loop has already stopped gets a
    no-op rather than an error -- teardown is best-effort. The consequence for tests is that
    calling it before the loop starts silently does nothing, which is a passing call and a
    failing assertion.
    """
    loop = asyncio.new_event_loop()  # created, never run

    class FakeSession:
        def __init__(self) -> None:
            self.closed = False
            self._loop = loop

        async def close(self) -> None:  # pragma: no cover - must not be reached
            self.closed = True

    session = FakeSession()
    fs = type("FakeFS", (), {"_session": session})()

    close_store(type("FakeStore", (), {"fs": fs})())  # type: ignore[arg-type]

    assert session.closed is False
    assert fs._session is session, "the handle is left alone when nothing was closed"
    loop.close()


def test_close_store_closes_async_session_on_its_loop() -> None:
    # An fsspec/gcsfs store's aiohttp session lives on some loop; close_store must close
    # it *on that loop* and drop the handle so gcsfs's finalizer is a no-op. Mocked so it
    # runs without a cloud backend: a live loop in a thread + a fake session.
    loop = asyncio.new_event_loop()
    # Wait for the loop to actually be running before closing: `close_store` returns early
    # on a loop that is not, so calling it while the thread is still starting up asserts
    # nothing. `call_soon` is queued before `run_forever` and fires as soon as it does.
    running = threading.Event()
    threading.Thread(
        target=lambda: (loop.call_soon(running.set), loop.run_forever()), daemon=True
    ).start()
    assert running.wait(5), "the loop never started"

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
