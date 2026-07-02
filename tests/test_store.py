"""Store constructors that need no live backend to test the wiring.

``arraylake_store`` is a thin promotion of the Arraylake/Icechunk session-store
recipe into the library; it needs a client + auth + network to run for real, so
here we mock the ``arraylake`` module and assert only the call chain + argument
threading -- the part that would break silently on a rename.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from insitubatch import arraylake_store


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
