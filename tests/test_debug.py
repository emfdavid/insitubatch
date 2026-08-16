"""The bug-report environment dump must never be the thing that breaks.

It runs on whatever env the reporter has -- core-only, one framework, or a 3.13t build with
no wheels for half the extras -- so the contract is: report every key, never import an
optional dependency, never raise.
"""

from __future__ import annotations

import io
import sys
import sysconfig

from insitubatch import debug_info, print_debug_info
from insitubatch.debug import _NOT_INSTALLED


def test_debug_info_reports_the_load_bearing_facts() -> None:
    info = debug_info()
    # The storage stack and the interpreter facts are what triage actually turns on.
    for key in ("insitubatch", "python", "free-threading", "platform", "cpu count"):
        assert key in info, key
    for key in ("zarr", "numpy", "xarray", "obstore"):
        assert info[key] != _NOT_INSTALLED, f"{key} is a core dependency and must resolve"
    assert info["insitubatch"]
    assert str(sys.version_info.major) in info["python"]


def test_absent_optionals_degrade_instead_of_raising() -> None:
    info = debug_info()
    # Every optional key is present regardless of installation state: "torch: not installed"
    # is itself a useful answer, so absent extras are reported, not dropped.
    for key in ("torch", "jax", "tensorflow", "icechunk", "cupy"):
        assert key in info
        assert isinstance(info[key], str) and info[key]


def test_free_threading_line_states_build_and_live_gil() -> None:
    line = debug_info()["free-threading"]
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        # On a 3.13t build we want *both* halves: the build flag and whether an import
        # (numcodecs, today) has since switched the GIL back on. A report that says only
        # "free-threaded" cannot distinguish a real FT run from a silently re-GIL'd one.
        assert "free-threaded build" in line
        assert "GIL currently" in line
        assert ("ON" in line) == bool(sys._is_gil_enabled())
    else:
        assert line == "GIL build (not free-threaded)"


def test_print_debug_info_writes_pasteable_text() -> None:
    buf = io.StringIO()
    print_debug_info(file=buf)
    text = buf.getvalue()
    assert text.startswith("insitubatch debug info")
    assert "zarr" in text and "free-threading" in text
    # One fact per line, aligned -- it goes straight into an issue form.
    assert all(" : " in line for line in text.splitlines()[1:])
