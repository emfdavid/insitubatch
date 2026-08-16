"""Environment report for bug reports -- one paste instead of a dozen questions.

``print_debug_info()`` dumps the versions and interpreter facts that actually change
insitubatch's behavior: the storage stack (zarr / obstore / numpy), whichever framework
adapter is installed, and the **free-threading state**. That last one matters more here
than in most libraries -- the ``ChunkPool``'s lock-free disjoint scatter is exercised very
differently on a 3.13t build with the GIL genuinely off, and "works for me" reports have
turned on exactly that difference::

    python -c "import insitubatch; insitubatch.print_debug_info()"

Nothing here **imports** an optional dependency: versions come from installed distribution
metadata. Importing torch and JAX in one process crashes (duplicate OpenMP/XLA runtimes),
so a debug helper that imported what it reports on would take the process down precisely
when someone is trying to report a bug.
"""

from __future__ import annotations

import os
import platform
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, version
from typing import TextIO

# Reported in this order. Left is the label, right is the *distribution* name (which is not
# always the import name: tensorflow, cupy-cuda12x, kvikio-cu12). Core deps first, then the
# optional extras -- absent ones are reported, not skipped, because "torch: not installed"
# is itself the answer to a good fraction of adapter reports.
_CORE: tuple[tuple[str, str], ...] = (
    ("zarr", "zarr"),
    ("numpy", "numpy"),
    ("xarray", "xarray"),
    ("obstore", "obstore"),
)

_OPTIONAL: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("jax", "jax"),
    ("tensorflow", "tensorflow"),
    ("icechunk", "icechunk"),
    ("virtualizarr", "virtualizarr"),
    ("kerchunk", "kerchunk"),
    ("fsspec", "fsspec"),
    ("gcsfs", "gcsfs"),
    ("s3fs", "s3fs"),
    ("cloudpickle", "cloudpickle"),
    ("cupy", "cupy-cuda12x"),
    ("kvikio", "kvikio-cu12"),
)

_NOT_INSTALLED = "not installed"


def _dist_version(dist: str) -> str:
    try:
        return version(dist)
    except PackageNotFoundError:
        return _NOT_INSTALLED


def _gil_state() -> str:
    """Free-threading build flag *and* whether the GIL is live right now.

    The two differ in practice: on a 3.13t build an extension that has not declared itself
    GIL-safe (numcodecs, today) re-enables the GIL on import, so a run that looks
    free-threaded is not. Report both.
    """
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    if not free_threaded:
        return "GIL build (not free-threaded)"
    # sys._is_gil_enabled() exists only on free-threading-capable interpreters (3.13+).
    is_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_enabled is None:  # pragma: no cover - unreachable on a real 3.13t build
        return "free-threaded build, live GIL state unknown"
    return f"free-threaded build, GIL currently {'ON' if is_enabled() else 'OFF'}"


def debug_info() -> dict[str, str]:
    """Return the environment report as an ordered ``label -> value`` mapping.

    The structured half of :func:`print_debug_info`; useful if you want to attach the same
    facts to a benchmark record rather than paste them into an issue.
    """
    # Deferred so this module can be imported from the package __init__ without a cycle;
    # __init__ owns the one version-resolution fallback, and we do not want a second.
    from . import __version__

    info = {
        "insitubatch": __version__,
        "python": f"{platform.python_version()} ({platform.python_implementation()})",
        "free-threading": _gil_state(),
        "platform": platform.platform(),
        "cpu count": str(os.cpu_count()),
    }
    info.update({label: _dist_version(dist) for label, dist in _CORE})
    info.update({label: _dist_version(dist) for label, dist in _OPTIONAL})
    return info


def print_debug_info(file: TextIO | None = None) -> None:
    """Print the environment report. Paste the output into a bug or performance report."""
    info = debug_info()
    width = max(len(label) for label in info)
    out = file if file is not None else sys.stdout
    print("insitubatch debug info", file=out)
    for label, value in info.items():
        print(f"  {label:<{width}} : {value}", file=out)
