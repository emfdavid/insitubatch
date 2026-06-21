"""py-spy sampling-profiler helper for the bench diagnostics.

Sampling (not deterministic, unlike yappi/cProfile) so it barely perturbs the
threading we want to study, and ``--native`` reaches into Rust and C -- the
obstore/tokio fetch and the numcodecs decode + numpy scatter memcpy -- exactly
the parts a Python-level profiler cannot see. py-spy samples *all* threads of the
target, so the named threads (``insitu-sched`` / ``insitu-dec`` / ``insitu-prefetch``)
show up as distinct stacks.

Attach model: py-spy uses ptrace, and here it is a *child* attaching to its
parent (this process). Under the common ``kernel.yama.ptrace_scope=1`` a child may
not trace its parent, so self-profiling needs::

    sudo sysctl kernel.yama.ptrace_scope=0      # or run the probe under sudo

If py-spy is missing or cannot attach, recording is skipped with a clear message
rather than failing the measurement run.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator


@contextlib.contextmanager
def record_pyspy(out: str, *, native: bool = True) -> Iterator[None]:
    """Record a py-spy profile of *this* process for the duration of the block.

    ``out`` ending in ``.json`` => speedscope (interactive); otherwise a flamegraph
    SVG. Best-effort: a missing binary or a failed ptrace attach degrades to a
    no-op + message, so a profiling typo never costs an S3 measurement run.
    """
    exe = shutil.which("py-spy")
    if exe is None:
        print("   [profile] py-spy not found; install with `uv sync --extra bench`. skipping.")
        yield
        return

    fmt = "speedscope" if out.endswith(".json") else "flamegraph"
    cmd = [exe, "record", "--pid", str(os.getpid()), "--format", fmt, "--output", out]
    if native:
        cmd.append("--native")
    proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
    time.sleep(1.0)  # let py-spy attach before the workload starts
    if proc.poll() is not None:  # exited already => attach failed (ptrace perms)
        print(
            "   [profile] py-spy could not attach (need kernel.yama.ptrace_scope=0 "
            "or sudo for --native self-attach). skipping."
        )
        yield
        return

    try:
        yield
    finally:
        proc.send_signal(signal.SIGINT)  # py-spy flushes the output file on SIGINT
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=30)
        if proc.poll() is None:
            proc.kill()
        print(f"   [profile] wrote {out}")
