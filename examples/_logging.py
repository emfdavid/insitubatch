"""Logging setup for the examples.

insitubatch emits one INFO line per epoch summarising both pools -- chunk-cache hit rate and
peak residency, batch-buffer count/kind/reuse -- but it never configures logging, because a
library that calls ``basicConfig`` hijacks its caller's setup. Applications do that, and the
examples are applications, so the wiring lives here and every example CLI gets the same flag.

Only the ``insitubatch`` logger is turned up. Setting the *root* level to INFO instead would
also unmute zarr, obstore, gcsfs and aiohttp, which bury the one line worth reading.
"""

from __future__ import annotations

import argparse
import logging

LEVELS = ("debug", "info", "warning", "error")
DEFAULT_LEVEL = "info"
"""Examples show the per-epoch summary by default -- seeing what the loader did is the point."""


def add_log_level(parser: argparse.ArgumentParser) -> None:
    """Add ``--log-level`` to an example's CLI."""
    parser.add_argument(
        "--log-level",
        choices=LEVELS,
        default=DEFAULT_LEVEL,
        help="insitubatch log level (default: info -- one summary line per epoch)",
    )


def configure_logging(level: str = DEFAULT_LEVEL) -> None:
    """Install a stderr handler and set the ``insitubatch`` logger's level.

    ``basicConfig`` supplies the handler and leaves everything else at WARNING; only the
    insitubatch logger is raised, so third-party INFO chatter stays out of the way.
    """
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.WARNING)
    logging.getLogger("insitubatch").setLevel(getattr(logging, level.upper()))
