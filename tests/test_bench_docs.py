"""Every documented `python -m bench` command must still parse.

The runbooks are copy-paste instructions for a run that costs an EC2 box and hours of
S3 reads, so a command that no longer parses is not a typo -- it is an operator pasting
a block, watching it die on ``unrecognized arguments``, and having to reconstruct the
intent from the source. That is what happened to ``--storage s3``: the suite started
deriving storage from the URL scheme and dropped the flag, and six documented commands
across the benchmarks page and the benchmark plan kept passing it.

Only flags are checked, not behaviour: the point is that the command *shape* in the docs
and the parser cannot drift apart silently. Values are not run, and shell variables are
substituted with a placeholder -- the docs define them in an earlier block.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from bench.run import build_parser

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "docs" / "benchmarks.md",
    ROOT / "bench" / "benchmark_plan.md",
    ROOT / "bench" / "ops_aws.md",
    ROOT / "bench" / "ops_gcp.md",
]

# `uv run python -m bench` (the suite), up to the end of the command: shell line
# continuations are joined. `-m bench.probe_*` is deliberately not matched -- the probes
# carry their own parsers, and a shared regex that silently matched them would check
# their flags against the wrong one.
COMMAND = re.compile(r"(?:uv run )?python -m bench(?![.\w])(?P<args>(?:\\\n|[^\n])*)")
SHELL_VAR = re.compile(r"\$\{?\w+\}?")


def _commands() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in DOCS:
        for block in re.findall(r"```(?:bash|sh)\n(.*?)```", path.read_text(), re.S):
            for match in COMMAND.finditer(block):
                args = match.group("args").replace("\\\n", " ")
                found.append((f"{path.name}: {match.group(0).splitlines()[0][:60]}", args))
    return found


def test_the_docs_actually_contain_run_commands() -> None:
    """The control: a regex that matches nothing would make every test below vacuous."""
    assert len(_commands()) >= 5


@pytest.mark.parametrize("where,args", _commands(), ids=lambda v: v if isinstance(v, str) else "")
def test_a_documented_command_parses(where: str, args: str) -> None:
    argv = [SHELL_VAR.sub("placeholder", tok) for tok in shlex.split(args)]
    parser = build_parser()
    # Value domains are dropped, flag names are not: the docs legitimately pass shell
    # variables into choice-valued flags (`--backend "$BK"`), and a placeholder is never
    # a valid choice. Checking that a flag *exists* is the drift this test is for.
    for action in parser._actions:
        action.choices = None
    try:
        parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits rather than raising on a bad flag
        pytest.fail(f"{where}\n  args: {args.strip()}\n  argparse rejected it ({exc})")
