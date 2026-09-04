"""The command-line contract.

Everything a shell can say to AgentGod is defined here, in one argparse
surface with a stable exit-code contract - and nothing here imports the rest
of the project, so `agentgod --version` and a usage error cost nothing and
work even on a machine where the dependencies are not installed yet.

The contract:

    agentgod                          interactive session
    agentgod "write a haiku"          one task, then exit
    agentgod -                        task text read from stdin
    agentgod library|stats|history    the free, offline commands
    agentgod --task ...               everything after --task is task text

    exit 0  task succeeded        exit 2    bad usage
    exit 1  task failed           exit 130  interrupted

`--task` keeps its historical meaning - every argument after it is task
text, never a flag - so a task containing "--json" cannot be eaten by the
parser. Flags therefore go before it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

__version__ = "0.2.0"

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

# Bare first words that are session commands, not tasks. `agentgod library`
# answers from disk, free, without an API key - the same handlers the
# interactive slash commands use.
COMMAND_VERBS = frozenset({"library", "stats", "history", "audit", "forget", "last"})

EFFORTS = ("low", "medium", "high", "xhigh", "max")
COUNCIL_MODES = ("auto", "always", "off")

_EPILOG = """\
examples:
  agentgod                                   interactive session
  agentgod "write a haiku about rain"        one task, then exit
  agentgod --json "compare sqlite and duckdb"   machine-readable result on stdout
  echo summarise this repo | agentgod -      task text from stdin
  agentgod library                           what the library holds (free, no key)
  agentgod --discard "one-off experiment"    run without growing the library

exit codes:
  0  the task succeeded    2    bad usage
  1  the task failed       130  interrupted
"""

# Invisible characters PowerShell and pipes prepend to text; they must never
# make a task unrecognisable.
_STDIN_NOISE = ("﻿", "​", "ï»¿")


@dataclass
class Invocation:
    """One parsed command line, ready for main.py to act on."""

    task: str | None = None
    command: tuple[str, str] | None = None  # (verb, argument) for free commands
    plain: bool = False
    quiet: bool = False
    json_output: bool = False
    no_input: bool = False
    keep: str | None = None  # None = ask / env default, "always", "never"
    model: str | None = None
    fast_model: str | None = None
    deep_model: str | None = None
    effort: str | None = None
    council: str | None = None

    @property
    def interactive(self) -> bool:
        return self.task is None and self.command is None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentgod",
        description=(
            "One permanent agent that builds, runs and retires task-specific "
            "agents. Give it a task; it plans the team, writes each agent as "
            "real code, runs them, and merges one answer."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "words",
        nargs="*",
        metavar="TASK",
        help="the task to run ('-' reads it from stdin); omit for an interactive session",
    )
    parser.add_argument(
        "--task",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="...",
        help="everything after this flag is task text, never a flag",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--plain", action="store_true", help="no color, no animation - log-file safe"
    )
    output.add_argument(
        "-q", "--quiet", action="store_true", help="print only the answer"
    )
    output.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="print one JSON object (answer, agents, cost, outcome) on stdout",
    )

    run = parser.add_argument_group("run controls")
    run.add_argument(
        "--model", metavar="NAME", help="the workhorse model for this invocation"
    )
    run.add_argument(
        "--fast-model",
        metavar="NAME",
        help="model for the mechanical checks (clarify, judge) - the cheap half",
    )
    run.add_argument(
        "--deep-model",
        metavar="NAME",
        help="model used only for tasks the planner grades 'deep'",
    )
    run.add_argument(
        "--effort",
        choices=EFFORTS,
        help="how hard the model works per call (this invocation only)",
    )
    run.add_argument(
        "--council",
        choices=COUNCIL_MODES,
        help="the adversarial answer review: auto (deep tasks), always, off",
    )

    keeping = parser.add_argument_group("library policy")
    policy = keeping.add_mutually_exclusive_group()
    policy.add_argument(
        "--keep",
        action="store_true",
        help="keep every newly built agent without asking",
    )
    policy.add_argument(
        "--discard",
        action="store_true",
        help="discard newly built agents without asking",
    )
    keeping.add_argument(
        "--no-input",
        action="store_true",
        help="never prompt (for scripts); defaults answer every question",
    )

    parser.add_argument(
        "-V", "--version", action="version", version=f"agentgod {__version__}"
    )
    return parser


def _clean(text: str) -> str:
    for noise in _STDIN_NOISE:
        text = text.replace(noise, "")
    return text.strip()


def parse(argv: list[str]) -> Invocation:
    """Read one command line into an Invocation.

    argparse handles --help/--version (SystemExit 0) and usage errors
    (SystemExit 2) itself; both are part of the exit-code contract, so they
    are left to propagate.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    task: str | None = None
    command: tuple[str, str] | None = None

    if args.task is not None:
        # The historical escape hatch: everything after --task is task text.
        task = _clean(" ".join(args.task))
        if not task:
            parser.error("--task needs the task text after it")
    elif args.words:
        first = args.words[0].lower()
        if first in COMMAND_VERBS:
            command = (first, " ".join(args.words[1:]).strip())
        else:
            task = _clean(" ".join(args.words))

    if task == "-":
        # The Unix spelling of "the task is on stdin".
        task = _clean(sys.stdin.read())
        if not task:
            parser.error("stdin carried no task text")

    if args.json_output and task is None and command is None:
        parser.error("--json needs a task (or a command) to report on")

    return Invocation(
        task=task or None,
        command=command,
        plain=args.plain,
        quiet=args.quiet,
        json_output=args.json_output,
        no_input=args.no_input,
        keep="always" if args.keep else ("never" if args.discard else None),
        model=args.model,
        fast_model=args.fast_model,
        deep_model=args.deep_model,
        effort=args.effort,
        council=args.council,
    )


def apply(invocation: Invocation) -> None:
    """Write the per-invocation overrides where the project reads them.

    config.py reads these names from the environment at import time, and
    main.py defers importing config until after parsing - so a --model here
    is indistinguishable from MODEL in .env, with no second configuration
    path to maintain.
    """
    import os

    if invocation.model:
        os.environ["MODEL"] = invocation.model
    if invocation.fast_model:
        os.environ["FAST_MODEL"] = invocation.fast_model
    if invocation.deep_model:
        os.environ["DEEP_MODEL"] = invocation.deep_model
    if invocation.effort:
        os.environ["LLM_EFFORT"] = invocation.effort
    if invocation.council:
        os.environ["COUNCIL"] = invocation.council
    if invocation.plain:
        os.environ["AGENTGOD_PLAIN"] = "1"
    if invocation.quiet or invocation.json_output:
        os.environ["AGENTGOD_QUIET"] = "1"
