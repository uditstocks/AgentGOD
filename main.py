"""AgentGod - entry point.

One permanent main agent that builds, runs, and manages
task-specific agents on the fly.

The only module that talks to a human - and it talks through `ui`:
every visual decision (color, animation, layout, degradation) lives
there, so this file stays about the conversation, not the paint.

Nothing here imports the rest of the project at module level: preflight()
has to be able to explain a missing dependency, and it cannot do that from
inside the traceback of the import that failed. `ui` is the one exception
by design - it needs nothing but the standard library.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MIN_PYTHON = (3, 10)
QUIT_WORDS = frozenset({"quit", "exit", "q"})
TASK_PROMPT = "What do you need done?\n> "

# import name -> pip name, for the one message that has to be right.
# `rich` is deliberately absent: the interface degrades without it.
REQUIREMENTS = {
    "langchain_openai": "langchain-openai",
    "langchain_core": "langchain-core",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
}

_ACTIVE_UI = None


def _ui():
    """The session's renderer, created on first use so tests can intercept."""
    global _ACTIVE_UI
    if _ACTIVE_UI is None:
        from ui import make_ui

        _ACTIVE_UI = make_ui()
    return _ACTIVE_UI


def _reset_ui() -> None:
    """Forget the renderer, so the next _ui() re-detects what is installed."""
    global _ACTIVE_UI
    _ACTIVE_UI = None


def _force_utf8_output() -> None:
    """Never let an em dash or an emoji kill a finished run.

    A model's answer routinely contains characters a legacy console encoding
    cannot represent. Redirected stdout defaults to that encoding, so printing
    the result would raise UnicodeEncodeError after every call has been paid for.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


# A UTF-8 BOM at the head of piped stdin (PowerShell adds one), in both the
# decoded form and the mojibake cp1252 form. Invisible characters must never
# make an answer unrecognisable.
_STDIN_NOISE = ("﻿", "​", "ï»¿")


def ask(message: str, raw: bool = False) -> str | None:
    """Prompt the user. Returns None when they end the session (EOF or Ctrl-C).

    `raw` keeps leading whitespace, which matters only inside a pasted block:
    stripping it would silently reindent the code or Markdown being pasted.
    """
    try:
        answer = _ui().input(message)
    except (EOFError, KeyboardInterrupt):
        _ui().blank()
        return None
    for noise in _STDIN_NOISE:
        answer = answer.replace(noise, "")
    return answer.rstrip() if raw else answer.strip()


def _confirm(message: str) -> bool:
    """Yes/no prompt that treats a non-interactive stdin as 'no'."""
    answer = ask(message)
    return answer is not None and answer.lower() in ("y", "yes")


# --------------------------------------------------------------------------
# preflight: everything that must be true before the project can start
# --------------------------------------------------------------------------


def _check_python() -> bool:
    if sys.version_info >= MIN_PYTHON:
        return True
    need = ".".join(str(part) for part in MIN_PYTHON)
    have = ".".join(str(part) for part in sys.version_info[:3])
    _ui().error(f"AgentGod needs Python {need} or newer. This is Python {have}.")
    _ui().note(f"  {sys.executable}")
    return False


def _missing_packages() -> list[str]:
    """pip names of the requirements that are not importable."""
    import importlib.util

    missing = []
    for module, package in REQUIREMENTS.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(package)
    return missing


def _check_dependencies() -> bool:
    missing = _missing_packages()
    if not missing:
        return True

    _ui().warn("Missing dependencies: " + ", ".join(missing))
    command = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]

    if not _confirm("Install them now? [y/N]: "):
        _ui().note("\nInstall them with:\n  pip install -r requirements.txt")
        return False

    _ui().blank()
    if subprocess.run(command, cwd=str(PROJECT_DIR)).returncode != 0:
        _ui().error("\npip failed. Install them manually:\n  pip install -r requirements.txt")
        return False

    still_missing = _missing_packages()
    if still_missing:
        _ui().error("\nStill missing after install: " + ", ".join(still_missing))
        return False

    # requirements.txt includes `rich`; re-detect so this very session
    # gets the full interface the install just made possible.
    _reset_ui()
    _ui().success("\nDependencies installed.\n")
    return True


def _looks_like_openrouter_key(value: str) -> bool:
    """A cheap shape check before anything is ever written to .env."""
    return value.startswith("sk-or-") and len(value) >= 24


def _check_api_key() -> bool:
    """Load .env, then make sure a usable key is present - offering to write one.

    The paste prompt only exists for a human at a terminal. Piped stdin is
    data, not a person: consuming a line of it here would persist arbitrary
    text as the key and silently destroy a working .env.
    """
    from dotenv import load_dotenv

    env_file = PROJECT_DIR / ".env"
    load_dotenv(env_file)

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key and not key.startswith("sk-or-..."):
        return True

    _ui().warn("No OPENROUTER_API_KEY found.")
    _ui().note("Get one at https://openrouter.ai/keys\n")

    if not sys.stdin.isatty():
        _ui().note("Set it in .env, or as an environment variable:")
        _ui().note("  OPENROUTER_API_KEY=sk-or-...")
        return False

    entered = ask("Paste your key here (or press Enter to exit): ")
    if not entered:
        _ui().note("\nSet it in .env, or as an environment variable:")
        _ui().note("  OPENROUTER_API_KEY=sk-or-...")
        return False

    if not _looks_like_openrouter_key(entered):
        _ui().warn("That does not look like an OpenRouter key (sk-or-...). Nothing was saved.")
        return False

    # Write it back so this only ever happens once.
    try:
        lines = []
        if env_file.exists():
            lines = [
                line
                for line in env_file.read_text(encoding="utf-8").splitlines()
                if not line.strip().startswith("OPENROUTER_API_KEY=")
            ]
        lines.append(f"OPENROUTER_API_KEY={entered}")
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _ui().success(f"Saved to {env_file.name} (gitignored).\n")
    except OSError as error:
        _ui().warn(f"Could not write .env ({error}); using the key for this session only.\n")

    os.environ["OPENROUTER_API_KEY"] = entered
    return True


def preflight() -> bool:
    """Verify the project can actually run. Explains anything it cannot fix."""
    return _check_python() and _check_dependencies() and _check_api_key()


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------


def _session_banner() -> None:
    """The startup screen: who this is, what it will spend, what it remembers."""
    from config import MAX_AGENTS, MODEL, RUNS_DIR
    from library import catalogue

    try:
        run_count = sum(1 for _ in RUNS_DIR.glob("*.md")) if RUNS_DIR.is_dir() else 0
    except OSError:
        run_count = 0
    _ui().banner(MODEL, MAX_AGENTS, len(catalogue()), run_count)
    _ui().hint()


def ask_keep(result) -> None:
    """Let the user decide the fate of every agent this run had to build.

    Only newly built agents are offered: anything reused was kept by an
    earlier decision, so re-asking would be noise. Nothing enters the library
    without the user saying so.
    """
    if not result.pending:
        return
    from library import remember

    names = ", ".join(sorted(result.pending))
    count = len(result.pending)
    noun = "agent" if count == 1 else "agents"

    while True:
        choice = ask(f"\nKeep the {count} new {noun} ({names}) for reuse? [keep/discard]: ")
        if choice is None:
            # Non-interactive or Ctrl-C: keeping is the non-destructive default,
            # and it is the whole point of the library.
            choice = "keep"
            _ui().note("  (no answer - keeping)")

        choice = choice.lower()
        if choice in ("keep", "k", ""):
            # The task goes in with the agent: it is what a later run checks
            # the agent against before handing it back as reusable.
            kept = [
                name
                for name, (role, source) in sorted(result.pending.items())
                if remember(name, role, source, task=getattr(result, "task", ""))
            ]
            if kept:
                _ui().success(f"  Kept for reuse: {', '.join(kept)}")
            else:
                _ui().warn("  Could not save.")
            return
        if choice in ("discard", "d", "delete"):
            _ui().note("  Discarded. They will be rebuilt if a future task needs them.")
            return
        _ui().warn("  Please type 'keep' or 'discard'.")


def cleanup(agent_paths: list[Path]) -> None:
    """Remove the working copies from generated_agents/.

    These are scratch either way: a kept agent lives in the library, and a
    discarded one should leave nothing behind.
    """
    if not agent_paths:
        return
    from inventory import delete_agents

    delete_agents(agent_paths)


def report(task: str, result) -> None:
    """Archive the answer, then hand everything to the interface to present.

    The answer is already paid for by the time this runs, so neither a broken
    archive write nor a rendering bug is allowed to lose it: both fall back
    rather than raise.
    """
    from runlog import save_run

    try:
        saved = save_run(task, result)
    except Exception:
        saved = None

    try:
        _ui().run_succeeded(result, saved)
    except Exception:
        from ui import PlainUI

        PlainUI().run_succeeded(result, saved)


def answer_directly(text: str) -> str | None:
    """The reply for a line that is conversation rather than work, or None.

    Nothing here calls a model. A question about this system is answered from
    this system - the previous behaviour was to build agents to answer it, and
    those agents, knowing nothing about AgentGod, described one that does not
    exist.
    """
    import identity
    from router import Intent, classify

    intent = classify(text)
    if intent is Intent.GREETING:
        return identity.describe_greeting()
    if intent is Intent.THANKS:
        return identity.describe_thanks()
    if intent is Intent.CAPABILITY:
        return identity.describe_capabilities()
    if intent is Intent.IDENTITY:
        return identity.describe_identity()
    if intent is Intent.LIVE_DATA:
        return identity.describe_live_data_limit(text)
    if intent is Intent.HELP:
        from commands import help_text

        return help_text()
    return None


def prepare(task: str, conversation=None) -> tuple[str, list[str], str]:
    """Turn what the user typed into what the pipeline should receive.

    Returns the prepared task, the labels of any local files that were read,
    and the earlier request this one is continuing (empty if it stands alone).
    Both steps are announced by the caller: reading a file sends it to a model
    provider, and folding in context changes what the answer is about.
    """
    from attachments import attach

    attached = attach(task)
    labels = [item.label() for item in attached.files]
    prepared = attached.task

    carried = ""
    if conversation is not None:
        prepared, used = conversation.contextualise(prepared)
        if used and conversation.last is not None:
            carried = conversation.last.task
    return prepared, labels, carried


def run_task(task: str, echo_task: bool = False, conversation=None) -> bool:
    """Run one task end to end. Returns False only if the task itself failed.

    A line that is conversation never reaches the pipeline: it is answered
    here, for nothing, and the session moves on.
    """
    from orchestrator import handle_task

    ui = _ui()

    reply = answer_directly(task)
    if reply is not None:
        ui.run_started(task, echo=echo_task)
        ui.reply(reply)
        return True

    agent_paths: list[Path] = []
    ok = False
    try:
        ui.run_started(task, echo=echo_task)
        prepared, labels, carried = prepare(task, conversation)
        if labels:
            ui.attachments_read(labels)
        if carried:
            ui.context_carried(carried)

        # The pipeline runs on `prepared`, which may carry a file's contents or
        # an earlier exchange. The guards judge against `task` - what the user
        # actually asked - because none of that folded-in material is theirs.
        result = handle_task(
            prepared, on_agent_created=agent_paths.append, events=ui, subject=task
        )
        # The archive records what the user asked, not the expanded form the
        # pipeline was given - the expansion is plumbing, not the request.
        report(task, result)
        if conversation is not None:
            conversation.remember(task, result.response)
        ask_keep(result)
        ok = True
    except KeyboardInterrupt:
        ui.run_cancelled()
    except Exception as error:  # one failed task must not end the session
        ui.run_failed(error)
    finally:
        # The live display must come down first, but scratch cleanup must
        # happen even if tearing it down somehow fails.
        try:
            ui.run_ended()
        finally:
            cleanup(agent_paths)
    return ok


# Openers that start a multi-line task, and the lines that end one. A single
# input() call reads one line, so a pasted paragraph used to arrive as several
# separate tasks - each line planned, built and billed on its own.
BLOCK_OPENERS = frozenset({'"""', "'''", "<<<", "```"})
BLOCK_CLOSERS = frozenset({'"""', "'''", ">>>", "```", "."})
CONTINUATION = "\\"


def read_block() -> str:
    """Collect lines until a closing marker, keeping them exactly as typed."""
    _ui().note("  (multi-line: finish with a line containing only . )")
    lines: list[str] = []
    while True:
        line = ask("  | ", raw=True)
        if line is None or line.strip() in BLOCK_CLOSERS:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def read_input() -> str | None:
    """One unit of user input, however many lines it takes to type."""
    line = ask("\n" + TASK_PROMPT)
    if line is None:
        return None
    if line in BLOCK_OPENERS:
        return read_block()
    # A trailing backslash means the sentence is not finished yet.
    while line.endswith(CONTINUATION):
        more = ask("  | ")
        if more is None:
            break
        line = line[: -len(CONTINUATION)].rstrip() + " " + more
    return line


def _strip_plain_flag(args: list[str]) -> tuple[list[str], bool]:
    """Remove --plain from the arguments that are ours.

    Anything at or after --task is task text and is never touched: a task
    that happens to contain the words "--plain" must reach the model intact.
    """
    boundary = args.index("--task") if "--task" in args else len(args)
    if "--plain" not in args[:boundary]:
        return args, False
    head = [argument for argument in args[:boundary] if argument != "--plain"]
    return head + args[boundary:], True


def main() -> int:
    _force_utf8_output()
    args, plain = _strip_plain_flag(sys.argv[1:])
    if plain:
        os.environ["AGENTGOD_PLAIN"] = "1"

    if args and args[0] in ("-h", "--help"):
        _ui().help()
        return 0

    if args and args[0].startswith("-") and args[0] != "--task":
        # A mistyped flag must fail loudly, not fall into a session that a
        # CI pipeline would then hang on.
        _ui().error(f"Unknown option: {args[0]}")
        _ui().note('Usage: python main.py [--plain] [--task "..."]')
        return 2

    if not preflight():
        return 1

    # Everything below is safe to import: preflight proved the deps exist.
    if args and args[0] == "--task":
        task = " ".join(args[1:]).strip()
        if not task:
            _ui().error('Usage: python main.py --task "what you need done"')
            return 1
        return 0 if run_task(task, echo_task=True) else 1

    from commands import PASTE, QUIT, handle, parse
    from conversation import Conversation
    from router import Intent, classify

    _session_banner()
    conversation = Conversation()
    while True:
        task = read_input()
        if task is None or task.lower() in QUIT_WORDS:
            _ui().farewell()
            return 0
        if not task:
            continue

        command = parse(task)
        if command is not None:
            output = handle(command, conversation)
            if output == QUIT:
                _ui().farewell()
                return 0
            if output != PASTE:
                _ui().reply(output)
                continue
            task = read_block()
            if not task:
                continue

        # "bye" is a farewell, not a task about the word "bye".
        if classify(task) is Intent.FAREWELL:
            _ui().farewell()
            return 0

        run_task(task, conversation=conversation)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
