"""AgentGod - entry point.

One permanent main agent that builds, runs, and manages
task-specific agents on the fly.

The only module that talks to a human - and it talks through `ui`:
every visual decision (color, animation, layout, degradation) lives
there, so this file stays about the conversation, not the paint.
The shape of the command line itself lives in `cli.py`; this module
acts on the parsed invocation.

Nothing here imports the rest of the project at module level: preflight()
has to be able to explain a missing dependency, and it cannot do that from
inside the traceback of the import that failed. `ui` and `cli` are the
exceptions by design - they need nothing but the standard library.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cli

PROJECT_DIR = Path(__file__).resolve().parent
MIN_PYTHON = (3, 10)
QUIT_WORDS = frozenset({"quit", "exit", "q"})
TASK_PROMPT = "What do you need done?\n> "

# import name -> pip name, for the one message that has to be right.
# `rich` is deliberately absent: the interface degrades without it.
REQUIREMENTS = {
    "anthropic": "anthropic",
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


def _interactive() -> bool:
    """Whether a person is on the other end of stdin."""
    try:
        return sys.stdin.isatty()
    except (ValueError, OSError):
        return False


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


def _looks_like_anthropic_key(value: str) -> bool:
    """A cheap shape check before anything is ever written to .env."""
    return value.startswith("sk-ant-") and len(value) >= 24


def _read_secret(message: str) -> str | None:
    """Read a credential without echoing it, where the terminal allows.

    An API key pasted in plain sight lives on in scrollback and screen
    recordings. getpass hides it; anywhere getpass cannot run, the ordinary
    prompt is the honest fallback.
    """
    if _interactive():
        import getpass

        try:
            return getpass.getpass(message).strip()
        except (EOFError, KeyboardInterrupt):
            _ui().blank()
            return None
        except Exception:
            pass  # odd shells; fall through to the visible prompt
    return ask(message)


def _probe_key(key: str) -> str:
    """Ask the API whether this key works: 'ok', 'rejected', or 'unreachable'.

    Called only for a key a person just pasted, so the cost is one cheap
    request at the one moment it can prevent persisting a dead credential.
    Offline is not evidence of a bad key, so 'unreachable' is accepted.
    """
    try:
        import anthropic

        anthropic.Anthropic(api_key=key, max_retries=0, timeout=8.0).models.list(limit=1)
        return "ok"
    except Exception as error:
        name = type(error).__name__
        if name in ("AuthenticationError", "PermissionDeniedError"):
            return "rejected"
        return "unreachable"


def _save_key(env_file: Path, key: str) -> None:
    """Write the key back to .env so this only ever happens once."""
    try:
        lines = []
        if env_file.exists():
            lines = [
                line
                for line in env_file.read_text(encoding="utf-8").splitlines()
                if not line.strip().startswith("ANTHROPIC_API_KEY=")
            ]
        lines.append(f"ANTHROPIC_API_KEY={key}")
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _ui().success(f"Saved to {env_file.name} (gitignored).\n")
    except OSError as error:
        _ui().warn(f"Could not write .env ({error}); using the key for this session only.\n")


def _check_api_key() -> bool:
    """Load .env, then make sure a usable key is present - offering to write one.

    The same shape check guards both paths now: a mangled key hand-edited
    into .env used to sail through preflight and die mid-run as a raw 401.
    The paste prompt only exists for a human at a terminal - piped stdin is
    data, not a person - and a slip re-prompts instead of ending the process.
    """
    from dotenv import load_dotenv

    env_file = PROJECT_DIR / ".env"
    load_dotenv(env_file)

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key and not key.startswith("sk-ant-..."):
        if _looks_like_anthropic_key(key):
            return True
        _ui().warn(
            "The ANTHROPIC_API_KEY that was found does not look like an "
            "Anthropic key (they start with sk-ant-). Check .env or the "
            "environment variable."
        )

    if not key or key.startswith("sk-ant-..."):
        _ui().warn("No ANTHROPIC_API_KEY found.")
    _ui().note("Get one at https://console.anthropic.com/settings/keys\n")

    if not _interactive():
        _ui().note("Set it in .env, or as an environment variable:")
        _ui().note("  ANTHROPIC_API_KEY=sk-ant-...")
        return False

    for _ in range(3):
        entered = _read_secret("Paste your key (input hidden; Enter to exit): ")
        if not entered:
            _ui().note("\nSet it in .env, or as an environment variable:")
            _ui().note("  ANTHROPIC_API_KEY=sk-ant-...")
            return False
        if not _looks_like_anthropic_key(entered):
            shown = entered[:6] + "..." if len(entered) > 6 else entered
            _ui().warn(
                f"That does not look like an Anthropic key (yours started "
                f"'{shown}'; they start 'sk-ant-'). Try again."
            )
            continue
        # Probe only when a person is present to act on the verdict; a
        # headless environment gets the shape check and the runtime
        # translation, never a surprise network call.
        verdict = _probe_key(entered) if _interactive() else "ok"
        if verdict == "rejected":
            _ui().warn("The API rejected that key - it may be revoked or mistyped. Try again.")
            continue
        if verdict == "unreachable":
            _ui().note("(could not reach the API to verify the key - keeping it anyway)")
        _save_key(env_file, entered)
        os.environ["ANTHROPIC_API_KEY"] = entered
        return True

    _ui().warn("Three attempts - stopping here. Nothing was saved.")
    return False


def preflight(require_key: bool = True) -> bool:
    """Verify the project can actually run. Explains anything it cannot fix.

    The free commands (`agentgod library`) read only the local disk, so they
    skip the key check: asking someone for a credential to list their own
    files would be absurd.
    """
    if not (_check_python() and _check_dependencies()):
        return False
    return _check_api_key() if require_key else True


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------


@dataclass
class Outcome:
    """What one run_task call actually produced, for the caller to act on.

    run_task presents everything itself; this exists so a caller with its
    own contract - the --json printer, the exit-code logic - can see the
    result instead of re-deriving it from what was printed.
    """

    ok: bool = False
    kind: str = "task"  # task | reply
    answer: str = ""
    result: object = None
    saved: Path | None = None
    kept: list[str] = field(default_factory=list)
    cancelled: bool = False
    error: BaseException | None = None
    problem: object = None


def _session_banner() -> None:
    """The startup screen: who this is, what it will spend, what it remembers."""
    from config import MAX_AGENTS, MODEL, RUNS_DIR
    from library import catalogue

    try:
        run_count = sum(1 for _ in RUNS_DIR.glob("*.md")) if RUNS_DIR.is_dir() else 0
    except OSError:
        run_count = 0
    kept = len(catalogue())
    _ui().banner(MODEL, MAX_AGENTS, kept, run_count)
    _ui().hint()
    if kept == 0 and run_count == 0:
        _ui().first_run_welcome()


def _keep_policy_from_env() -> str:
    """The standing keep policy: 'always', 'never', or 'ask'."""
    value = os.environ.get("AGENTGOD_KEEP", "").strip().lower()
    return value if value in ("always", "never") else "ask"


def _persist_keep_always() -> None:
    """Record 'always keep' in .env, so the question never comes back."""
    os.environ["AGENTGOD_KEEP"] = "always"
    env_file = PROJECT_DIR / ".env"
    try:
        lines = []
        if env_file.exists():
            lines = [
                line
                for line in env_file.read_text(encoding="utf-8").splitlines()
                if not line.strip().startswith("AGENTGOD_KEEP=")
            ]
        lines.append("AGENTGOD_KEEP=always")
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass  # the session still remembers; only the persistence failed


def ask_keep(result, policy: str = "ask", interactive: bool | None = None) -> list[str]:
    """Decide the fate of every agent this run had to build. Returns the kept names.

    Only newly built agents are offered: anything reused was kept by an
    earlier decision, so re-asking would be noise. `policy` is the standing
    answer - 'always' and 'never' skip the prompt entirely, and headless
    runs keep by default because keeping is non-destructive and reuse is
    the whole point of the library.
    """
    if not result.pending:
        return []
    from library import remember

    def store() -> list[str]:
        return [
            name
            for name, (role, source) in sorted(result.pending.items())
            if remember(name, role, source, task=getattr(result, "task", ""))
        ]

    names = ", ".join(sorted(result.pending))
    count = len(result.pending)
    noun = "agent" if count == 1 else "agents"

    if policy == "never":
        _ui().note(f"  Discarded {count} new {noun} ({names}) - keep policy is 'never'.")
        return []
    if policy == "always":
        kept = store()
        if kept:
            _ui().note(f"  Kept for reuse: {', '.join(kept)} (keep policy is 'always')")
        return kept
    if interactive is None:
        interactive = _interactive()
    if not interactive:
        kept = store()
        if kept:
            _ui().note(f"  Kept for reuse: {', '.join(kept)} (non-interactive default)")
        return kept

    while True:
        choice = ask(
            f"\nKeep the {count} new {noun} ({names}) for reuse? "
            "[Keep/discard/always] (Enter = keep): "
        )
        if choice is None:
            # Ctrl-C mid-question: keeping is the non-destructive default,
            # and it is the whole point of the library.
            choice = "keep"
            _ui().note("  (no answer - keeping)")

        choice = choice.lower()
        if choice in ("keep", "k", "y", "yes", ""):
            kept = store()
            if kept:
                _ui().success(f"  Kept for reuse: {', '.join(kept)}")
            else:
                _ui().warn("  Could not save.")
            return kept
        if choice in ("always", "a"):
            _persist_keep_always()
            kept = store()
            if kept:
                _ui().success(f"  Kept for reuse: {', '.join(kept)}")
            _ui().note("  Keeping automatically from now on (AGENTGOD_KEEP=always in .env).")
            return kept
        if choice in ("discard", "d", "n", "no", "delete"):
            _ui().note("  Discarded. They will be rebuilt if a future task needs them.")
            return []
        _ui().warn("  Please answer keep, discard, or always (Enter keeps).")


def cleanup(agent_paths: list[Path]) -> None:
    """Remove the working copies from generated_agents/.

    These are scratch either way: a kept agent lives in the library, and a
    discarded one should leave nothing behind.
    """
    if not agent_paths:
        return
    from inventory import delete_agents

    delete_agents(agent_paths)


def archive(task: str, result) -> Path | None:
    """Write the run to runs/, never letting a broken archive lose the answer."""
    from runlog import save_run

    try:
        return save_run(task, result)
    except Exception:
        return None


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
    if intent is Intent.ACKNOWLEDGEMENT:
        return identity.describe_acknowledgement()
    if intent is Intent.CAPABILITY:
        return identity.describe_capabilities()
    if intent is Intent.IDENTITY:
        return identity.describe_identity()
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


def clarify(task: str, conversation=None, allow_prompt: bool = True):
    """Ask the one question worth asking before the pipeline starts.

    Returns the task to run and what the asking cost. Skipped entirely when
    stdin is not a person - a question nobody can answer is a stalled run -
    and for follow-ups, where the previous exchange IS the disambiguation.
    An empty reply means "just get on with it", which is a perfectly good
    answer and is not asked about twice.
    """
    from config import CLARIFY, Usage

    usage = Usage()
    if not allow_prompt or not _interactive() or CLARIFY == "off":
        return task, usage
    if conversation is not None:
        from conversation import is_follow_up

        if is_follow_up(task):
            return task, usage

    from judgment import clarifying_question

    try:
        with _ui().status("sizing up the task..."):
            question = clarifying_question(task, usage=usage)
    except Exception:
        # Never let the optional question be the thing that kills a run.
        return task, usage
    if not question:
        return task, usage

    _ui().note(f"\nBefore I start - {question}")
    answer = ask("> ")
    if not answer:
        return task, usage
    return f"{task}\n\n({question} {answer})", usage


def run_task(
    task: str,
    echo_task: bool = False,
    conversation=None,
    keep_policy: str = "ask",
    no_input: bool = False,
    announce: bool = True,
    outcome: Outcome | None = None,
) -> bool:
    """Run one task end to end. Returns False only if the task itself failed.

    A line that is conversation never reaches the pipeline: it is answered
    here, for nothing, and the session moves on. `outcome`, when given, is
    filled with what happened so callers with their own contract (--json,
    exit codes) do not have to re-derive it from the printed output.
    """
    from orchestrator import handle_task

    ui = _ui()
    outcome = outcome if outcome is not None else Outcome()

    reply = answer_directly(task)
    if reply is not None:
        ui.run_started(task, echo=echo_task)
        if announce:
            ui.reply(reply)
        outcome.ok = True
        outcome.kind = "reply"
        outcome.answer = reply
        return True

    # Asked before the live display goes up, and before anything is built:
    # a question is only worth asking while it can still change the plan.
    task, clarify_usage = clarify(task, conversation, allow_prompt=not no_input)

    agent_paths: list[Path] = []
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
        result.usage.merge(clarify_usage)
        # The archive records what the user asked, not the expanded form the
        # pipeline was given - the expansion is plumbing, not the request.
        saved = archive(task, result)
        if announce:
            try:
                ui.run_succeeded(result, saved)
            except Exception:
                from ui import PlainUI

                PlainUI().run_succeeded(result, saved)
        if conversation is not None:
            conversation.remember(task, result.response)
        interactive = None if not no_input else False
        outcome.kept = ask_keep(result, policy=keep_policy, interactive=interactive)
        outcome.ok = True
        outcome.answer = result.response
        outcome.result = result
        outcome.saved = saved
    except KeyboardInterrupt:
        ui.run_cancelled()
        outcome.cancelled = True
    except Exception as error:  # one failed task must not end the session
        from problems import explain

        try:
            problem = explain(error)
        except Exception:
            problem = None
        ui.run_failed(error, problem)
        outcome.error = error
        outcome.problem = problem
    finally:
        # The live display must come down first, but scratch cleanup must
        # happen even if tearing it down somehow fails.
        try:
            ui.run_ended()
        finally:
            cleanup(agent_paths)
    return outcome.ok


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


def _buffered_lines() -> list[str]:
    """Whatever whole lines a paste has already put in the stdin buffer.

    A multi-line paste lands in the terminal buffer all at once, but input()
    hands back only the first line - and each leftover line used to become
    its own billed task, then bogus answers to the keep prompt. This drains
    what is already there, and only there: it never waits for typing.
    """
    if not _interactive():
        return []
    lines: list[str] = []
    try:
        if os.name == "nt":
            import msvcrt

            chars: list[str] = []
            while msvcrt.kbhit():
                chars.append(msvcrt.getwch())
            text = "".join(chars).replace("\r\n", "\n").replace("\r", "\n")
            lines = [line for line in text.split("\n") if line.strip()]
        else:
            import select

            while select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line:
                    break
                if line.strip():
                    lines.append(line.rstrip("\n"))
    except Exception:
        return []
    return lines


def read_input() -> str | None:
    """One unit of user input, however many lines it takes to type.

    A pasted paragraph is recognised by the extra lines already sitting in
    the buffer and treated as ONE task, announced as such.
    """
    line = ask("\n" + TASK_PROMPT)
    if line is None:
        return None
    if not line:
        return ""
    if line in BLOCK_OPENERS:
        return read_block()
    # A trailing backslash means the sentence is not finished yet.
    while line.endswith(CONTINUATION):
        more = ask("  | ")
        if more is None:
            break
        line = line[: -len(CONTINUATION)].rstrip() + " " + more

    rest = _buffered_lines()
    if rest:
        _ui().note(f"  (treating your {len(rest) + 1} pasted lines as one task)")
        line = "\n".join([line, *rest]).strip()
    return line


def _json_payload(task: str, outcome: Outcome) -> dict:
    """The machine-readable story of one run, for --json."""
    if outcome.error is not None:
        problem = outcome.problem
        return {
            "ok": False,
            "task": task,
            "error": {
                "type": type(outcome.error).__name__,
                "headline": getattr(problem, "headline", str(outcome.error)),
                "advice": getattr(problem, "advice", ""),
                "detail": getattr(problem, "technical", "") or str(outcome.error),
            },
        }
    if outcome.kind == "reply":
        return {"ok": True, "task": task, "answer": outcome.answer, "conversational": True}

    result = outcome.result
    usage = getattr(result, "usage", None)
    main_cost = getattr(usage, "cost_usd", None) if usage else None
    agent_cost = getattr(result, "agent_cost_usd", 0.0)
    return {
        "ok": outcome.ok,
        "task": task,
        "answer": outcome.answer,
        "complexity": getattr(result, "complexity", "standard"),
        "agents": [
            {
                "name": spec.name,
                "reused": spec.name in getattr(result, "reused", []),
                "ok": spec.name not in getattr(result, "failures", {}),
            }
            for spec in (getattr(getattr(result, "plan", None), "agents", None) or [])
        ],
        "kept": outcome.kept,
        "failures": dict(getattr(result, "failures", {})),
        "revisions": getattr(result, "revisions", 0),
        "council_improved": getattr(result, "council_improved", False),
        "caveat": getattr(result, "caveat", ""),
        "cost": {
            "llm_calls": (usage.calls if usage else 0) + getattr(result, "agent_calls", 0),
            "input_tokens": (usage.input_tokens if usage else 0)
            + getattr(result, "agent_input_tokens", 0),
            "output_tokens": (usage.output_tokens if usage else 0)
            + getattr(result, "agent_output_tokens", 0),
            # Cached input is billed separately and is not part of input_tokens;
            # reporting it is what makes the number above add up.
            "cache_write_tokens": getattr(usage, "cache_write_tokens", 0) if usage else 0,
            "cache_read_tokens": getattr(usage, "cache_read_tokens", 0) if usage else 0,
            "usd": round(agent_cost + main_cost, 6) if main_cost is not None else None,
        },
        "duration_seconds": round(getattr(result, "duration_seconds", 0.0), 2),
        "saved": str(outcome.saved) if outcome.saved else None,
    }


def _run_command(verb: str, argument: str) -> int:
    """One free command - `agentgod library` - straight to the handler and out."""
    from commands import Command, handle

    _ui().reply(handle(Command(name=verb, argument=argument)))
    return cli.EXIT_OK


def _run_one_shot(invocation: cli.Invocation) -> int:
    """The scripting path: one task, one exit code, optionally one JSON object."""
    task = invocation.task or ""

    # A slash command as the one-shot task is a question about the session,
    # answered free - never a billed pipeline run.
    from commands import PASTE, QUIT, handle, parse

    command = parse(task)
    if command is not None:
        output = handle(command)
        if output not in (QUIT, PASTE):
            _ui().reply(output)
        return cli.EXIT_OK

    keep_policy = invocation.keep or _keep_policy_from_env()
    outcome = Outcome()
    run_task(
        task,
        echo_task=not (invocation.quiet or invocation.json_output),
        keep_policy=keep_policy,
        no_input=invocation.no_input,
        announce=not invocation.json_output,
        outcome=outcome,
    )
    if invocation.json_output:
        import json

        print(json.dumps(_json_payload(task, outcome), ensure_ascii=False))
    if outcome.cancelled:
        return cli.EXIT_INTERRUPTED
    return cli.EXIT_OK if outcome.ok else cli.EXIT_FAILURE


def _session(invocation: cli.Invocation) -> int:
    """The interactive loop: the product most people meet."""
    from commands import PASTE, QUIT, handle, parse
    from conversation import Conversation
    from router import Intent, classify

    _session_banner()
    conversation = Conversation()
    keep_policy = invocation.keep or _keep_policy_from_env()
    taught_follow_up = False
    taught_library = False

    while True:
        task = read_input()
        if task is None or task.lower() in QUIT_WORDS:
            _ui().farewell()
            return cli.EXIT_OK
        if not task:
            _ui().note("  (type a task, /help for commands, or 'quit' to leave)")
            continue

        command = parse(task)
        if command is not None:
            output = handle(command, conversation)
            if output == QUIT:
                _ui().farewell()
                return cli.EXIT_OK
            if output != PASTE:
                _ui().reply(output)
                continue
            task = read_block()
            if not task:
                continue

        # "bye" is a farewell, not a task about the word "bye".
        if classify(task) is Intent.FAREWELL:
            _ui().farewell()
            return cli.EXIT_OK

        outcome = Outcome()
        run_task(task, conversation=conversation, keep_policy=keep_policy, outcome=outcome)
        if outcome.kept and keep_policy == "ask" and _keep_policy_from_env() == "always":
            keep_policy = "always"  # the user just answered 'always'
        if outcome.ok and outcome.kind == "task" and not taught_follow_up:
            _ui().note(
                "  tip: follow-ups work - 'make it shorter' or 'now in French' "
                "continues from this answer"
            )
            taught_follow_up = True
        elif outcome.kept and not taught_library:
            _ui().note(
                "  tip: kept agents come back free - /library lists them, "
                "/stats shows their record"
            )
            taught_library = True


def main() -> int:
    _force_utf8_output()
    try:
        invocation = cli.parse(sys.argv[1:])
    except SystemExit as exit_code:
        # argparse already printed the message; honour its exit code.
        return int(exit_code.code or 0)

    cli.apply(invocation)
    if invocation.plain or invocation.quiet or invocation.json_output:
        _reset_ui()  # the env vars just set must influence renderer choice

    if invocation.command is not None:
        verb, argument = invocation.command
        if not preflight(require_key=False):
            return cli.EXIT_FAILURE
        return _run_command(verb, argument)

    if not preflight():
        return cli.EXIT_FAILURE

    if invocation.task is not None:
        return _run_one_shot(invocation)
    return _session(invocation)


def cli_main() -> None:
    """Console-script entry point (`agentgod`), honouring the exit contract."""
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(cli.EXIT_INTERRUPTED)


if __name__ == "__main__":
    cli_main()
