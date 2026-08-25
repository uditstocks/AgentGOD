"""AgentGod - entry point.

One permanent main agent that builds, runs, and manages
task-specific agents on the fly.

The only module that talks to a human.

Nothing here imports the rest of the project at module level: preflight() has
to be able to explain a missing dependency, and it cannot do that from inside
the traceback of the import that failed.
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

# import name -> pip name, for the one message that has to be right.
REQUIREMENTS = {
    "langchain_openai": "langchain-openai",
    "langchain_core": "langchain-core",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
}


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


def banner() -> str:
    """A short, honest header: what model is about to spend your money."""
    from config import MAX_AGENTS, MODEL

    return (
        "\n  A G E N T   G O D\n"
        "  " + "─" * 52 + "\n"
        f"  {MODEL}  ·  up to {MAX_AGENTS} agents per task\n"
    )


def ask(message: str) -> str | None:
    """Prompt the user. Returns None when they end the session (EOF or Ctrl-C)."""
    try:
        return input(message).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


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
    print(f"AgentGod needs Python {need} or newer. This is Python {have}.")
    print(f"  {sys.executable}")
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

    print("Missing dependencies: " + ", ".join(missing))
    command = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]

    if not _confirm("Install them now? [y/N]: "):
        print("\nInstall them with:\n  pip install -r requirements.txt")
        return False

    print()
    if subprocess.run(command, cwd=str(PROJECT_DIR)).returncode != 0:
        print("\npip failed. Install them manually:\n  pip install -r requirements.txt")
        return False

    still_missing = _missing_packages()
    if still_missing:
        print("\nStill missing after install: " + ", ".join(still_missing))
        return False

    print("\nDependencies installed.\n")
    return True


def _check_api_key() -> bool:
    """Load .env, then make sure a usable key is present - offering to write one."""
    from dotenv import load_dotenv

    env_file = PROJECT_DIR / ".env"
    load_dotenv(env_file)

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key and not key.startswith("sk-or-..."):
        return True

    print("No OPENROUTER_API_KEY found.")
    print("Get one at https://openrouter.ai/keys\n")

    entered = ask("Paste your key here (or press Enter to exit): ")
    if not entered:
        print("\nSet it in .env, or as an environment variable:")
        print("  OPENROUTER_API_KEY=sk-or-...")
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
        print(f"Saved to {env_file.name} (gitignored).\n")
    except OSError as error:
        print(f"Could not write .env ({error}); using the key for this session only.\n")

    os.environ["OPENROUTER_API_KEY"] = entered
    return True


def preflight() -> bool:
    """Verify the project can actually run. Explains anything it cannot fix."""
    return _check_python() and _check_dependencies() and _check_api_key()


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------


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
            print("  (no answer - keeping)")

        choice = choice.lower()
        if choice in ("keep", "k", ""):
            kept = [
                name
                for name, (role, source) in sorted(result.pending.items())
                if remember(name, role, source)
            ]
            print(f"  Kept for reuse: {', '.join(kept)}" if kept else "  Could not save.")
            return
        if choice in ("discard", "d", "delete"):
            print("  Discarded. They will be rebuilt if a future task needs them.")
            return
        print("  Please type 'keep' or 'discard'.")


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
    """Print the final answer, archive it, and note anything that failed."""
    from runlog import save_run

    print("\n" + "=" * 60)
    print("FINAL RESPONSE")
    print("=" * 60)
    print(result.response)

    if result.failures:
        print("\nAgents that failed (their output was excluded):")
        for name, error in result.failures.items():
            print(f"  - {name}: {error.splitlines()[0][:150]}")

    print(f"\n{result.duration_seconds:.1f}s - {result.cost_summary()}")

    if result.reused:
        print(f"Reused free from library: {', '.join(result.reused)}")
    if result.built:
        print(f"Newly built this run: {', '.join(result.built)}")

    saved = save_run(task, result)
    if saved is not None:
        print(f"Saved to {saved.parent.name}/{saved.name}")
    else:
        print("(could not write the run archive)")


def run_task(task: str) -> bool:
    """Run one task end to end. Returns False only if the task itself failed."""
    from orchestrator import handle_task

    agent_paths: list[Path] = []
    ok = False
    try:
        result = handle_task(task, on_agent_created=agent_paths.append)
        report(task, result)
        ask_keep(result)
        ok = True
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as error:  # one failed task must not end the session
        print(f"\nTask failed: {type(error).__name__}: {error}")
    finally:
        cleanup(agent_paths)
    return ok


def main() -> int:
    _force_utf8_output()
    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help"):
        print("AgentGod - builds the agents a task needs, runs them, lets them go.")
        print("\nUsage:")
        print("  python main.py                 interactive session")
        print('  python main.py --task "..."    run one task and exit')
        return 0

    if not preflight():
        return 1

    # Everything below is safe to import: preflight proved the deps exist.
    if args and args[0] == "--task":
        task = " ".join(args[1:]).strip()
        if not task:
            print('Usage: python main.py --task "what you need done"')
            return 1
        return 0 if run_task(task) else 1

    print(banner())
    print("  Type your task, or 'quit' to exit.")
    while True:
        task = ask("\nWhat do you need done?\n> ")
        if task is None or task.lower() in QUIT_WORDS:
            print("Goodbye.")
            return 0
        if not task:
            continue
        run_task(task)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
