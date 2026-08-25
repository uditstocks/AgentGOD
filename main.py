"""Dynamic Agent Creator - entry point.

One permanent main agent that builds, runs, and manages
task-specific agents on the fly.

The only module that talks to a human.
"""

from __future__ import annotations

from pathlib import Path

from config import require_api_key
from inventory import delete_agents, save_to_inventory
from orchestrator import TaskResult, handle_task

QUIT_WORDS = frozenset({"quit", "exit", "q"})


def ask(message: str) -> str | None:
    """Prompt the user. Returns None when they end the session (EOF or Ctrl-C)."""
    try:
        return input(message).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def ask_cleanup(agent_paths: list[Path], task: str) -> None:
    """Let the user decide the fate of the generated agents.

    Defaults to deleting on EOF/Ctrl-C: the scratch directory must not be left
    holding files the user never chose to keep.
    """
    if not agent_paths:
        return
    while True:
        choice = ask("\nDelete the generated agents or save them to inventory? [delete/save]: ")
        if choice is None:
            delete_agents(agent_paths)
            return
        choice = choice.lower()
        if choice in ("delete", "d"):
            delete_agents(agent_paths)
            return
        if choice in ("save", "s"):
            save_to_inventory(agent_paths, task)
            return
        print("Please type 'delete' or 'save'.")


def report(result: TaskResult) -> None:
    """Print the final answer, plus anything that went wrong along the way."""
    print("\n" + "=" * 60)
    print("FINAL RESPONSE")
    print("=" * 60)
    print(result.response)

    if result.failures:
        print("\nAgents that failed (their output was excluded):")
        for name, error in result.failures.items():
            print(f"  - {name}: {error.splitlines()[0][:150]}")

    print(f"\n{result.duration_seconds:.1f}s · {result.cost_summary()}")


def main() -> None:
    require_api_key()

    print("Dynamic Agent Creator (type 'quit' to exit)")
    while True:
        task = ask("\nWhat do you need done?\n> ")
        if task is None or task.lower() in QUIT_WORDS:
            break
        if not task:
            continue

        # Collected as files are written, so a mid-run failure still cleans up.
        agent_paths: list[Path] = []
        try:
            result = handle_task(task, on_agent_created=agent_paths.append)
            report(result)
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as error:  # one failed task must not end the session
            print(f"\nTask failed: {type(error).__name__}: {error}")
        finally:
            ask_cleanup(agent_paths, task)


if __name__ == "__main__":
    main()
