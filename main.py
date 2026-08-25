"""Dynamic Agent Creator — entry point.

One permanent main agent that builds, runs, and manages
task-specific LangChain agents on the fly.
"""

from pathlib import Path

from config import require_api_key
from inventory import delete_agents, save_to_inventory
from orchestrator import handle_task


def ask_cleanup(agent_paths: list[Path], task: str) -> None:
    """Let the user decide the fate of the generated agents."""
    while True:
        choice = input("\nDelete the generated agents or save them to inventory? [delete/save]: ")
        choice = choice.strip().lower()
        if choice in ("delete", "d"):
            delete_agents(agent_paths)
            return
        if choice in ("save", "s"):
            save_to_inventory(agent_paths, task)
            return
        print("Please type 'delete' or 'save'.")


def main() -> None:
    require_api_key()

    print("Dynamic Agent Creator (type 'quit' to exit)")
    while True:
        task = input("\nWhat do you need done?\n> ").strip()
        if not task:
            continue
        if task.lower() in ("quit", "exit", "q"):
            break

        final_response, agent_paths = handle_task(task)

        print("\n" + "=" * 60)
        print("FINAL RESPONSE")
        print("=" * 60)
        print(final_response)

        ask_cleanup(agent_paths, task)


if __name__ == "__main__":
    main()
