"""Step 7: After the task, delete the generated agents or save them for reuse."""

import shutil
from datetime import datetime
from pathlib import Path

from config import INVENTORY_DIR


def delete_agents(agent_paths: list[Path]) -> None:
    """Remove the generated agent files."""
    for path in agent_paths:
        path.unlink(missing_ok=True)
    print("Agents deleted.")


def save_to_inventory(agent_paths: list[Path], task: str) -> Path:
    """Move the generated agents into a timestamped inventory folder."""
    folder = INVENTORY_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)

    for path in agent_paths:
        shutil.move(str(path), folder / path.name)

    # Keep a note of which task this team was built for.
    (folder / "TASK.txt").write_text(task, encoding="utf-8")
    print(f"Agents saved to {folder}")
    return folder
