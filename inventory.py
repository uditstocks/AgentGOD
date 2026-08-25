"""Step 7: After the task, delete the generated agents or save them for reuse."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from config import INVENTORY_DIR


def delete_agents(agent_paths: list[Path]) -> int:
    """Remove the generated agent files. Returns how many were deleted."""
    removed = 0
    for path in agent_paths:
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError as error:
            print(f"  Could not delete {path.name}: {error}")
    print(f"Agents deleted ({removed}).")
    return removed


def _unique_folder(stamp: str) -> Path:
    """A fresh folder for this save, even if another landed the same second."""
    folder = INVENTORY_DIR / stamp
    suffix = 2
    while folder.exists():
        folder = INVENTORY_DIR / f"{stamp}_{suffix}"
        suffix += 1
    return folder


def save_to_inventory(agent_paths: list[Path], task: str) -> Path:
    """Move the generated agents into a timestamped inventory folder."""
    folder = _unique_folder(datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True)

    saved = 0
    for path in agent_paths:
        if not path.is_file():
            continue
        shutil.move(str(path), str(folder / path.name))
        saved += 1

    # Keep a note of which task this team was built for.
    (folder / "TASK.txt").write_text(task, encoding="utf-8")
    print(f"Saved {saved} agent(s) to {folder}")
    return folder
