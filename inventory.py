"""Removal of the working copies in generated_agents/.

Keeping an agent is library.remember's job; this module only clears scratch.
"""

from __future__ import annotations

from pathlib import Path


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
