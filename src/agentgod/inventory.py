"""Removal of the working copies in generated_agents/.

Keeping an agent is library.remember's job; this module only clears scratch.
Silent by design: scratch removal is bookkeeping, not news, and what (if
anything) the user hears about it is the interface's decision, not this one's.
"""

from __future__ import annotations

from pathlib import Path


def delete_agents(agent_paths: list[Path]) -> int:
    """Remove the generated agent files. Returns how many are gone."""
    removed = 0
    for path in agent_paths:
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError:
            pass  # locked right now; the next run overwrites it anyway
    return removed
