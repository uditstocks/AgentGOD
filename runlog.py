"""Persist the result of every completed task to runs/.

The answer is the thing the user actually paid for, so it must outlive the
terminal it was printed in. One readable Markdown file per run: the task, the
team that was built for it, the final response, and what it cost.

No AI and no network here, so this module is fully unit-testable.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from config import MODEL, RUNS_DIR

# Task text is user input and becomes part of a filename, so it is reduced to
# a safe slug the same way agent names are.
_UNSAFE = re.compile(r"[^a-z0-9]+")
MAX_SLUG_LENGTH = 48


def slugify(text: str) -> str:
    """Reduce arbitrary task text to a short, filesystem-safe slug."""
    slug = _UNSAFE.sub("-", text.strip().lower()).strip("-")
    if len(slug) > MAX_SLUG_LENGTH:
        slug = slug[:MAX_SLUG_LENGTH].rsplit("-", 1)[0].strip("-")
    return slug or "task"


def _unique_path(folder: Path, stem: str) -> Path:
    """A free filename, even for two runs inside the same second."""
    path = folder / f"{stem}.md"
    suffix = 2
    while path.exists():
        path = folder / f"{stem}_{suffix}.md"
        suffix += 1
    return path


def render(task: str, result, when: datetime | None = None) -> str:
    """Build the Markdown record for one completed task."""
    stamp = (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    plan = getattr(result, "plan", None)

    lines = [
        f"# {task.strip().splitlines()[0][:100]}",
        "",
        f"- **When** {stamp}",
        f"- **Model** {MODEL}",
        f"- **Took** {result.duration_seconds:.1f}s",
        f"- **Cost** {result.cost_summary()}",
        "",
        "## Task",
        "",
        task.strip(),
        "",
        "## Answer",
        "",
        result.response.strip(),
        "",
    ]

    if plan is not None and getattr(plan, "agents", None):
        lines += ["## Agents built for this", ""]
        if getattr(plan, "reasoning", ""):
            lines += [f"> {plan.reasoning.strip()}", ""]
        lines += [f"- `{spec.name}` - {spec.role}" for spec in plan.agents]
        lines.append("")

    if result.failures:
        lines += ["## Agents that failed", ""]
        lines += [
            f"- `{name}` - {error.splitlines()[0][:200]}" for name, error in result.failures.items()
        ]
        lines.append("")

    return "\n".join(lines)


def save_run(task: str, result) -> Path | None:
    """Write the run to runs/ and return the path.

    Returns None instead of raising if the write fails: the answer has already
    been produced and printed, and losing the archive must not lose the run.
    """
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(task)}"
        path = _unique_path(RUNS_DIR, stem)
        path.write_text(render(task, result), encoding="utf-8")
        return path
    except OSError:
        return None
