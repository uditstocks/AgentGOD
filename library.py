"""The reusable agent library.

Building `research_agent` again for every new report is the most expensive
thing this system can do: one generator call per agent, every run, forever.
The library makes that a one-time cost. An agent is written once, remembered
by capability, and handed back for free on every later task that needs it.

This only works because generated agents are topic-agnostic: the subject
arrives at runtime on stdin, so the same `research_agent` serves a report on
electric scooters and a report on anything else.

No AI and no network here, so this module is fully unit-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import AGENT_RUNTIME_VERSION, INVENTORY_DIR

LIBRARY_DIR = INVENTORY_DIR / "agents"
INDEX_PATH = INVENTORY_DIR / "index.json"
INDEX_VERSION = 1


@dataclass
class LibraryEntry:
    """One remembered agent, addressable by the capability it provides."""

    name: str
    role: str = ""
    uses: int = 0
    created: str = ""
    last_used: str = ""
    # The task this agent was written for. Kept so the agent can be re-checked
    # against it on every reuse: an agent that hardcoded that task's subject
    # is wrong for every task after it, and the only way to notice is to
    # remember what it was originally built to do.
    built_for: str = ""
    # Which version of the trusted agent runtime this agent was written
    # against. 0 means it predates the stamp, which is treated as too old:
    # an agent cannot call a helper that did not exist when it was written.
    runtime: int = 0

    @property
    def path(self) -> Path:
        return LIBRARY_DIR / f"{self.name}.py"

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "uses": self.uses,
            "created": self.created,
            "last_used": self.last_used,
            "built_for": self.built_for,
            "runtime": self.runtime,
        }


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_index() -> dict[str, LibraryEntry]:
    """Load the index, rebuilding from disk if it is missing or unreadable.

    The .py files are the source of truth; index.json is only a convenience,
    so a corrupt index must never cost the user their library.
    """
    entries: dict[str, LibraryEntry] = {}
    raw: dict = {}
    if INDEX_PATH.is_file():
        try:
            raw = json.loads(INDEX_PATH.read_text(encoding="utf-8")).get("agents", {})
        except (OSError, ValueError):
            raw = {}

    if not LIBRARY_DIR.is_dir():
        return entries

    for source_file in sorted(LIBRARY_DIR.glob("*.py")):
        name = source_file.stem
        record = raw.get(name, {}) if isinstance(raw.get(name), dict) else {}
        entries[name] = LibraryEntry(
            name=name,
            role=str(record.get("role", "")),
            uses=int(record.get("uses", 0) or 0),
            created=str(record.get("created", "")),
            last_used=str(record.get("last_used", "")),
            built_for=str(record.get("built_for", "")),
            runtime=int(record.get("runtime", 0) or 0),
        )
    return entries


def _write_index(entries: dict[str, LibraryEntry]) -> None:
    try:
        INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "agents": {name: entry.to_dict() for name, entry in sorted(entries.items())},
        }
        INDEX_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass  # the .py files are what matter; the index can be rebuilt


def catalogue() -> list[LibraryEntry]:
    """Every remembered agent, most-used first - what the planner is shown."""
    return sorted(_read_index().values(), key=lambda entry: (-entry.uses, entry.name))


def lookup(name: str) -> str | None:
    """Return the stored source for `name`, or None if it was never built."""
    source_file = LIBRARY_DIR / f"{name}.py"
    try:
        if source_file.is_file():
            return source_file.read_text(encoding="utf-8")
    except OSError:
        pass
    return None


def entry_for(name: str) -> LibraryEntry | None:
    """The index record for one agent, or None if it is not in the library."""
    return _read_index().get(name)


def reusable(name: str) -> bool:
    """Whether the stored agent still does its job on a subject it has not seen.

    An agent is admitted to the library on the promise that it is topic-
    agnostic. This re-checks that promise against the task it was written for,
    every time it is about to be handed back, because the promise was once
    broken silently: a `summary_agent` with "one benefit of code review" in its
    prompt served sixteen later tasks before anyone noticed.

    An agent recorded before the task was tracked cannot be checked, and is
    given the benefit of the doubt - `audit()` is how those are found.
    """
    entry = entry_for(name)
    if entry is None or not entry.built_for:
        return True
    source = lookup(name)
    if source is None:
        return True

    from topicguard import is_reusable

    return is_reusable(source, entry.built_for)


def up_to_date(name: str) -> bool:
    """Whether the stored agent can still do everything the planner may ask of it.

    The second promise the library makes, alongside `reusable`. When the
    trusted runtime gains a capability, every agent already on disk was
    written without it - and nothing in the code or the plan reveals that.
    A research agent built before web search existed answers from memory and
    looks like it worked, which is the worst way for this to fail.
    """
    entry = entry_for(name)
    return entry is not None and entry.runtime >= AGENT_RUNTIME_VERSION


def audit() -> dict[str, list[str]]:
    """Every library agent that hardcoded the task it was built for.

    Maps agent name to the problems found. An empty dict means the library is
    clean. Agents with no recorded task are skipped, not accused.
    """
    from topicguard import check_topic_leakage

    found: dict[str, list[str]] = {}
    for entry in catalogue():
        if not entry.built_for:
            continue
        source = lookup(entry.name)
        if source is None:
            continue
        problems = check_topic_leakage(source, entry.built_for)
        if problems:
            found[entry.name] = problems
    return found


def remember(name: str, role: str, source: str, task: str = "") -> bool:
    """Store (or refresh) one agent. Returns False if it could not be written."""
    try:
        LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        target = (LIBRARY_DIR / f"{name}.py").resolve()
        if target.parent != LIBRARY_DIR.resolve():
            return False
        target.write_text(source, encoding="utf-8")
    except OSError:
        return False

    entries = _read_index()
    entry = entries.get(name) or LibraryEntry(name=name, created=_now())
    entry.role = role or entry.role
    entry.created = entry.created or _now()
    entry.built_for = task.strip() or entry.built_for
    entry.runtime = AGENT_RUNTIME_VERSION
    entries[name] = entry
    _write_index(entries)
    return True


def record_use(name: str) -> None:
    """Count one reuse, so the catalogue can rank by what actually gets used."""
    entries = _read_index()
    entry = entries.get(name)
    if entry is None:
        return
    entry.uses += 1
    entry.last_used = _now()
    _write_index(entries)


def forget(name: str) -> bool:
    """Drop one agent from the library."""
    entries = _read_index()
    if name not in entries:
        return False
    try:
        entries[name].path.unlink(missing_ok=True)
    except OSError:
        return False
    del entries[name]
    _write_index(entries)
    return True


def describe_for_planner(limit: int = 25) -> str:
    """The catalogue, rendered for the planner prompt."""
    entries = catalogue()[:limit]
    if not entries:
        return "(none yet - this is the first run)"
    return "\n".join(
        f"  - {entry.name}: {entry.role or 'no description recorded'}" for entry in entries
    )
