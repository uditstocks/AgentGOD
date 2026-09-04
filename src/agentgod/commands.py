"""Slash commands: the questions about the session itself.

A user who wants to know what is in the library, or to drop an agent that has
gone bad, should not have to spend a pipeline run finding out. These are
answered from disk, instantly, and none of them costs anything.

Everything here returns text for the caller to display, so the module has no
opinion about how a session is presented and stays unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

PREFIX = "/"

# name -> (argument, one-line description), in the order /help lists them.
COMMANDS: dict[str, tuple[str, str]] = {
    "help": ("", "this list"),
    "library": ("", "the agents kept for reuse, with each one's record"),
    "stats": ("", "the lifetime dashboard: runs, reuse savings, reliability"),
    "forget": ("<agent>", "drop one agent from the library"),
    "audit": ("", "check the library for agents that hardcoded their first task"),
    "history": ("[n]", "the recent archived runs, or reopen run n"),
    "last": ("", "show the most recent answer again"),
    "clear": ("", "forget the conversation so far"),
    "paste": ("", "start a multi-line task; end it with a line containing only ."),
    "quit": ("", "leave"),
}

# Returned by handle() when the session should end.
QUIT = "__quit__"

# Returned by handle() when a multi-line task should be collected.
PASTE = "__paste__"


@dataclass(frozen=True)
class Command:
    """One parsed slash command."""

    name: str
    argument: str = ""


def parse(line: str) -> Command | None:
    """Read a slash command, or None if `line` is ordinary input."""
    text = line.strip()
    if not text.startswith(PREFIX) or text == PREFIX:
        return None
    body = text[len(PREFIX) :].strip()
    name, _, argument = body.partition(" ")
    return Command(name=name.lower(), argument=argument.strip())


def help_text() -> str:
    """The usage screen: how to run a task, and every command."""
    lines = [
        "**Give me a task** - describe what you need, in as much detail as you like.",
        "",
        "    write a 200-word investor memo on X, cover the risks, end with a recommendation",
        "    summarise @report.md in five bullets",
        "    compare Postgres and SQLite for a small web app",
        "",
        "Follow-ups work: after an answer, `make it shorter` or `now do it in French`",
        "continues from what came before.",
        "",
        "Name a file in your task (`README.md`, or `@path/to/file`) and I will read it",
        "and work from its contents.",
        "",
        "**Commands**",
        "",
    ]
    for name, (argument, description) in COMMANDS.items():
        invocation = f"{PREFIX}{name} {argument}".strip()
        lines.append(f"    {invocation:<18} {description}")
    lines += [
        "",
        "`quit` also leaves. Ask `what can you do?` for what I am actually good at.",
    ]
    return "\n".join(lines)


def _library_text() -> str:
    from library import catalogue

    entries = catalogue()
    if not entries:
        return (
            "The library is empty. Agents go in when you answer `keep` after a run, "
            "and every task that reuses one is that much cheaper."
        )

    lines = [f"**{len(entries)} agent{'s' if len(entries) != 1 else ''} kept for reuse**", ""]
    for entry in entries:
        uses = f"{entry.uses} use{'s' if entry.uses != 1 else ''}"
        record = _record(entry)
        lines.append(
            f"    {entry.name:<22} {uses:<9} {record:<14} "
            f"{entry.role or 'no description recorded'}"
        )
    lines += ["", f"Drop one with `{PREFIX}forget <name>`."]
    return "\n".join(lines)


def _record(entry) -> str:
    """One agent's reliability, compact: `3W/1L · gen 2`. Blank while unproven."""
    parts = []
    if entry.wins or entry.losses:
        parts.append(f"{entry.wins}W/{entry.losses}L")
    if entry.generation > 1:
        parts.append(f"gen {entry.generation}")
    return " · ".join(parts)


def _stats_text() -> str:
    """The lifetime dashboard, entirely from disk - nothing here costs a call."""
    from config import RUNS_DIR
    from library import catalogue

    entries = catalogue()
    try:
        run_count = sum(1 for _ in RUNS_DIR.glob("*.md")) if RUNS_DIR.is_dir() else 0
    except OSError:
        run_count = 0

    reuses = sum(entry.uses for entry in entries)
    evolved = [entry for entry in entries if entry.generation > 1]
    proven = [entry for entry in entries if entry.wins or entry.losses]

    lines = [
        "**AgentGod - lifetime stats**",
        "",
        f"    runs archived      {run_count}",
        f"    agents in library  {len(entries)}",
        f"    free reuses        {reuses}  (each one skipped a full code-generation call)",
        f"    evolved agents     {len(evolved)}"
        + (f"  ({', '.join(e.name for e in evolved)})" if evolved else ""),
    ]
    if proven:
        lines += ["", "**Reliability** - wins/losses since each agent was kept or last evolved", ""]
        ranked = sorted(proven, key=lambda e: (-(e.wins - e.losses), -e.uses, e.name))
        for entry in ranked:
            record = _record(entry) or "-"
            lines.append(f"    {entry.name:<22} {record}")
    if not entries and not run_count:
        return (
            "Nothing to count yet. Run a task, keep an agent, and this "
            "dashboard starts filling in."
        )
    lines += ["", f"`{PREFIX}library` lists every kept agent; `{PREFIX}history` the recent runs."]
    return "\n".join(lines)


def _forget_text(name: str) -> str:
    from library import forget

    if not name:
        return f"Which one? `{PREFIX}forget <agent>` - see `{PREFIX}library`."
    if forget(name):
        return f"Dropped `{name}`. It will be rebuilt if a future task needs it."
    return f"There is no `{name}` in the library. See `{PREFIX}library`."


def _audit_text() -> str:
    from library import audit, catalogue

    checked = [entry for entry in catalogue() if entry.built_for]
    unchecked = [entry.name for entry in catalogue() if not entry.built_for]
    problems = audit()

    lines = []
    if problems:
        count = len(problems)
        lines += [
            f"**{count} agent{'s' if count != 1 else ''} hardcoded the task {'they were' if count != 1 else 'it was'} built for.**",
            "",
            "An agent like this answers every later task as if it were still doing "
            "its first one. Drop it and the next run rebuilds it clean.",
            "",
        ]
        for name, found in problems.items():
            lines.append(f"    {name}")
            lines.append(f"        {found[0]}")
        lines += ["", f"Drop one with `{PREFIX}forget <name>`."]
    else:
        lines.append(f"Library clean - {len(checked)} agent(s) checked, none hardcoded its task.")

    if unchecked:
        lines += [
            "",
            f"Not checkable ({len(unchecked)}): {', '.join(unchecked)} - kept before the "
            "task was recorded, so there is nothing to check them against.",
        ]
    return "\n".join(lines)


def _archived_runs() -> list:
    """Every archived run, most recent first."""
    from config import RUNS_DIR

    try:
        return sorted(RUNS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []


def _title_of(path) -> str:
    """The run's own heading, or its filename if the file cannot be read."""
    try:
        first = path.read_text(encoding="utf-8").lstrip().splitlines()[0]
        return first.lstrip("# ").strip() or path.stem
    except (OSError, IndexError):
        return path.stem


def _read_run(path) -> str:
    """One archived run, verbatim - it is already Markdown."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        return f"Could not read `{path.name}` ({error})."


def _history_text(argument: str = "", limit: int = 10) -> str:
    """The recent runs, numbered - or one of them, reopened.

    Numbering is the whole point: an archive nobody can address is a folder,
    not a history. `/history 3` prints run 3 exactly as it was saved.
    """
    files = _archived_runs()
    if not files:
        return "No runs archived yet. Every completed task is saved to `runs/`."

    wanted = argument.strip()
    if wanted:
        if not wanted.isdigit() or not 1 <= int(wanted) <= len(files):
            return (
                f"There is no run {wanted}. `{PREFIX}history` lists them "
                f"(1 to {len(files)})."
            )
        return _read_run(files[int(wanted) - 1])

    lines = [
        f"**{len(files)} run{'s' if len(files) != 1 else ''} archived** - most recent first",
        "",
    ]
    for index, path in enumerate(files[:limit], start=1):
        lines.append(f"    {index:>2}. {_title_of(path)[:66]}")
    if len(files) > limit:
        lines.append(f"    ... and {len(files) - limit} more in runs/")
    lines += ["", f"Reopen one with `{PREFIX}history <number>`, or `{PREFIX}last`."]
    return "\n".join(lines)


def _last_text() -> str:
    """The most recent answer, shown again without paying for it twice."""
    files = _archived_runs()
    if not files:
        return "Nothing archived yet - finish a task and it will be saved to `runs/`."
    return _read_run(files[0])


def handle(command: Command, conversation=None) -> str:
    """Run one command and return what to show. Unknown names say so."""
    if command.name in ("help", "h", "?"):
        return help_text()
    if command.name in ("library", "agents", "ls"):
        return _library_text()
    if command.name in ("stats", "dashboard"):
        return _stats_text()
    if command.name == "forget":
        return _forget_text(command.argument)
    if command.name == "audit":
        return _audit_text()
    if command.name in ("history", "runs"):
        return _history_text(command.argument)
    if command.name in ("last", "again"):
        return _last_text()
    if command.name == "clear":
        if conversation is None:
            return "Nothing to forget."
        dropped = conversation.clear()
        if not dropped:
            return "Nothing to forget - this is a fresh session."
        return f"Forgotten {dropped} exchange{'s' if dropped != 1 else ''}. Starting clean."
    if command.name == "paste":
        return PASTE
    if command.name in ("quit", "exit", "q", "bye"):
        return QUIT

    known = ", ".join(f"{PREFIX}{name}" for name in COMMANDS)
    return f"Unknown command `{PREFIX}{command.name}`. Try: {known}"
