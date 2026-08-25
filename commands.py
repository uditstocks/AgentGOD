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
    "library": ("", "the agents kept for reuse, and how often each is used"),
    "forget": ("<agent>", "drop one agent from the library"),
    "audit": ("", "check the library for agents that hardcoded their first task"),
    "history": ("", "the most recent archived runs"),
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
        lines.append(f"    {entry.name:<22} {uses:<10} {entry.role or 'no description recorded'}")
    lines += ["", f"Drop one with `{PREFIX}forget <name>`."]
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


def _history_text(limit: int = 10) -> str:
    from config import RUNS_DIR

    try:
        files = sorted(RUNS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        files = []
    if not files:
        return "No runs archived yet. Every completed task is saved to `runs/`."

    lines = [f"**{len(files)} run{'s' if len(files) != 1 else ''} archived** - most recent first", ""]
    for path in files[:limit]:
        title = path.stem
        try:
            first = path.read_text(encoding="utf-8").lstrip().splitlines()[0]
            title = first.lstrip("# ").strip() or title
        except (OSError, IndexError):
            pass
        lines.append(f"    {title[:70]}")
    if len(files) > limit:
        lines.append(f"    ... and {len(files) - limit} more in runs/")
    return "\n".join(lines)


def handle(command: Command, conversation=None) -> str:
    """Run one command and return what to show. Unknown names say so."""
    if command.name in ("help", "h", "?"):
        return help_text()
    if command.name in ("library", "agents", "ls"):
        return _library_text()
    if command.name == "forget":
        return _forget_text(command.argument)
    if command.name == "audit":
        return _audit_text()
    if command.name in ("history", "runs"):
        return _history_text()
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
