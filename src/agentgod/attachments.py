"""Reading the files a task names.

"read README.md and tell me the project name" used to fail, because the
generated agents cannot open files - codeguard forbids it, deliberately, and
that rule is worth keeping. But the restriction belongs to the *agents*, not
to the session: the user naming a file at their own prompt is not untrusted
code reaching for the disk.

So the file is read here, in the main process, before any agent exists, and
its contents travel with the task as ordinary text. The agents stay sealed.

Two rules make that safe to do quietly. Nothing is read that the user did not
name; and whatever is read is announced, because it is about to be sent to a
model provider and the user is entitled to know that before it happens.

No AI and no network here, so this module is fully unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Extensions that are plainly text a person might want reasoned about.
# An unlisted extension can still be read - the user just has to point at it
# with @, which is an unambiguous instruction rather than a guess.
TEXT_SUFFIXES = frozenset(
    {
        ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp",
        ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".sh", ".ps1",
        ".sql", ".html", ".htm", ".css", ".scss", ".xml", ".svg", ".tex",
    }
)

# Never read on a guess, and never read on an @ either. A task that mentions
# .env means the file's role, not its contents, and uploading a key to a
# model provider is not a mistake that can be taken back.
SECRET_NAMES = frozenset(
    {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519", ".npmrc", ".netrc"}
)
SECRET_SUFFIXES = frozenset({".pem", ".key", ".pfx", ".p12", ".crt", ".keystore"})

MAX_FILES = 5
MAX_BYTES_PER_FILE = 100_000
MAX_BYTES_TOTAL = 250_000

# @path, or "quoted path", or a bare token that looks like a filename.
_EXPLICIT = re.compile(r"@([^\s'\"]+)")
_QUOTED = re.compile(r"['\"]([^'\"\n]{1,200}?\.[A-Za-z0-9]{1,8})['\"]")
_BARE = re.compile(r"(?<![\w@/\\.])((?:[\w.\-]+[/\\])*[\w.\-]+\.[A-Za-z0-9]{1,8})(?![\w/\\])")


@dataclass
class Attachment:
    """One file that was read, or one that could not be."""

    reference: str
    path: Path | None = None
    text: str = ""
    bytes_read: int = 0
    truncated: bool = False
    problem: str = ""

    @property
    def ok(self) -> bool:
        return not self.problem

    def label(self) -> str:
        if self.problem:
            return f"{self.reference} - {self.problem}"
        size = f"{self.bytes_read / 1000:.1f} KB" if self.bytes_read >= 1000 else f"{self.bytes_read} B"
        return f"{self.reference} ({size}{', truncated' if self.truncated else ''})"


@dataclass
class Attached:
    """The result of resolving every file reference in one task."""

    task: str = ""
    files: list[Attachment] = field(default_factory=list)

    @property
    def read(self) -> list[Attachment]:
        return [item for item in self.files if item.ok]

    @property
    def any_read(self) -> bool:
        return bool(self.read)


def _is_secret(path: Path) -> bool:
    return path.name.lower() in SECRET_NAMES or path.suffix.lower() in SECRET_SUFFIXES


def find_references(text: str, root: Path | None = None) -> list[str]:
    """Every file reference in `text`, explicit ones first, in order.

    A bare token only counts when a file by that name actually exists: the
    alternative is treating "the fix landed in main.py" as a request to read
    the file, which is a guess the user did not ask for.
    """
    root = root or Path.cwd()
    seen: list[str] = []

    def add(candidate: str) -> None:
        # Trailing sentence punctuation only. A leading dot is part of the
        # name: stripping it turns "@.env" into "env", which is a different
        # file and, worse, one that is not on the refused list.
        candidate = candidate.strip().rstrip(".,;:)")
        if candidate and candidate not in seen:
            seen.append(candidate)

    for match in _EXPLICIT.finditer(text):
        add(match.group(1))
    for match in _QUOTED.finditer(text):
        add(match.group(1))

    explicit = set(seen)
    for match in _BARE.finditer(text):
        candidate = match.group(1).strip().strip(".,;:)")
        if candidate in explicit:
            continue
        if Path(candidate).suffix.lower() not in TEXT_SUFFIXES:
            continue
        # Only a file that is really there; a mention is not a request.
        if (root / candidate).is_file():
            add(candidate)
    return seen[: MAX_FILES * 2]


def _read_one(reference: str, root: Path, budget: int) -> Attachment:
    """Read one referenced file, refusing anything unsafe or unreadable."""
    try:
        path = (root / reference).expanduser()
        resolved = path.resolve()
    except (OSError, ValueError, RuntimeError):
        return Attachment(reference=reference, problem="not a usable path")

    if _is_secret(resolved):
        return Attachment(reference=reference, problem="refused (looks like a secret)")
    if not resolved.is_file():
        return Attachment(reference=reference, problem="not found")

    try:
        size = resolved.stat().st_size
        limit = min(MAX_BYTES_PER_FILE, budget)
        if limit <= 0:
            return Attachment(reference=reference, path=resolved, problem="skipped (size budget)")
        raw = resolved.read_bytes()[: limit + 1]
    except OSError as error:
        return Attachment(reference=reference, path=resolved, problem=f"unreadable ({error.strerror or error})")

    if b"\x00" in raw[:4096]:
        return Attachment(reference=reference, path=resolved, problem="binary file, not read")

    truncated = len(raw) > limit or size > limit
    text = raw[:limit].decode("utf-8", errors="replace")
    if truncated:
        text = text.rstrip() + "\n[...truncated...]"

    return Attachment(
        reference=reference,
        path=resolved,
        text=text,
        bytes_read=min(size, limit),
        truncated=truncated,
    )


def _render(task: str, files: list[Attachment]) -> str:
    """Put the file contents in front of the task, clearly labelled."""
    sections = [
        f"--- contents of {item.reference} ---\n{item.text}"
        for item in files
    ]
    return (
        "The user's request refers to the following file(s), read from their machine.\n"
        "Use the contents as the material for the task.\n\n"
        + "\n\n".join(sections)
        + "\n\n--- the user's request ---\n"
        + task.strip()
    )


def attach(task: str, root: Path | None = None) -> Attached:
    """Resolve and read the files `task` names, returning the task to run.

    When nothing is referenced - or nothing referenced could be read - the
    task is returned exactly as written.
    """
    root = root or Path.cwd()
    references = find_references(task, root)
    if not references:
        return Attached(task=task, files=[])

    files: list[Attachment] = []
    budget = MAX_BYTES_TOTAL
    for reference in references:
        if len([item for item in files if item.ok]) >= MAX_FILES:
            break
        attachment = _read_one(reference, root, budget)
        if attachment.ok:
            budget -= attachment.bytes_read
        files.append(attachment)

    usable = [item for item in files if item.ok]
    if not usable:
        return Attached(task=task, files=files)
    return Attached(task=_render(task, usable), files=files)
