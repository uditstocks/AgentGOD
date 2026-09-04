"""What the session remembers between tasks.

Each task used to arrive with no past. "write a haiku about rain" worked;
"now make it about snow instead" was planned from scratch, so the word haiku
was never in the prompt and the answer came back as an essay about snow.

This module is the smallest thing that fixes that: the last few exchanges,
and a rule for deciding when a new line is leaning on them. A line that
stands on its own is passed through untouched - carrying context into a task
that does not need it is how a session starts drifting.

No AI and no I/O, so every rule here is unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# How many exchanges to keep. Only the most recent is ever sent to the model;
# the rest exist so `/history` has something to show.
MEMORY_TURNS = 8

# How much of the previous answer to carry forward. Enough to revise, not
# enough to crowd out the new request.
MAX_CONTEXT_CHARS = 2500

# A follow-up is short. A long message brings its own context with it.
MAX_FOLLOW_UP_WORDS = 25

# Words that only mean something if something came before. Split by kind so
# the rule can require an actual reference rather than one stray "it".
_REFERENTS = frozenset(
    {
        "it", "its", "that", "this", "these", "those", "them", "they",
        "the same", "above", "previous", "last", "former", "yours",
    }
)

_CONTINUATIONS = frozenset(
    {
        "instead", "again", "also", "too", "another", "more", "less", "further",
        "next", "then", "now", "still", "rather", "actually", "but",
    }
)

_REVISIONS = frozenset(
    {
        "shorter", "longer", "briefer", "simpler", "clearer", "better", "tighter",
        "expand", "shorten", "lengthen", "condense", "rewrite", "redo", "revise",
        "rephrase", "reword", "change", "adjust", "tweak", "fix", "improve",
        "polish", "formal", "casual", "friendlier", "softer", "harder", "punchier",
        "bullet", "bullets", "table", "list", "translate", "continue", "keep",
        # Corrections. "that's wrong" without the previous answer is nothing;
        # treating it as a fresh task built agents to research the word "wrong".
        "wrong", "incorrect", "missed", "forgot", "mistake",
    }
)

_PHRASES = (
    re.compile(r"\b(?:make|do) (?:it|that|this|them)\b"),
    re.compile(r"\b(?:try|say|write|do) (?:it )?again\b"),
    re.compile(r"\bsame (?:but|thing|again|as)\b"),
    re.compile(r"\bwhat about\b"),
    re.compile(r"\bhow about\b"),
    re.compile(r"\b(?:can|could) you (?:also|instead)\b"),
    re.compile(r"\bin (?:french|spanish|german|hindi|japanese|italian)\b"),
    # Corrections: only meaningful against the answer that just arrived.
    re.compile(r"\bthat'?s? (?:is )?(?:wrong|incorrect|not right|not what i asked)\b"),
    re.compile(r"\bnot what i (?:asked|meant|wanted)\b"),
    re.compile(r"\byou (?:missed|forgot|left out|got .{1,40} wrong)\b"),
    # Elaborations: a request for more of what came before.
    re.compile(r"\btell me more\b"),
    re.compile(r"\bgo (?:on|deeper)\b"),
    re.compile(r"\bkeep going\b"),
    re.compile(r"\b(?:more|extra) detail(?:s)?\b"),
    re.compile(r"\belaborate\b"),
)

_WORD = re.compile(r"[a-z']+")

CONTEXT_TEMPLATE = """The user is continuing an earlier exchange. The previous request and
answer are given only so you can resolve what the new request refers to -
including its subject, form and length. The NEW REQUEST is the task.

--- previous request ---
{previous_task}

--- previous answer ---
{previous_answer}

--- NEW REQUEST (this is the task) ---
{task}
"""


@dataclass(frozen=True)
class Turn:
    """One completed exchange."""

    task: str
    answer: str


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def is_follow_up(text: str) -> bool:
    """True when `text` cannot be understood without the exchange before it."""
    lowered = text.lower().strip()
    if not lowered:
        return False

    words = _words(lowered)
    if not words or len(words) > MAX_FOLLOW_UP_WORDS:
        return False

    if any(phrase.search(lowered) for phrase in _PHRASES):
        return True

    unique = set(words)
    has_referent = bool(unique & _REFERENTS)
    has_revision = bool(unique & _REVISIONS)
    has_continuation = bool(unique & _CONTINUATIONS)

    # A bare referent ("shorten it") or a revision verb with a continuation
    # ("now shorter") is a follow-up. A revision word alone is not: "write a
    # shorter version of the attached brief" is a complete instruction.
    if has_referent and (has_revision or has_continuation or len(words) <= 6):
        return True
    return has_revision and has_continuation


def _clip(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip() + "\n[...truncated...]"


@dataclass
class Conversation:
    """The session's short-term memory."""

    turns: list[Turn] = field(default_factory=list)
    limit: int = MEMORY_TURNS

    def remember(self, task: str, answer: str) -> None:
        """Record one completed exchange, dropping the oldest if full."""
        self.turns.append(Turn(task=task.strip(), answer=answer.strip()))
        if len(self.turns) > self.limit:
            del self.turns[: len(self.turns) - self.limit]

    @property
    def last(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def clear(self) -> int:
        """Forget everything. Returns how many exchanges were dropped."""
        dropped = len(self.turns)
        self.turns.clear()
        return dropped

    def contextualise(self, task: str) -> tuple[str, bool]:
        """Return the task to run, and whether earlier context was folded in.

        The original text is returned unchanged unless the line is genuinely
        leaning on the previous exchange - a task that stands alone must
        reach the planner exactly as the user wrote it.
        """
        previous = self.last
        if previous is None or not is_follow_up(task):
            return task, False
        return (
            CONTEXT_TEMPLATE.format(
                previous_task=_clip(previous.task, 500),
                previous_answer=_clip(previous.answer),
                task=task.strip(),
            ),
            True,
        )
