"""Step 6: Merge all agent outputs into one final response for the user."""

from __future__ import annotations

import re

from .config import MAX_CHARS_PER_INPUT, Usage, complete

MERGER_PROMPT = """You are the coordinator of a multi-agent system.
Several specialized agents each completed one part of the user's task.
Merge their outputs into ONE clear, complete final answer for the user.

Rules:
- Obey every explicit constraint in the user's task exactly: length limits,
  word counts, format, structure and tone.
- Do not mention the agents or the process - just deliver the answer.
- Add nothing the agent outputs do not support.
- If one of the outputs below is off-topic, contradicts the task, or is
  plainly about something the user never asked for, IGNORE IT COMPLETELY.
  Use the outputs that do answer the task and write as though the stray one
  was never there. Never tell the user that an output was wrong, unrelated,
  mistaken or disregarded, and never apologise for it - the user asked a
  question, not for a report on how the work went. A caveat about the
  machinery is never part of the answer.
- Deliver the thing that was asked for. If the task asks for working code, a
  plan for code is not the answer; write the code the outputs support.

User task:
{task}

Agent outputs:
{outputs}
"""


def _format_outputs(outputs: dict[str, str]) -> str:
    """Label each agent's result and cap it, so the prompt cannot run away."""
    sections = []
    for name, output in outputs.items():
        text = output.strip()
        if len(text) > MAX_CHARS_PER_INPUT:
            text = text[:MAX_CHARS_PER_INPUT] + "\n[...truncated...]"
        sections.append(f"--- {name} ---\n{text}")
    return "\n\n".join(sections)


def merge_outputs(
    task: str,
    outputs: dict[str, str],
    usage: Usage | None = None,
    effort: str | None = None,
    model: str | None = None,
) -> str:
    """Combine every agent's output into a single final response.

    The merger always runs, including for a single agent: it is the only
    stage that still holds the user's original wording, so it is what
    enforces the task's own formatting constraints.
    """
    if not outputs:
        raise ValueError("no agent produced an output to merge")

    return complete(
        MERGER_PROMPT.format(task=task, outputs=_format_outputs(outputs)),
        usage=usage,
        effort=effort,
        model=model,
    ).strip()


# The characters that open and close a Markdown code block.
FENCE = "`" * 3

# Lines that only source code has. Every pattern is anchored at the start of
# the line, so a sentence that merely contains the word - "import the CSV
# first" - cannot match; only a line that *begins* as a declaration does.
_DECLARATION = re.compile(
    r"^(?:#!/|@\w|(?:from|import|def|class|async|package|using|#include)\s+\S"
    r"|(?:public|private|protected|static|func|fn|const|let|var|function)\s+\w)"
)

# Lines carrying code's punctuation rather than a sentence's: a closing
# bracket, a trailing brace or colon, a keyword statement, an assignment,
# a bare call.
_SYNTAX = re.compile(
    r"(?:^\s*[)\]}]|[;{]\s*$|:\s*$"
    r"|^\s*(?:return|if|for|while|try|except|with|elif|else|raise|yield)\b"
    r"|^\s*[\w.\[\]]+\s*(?:=|\+=|-=)\s*\S|\w\([^)]*\)\s*$)"
)

# How much of an answer must read as code before it is treated as code, and
# how many outright declarations it must contain. Both thresholds were set
# against every answer this installation had archived: 51 prose answers and
# 13 code answers, with no prose answer reaching them.
_CODE_LINE_RATIO = 0.45
_MIN_DECLARATIONS = 2


def looks_like_source(answer: str) -> bool:
    """Whether `answer` is source code rather than prose about it."""
    lines = [line for line in answer.splitlines() if line.strip()]
    if not lines:
        return False
    declarations = sum(1 for line in lines if _DECLARATION.match(line))
    if declarations < _MIN_DECLARATIONS:
        return False
    codey = sum(
        1
        for line in lines
        if _DECLARATION.match(line) or _SYNTAX.search(line) or line[:1] in " \t"
    )
    return codey / len(lines) >= _CODE_LINE_RATIO


def as_markdown(answer: str) -> str:
    """The answer as a Markdown document, with raw source code fenced.

    Every answer is read as Markdown - by the terminal that renders it and by
    anything that opens the .md archive. Markdown joins a run of unfenced
    lines into one paragraph, so raw code arrives with its newlines collapsed
    and its indentation stripped: a whole function lands on a single line and
    a `#` comment becomes a centred heading. That is not a rendering wobble;
    it is the code being destroyed on the way to the user, in the terminal
    and in the archive alike.

    So the answer is made into valid Markdown once, here, at the point it is
    produced - rather than each reader guessing. An answer that already
    carries a fence is left exactly as written: the model did the job itself,
    and second-guessing it is how a formatter starts corrupting good output.
    """
    text = answer.strip()
    if not text or FENCE in text:
        return answer
    if not looks_like_source(text):
        return answer
    return f"{FENCE}\n{text}\n{FENCE}"
