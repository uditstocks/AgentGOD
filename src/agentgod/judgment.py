"""The two decisions AgentGod makes about its own work.

Everything else in this project is judgement about the *subject*: which
agents a task needs, what code they should contain, how their outputs
combine. This module is the main agent judging itself - before it spends
anything, and after it has spent everything:

- `clarifying_question` - is this task actually clear enough to plan?
- `judge` - does the finished answer answer the question that was asked?

Both are deliberately biased toward silence. An agent that asks about every
task is worse than one that never asks, and an agent that finds every answer
wanting will bill twice for the same work.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from config import Usage, complete_structured, model_for

# How many chars of an answer the judge is shown. A judge that reads a
# 40 KB answer costs more than the run it is checking.
MAX_ANSWER_CHARS = 8000

# A demand the judge can actually measure an answer against: a number, or a
# named shape. These are what the check exists for - the 200-word brief that
# came back at 600 words - and they are the only reason worth a round trip on
# a task the planner already graded as trivially simple.
#
# Any digit counts, deliberately. "3 benefits", "200 words", "top 5" and even
# "the 2008 crisis" all send the answer to the judge: over-judging costs one
# cheap call, while under-judging ships an answer that quietly broke a stated
# limit. The bias belongs on the safe side.
_COUNT_DEMAND = re.compile(r"\d")
_SHAPE_DEMAND = re.compile(
    r"\b(?:bullets?|tables?|lists?|json|csv|yaml|markdown|outlines?|haikus?|"
    r"sonnets?|poems?|emails?|memos?|essays?|summar(?:y|ies)|step by step|"
    r"exactly|at least|at most|no more than|in the form of|as a)\b",
    re.IGNORECASE,
)


def has_checkable_demand(task: str) -> bool:
    """Whether the task states something an answer can be measured against."""
    return bool(_COUNT_DEMAND.search(task) or _SHAPE_DEMAND.search(task))


def should_judge(complexity: str, task: str) -> bool:
    """Whether this answer is worth the price of reading it back.

    The check earns its cost wherever the answer could quietly miss the
    request. On a task the planner itself graded trivially simple - one
    obvious step, a short answer - with no stated count or format to break,
    there is nothing for the judge to catch, and it is a full round trip
    billed on the cheapest task in the product. Anything else is judged.
    """
    return complexity != "simple" or has_checkable_demand(task)


class Clarification(BaseModel):
    """The one question worth asking before any money is spent, if there is one."""

    question: str = Field(
        description="The single question to ask the user, or an empty string "
        "if the task is clear enough to start on"
    )


class Verdict(BaseModel):
    """Whether a finished answer actually answers the request.

    `missing` is declared before `done` on purpose: structured output is
    generated in field order, so the model states what it found before it
    rules on it, rather than announcing a verdict and then justifying it.
    """

    missing: str = Field(
        description="What the answer fails to deliver, as an instruction the "
        "next attempt can act on. Empty when the answer is complete."
    )
    done: bool = Field(description="True when the answer fully satisfies the request")


CLARIFY_PROMPT = """You are about to build a team of agents for the task below.
Before anything is spent, decide whether you need to ask the user ONE question.

Ask only if the task has two reasonable readings that would produce materially
different work, and the wording cannot settle which one is meant.

Do NOT ask:
- for a detail you can reasonably choose yourself - length, format, tone, depth
- to confirm something the task already states
- because the task is broad. Broad is not ambiguous; a broad task has one
  reading, it is just a big one
- for anything an agent could look up instead
- more than one question, ever

Most tasks need no question at all. An empty string is the normal answer, and
asking when you did not need to wastes the user's time and trust.

Task:
{task}
"""

JUDGE_PROMPT = """You are checking one finished answer against the request it was
written for. Be strict about the request and indifferent to style.

The answer is NOT done if it:
- ignores an explicit demand - a word count, a format, a required section, a
  number of items, a language
- answers a narrower or different question than the one asked
- leaves part of a multi-part request unanswered
- delivers an outline, a plan or a promise where the thing itself was asked for

The answer IS done if it delivers what was asked. Do not withhold that because
you would have written it differently, or because more could be said. "Could be
better" is not "not done".

When it is not done, `missing` must name the unmet demand as an instruction the
next attempt can act on - not a critique of the writing.

The request:
{task}

The answer:
{answer}
"""


def clarifying_question(task: str, usage: Usage | None = None) -> str | None:
    """The one question worth asking before planning, or None to just start.

    Never called for a non-interactive run: a question nobody can answer is a
    stalled pipeline, not a careful one.
    """
    if not task.strip():
        return None
    # The cheapest call in the product, deliberately: it sits between the user
    # pressing Enter and anything appearing on screen, and its whole job is a
    # yes/no plus one short question. Low effort, on the fast model.
    result = complete_structured(
        CLARIFY_PROMPT.format(task=task),
        Clarification,
        max_tokens=1000,
        usage=usage,
        effort="low",
        model=model_for("clarify"),
    )
    question = result.question.strip()
    return question or None


def judge(
    task: str, answer: str, usage: Usage | None = None, effort: str | None = None
) -> Verdict:
    """Whether `answer` actually answers `task`.

    A judge that cannot see the answer cannot judge it, but a judge that reads
    all of a very long one costs more than the run it is checking - so the
    answer is capped the same way every other forwarded text is.
    """
    shown = answer.strip()
    if len(shown) > MAX_ANSWER_CHARS:
        shown = shown[:MAX_ANSWER_CHARS] + "\n[...truncated...]"

    # Comparing an answer against the demands the task actually stated is a
    # checking job, not a writing one: it needs care, not depth. It runs on
    # the fast model, which is where the architect model's price stops being
    # worth paying. The council - which judges substance rather than
    # compliance - stays on the main model.
    verdict = complete_structured(
        JUDGE_PROMPT.format(task=task, answer=shown),
        Verdict,
        max_tokens=1500,
        usage=usage,
        effort=effort,
        model=model_for("judge"),
    )
    # A "not done" with nothing to act on cannot drive a second attempt, so it
    # is treated as done - retrying on it would bill for the same answer twice.
    if not verdict.done and not verdict.missing.strip():
        return Verdict(missing="", done=True)
    return verdict


def revision_task(task: str, missing: str) -> str:
    """The task as the agents should receive it on a second attempt.

    The original wording is kept intact and the gap is added to it. Replacing
    the task with the critique is how a revision drifts: the agents would
    then work on the complaint instead of on what the user asked for.
    """
    return (
        f"{task}\n\n"
        "The previous attempt fell short in this specific way:\n"
        f"{missing.strip()}\n"
        "Produce the complete answer this time, meeting every demand above."
    )
