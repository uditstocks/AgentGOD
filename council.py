"""The council: an adversarial reading of the answer before the user sees it.

The judge (judgment.py) checks *compliance* - was the word count met, was
every part answered. It is deliberately indifferent to quality, because
"could be better" must not bill a full rerun of every agent. That leaves a
gap on exactly the tasks where quality is the point: a deep analysis can
meet every stated demand and still be shallow, one-sided, or quietly wrong.

The council fills that gap for deep tasks only. One critic reads the merged
answer looking for real faults - claims that need support, considerations a
competent reviewer would demand, reasoning that does not hold - and if it
finds any, one refinement pass repairs those faults and nothing else. Two
calls at most, no agent reruns, and the original answer stands whenever the
critic finds nothing worth the cost of saying.

Biased toward acquittal on purpose, like every self-check in this project:
a critic that always finds fault bills every deep run twice and teaches the
user to ignore it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config import COUNCIL, Usage, complete, complete_structured, model_for

# How much of the answer the critic reads. Same reasoning as the judge's cap:
# a critic that reads a 40 KB answer costs more than the run it is checking.
MAX_ANSWER_CHARS = 8000


class Challenge(BaseModel):
    """What the critic found, stated before it rules.

    `weaknesses` is declared before `flawed` on purpose: structured output is
    generated in field order, so the critic must name the faults before it
    decides whether they matter - not announce a verdict and then justify it.
    """

    weaknesses: str = Field(
        description="The genuine faults, as numbered, concrete instructions a "
        "revision could act on. Empty when the answer holds up."
    )
    flawed: bool = Field(
        description="True only when at least one fault would matter to the "
        "person who asked - not for style, and not for 'could add more'."
    )


CHALLENGE_PROMPT = """You are the council: the adversarial reviewer of a finished answer,
the last reader before the person who asked sees it. Your job is to find the
faults that person would eventually find - before they do.

Read the answer against the request and hunt for:
- claims stated as fact that the answer gives no basis for
- reasoning that does not actually support the conclusion drawn from it
- a one-sided treatment where the request deserves the strongest counter-case
- an important consideration any competent reviewer would demand
- internal contradictions, or numbers that do not add up

Do NOT raise: style, tone, formatting, length the request did not constrain,
or anything that amounts to "more could be said". More can always be said.
A sound answer must come back unflawed - a council that always objects is a
council nobody convenes.

When it is flawed, each weakness must be a concrete instruction a revision
could act on, not a critique of the writing.

The request:
{task}

The answer:
{answer}
"""

REFINE_PROMPT = """You are revising one finished answer after an adversarial review.
The reviewer's objections are listed below. Fix exactly what they name and
preserve everything else - the structure, the voice, the parts that were
right. Do not mention the review, the process, or that anything changed.
Honour every explicit constraint in the original request: length limits,
format, structure, tone.

The request:
{task}

The current answer:
{answer}

The objections to fix:
{weaknesses}
"""


def should_convene(complexity: str) -> bool:
    """Whether this run earns the council's two extra calls."""
    if COUNCIL == "off":
        return False
    if COUNCIL == "always":
        return True
    return complexity == "deep"


def _clip(answer: str) -> str:
    shown = answer.strip()
    if len(shown) > MAX_ANSWER_CHARS:
        shown = shown[:MAX_ANSWER_CHARS] + "\n[...truncated...]"
    return shown


def challenge(
    task: str,
    answer: str,
    usage: Usage | None = None,
    effort: str | None = None,
    model: str | None = None,
) -> Challenge:
    """The critic's reading of the answer: faults first, verdict second.

    A "flawed" with nothing actionable cannot drive a refinement, so it is
    treated as sound - exactly the rule the judge applies to its own verdicts.
    """
    found = complete_structured(
        CHALLENGE_PROMPT.format(task=task, answer=_clip(answer)),
        Challenge,
        max_tokens=2000,
        usage=usage,
        effort=effort,
        # The council only ever sits on deep work, so it is entitled to the
        # deep model: this is the one check that judges substance, not shape.
        model=model or model_for("council", "deep"),
    )
    if found.flawed and not found.weaknesses.strip():
        return Challenge(weaknesses="", flawed=False)
    return found


def refine(
    task: str,
    answer: str,
    weaknesses: str,
    usage: Usage | None = None,
    effort: str | None = None,
    model: str | None = None,
) -> str:
    """One repair pass over the answer, fixing the named faults and nothing else."""
    return complete(
        REFINE_PROMPT.format(task=task, answer=answer, weaknesses=weaknesses),
        usage=usage,
        effort=effort,
        model=model or model_for("council", "deep"),
    ).strip()


def deliberate(
    task: str,
    answer: str,
    usage: Usage | None = None,
    effort: str | None = None,
    model: str | None = None,
) -> tuple[str, bool, str]:
    """The whole sitting: challenge, then refine only if something real was found.

    Returns the answer to carry forward, whether it was improved, and the
    weaknesses that drove the improvement (empty when the answer stood).
    A refinement that comes back empty is discarded: a critic must never
    cost the user the answer it was reviewing.
    """
    verdict = challenge(task, answer, usage=usage, effort=effort, model=model)
    if not verdict.flawed:
        return answer, False, ""
    revised = refine(
        task, answer, verdict.weaknesses, usage=usage, effort=effort, model=model
    )
    if not revised:
        return answer, False, ""
    return revised, True, verdict.weaknesses
