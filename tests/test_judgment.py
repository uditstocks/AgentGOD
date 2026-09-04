"""The main agent's judgement about its own work, with the model faked.

Every rule here is about what the code does with a verdict, never about what
the model decides - so none of it needs an API key.
"""

from __future__ import annotations

import pytest

import judgment
from judgment import Clarification, Verdict, clarifying_question, judge, revision_task


def fake_reply(monkeypatch, reply):
    """Answer the next complete_structured call with `reply`, and record the prompt."""
    seen: dict = {}

    def stub(prompt, output_format, system=None, max_tokens=None, usage=None, effort=None, model=None):
        seen["prompt"] = prompt
        if usage is not None:
            usage.add(10, 5)
        return reply

    monkeypatch.setattr(judgment, "complete_structured", stub)
    return seen


# --- the clarifying question ---------------------------------------------------


def test_no_question_means_none(monkeypatch):
    fake_reply(monkeypatch, Clarification(question=""))
    assert clarifying_question("write a haiku about rain") is None


def test_a_blank_question_is_treated_as_no_question(monkeypatch):
    """The model answering with whitespace must not stall the run on an empty prompt."""
    fake_reply(monkeypatch, Clarification(question="   \n "))
    assert clarifying_question("write a haiku") is None


def test_a_real_question_comes_back_stripped(monkeypatch):
    fake_reply(monkeypatch, Clarification(question="  Which language?  "))
    assert clarifying_question("translate this") == "Which language?"


def test_an_empty_task_is_never_worth_a_question(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("an empty task must not cost an LLM call")

    monkeypatch.setattr(judgment, "complete_structured", explode)
    assert clarifying_question("   ") is None


def test_asking_is_billed(monkeypatch):
    from config import Usage

    fake_reply(monkeypatch, Clarification(question="Which language?"))
    usage = Usage()
    clarifying_question("translate this", usage=usage)
    assert usage.calls == 1


# --- the verdict ---------------------------------------------------------------


def test_a_complete_answer_passes(monkeypatch):
    fake_reply(monkeypatch, Verdict(missing="", done=True))
    assert judge("t", "a").done


def test_an_incomplete_answer_carries_an_instruction(monkeypatch):
    fake_reply(monkeypatch, Verdict(missing="the word limit was ignored", done=False))
    verdict = judge("write 50 words", "a very long answer")
    assert not verdict.done
    assert "word limit" in verdict.missing


def test_a_rejection_with_nothing_to_act_on_is_treated_as_done(monkeypatch):
    """A second attempt needs an instruction; without one it just bills twice."""
    fake_reply(monkeypatch, Verdict(missing="   ", done=False))
    assert judge("t", "a").done


def test_a_very_long_answer_is_capped_before_it_is_judged(monkeypatch):
    """A judge that reads all of a 40 KB answer costs more than the run it checks."""
    seen = fake_reply(monkeypatch, Verdict(missing="", done=True))
    judge("t", "x" * (judgment.MAX_ANSWER_CHARS * 3))
    assert "[...truncated...]" in seen["prompt"]
    assert len(seen["prompt"]) < judgment.MAX_ANSWER_CHARS * 2


def test_the_judge_sees_both_the_request_and_the_answer(monkeypatch):
    seen = fake_reply(monkeypatch, Verdict(missing="", done=True))
    judge("the original request", "the finished answer")
    assert "the original request" in seen["prompt"]
    assert "the finished answer" in seen["prompt"]


# --- the revised task ----------------------------------------------------------


def test_the_revised_task_keeps_the_original_wording():
    revised = revision_task("write 200 words on logging", "the limit was ignored")
    assert "write 200 words on logging" in revised
    assert "the limit was ignored" in revised


def test_the_revised_task_does_not_replace_the_request_with_the_complaint():
    """Agents that work on the critique stop working on what the user asked for."""
    revised = revision_task("write about logging", "no sources were cited")
    assert revised.index("write about logging") < revised.index("no sources were cited")


# --- the check is skipped where it has nothing to catch ------------------------


@pytest.mark.parametrize(
    "task",
    [
        "write a 200-word brief on logging",
        "give me exactly 5 test cases",
        "summarise this in three bullets",
        "reply as a table",
        "write a haiku about rain",
        "return it as json",
        "no more than 2 sentences",
    ],
)
def test_a_stated_demand_is_recognised(task):
    assert judgment.has_checkable_demand(task) is True


@pytest.mark.parametrize(
    "task",
    [
        "translate good morning into French",
        "what is the capital of France",
        "name one benefit of code review",
    ],
)
def test_a_task_with_nothing_measurable_is_recognised(task):
    assert judgment.has_checkable_demand(task) is False


def test_only_a_simple_task_with_no_demand_skips_the_judge():
    """The judge earns its round trip everywhere it could catch something."""
    assert judgment.should_judge("simple", "name one benefit of logging") is False
    # A stated count is exactly what the judge exists for, however simple.
    assert judgment.should_judge("simple", "name 3 benefits of logging") is True
    # Anything the planner did not call trivial is always judged.
    assert judgment.should_judge("standard", "name one benefit of logging") is True
    assert judgment.should_judge("deep", "name one benefit of logging") is True
