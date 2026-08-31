"""The council's sitting, with the model faked.

Every rule is about what the code does with a challenge, never about what
the critic decides - so none of it needs an API key.
"""

from __future__ import annotations

import council
from council import Challenge, deliberate, should_convene


def fake_challenge(monkeypatch, reply: Challenge):
    seen: dict = {}

    def stub(prompt, output_format, system=None, max_tokens=None, usage=None, effort=None):
        seen["prompt"] = prompt
        return reply

    monkeypatch.setattr(council, "complete_structured", stub)
    return seen


def fake_refinement(monkeypatch, reply: str):
    def stub(prompt, system=None, max_tokens=None, usage=None, search=False, effort=None):
        return reply

    monkeypatch.setattr(council, "complete", stub)


# --- when the council convenes at all ------------------------------------------


def test_auto_convenes_only_for_deep_tasks(monkeypatch):
    monkeypatch.setattr(council, "COUNCIL", "auto")
    assert should_convene("deep")
    assert not should_convene("standard")
    assert not should_convene("simple")


def test_off_and_always_do_what_they_say(monkeypatch):
    monkeypatch.setattr(council, "COUNCIL", "off")
    assert not should_convene("deep")
    monkeypatch.setattr(council, "COUNCIL", "always")
    assert should_convene("simple")


# --- the sitting ----------------------------------------------------------------


def test_a_sound_answer_stands_and_costs_no_refinement(monkeypatch):
    fake_challenge(monkeypatch, Challenge(weaknesses="", flawed=False))

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a sound answer must not be refined")

    monkeypatch.setattr(council, "complete", explode)
    answer, improved, weaknesses = deliberate("the task", "the answer")
    assert (answer, improved, weaknesses) == ("the answer", False, "")


def test_a_flawed_answer_is_refined(monkeypatch):
    fake_challenge(
        monkeypatch, Challenge(weaknesses="1. cite the source", flawed=True)
    )
    fake_refinement(monkeypatch, "the better answer")
    answer, improved, weaknesses = deliberate("the task", "the answer")
    assert answer == "the better answer"
    assert improved
    assert weaknesses == "1. cite the source"


def test_a_flawed_verdict_with_nothing_actionable_is_treated_as_sound(monkeypatch):
    """A critic that objects without naming a fault cannot drive a revision."""
    fake_challenge(monkeypatch, Challenge(weaknesses="   ", flawed=True))

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("nothing actionable, so nothing to refine")

    monkeypatch.setattr(council, "complete", explode)
    answer, improved, _ = deliberate("the task", "the answer")
    assert (answer, improved) == ("the answer", False)


def test_an_empty_refinement_never_replaces_the_answer(monkeypatch):
    """The critic must not cost the user the answer it was reviewing."""
    fake_challenge(monkeypatch, Challenge(weaknesses="1. add the risks", flawed=True))
    fake_refinement(monkeypatch, "")
    answer, improved, _ = deliberate("the task", "the answer")
    assert (answer, improved) == ("the answer", False)


def test_the_critic_sees_both_the_request_and_the_answer(monkeypatch):
    seen = fake_challenge(monkeypatch, Challenge(weaknesses="", flawed=False))
    deliberate("write the memo", "here is the memo")
    assert "write the memo" in seen["prompt"]
    assert "here is the memo" in seen["prompt"]


def test_a_very_long_answer_is_capped_before_the_critic_reads_it(monkeypatch):
    seen = fake_challenge(monkeypatch, Challenge(weaknesses="", flawed=False))
    deliberate("the task", "x" * 50_000)
    assert len(seen["prompt"]) < 20_000
    assert "[...truncated...]" in seen["prompt"]
