"""Tests for what the session remembers between tasks.

The bug being pinned: "write a haiku about rain" worked, and the follow-up
"now make it about snow instead" came back as a factual essay, because the
word haiku was never in the second prompt.
"""

from __future__ import annotations

import pytest

from conversation import MAX_CONTEXT_CHARS, Conversation, is_follow_up


@pytest.mark.parametrize(
    "text",
    [
        "now make it about snow instead",
        "make it shorter",
        "shorten it",
        "translate that into french",
        "try again",
        "same but funnier",
        "what about the risks",
        "expand on that",
        "rewrite it as bullets",
        "can you also add a conclusion",
        "in spanish",
    ],
)
def test_follow_ups_are_recognised(text):
    assert is_follow_up(text)


@pytest.mark.parametrize(
    "text",
    [
        "write a haiku about rain",
        "compare postgres and sqlite for a small web app",
        "summarise the attached brief in five bullets",
        "name one benefit of static typing",
        "draft a 200-word memo on the electric scooter market and list the risks",
    ],
)
def test_standalone_tasks_are_not_follow_ups(text):
    assert not is_follow_up(text)


def test_a_long_message_carries_its_own_context():
    assert not is_follow_up("make it shorter " + " ".join(["word"] * 40))


def test_empty_input_is_not_a_follow_up():
    assert not is_follow_up("")


# --- the memory itself ---------------------------------------------------------


def test_a_follow_up_is_given_the_previous_exchange():
    session = Conversation()
    session.remember("write a haiku about rain", "Whispers on the pane...")

    prepared, used = session.contextualise("now make it about snow instead")

    assert used
    assert "haiku" in prepared
    assert "Whispers on the pane" in prepared
    assert "now make it about snow instead" in prepared


def test_a_standalone_task_is_passed_through_untouched():
    session = Conversation()
    session.remember("write a haiku about rain", "Whispers on the pane...")

    task = "compare postgres and sqlite for a small web app"
    prepared, used = session.contextualise(task)

    assert not used
    assert prepared == task


def test_the_first_task_of_a_session_has_nothing_to_lean_on():
    prepared, used = Conversation().contextualise("make it shorter")
    assert not used
    assert prepared == "make it shorter"


def test_a_long_previous_answer_is_clipped():
    session = Conversation()
    session.remember("write something", "x" * (MAX_CONTEXT_CHARS * 3))

    prepared, used = session.contextualise("make it shorter")

    assert used
    assert "[...truncated...]" in prepared
    assert len(prepared) < MAX_CONTEXT_CHARS * 2


def test_only_the_most_recent_turns_are_kept():
    session = Conversation(limit=3)
    for index in range(6):
        session.remember(f"task {index}", f"answer {index}")

    assert len(session.turns) == 3
    assert session.turns[0].task == "task 3"
    assert session.last is not None
    assert session.last.task == "task 5"


def test_clearing_reports_what_it_dropped():
    session = Conversation()
    session.remember("a", "b")
    session.remember("c", "d")

    assert session.clear() == 2
    assert session.turns == []
    assert session.last is None
    assert session.clear() == 0
