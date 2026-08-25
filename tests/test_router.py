"""Tests for the line between conversation and work.

Two failure directions, and they are not equally bad. Missing a
conversational line costs a pipeline run and a wrong answer; claiming a real
task costs the user their task entirely. So the task cases below are the
ones that matter most.
"""

from __future__ import annotations

import pytest

from router import Intent, classify, normalise

# --- small talk ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "hi", "Hello", "hello!", "hey", "HI THERE", "hiii", "helloooo",
        "good morning", "good evening", "namaste", "hola", "howdy",
        "how are you?", "hows it going", "sup", "yo", "ping", "test",
    ],
)
def test_greetings(text):
    assert classify(text) is Intent.GREETING


@pytest.mark.parametrize(
    "text",
    ["thanks", "Thank you!", "thank you so much", "thx", "ty", "cheers",
     "perfect", "awesome", "nice one", "got it", "ok thanks"],
)
def test_thanks(text):
    assert classify(text) is Intent.THANKS


@pytest.mark.parametrize(
    "text", ["bye", "goodbye", "see you later", "cya", "good night", "thats all"]
)
def test_farewells(text):
    assert classify(text) is Intent.FAREWELL


# --- questions about this system -----------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "what can you do?",
        "so what can you do",
        "what can u do",
        "what do you do",
        "what are your capabilities",
        "what services do you provide",
        "what features do you offer",
        "what kind of tasks can you do",
        "what are you good at",
        "how do you work",
        "how does this work",
        "list your features",
        "capabilities",
        # the misspelt form the fallback rule exists for
        "wnat service you propvide",
        "what all can you do",
    ],
)
def test_capability_questions(text):
    assert classify(text) is Intent.CAPABILITY


@pytest.mark.parametrize(
    "text",
    [
        "who are you", "who r u", "what are you", "what is this",
        "tell me about yourself", "describe yourself", "introduce yourself",
        "who made you", "what model are you using", "are you an ai",
    ],
)
def test_identity_questions(text):
    assert classify(text) is Intent.IDENTITY


@pytest.mark.parametrize(
    "text", ["help", "help me", "how do i use this", "usage", "commands", "what should i type"]
)
def test_help_questions(text):
    assert classify(text) is Intent.HELP


@pytest.mark.parametrize(
    "text",
    [
        "what is the stock price of itc today",
        "whats the weather right now",
        "what is the current temperature",
        "what time is it",
        "what is todays date",
    ],
)
def test_requests_for_live_readings(text):
    assert classify(text) is Intent.LIVE_DATA


# --- real work must survive every rule above -----------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "write a haiku about rain",
        "can you write a poem about how you are feeling",
        "summarize this contract",
        "what is 17 * 23",
        "explain the french revolution",
        "tell me about the french revolution",
        "write a report on current AI trends",
        "hello, summarise this contract for me",
        "compare react and vue for a new project",
        "name one benefit of static typing",
        "give me a list of your favourite books",
        "what is the capital of France",
        "how do i use python decorators",
        "translate good morning into french",
        "write a 200-word investor memo on the scooter market",
        "help me write a cover letter",
        "research the history of the printing press and summarise it",
    ],
)
def test_real_tasks_are_never_claimed_by_a_conversational_rule(text):
    assert classify(text) is Intent.TASK


def test_a_long_message_is_always_work():
    """No conversational rule may claim a message long enough to carry orders."""
    text = "hi " + " ".join(["word"] * 40)
    assert classify(text) is Intent.TASK


def test_empty_input_is_left_to_the_caller():
    assert classify("") is Intent.TASK
    assert classify("   ") is Intent.TASK


# --- normalisation -------------------------------------------------------------


def test_leading_filler_is_stripped():
    assert normalise("So, bhai, what can you do?") == "what can you do"


def test_trailing_time_words_survive():
    """'now' is filler at the end of a question and the whole point of another."""
    assert normalise("whats the weather right now") == "whats the weather right now"
    assert normalise("what can you do please") == "what can you do"


def test_intent_is_task_reports_itself():
    assert Intent.TASK.is_task
    assert not Intent.GREETING.is_task
