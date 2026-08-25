"""Tests for the reusability check that keeps the library from being poisoned.

The fixtures here are not invented. POISONED_SUMMARY is the agent that was
actually found in inventory/agents/summary_agent.py after sixteen reuses,
answering every task - including "hi" - with an essay about code review.
"""

from __future__ import annotations

import pytest

from topicguard import (
    check_task_is_used,
    check_topic_leakage,
    is_reusable,
    phrases,
    subject_words,
)

# --- the real thing, copied out of the library it poisoned ---------------------

POISONED_SUMMARY = '''
def run(task: str, previous_outputs: dict) -> str:
    prompt = (
        "You are a summary agent. Condense the findings into a single line "
        "that states one benefit of code review. "
        f"User task: {task}. Previous outputs: {format_previous(previous_outputs)}"
    )
    return call_llm(prompt)
'''

CLEAN_SUMMARY = '''
def run(task: str, previous_outputs: dict) -> str:
    prompt = (
        "You are a summary agent. Condense the supplied material to the length "
        "the task asks for.\\n"
        f"Task: {task}\\n"
        f"Material:\\n{format_previous(previous_outputs)}"
    )
    return call_llm(prompt)
'''

CLEAN_RESEARCH = '''
def run(task: str, previous_outputs: dict) -> str:
    formatted = format_previous(previous_outputs)
    prompt = f"You are a research agent. Gather key facts for the task below.\\nTask: {task}\\nEarlier: {formatted}"
    return call_llm(prompt)
'''

POISONED_TRANSLATE = '''
def run(task: str, previous_outputs: dict) -> str:
    prompt = f"You are a translation agent. Translate the phrase into French.\\nTask: {task}"
    return call_llm(prompt)
'''

CLEAN_TRANSLATE = '''
def run(task: str, previous_outputs: dict) -> str:
    prompt = f"You are a translation agent. Translate the supplied text into the language the task names.\\nTask: {task}"
    return call_llm(prompt)
'''

POISONED_SCOOTER = '''
def run(task: str, previous_outputs: dict) -> str:
    prompt = f"Research the electric scooter industry and report on its market share.\\nTask: {task}"
    return call_llm(prompt)
'''

DOCSTRING_ONLY = '''
def run(task: str, previous_outputs: dict) -> str:
    """Summarise a report on the electric scooter industry."""
    return call_llm(f"You are a summary agent. Condense what you are given.\\nTask: {task}")
'''

CODE_REVIEW_TASK = "name one benefit of code review in one line"
SCOOTER_TASK = "write a report on the electric scooter industry"
TRANSLATE_TASK = "translate the phrase good morning into french"


# --- leakage is caught ---------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "task"),
    [
        (POISONED_SUMMARY, CODE_REVIEW_TASK),
        (POISONED_TRANSLATE, TRANSLATE_TASK),
        (POISONED_SCOOTER, SCOOTER_TASK),
    ],
)
def test_hardcoded_subject_is_reported(source, task):
    problems = check_topic_leakage(source, task)
    assert problems, "a hardcoded subject must be reported"
    assert not is_reusable(source, task)


def test_the_report_names_the_leaked_words_and_the_line():
    problems = check_topic_leakage(POISONED_TRANSLATE, TRANSLATE_TASK)
    assert any("french" in problem.lower() for problem in problems)
    assert any(problem.startswith("line ") for problem in problems)


def test_the_report_explains_what_to_do_instead():
    """The message is fed back to the model, so it has to be actionable."""
    problems = check_topic_leakage(POISONED_SUMMARY, CODE_REVIEW_TASK)
    assert any("task" in problem and "runtime" in problem for problem in problems)


# --- role-only agents are left alone -------------------------------------------


@pytest.mark.parametrize(
    ("source", "task"),
    [
        (CLEAN_SUMMARY, CODE_REVIEW_TASK),
        (CLEAN_SUMMARY, SCOOTER_TASK),
        (CLEAN_RESEARCH, CODE_REVIEW_TASK),
        (CLEAN_RESEARCH, SCOOTER_TASK),
        (CLEAN_RESEARCH, TRANSLATE_TASK),
        (CLEAN_TRANSLATE, TRANSLATE_TASK),
    ],
)
def test_role_only_agents_pass(source, task):
    assert check_topic_leakage(source, task) == []
    assert is_reusable(source, task)


def test_a_docstring_may_mention_the_subject():
    """Docstrings describe the agent to a reader; they never reach the model."""
    assert check_topic_leakage(DOCSTRING_ONLY, SCOOTER_TASK) == []


# --- edges ---------------------------------------------------------------------


def test_an_empty_task_disables_the_check():
    assert check_topic_leakage(POISONED_SUMMARY, "") == []
    assert check_topic_leakage(POISONED_SUMMARY, "   ") == []


def test_unparseable_source_is_left_to_codeguard():
    """A syntax error is codeguard's complaint to make, not this module's."""
    assert check_topic_leakage("def run(:", SCOOTER_TASK) == []


def test_craft_vocabulary_is_not_a_subject():
    words = ["summary", "agent", "condense", "the", "findings", "into", "one", "line"]
    assert subject_words(words) == set()


def test_a_real_subject_survives_the_craft_filter():
    assert "scooter" in subject_words(["write", "about", "the", "scooter", "market"])


def test_phrases_ignore_connectives_so_wording_can_shift():
    """'one benefit of code review' must match 'states one benefit of code review'."""
    task = phrases(["one", "benefit", "of", "code", "review"])
    literal = phrases(["that", "states", "one", "benefit", "of", "code", "review"])
    assert task.issubset(literal)


# --- an agent that never looks at the task -------------------------------------
#
# Subtler than a hardcoded subject and worse. This writer_agent was really
# generated, really passed every other check, and really went into the library:
# it builds a careful, well-structured prompt entirely out of string literals,
# and passes `task` only to constraints(). It returns the same paragraph for
# every request ever made of it.

IGNORES_TASK = '''
def run(task: str, previous_outputs: dict) -> str:
    system = "You are a writing agent. Compose concise, informative text."
    material = "Explain the importance of reliability and quality."
    limits = " ".join(constraints(task))
    prompt = (
        f"OBJECTIVE: Write a brief on the importance of reliability.\\n"
        f"MATERIAL: {material}\\n"
        f"CONSTRAINTS: {limits}"
    )
    return call_llm(prompt, system=system)
'''

USES_TASK_IN_AN_FSTRING = '''
def run(task: str, previous_outputs: dict) -> str:
    return call_llm(f"You are a writing agent.\\nOBJECTIVE\\n{task}")
'''

USES_TASK_BY_CONCATENATION = '''
def run(task: str, previous_outputs: dict) -> str:
    return call_llm("You are a writing agent.\\nOBJECTIVE\\n" + task)
'''

USES_TASK_BY_JOIN = '''
def run(task: str, previous_outputs: dict) -> str:
    return call_llm("\\n".join(["You are a writing agent.", "OBJECTIVE", task]))
'''

USES_TASK_DIRECTLY = '''
def run(task: str, previous_outputs: dict) -> str:
    return call_llm(task)
'''


def test_an_agent_that_never_puts_the_task_in_its_prompt_is_rejected():
    problems = check_task_is_used(IGNORES_TASK)
    assert problems
    assert "constraints(task) is not enough" in problems[0]


def test_it_is_not_reusable_even_though_it_leaks_no_subject():
    """It passes the leakage check and is still worthless. Both must run."""
    assert check_topic_leakage(IGNORES_TASK, "write a 60-word brief on unit tests") == []
    assert not is_reusable(IGNORES_TASK, "write a 60-word brief on unit tests")


@pytest.mark.parametrize(
    "source",
    [
        USES_TASK_IN_AN_FSTRING,
        USES_TASK_BY_CONCATENATION,
        USES_TASK_BY_JOIN,
        USES_TASK_DIRECTLY,
        CLEAN_RESEARCH,
        CLEAN_SUMMARY,
        CLEAN_TRANSLATE,
    ],
)
def test_every_way_of_getting_the_task_into_a_prompt_is_accepted(source):
    assert check_task_is_used(source) == []


def test_source_without_a_run_function_is_left_to_codeguard():
    assert check_task_is_used("x = 1") == []


def test_unparseable_source_is_left_to_codeguard_here_too():
    assert check_task_is_used("def run(:") == []


# --- a task's figures are part of its subject ----------------------------------
#
# A real writer_agent shipped with "A concise 60-word paragraph." in its output
# format. It passed every word-based check, and would have answered every later
# request at sixty words no matter what was asked.

HARDCODES_THE_LIMIT = '''
def run(task: str, previous_outputs: dict) -> str:
    return call_llm(f"OBJECTIVE: {task} OUTPUT FORMAT: A concise 60-word paragraph.")
'''

READS_THE_LIMIT_AT_RUNTIME = '''
def run(task: str, previous_outputs: dict) -> str:
    limits = ", ".join(constraints(task)) or "none stated"
    return call_llm(f"OBJECTIVE: {task} CONSTRAINTS: {limits}")
'''


def test_a_hardcoded_length_limit_is_reported():
    problems = check_topic_leakage(HARDCODES_THE_LIMIT, "write a 60-word brief on unit tests")
    assert problems
    assert "60" in problems[0]
    assert "constraints(task)" in problems[0]


def test_reading_the_limit_from_the_task_passes():
    assert check_topic_leakage(READS_THE_LIMIT_AT_RUNTIME, "write a 60-word brief") == []


def test_a_different_number_is_not_a_leak():
    """160 must not match a task that said 60."""
    source = '''
def run(task: str, previous_outputs: dict) -> str:
    return call_llm("Keep lines under 160 characters. " + task)
'''
    assert check_topic_leakage(source, "write a 60-word brief") == []


def test_a_task_with_no_figures_is_unaffected():
    assert check_topic_leakage(HARDCODES_THE_LIMIT, "write a brief on unit tests") == []
