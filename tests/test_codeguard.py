"""Codeguard tests: generated code is executed, so it is checked first."""

from __future__ import annotations

import pytest

from codeguard import check_agent_source

VALID = "import json\n\n\ndef run(task, previous_outputs):\n    return 'ok'\n"


def test_valid_agent_passes():
    assert check_agent_source(VALID) == []


# --- structure: the run(task, previous_outputs) contract ------------------------


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("def run(a, b)\n    return 1\n", "syntax error"),
        ("def go(a, b):\n    return 1\n", "no top-level run"),
        ("def run(task):\n    return 1\n", "positional argument"),
        ("def run(a, b, c):\n    return 1\n", "positional argument"),
        ("async def run(a, b):\n    return 1\n", "must not be async"),
        ("run = lambda a, b: 1\n", "no top-level run"),
    ],
)
def test_contract_violations_are_reported(source, fragment):
    problems = check_agent_source(source)
    assert problems and fragment in problems[0]


# --- safety: imports are an allowlist, not a denylist ---------------------------


@pytest.mark.parametrize(
    "module", ["subprocess", "socket", "shutil", "ctypes", "multiprocessing", "pickle"]
)
def test_dangerous_imports_are_refused(module):
    problems = check_agent_source(f"import {module}\n{VALID}")
    assert any("not allowed" in problem for problem in problems)


@pytest.mark.parametrize("module", ["json", "os", "re", "urllib.request", "pathlib", "requests"])
def test_vetted_imports_are_allowed(module):
    assert check_agent_source(f"import {module}\n{VALID}") == []


def test_from_import_is_checked_too():
    assert check_agent_source(f"from subprocess import run as r\n{VALID}")
    assert check_agent_source(f"from pathlib import Path\n{VALID}") == []


def test_relative_import_is_refused():
    problems = check_agent_source(f"from . import sibling\n{VALID}")
    assert any("relative" in problem for problem in problems)


# --- safety: no shelling out, no eval, no writing ------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "eval(task)",
        "exec(task)",
        "compile(task, 'x', 'exec')",
        "__import__('os')",
        "input('hi')",
    ],
)
def test_banned_builtins(statement):
    source = f"def run(task, previous_outputs):\n    return {statement}\n"
    problems = check_agent_source(source)
    assert any("not allowed" in problem for problem in problems)


@pytest.mark.parametrize(
    "statement",
    ["os.system('dir')", "os.popen('dir')", "os.remove('x')", "os.rmdir('x')", "os.execv('x', [])"],
)
def test_banned_attribute_calls(statement):
    source = f"import os\n\n\ndef run(task, previous_outputs):\n    {statement}\n    return ''\n"
    problems = check_agent_source(source)
    assert any("not allowed" in problem for problem in problems)


@pytest.mark.parametrize("mode", ["'w'", "'a'", "'x'", "'r+'", "'wb'"])
def test_open_for_writing_is_refused(mode):
    source = f"def run(task, previous_outputs):\n    open('f', {mode})\n    return ''\n"
    problems = check_agent_source(source)
    assert any("open() for writing" in problem for problem in problems)


def test_open_for_reading_is_allowed():
    source = "def run(task, previous_outputs):\n    return open('f').read()\n"
    assert check_agent_source(source) == []


def test_non_literal_open_mode_is_refused():
    source = "def run(task, previous_outputs):\n    open('f', task)\n    return ''\n"
    assert check_agent_source(source)


def test_every_problem_is_reported_not_just_the_first():
    source = "import socket\nimport subprocess\n\n\ndef run(a, b):\n    return eval(a)\n"
    assert len(check_agent_source(source)) >= 3
