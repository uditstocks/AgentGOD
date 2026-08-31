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


# --- a filesystem name is only a filesystem call on a filesystem receiver ------
#
# `.replace()` was once banned outright. Building a prompt is string work, and
# `prompt.replace(...)` is the most natural line in it, so whole agents were
# being rejected over a method that cannot touch a disk.


def _probe(line: str) -> list[str]:
    source = (
        "import os\n"
        "from pathlib import Path\n"
        "def run(task, previous_outputs):\n"
        f"    {line}\n"
        "    return ''\n"
    )
    return check_agent_source(source)


@pytest.mark.parametrize(
    "line",
    [
        'prompt = task.replace("a", "b")',
        'text = "x".replace("x", "y")',
        "items = [1]; items.remove(1)",
        'parts = task.split(","); parts.remove("x")',
    ],
)
def test_string_and_list_methods_are_allowed(line):
    assert _probe(line) == []


@pytest.mark.parametrize(
    "line",
    [
        'os.replace("a", "b")',
        'os.remove("a")',
        'os.rename("a", "b")',
        'Path("a").replace("b")',
        'os.path.replace("a", "b")',
    ],
)
def test_the_same_names_are_refused_on_a_filesystem_receiver(line):
    assert _probe(line)


@pytest.mark.parametrize(
    "line", ['os.unlink("a")', 'os.system("x")', 'os.rmdir("a")', 'os.chmod("a", 0)']
)
def test_names_that_are_only_ever_filesystem_calls_stay_banned_outright(line):
    assert _probe(line)


def test_the_installer_and_the_import_check_share_one_list():
    """A package that installs but cannot be imported fails after the install."""
    import codeguard
    import executor

    assert executor.ALLOWED_PACKAGES is codeguard.ALLOWED_PACKAGES
    assert set(codeguard.ALLOWED_PACKAGES.values()) == set(codeguard.ALLOWED_THIRD_PARTY)


# --- the standard library is allowed wholesale, minus a named few ---------------


@pytest.mark.parametrize(
    "module",
    ["csv", "sqlite3", "hashlib", "zipfile", "difflib", "argparse", "asyncio", "secrets"],
)
def test_ordinary_stdlib_is_allowed(module):
    """These were refused for no reason but absence from a curated list."""
    assert check_agent_source(f"import {module}\n{VALID}") == []


def test_a_package_nobody_vetted_is_still_refused():
    problems = check_agent_source(f"import leftpad\n{VALID}")
    assert any("not allowed" in problem for problem in problems)


def test_every_vetted_package_is_importable_by_its_import_name():
    from codeguard import ALLOWED_PACKAGES

    for pip_name, import_name in ALLOWED_PACKAGES.items():
        assert check_agent_source(f"import {import_name}\n{VALID}") == [], pip_name


def test_the_refusal_says_why_without_reciting_the_allowlist():
    """The message goes into a repair prompt; the standard library is 300 names."""
    problem = check_agent_source(f"import subprocess\n{VALID}")[0]
    assert "shells out" in problem
    assert len(problem) < 200
