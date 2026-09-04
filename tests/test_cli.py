"""The command-line contract: parsing, verbs, policies, exit codes.

cli.py imports nothing from the project, so these tests run without the SDK,
a key, or a network - the contract is pure argument logic.
"""

from __future__ import annotations

import io

import pytest

import cli


def parse(*argv: str) -> cli.Invocation:
    return cli.parse(list(argv))


# --- the task, in every spelling ------------------------------------------------


def test_bare_words_are_the_task():
    assert parse("write", "a", "haiku").task == "write a haiku"


def test_a_quoted_task_is_one_word_to_the_shell_and_still_works():
    assert parse("write a haiku about rain").task == "write a haiku about rain"


def test_no_arguments_means_interactive():
    invocation = parse()
    assert invocation.interactive
    assert invocation.task is None and invocation.command is None


def test_everything_after_task_flag_is_task_text():
    """The historical escape hatch: a task may contain flag-shaped words."""
    invocation = parse("--task", "explain", "what", "--json", "means")
    assert invocation.task == "explain what --json means"
    assert invocation.json_output is False


def test_task_flag_with_nothing_after_it_is_a_usage_error():
    with pytest.raises(SystemExit) as caught:
        parse("--task")
    assert caught.value.code == 2


def test_dash_reads_the_task_from_stdin(monkeypatch):
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("summarise this\n"))
    assert parse("-").task == "summarise this"


def test_stdin_noise_is_stripped_from_a_piped_task(monkeypatch):
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("﻿write a haiku"))
    assert parse("-").task == "write a haiku"


def test_an_empty_stdin_task_is_a_usage_error(monkeypatch):
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(""))
    with pytest.raises(SystemExit) as caught:
        parse("-")
    assert caught.value.code == 2


# --- the free command verbs -----------------------------------------------------


def test_bare_verbs_become_commands_not_tasks():
    invocation = parse("library")
    assert invocation.command == ("library", "")
    assert invocation.task is None


def test_forget_carries_its_argument():
    assert parse("forget", "research_agent").command == ("forget", "research_agent")


def test_a_verb_after_task_flag_is_still_a_task():
    assert parse("--task", "library").task == "library"
    assert parse("--task", "library").command is None


def test_a_verb_with_a_sentence_around_it_is_a_task():
    invocation = parse("what", "does", "the", "library", "hold")
    assert invocation.command is None
    assert invocation.task == "what does the library hold"


# --- flags and policies ---------------------------------------------------------


def test_keep_and_discard_map_to_policies():
    assert parse("--keep", "do", "a", "thing").keep == "always"
    assert parse("--discard", "do", "a", "thing").keep == "never"
    assert parse("do", "a", "thing").keep is None


def test_keep_and_discard_are_mutually_exclusive():
    with pytest.raises(SystemExit) as caught:
        parse("--keep", "--discard", "task")
    assert caught.value.code == 2


def test_json_requires_something_to_report_on():
    with pytest.raises(SystemExit) as caught:
        parse("--json")
    assert caught.value.code == 2


def test_unknown_flags_exit_2():
    with pytest.raises(SystemExit) as caught:
        parse("--tsak", "oops")
    assert caught.value.code == 2


def test_version_exits_0(capsys):
    with pytest.raises(SystemExit) as caught:
        parse("--version")
    assert caught.value.code == 0
    assert cli.__version__ in capsys.readouterr().out


def test_effort_and_council_are_validated():
    assert parse("--effort", "high", "task").effort == "high"
    assert parse("--council", "off", "task").council == "off"
    with pytest.raises(SystemExit):
        parse("--effort", "turbo", "task")


# --- apply(): overrides reach the environment config reads -----------------------


def _clean_env(monkeypatch) -> None:
    """Unset the override names AND register their restoration.

    delenv on an absent variable records nothing to restore, so a value the
    test writes afterwards would leak into every later test. Setting then
    deleting registers both, and teardown ends with the variable absent.
    """
    for name in ("MODEL", "LLM_EFFORT", "COUNCIL", "AGENTGOD_PLAIN", "AGENTGOD_QUIET"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def test_apply_writes_the_override_environment(monkeypatch):
    _clean_env(monkeypatch)
    cli.apply(
        cli.Invocation(
            task="t", model="claude-x", effort="low", council="off", plain=True, quiet=True
        )
    )
    import os

    assert os.environ["MODEL"] == "claude-x"
    assert os.environ["LLM_EFFORT"] == "low"
    assert os.environ["COUNCIL"] == "off"
    assert os.environ["AGENTGOD_PLAIN"] == "1"
    assert os.environ["AGENTGOD_QUIET"] == "1"


def test_apply_touches_nothing_without_overrides(monkeypatch):
    _clean_env(monkeypatch)
    cli.apply(cli.Invocation(task="t"))
    import os

    assert "MODEL" not in os.environ
    assert "AGENTGOD_QUIET" not in os.environ
