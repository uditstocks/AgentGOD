"""The human-in-the-loop decision: nothing is kept without the user saying so."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import library
import main


@dataclass
class _Result:
    pending: dict = field(default_factory=dict)
    reused: list = field(default_factory=list)
    built: list = field(default_factory=list)


SOURCE = "def run(task, previous_outputs):\n    return 'ok'\n"


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    root = tmp_path / "inventory"
    monkeypatch.setattr(library, "INVENTORY_DIR", root)
    monkeypatch.setattr(library, "LIBRARY_DIR", root / "agents")
    monkeypatch.setattr(library, "INDEX_PATH", root / "index.json")
    return root


def answers(monkeypatch, *replies):
    """Feed scripted answers to main.ask, then EOF."""
    queue = list(replies)
    monkeypatch.setattr(main, "ask", lambda _message: queue.pop(0) if queue else None)


def two_pending() -> _Result:
    return _Result(
        pending={
            "research_agent": ("Gather facts", SOURCE),
            "summary_agent": ("Condense", SOURCE),
        },
        built=["research_agent", "summary_agent"],
    )


# --- the decision is real in both directions ---------------------------------


def test_keep_stores_every_new_agent(monkeypatch):
    answers(monkeypatch, "keep")
    main.ask_keep(two_pending())
    assert library.lookup("research_agent") == SOURCE
    assert library.lookup("summary_agent") == SOURCE


def test_discard_stores_nothing(monkeypatch):
    answers(monkeypatch, "discard")
    main.ask_keep(two_pending())
    assert library.lookup("research_agent") is None
    assert library.lookup("summary_agent") is None


@pytest.mark.parametrize("reply", ["keep", "k", "KEEP", "  Keep  ", ""])
def test_keep_synonyms(monkeypatch, reply):
    answers(monkeypatch, reply)
    main.ask_keep(two_pending())
    assert library.lookup("research_agent") == SOURCE


@pytest.mark.parametrize("reply", ["discard", "d", "delete", "DISCARD"])
def test_discard_synonyms(monkeypatch, reply):
    answers(monkeypatch, reply)
    main.ask_keep(two_pending())
    assert library.lookup("research_agent") is None


def test_unrecognised_answer_reasks(monkeypatch, capsys):
    answers(monkeypatch, "maybe", "yes please", "discard")
    main.ask_keep(two_pending())
    assert capsys.readouterr().out.count("Please type 'keep' or 'discard'") == 2
    assert library.lookup("research_agent") is None


# --- it only asks when there is something to decide --------------------------


def test_no_prompt_when_everything_was_reused(monkeypatch):
    def explode(_message):  # pragma: no cover - must never run
        raise AssertionError("must not prompt when nothing new was built")

    monkeypatch.setattr(main, "ask", explode)
    main.ask_keep(_Result(reused=["research_agent"]))


# --- non-interactive use must not hang or silently lose work -----------------


def test_eof_keeps_rather_than_hanging(monkeypatch, capsys):
    monkeypatch.setattr(main, "ask", lambda _message: None)
    main.ask_keep(two_pending())
    assert "keeping" in capsys.readouterr().out
    assert library.lookup("research_agent") == SOURCE


def test_a_failed_write_is_reported_not_crashed(monkeypatch, capsys):
    answers(monkeypatch, "keep")
    monkeypatch.setattr(library, "remember", lambda *args: False)
    monkeypatch.setattr(main, "ask", lambda _message: "keep")
    main.ask_keep(two_pending())
    assert "Could not save" in capsys.readouterr().out


# --- cleanup always clears scratch, whichever way the user decided -----------


def test_cleanup_removes_working_copies(tmp_path):
    paths = []
    for name in ("research_agent", "summary_agent"):
        path = tmp_path / f"{name}.py"
        path.write_text(SOURCE, encoding="utf-8")
        paths.append(path)

    main.cleanup(paths)
    assert not any(path.exists() for path in paths)


def test_cleanup_with_nothing_to_do_is_silent(capsys):
    main.cleanup([])
    assert capsys.readouterr().out == ""


# --- argument handling: flags are ours, task text is the model's -------------


def test_plain_flag_is_stripped_before_task():
    args, plain = main._strip_plain_flag(["--plain", "--task", "do a thing"])
    assert args == ["--task", "do a thing"]
    assert plain is True


def test_plain_inside_task_text_is_not_consumed():
    args, plain = main._strip_plain_flag(["--task", "explain what --plain means"])
    assert args == ["--task", "explain what --plain means"]
    assert plain is False


def test_unknown_flag_fails_loudly(monkeypatch, capsys):
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--tsak", "oops"])
    assert main.main() == 2
    assert "Unknown option" in capsys.readouterr().out


# --- stdin noise: invisible characters must never change an answer -----------


class _ScriptedUI:
    def __init__(self, reply):
        self.reply = reply

    def input(self, _message):
        return self.reply

    def blank(self):
        print()


@pytest.mark.parametrize("raw", ["discard", "﻿discard", "ï»¿discard", "  discard \r"])
def test_ask_strips_stdin_noise(monkeypatch, raw):
    monkeypatch.setattr(main, "_ACTIVE_UI", _ScriptedUI(raw))
    assert main.ask("> ") == "discard"


# --- the key prompt must never eat piped data and write it to .env -----------


def test_key_prompt_refuses_non_interactive_stdin(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: False)

    def explode(_message):  # pragma: no cover - must never run
        raise AssertionError("must not prompt when stdin is a pipe")

    monkeypatch.setattr(main, "ask", explode)
    assert main._check_api_key() is False
    assert not (tmp_path / ".env").exists()


def test_key_prompt_rejects_text_that_is_not_a_key(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(main, "ask", lambda _message: "discard")
    assert main._check_api_key() is False
    assert not (tmp_path / ".env").exists()


def test_key_prompt_saves_a_plausible_key(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: True)
    entered = "sk-or-v1-" + "a" * 64
    monkeypatch.setattr(main, "ask", lambda _message: entered)
    assert main._check_api_key() is True
    assert entered in (tmp_path / ".env").read_text(encoding="utf-8")
