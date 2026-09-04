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
    # Each test builds its renderer fresh under the current environment; a
    # cached UI from another test (or another test file's env) must not leak.
    monkeypatch.setattr(main, "_ACTIVE_UI", None)
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
    main.ask_keep(two_pending(), interactive=True)
    assert library.lookup("research_agent") == SOURCE
    assert library.lookup("summary_agent") == SOURCE


def test_discard_stores_nothing(monkeypatch):
    answers(monkeypatch, "discard")
    main.ask_keep(two_pending(), interactive=True)
    assert library.lookup("research_agent") is None
    assert library.lookup("summary_agent") is None


@pytest.mark.parametrize("reply", ["keep", "k", "KEEP", "  Keep  ", ""])
def test_keep_synonyms(monkeypatch, reply):
    answers(monkeypatch, reply)
    main.ask_keep(two_pending(), interactive=True)
    assert library.lookup("research_agent") == SOURCE


@pytest.mark.parametrize("reply", ["discard", "d", "delete", "DISCARD"])
def test_discard_synonyms(monkeypatch, reply):
    answers(monkeypatch, reply)
    main.ask_keep(two_pending(), interactive=True)
    assert library.lookup("research_agent") is None


def test_unrecognised_answer_reasks(monkeypatch, capsys):
    answers(monkeypatch, "maybe", "yes please", "discard")
    main.ask_keep(two_pending(), interactive=True)
    assert capsys.readouterr().err.count("Please answer keep, discard, or always") == 2
    assert library.lookup("research_agent") is None


# --- it only asks when there is something to decide --------------------------


def test_no_prompt_when_everything_was_reused(monkeypatch):
    def explode(_message):  # pragma: no cover - must never run
        raise AssertionError("must not prompt when nothing new was built")

    monkeypatch.setattr(main, "ask", explode)
    main.ask_keep(_Result(reused=["research_agent"]), interactive=True)


# --- non-interactive use must not hang or silently lose work -----------------


def test_eof_keeps_rather_than_hanging(monkeypatch, capsys):
    monkeypatch.setattr(main, "ask", lambda _message: None)
    main.ask_keep(two_pending(), interactive=True)
    assert "keeping" in capsys.readouterr().err
    assert library.lookup("research_agent") == SOURCE


def test_a_failed_write_is_reported_not_crashed(monkeypatch, capsys):
    answers(monkeypatch, "keep")
    monkeypatch.setattr(library, "remember", lambda *args, **kwargs: False)
    monkeypatch.setattr(main, "ask", lambda _message: "keep")
    main.ask_keep(two_pending(), interactive=True)
    assert "Could not save" in capsys.readouterr().err


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


def test_unknown_flag_fails_loudly(monkeypatch, capsys):
    monkeypatch.setattr(main.sys, "argv", ["main.py", "--tsak", "oops"])
    assert main.main() == 2
    err = capsys.readouterr().err
    assert "usage" in err or "unrecognized" in err


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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: False)

    def explode(_message):  # pragma: no cover - must never run
        raise AssertionError("must not prompt when stdin is a pipe")

    monkeypatch.setattr(main, "ask", explode)
    assert main._check_api_key() is False
    assert not (tmp_path / ".env").exists()


def test_key_prompt_rejects_text_that_is_not_a_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(main, "_read_secret", lambda _message: "discard")
    assert main._check_api_key() is False
    assert not (tmp_path / ".env").exists()
    # Three attempts, not a hard exit on the first slip.
    assert capsys.readouterr().err.count("does not look like an Anthropic key") == 3


def test_key_prompt_saves_a_plausible_key(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: True)
    entered = "sk-ant-api03-" + "a" * 64
    monkeypatch.setattr(main, "_read_secret", lambda _message: entered)
    monkeypatch.setattr(main, "_probe_key", lambda _key: "ok")
    assert main._check_api_key() is True
    assert entered in (tmp_path / ".env").read_text(encoding="utf-8")


def test_a_rejected_key_gets_another_attempt(monkeypatch, tmp_path, capsys):
    """The API saying 401 on a pasted key re-prompts instead of persisting it."""
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: True)
    keys = iter(["sk-ant-revoked-" + "a" * 32, "sk-ant-good-" + "b" * 32])
    monkeypatch.setattr(main, "_read_secret", lambda _message: next(keys))
    verdicts = iter(["rejected", "ok"])
    monkeypatch.setattr(main, "_probe_key", lambda _key: next(verdicts))
    assert main._check_api_key() is True
    assert "rejected that key" in capsys.readouterr().err
    assert "sk-ant-good-" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_a_malformed_env_key_no_longer_sails_through(monkeypatch, tmp_path, capsys):
    """A junk key hand-edited into .env used to pass preflight and die mid-run."""
    monkeypatch.setattr(main, "PROJECT_DIR", tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-proj-this-is-an-openai-key")
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: False)
    assert main._check_api_key() is False
    assert "does not look like" in capsys.readouterr().err


# --- paste-burst protection: a pasted paragraph is ONE task -------------------


def test_pasted_lines_become_one_task(monkeypatch, capsys):
    monkeypatch.setattr(main, "ask", lambda _message, raw=False: "write a brief on:")
    monkeypatch.setattr(main, "_buffered_lines", lambda: ["- solar", "- wind"])
    task = main.read_input()
    assert task == "write a brief on:\n- solar\n- wind"
    assert "3 pasted lines as one task" in capsys.readouterr().err


def test_typed_input_is_untouched_by_the_drain(monkeypatch):
    monkeypatch.setattr(main, "ask", lambda _message, raw=False: "just one line")
    monkeypatch.setattr(main, "_buffered_lines", lambda: [])
    assert main.read_input() == "just one line"


def test_the_drain_never_fires_off_a_terminal(monkeypatch):
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: False)
    assert main._buffered_lines() == []


# --- one-shot slash commands are free -----------------------------------------


def test_a_slash_command_as_the_one_shot_task_never_bills(monkeypatch, capsys):
    import cli

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a slash command must not start the pipeline")

    monkeypatch.setattr(main, "run_task", explode)
    code = main._run_one_shot(cli.Invocation(task="/help"))
    assert code == 0
    assert "Commands" in capsys.readouterr().out


# --- the machine-readable story of a run --------------------------------------


def test_json_payload_for_a_failure_carries_the_translation():
    from problems import Problem

    outcome = main.Outcome(
        error=RuntimeError("boom"),
        problem=Problem(headline="It broke.", advice="Try again.", technical="boom"),
    )
    payload = main._json_payload("do a thing", outcome)
    assert payload["ok"] is False
    assert payload["error"]["headline"] == "It broke."
    assert payload["error"]["advice"] == "Try again."


def test_json_payload_for_a_conversational_reply():
    outcome = main.Outcome(ok=True, kind="reply", answer="Hello.")
    payload = main._json_payload("hi", outcome)
    assert payload == {"ok": True, "task": "hi", "answer": "Hello.", "conversational": True}
