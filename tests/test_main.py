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
