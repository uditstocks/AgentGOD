"""Tests for the slash commands.

None of these may cost an LLM call: the whole point is that a question about
the session is answered from disk, instantly.
"""

from __future__ import annotations

import pytest

import commands
from commands import PASTE, QUIT, Command, handle, help_text, parse
from conversation import Conversation

# --- parsing -------------------------------------------------------------------


def test_a_slash_line_is_a_command():
    assert parse("/library") == Command(name="library")


def test_an_argument_is_kept():
    assert parse("/forget research_agent") == Command(name="forget", argument="research_agent")


def test_case_is_ignored():
    assert parse("/HELP").name == "help"


@pytest.mark.parametrize("text", ["write a haiku", "", "   ", "/", "http://x/y"])
def test_ordinary_input_is_not_a_command(text):
    assert parse(text) is None


# --- help ----------------------------------------------------------------------


def test_help_lists_every_command():
    text = help_text()
    for name in commands.COMMANDS:
        assert f"/{name}" in text


def test_help_explains_how_to_give_a_task_not_only_the_commands():
    text = help_text()
    assert "task" in text.lower()
    assert "follow-up" in text.lower() or "Follow-ups" in text


# --- dispatch ------------------------------------------------------------------


def test_quit_is_signalled_not_printed():
    assert handle(Command("quit")) == QUIT
    assert handle(Command("exit")) == QUIT


def test_paste_is_signalled():
    assert handle(Command("paste")) == PASTE


def test_an_unknown_command_says_so_and_lists_the_real_ones():
    text = handle(Command("nonsense"))
    assert "Unknown command" in text
    assert "/library" in text


def test_clear_reports_what_it_forgot():
    session = Conversation()
    session.remember("a", "b")
    assert "1 exchange" in handle(Command("clear"), session)


def test_clear_on_a_fresh_session_says_there_was_nothing():
    assert "fresh session" in handle(Command("clear"), Conversation())


def test_forget_without_an_argument_asks_which():
    assert "Which one" in handle(Command("forget"))


# --- the library views, against a real temporary library -----------------------


@pytest.fixture
def library_dir(tmp_path, monkeypatch):
    import library

    monkeypatch.setattr(library, "LIBRARY_DIR", tmp_path / "agents")
    monkeypatch.setattr(library, "INDEX_PATH", tmp_path / "index.json")
    monkeypatch.setattr(library, "INVENTORY_DIR", tmp_path)
    return library


def test_library_view_on_an_empty_library_explains_how_it_fills(library_dir):
    assert "empty" in handle(Command("library")).lower()


def test_library_view_lists_what_is_kept(library_dir):
    library_dir.remember("research_agent", "gather facts", "def run(t, p): return ''", task="x")
    text = handle(Command("library"))
    assert "research_agent" in text
    assert "gather facts" in text


def test_forget_removes_a_kept_agent(library_dir):
    library_dir.remember("research_agent", "gather", "def run(t, p): return ''")
    assert "Dropped" in handle(Command("forget", "research_agent"))
    assert library_dir.lookup("research_agent") is None


def test_forget_on_an_unknown_agent_says_so(library_dir):
    assert "no `ghost_agent`" in handle(Command("forget", "ghost_agent"))


def test_audit_reports_a_clean_library(library_dir):
    library_dir.remember(
        "summary_agent",
        "condense",
        'def run(task, previous_outputs):\n    return call_llm("You are a summary agent. " + task)\n',
        task="name one benefit of code review",
    )
    assert "clean" in handle(Command("audit")).lower()


def test_audit_names_an_agent_that_hardcoded_its_first_task(library_dir):
    poisoned = (
        "def run(task, previous_outputs):\n"
        '    return call_llm("Condense the findings into one line that states '
        'one benefit of code review. " + task)\n'
    )
    library_dir.remember("summary_agent", "condense", poisoned, task="name one benefit of code review")

    text = handle(Command("audit"))

    assert "summary_agent" in text
    assert "hardcoded" in text
    assert "/forget" in text


def test_audit_does_not_accuse_agents_it_cannot_check(library_dir):
    """An agent kept before the task was recorded has nothing to check against."""
    library_dir.remember("legacy_agent", "does things", "def run(t, p): return ''")
    text = handle(Command("audit"))
    assert "legacy_agent" in text
    assert "Not checkable" in text
