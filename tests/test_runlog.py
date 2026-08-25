"""Run archiving: the answer must outlive the terminal it was printed in."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pytest

import runlog
from runlog import render, save_run, slugify


@dataclass
class _Spec:
    name: str
    role: str


@dataclass
class _Plan:
    agents: list = field(default_factory=list)
    reasoning: str = ""


@dataclass
class _Result:
    response: str = "the answer"
    plan: object = None
    failures: dict = field(default_factory=dict)
    duration_seconds: float = 12.3

    def cost_summary(self) -> str:
        return "4 LLM calls - 900 in / 150 out tokens"


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    folder = tmp_path / "runs"
    monkeypatch.setattr(runlog, "RUNS_DIR", folder)
    return folder


# --- filenames are built from user input, so they must be safe ---------------


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("Write a memo", "write-a-memo"),
        ("  Spaced   out  ", "spaced-out"),
        ("../../etc/passwd", "etc-passwd"),
        ("C:\\Windows\\system32", "c-windows-system32"),
        ("emoji 🚀 and ✓", "emoji-and"),
        ("!!!", "task"),
        ("", "task"),
    ],
)
def test_slugify_is_filesystem_safe(task, expected):
    assert slugify(task) == expected


def test_long_task_is_truncated():
    slug = slugify("word " * 100)
    assert len(slug) <= runlog.MAX_SLUG_LENGTH
    assert not slug.endswith("-")


# --- the archive contains what the user paid for -----------------------------


def test_render_contains_task_answer_and_cost():
    text = render("Summarise the market", _Result(), when=datetime(2026, 8, 25, 18, 30))
    assert "Summarise the market" in text
    assert "the answer" in text
    assert "4 LLM calls" in text
    assert "2026-08-25 18:30" in text
    assert "12.3s" in text


def test_render_lists_the_team():
    plan = _Plan(
        agents=[_Spec("research_agent", "Gather facts"), _Spec("memo_writer", "Write it up")],
        reasoning="Split into research and writing.",
    )
    text = render("t", _Result(plan=plan))
    assert "`research_agent` - Gather facts" in text
    assert "`memo_writer` - Write it up" in text
    assert "Split into research and writing." in text


def test_render_records_failures():
    text = render("t", _Result(failures={"broken_agent": "timed out after 300s"}))
    assert "Agents that failed" in text
    assert "timed out after 300s" in text


def test_render_survives_a_result_without_a_plan():
    assert "the answer" in render("t", _Result(plan=None))


def test_render_handles_non_ascii():
    text = render("Résumé — 2026 ✓", _Result(response="Café ✓ 日本語"))
    assert "Café ✓ 日本語" in text


# --- writing --------------------------------------------------------------


def test_save_run_writes_a_readable_file(runs_dir):
    path = save_run("Write a memo", _Result())
    assert path is not None
    assert path.parent == runs_dir
    assert path.suffix == ".md"
    assert "the answer" in path.read_text(encoding="utf-8")


def test_two_runs_in_the_same_second_do_not_collide(runs_dir):
    paths = [save_run("same task", _Result()) for _ in range(3)]
    assert len(set(paths)) == 3
    assert all(p is not None and p.is_file() for p in paths)


def test_save_run_returns_none_instead_of_raising(runs_dir, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runlog.Path, "write_text", explode)
    assert save_run("t", _Result()) is None  # the answer is already printed
