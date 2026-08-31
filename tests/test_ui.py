"""The presentation layer: degradation, the plain renderer, and the rich one.

Everything here runs without a terminal. PlainUI is asserted line by line
because pipes and CI depend on it; RichUI is driven through the same event
sequence into a capture buffer to prove it renders without crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

import ui as ui_module
from events import TaskEvents
from executor import AgentResult, DependencyReport
from ui import PlainUI, first_line, make_ui


@dataclass
class FakeResult:
    """Just enough of TaskResult for the presentation surface."""

    response: str = "The answer."
    failures: dict = field(default_factory=dict)
    reused: list = field(default_factory=list)
    built: list = field(default_factory=list)
    duration_seconds: float = 12.3

    def cost_summary(self) -> str:
        return "4 LLM calls · 1,000 in / 500 out tokens · ~$0.0010"


def fake_plan():
    return SimpleNamespace(
        reasoning="research feeds summary",
        agents=[
            SimpleNamespace(name="research_agent", role="gather facts"),
            SimpleNamespace(name="summary_agent", role="condense them"),
        ],
    )


def drive_events(surface) -> None:
    """Every pipeline event once, repair and failure included."""
    surface.phase_started(1, 5, "Planning agents")
    surface.plan_ready(fake_plan())
    surface.phase_started(2, 5, "Generating agent code")
    surface.agent_build_started("research_agent")
    surface.agent_ready("research_agent", "research_agent.py", reused=False)
    surface.agent_ready("summary_agent", "summary_agent.py", reused=True)
    surface.phase_started(3, 5, "Checking dependencies")
    surface.deps_checked(DependencyReport(refused=["leftpad"], installed=["requests"]))
    surface.phase_started(4, 5, "Executing agents")
    surface.agent_started("research_agent", 1, 2)
    surface.agent_repairing("research_agent", 1, 2, "Traceback: boom")
    surface.agent_finished(
        AgentResult(
            name="research_agent", path=Path("research_agent.py"), ok=True,
            output="facts", duration_seconds=6.3, input_tokens=900, output_tokens=400,
        )
    )
    surface.agent_started("summary_agent", 2, 2)
    surface.agent_finished(
        AgentResult(
            name="summary_agent", path=Path("summary_agent.py"), ok=False,
            error="timed out after 300s",
        )
    )
    surface.phase_started(5, 5, "Merging outputs")
    surface.merge_started(1)


def drive_full_run(surface, saved: Path | None) -> None:
    """One complete task through the whole presentation surface."""
    surface.run_started("write a report", echo=True)
    drive_events(surface)
    surface.run_succeeded(
        FakeResult(
            failures={"summary_agent": "timed out after 300s"},
            reused=["summary_agent"],
            built=["research_agent"],
        ),
        saved,
    )
    surface.run_ended()


# --- the base event surface ---------------------------------------------------


def test_task_events_is_a_complete_no_op(capsys):
    drive_events(TaskEvents())
    assert capsys.readouterr().out == ""


def test_first_line_takes_the_first_and_caps_it():
    assert first_line("alpha\nbeta") == "alpha"
    assert first_line("   \n  beta  ") == "beta"
    assert first_line("", 10) == ""
    assert len(first_line("x" * 500, 40)) == 40


# --- renderer selection -------------------------------------------------------


def test_plain_is_forced_by_environment(monkeypatch):
    monkeypatch.setenv("AGENTGOD_PLAIN", "1")
    assert type(make_ui()) is PlainUI


def test_plain_is_chosen_without_a_terminal(monkeypatch):
    monkeypatch.delenv("AGENTGOD_PLAIN", raising=False)
    # Under pytest stdout is captured, so a rich console reports no terminal.
    assert type(make_ui()) is PlainUI


def test_plain_is_chosen_when_rich_is_missing(monkeypatch):
    monkeypatch.delenv("AGENTGOD_PLAIN", raising=False)

    def refuse(name):
        raise ImportError(name)

    monkeypatch.setattr(ui_module.importlib, "import_module", refuse)
    assert type(make_ui()) is PlainUI


def test_plain_is_chosen_when_rich_is_broken(monkeypatch):
    """An ancient or corrupted rich must degrade exactly like a missing one."""
    monkeypatch.delenv("AGENTGOD_PLAIN", raising=False)

    def explode(name):
        raise RuntimeError("rich is installed but unusable")

    monkeypatch.setattr(ui_module.importlib, "import_module", explode)
    assert type(make_ui()) is PlainUI


# --- the plain renderer: pipes and CI depend on these exact shapes ------------


def test_plain_full_run_reads_like_a_transcript(capsys, tmp_path):
    saved = tmp_path / "runs" / "20260825_report.md"
    drive_full_run(PlainUI(), saved)
    out = capsys.readouterr().out

    assert "[1/5] Planning agents..." in out
    assert "- research_agent: gather facts" in out
    assert "writing research_agent..." in out
    assert "wrote research_agent.py" in out
    assert "reused summary_agent.py (from library, free)" in out
    assert "refused (not on the allowlist): leftpad" in out
    assert "installed: requests" in out
    assert "repairing 1/2" in out
    assert "done in 6.3s" in out
    assert "! summary_agent failed: timed out after 300s" in out
    assert "ANSWER" in out
    assert "The answer." in out
    assert "summary_agent: timed out after 300s" in out
    assert "12.3s · 4 LLM calls" in out
    assert "Saved to runs/20260825_report.md" in out


def test_plain_reports_a_lost_archive(capsys):
    PlainUI().run_succeeded(FakeResult(), None)
    assert "(could not write the run archive)" in capsys.readouterr().out


def test_plain_failure_and_cancellation(capsys):
    surface = PlainUI()
    surface.run_failed(RuntimeError("every agent failed"))
    surface.run_cancelled()
    out = capsys.readouterr().out
    assert "Task failed: RuntimeError: every agent failed" in out
    assert "Cancelled" in out


def test_plain_banner_covers_the_empty_state(capsys):
    surface = PlainUI()
    surface.banner("claude-sonnet-5", 4, 0, 0)
    surface.banner("claude-sonnet-5", 4, 7, 12)
    out = capsys.readouterr().out
    assert "library empty" in out
    assert "runs none yet" in out
    assert "library 7 kept" in out


# --- the rich renderer: same events, no terminal, must not crash --------------


@pytest.fixture
def rich_surface():
    pytest.importorskip("rich")
    import io

    from rich.console import Console

    from richui import THEME, RichUI

    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=100, theme=THEME)
    return RichUI(console), buffer


def test_rich_full_run_renders_answer_and_transcript(rich_surface, tmp_path):
    surface, buffer = rich_surface
    drive_full_run(surface, tmp_path / "runs" / "20260825_report.md")
    out = buffer.getvalue()
    assert "A N S W E R" in out
    assert "The answer." in out
    assert "research_agent" in out
    assert "20260825_report.md" in out


def test_rich_failure_path_renders_a_panel(rich_surface):
    surface, buffer = rich_surface
    surface.run_started("doomed task")
    surface.phase_started(1, 5, "Planning agents")
    surface.run_failed(RuntimeError("every agent failed:\n  - a: boom"))
    surface.run_ended()
    out = buffer.getvalue()
    assert "RuntimeError" in out
    assert "every agent failed" in out


def test_rich_cancellation_stops_the_live_display(rich_surface):
    surface, buffer = rich_surface
    surface.run_started("interrupted task")
    surface.run_cancelled()
    surface.run_ended()
    surface.run_ended()  # idempotent
    assert "interrupted" in buffer.getvalue()


def test_rich_events_degrade_when_no_run_was_started(rich_surface, capsys):
    surface, _ = rich_surface
    surface.phase_started(1, 5, "Planning agents")  # falls back to the plain line
    assert "[1/5] Planning agents..." in capsys.readouterr().out


def test_rich_banner_and_help_render(rich_surface, monkeypatch):
    surface, buffer = rich_surface
    monkeypatch.setattr(surface, "_beat", lambda: None)
    surface.banner("claude-sonnet-5", 4, 0, 0)
    surface.help()
    out = buffer.getvalue()
    assert "claude-sonnet-5" in out
    assert "--task" in out


def test_rich_activity_spinner_is_cached_so_it_animates():
    """A fresh Spinner every refresh would reset its clock: frame zero forever."""
    pytest.importorskip("rich")
    from rich.padding import Padding

    from richui import GLYPHS, _Board

    board = _Board(GLYPHS)
    first = board._activity()
    second = board._activity()
    assert isinstance(first, Padding) and isinstance(second, Padding)
    assert first.renderable is second.renderable
