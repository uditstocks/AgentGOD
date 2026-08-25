"""handle_task's sequencing and event stream, with every collaborator faked.

The orchestrator owns retry policy and the order of the pipeline; these
tests pin both without a network, an API key, or a real subprocess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import orchestrator
from events import TaskEvents
from executor import AgentResult, DependencyReport
from planner import AgentSpec, Plan

SOURCE = "def run(task, previous_outputs):\n    return 'ok'\n"


class Recorder(TaskEvents):
    """Writes every event to one flat log, so order can be asserted."""

    def __init__(self) -> None:
        self.log: list[tuple] = []

    def phase_started(self, index, total, title):
        self.log.append(("phase", index))

    def plan_ready(self, plan):
        self.log.append(("plan", [spec.name for spec in plan.agents]))

    def agent_build_started(self, name):
        self.log.append(("build", name))

    def agent_ready(self, name, filename, reused):
        self.log.append(("ready", name, reused))

    def deps_checked(self, report):
        self.log.append(("deps",))

    def agent_started(self, name, index, total):
        self.log.append(("start", name))

    def agent_repairing(self, name, attempt, attempts, error):
        self.log.append(("repair", name, attempt))

    def agent_unrepairable(self, name, reason):
        self.log.append(("unrepairable", name))

    def agent_finished(self, result):
        self.log.append(("finish", result.name, result.ok))

    def merge_started(self, survivors):
        self.log.append(("merge", survivors))

    def kinds(self) -> list[str]:
        return [entry[0] for entry in self.log]


def two_agent_plan() -> Plan:
    return Plan(
        agents=[
            AgentSpec(name="research_agent", role="gather", instructions="gather facts"),
            AgentSpec(name="summary_agent", role="condense", instructions="condense them"),
        ],
        reasoning="research feeds summary",
    )


def ok_result(path: Path) -> AgentResult:
    return AgentResult(
        name=path.stem, path=path, ok=True, output=f"output of {path.stem}",
        duration_seconds=0.1, input_tokens=10, output_tokens=5,
    )


def failed_result(path: Path) -> AgentResult:
    return AgentResult(name=path.stem, path=path, ok=False, error="boom")


@pytest.fixture
def collaborators(monkeypatch, tmp_path):
    """Fake every module the orchestrator sequences. Individual tests override."""

    def fake_save(spec, code):
        path = tmp_path / f"{spec.name}.py"
        path.write_text(code, encoding="utf-8")
        return path

    monkeypatch.setattr(orchestrator, "plan_agents", lambda task, usage=None: two_agent_plan())
    monkeypatch.setattr(orchestrator, "lookup", lambda name: None)
    monkeypatch.setattr(orchestrator, "record_use", lambda name: None)
    monkeypatch.setattr(orchestrator, "remember", lambda name, role, source: True)
    monkeypatch.setattr(
        orchestrator,
        "generate_agent_code",
        lambda spec, upstream=None, feedback=None, usage=None: SOURCE,
    )
    monkeypatch.setattr(orchestrator, "save_agent_file", fake_save)
    monkeypatch.setattr(
        orchestrator, "install_dependencies", lambda specs: DependencyReport()
    )
    monkeypatch.setattr(
        orchestrator,
        "execute_agent",
        lambda path, task, outputs, python_exe=None: ok_result(path),
    )
    monkeypatch.setattr(
        orchestrator, "merge_outputs", lambda task, outputs, usage=None: "the answer"
    )
    return monkeypatch


def test_happy_path_emits_the_full_event_sequence(collaborators):
    recorder = Recorder()
    result = orchestrator.handle_task("do a thing", events=recorder)

    assert result.response == "the answer"
    assert result.built == ["research_agent", "summary_agent"]
    assert result.failures == {}
    assert set(result.pending) == {"research_agent", "summary_agent"}
    assert recorder.kinds() == [
        "phase", "plan",
        "phase", "build", "ready", "build", "ready",
        "phase", "deps",
        "phase", "start", "finish", "start", "finish",
        "phase", "merge",
    ]
    assert ("merge", 2) in recorder.log


def test_runs_silently_without_an_events_object(collaborators, capsys):
    orchestrator.handle_task("do a thing")
    assert capsys.readouterr().out == ""


def test_library_hit_skips_generation(collaborators, monkeypatch):
    monkeypatch.setattr(
        orchestrator, "lookup", lambda name: SOURCE if name == "research_agent" else None
    )
    recorder = Recorder()
    result = orchestrator.handle_task("do a thing", events=recorder)

    assert result.reused == ["research_agent"]
    assert result.built == ["summary_agent"]
    assert "research_agent" not in result.pending
    assert ("build", "research_agent") not in recorder.log
    assert ("ready", "research_agent", True) in recorder.log
    assert ("ready", "summary_agent", False) in recorder.log


def test_failed_agent_is_repaired_and_events_say_so(collaborators, monkeypatch):
    attempts: dict[str, int] = {}

    def flaky_execute(path, task, outputs, python_exe=None):
        attempts[path.stem] = attempts.get(path.stem, 0) + 1
        if path.stem == "research_agent" and attempts[path.stem] == 1:
            return failed_result(path)
        return ok_result(path)

    monkeypatch.setattr(orchestrator, "execute_agent", flaky_execute)
    recorder = Recorder()
    result = orchestrator.handle_task("do a thing", events=recorder)

    assert result.failures == {}
    assert ("repair", "research_agent", 1) in recorder.log
    assert ("finish", "research_agent", True) in recorder.log
    # The repaired source replaces the broken one awaiting the keep decision.
    assert result.pending["research_agent"] == ("gather", SOURCE)


def test_every_agent_failing_raises_before_merge(collaborators, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "execute_agent",
        lambda path, task, outputs, python_exe=None: failed_result(path),
    )
    monkeypatch.setattr(
        orchestrator,
        "generate_agent_code",
        lambda spec, upstream=None, feedback=None, usage=None: SOURCE,
    )
    recorder = Recorder()
    with pytest.raises(RuntimeError, match="every agent failed"):
        orchestrator.handle_task("do a thing", events=recorder)
    assert "merge" not in recorder.kinds()


def test_on_agent_created_sees_every_written_file(collaborators):
    seen: list[Path] = []
    orchestrator.handle_task("do a thing", on_agent_created=seen.append)
    assert [path.stem for path in seen] == ["research_agent", "summary_agent"]
