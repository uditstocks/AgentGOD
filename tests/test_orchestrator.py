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
from judgment import Verdict
from planner import AgentSpec, Plan, canonical_role

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

    def agent_retired(self, name, reason):
        self.log.append(("retired", name))

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

    def answer_judged(self, done, missing):
        self.log.append(("judged", done))

    def revision_started(self, attempt, attempts, missing):
        self.log.append(("revision", attempt))

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
    monkeypatch.setattr(orchestrator, "remember", lambda *a, **k: True)
    # A library agent is only handed back if it still checks out; these
    # tests are about sequencing, so the check always passes here.
    monkeypatch.setattr(orchestrator, "reusable", lambda name: True)
    # Likewise: a stale-runtime retirement has its own test below.
    monkeypatch.setattr(orchestrator, "up_to_date", lambda name: True)
    monkeypatch.setattr(orchestrator, "forget", lambda name: True)
    monkeypatch.setattr(
        orchestrator,
        "generate_agent_code",
        lambda spec, upstream=None, feedback=None, usage=None, task="": SOURCE,
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
    # The answer passes its own check unless a test says otherwise; the
    # revision path has its own tests below.
    monkeypatch.setattr(
        orchestrator, "judge", lambda task, answer, usage=None: Verdict(missing="", done=True)
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
        "phase", "judged",
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
    # The repaired source replaces the broken one awaiting the keep decision,
    # under the standard description of the capability rather than the
    # planner's wording for this one task.
    role, source = result.pending["research_agent"]
    assert source == SOURCE
    assert role == canonical_role("research_agent")


def test_every_agent_failing_raises_before_merge(collaborators, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "execute_agent",
        lambda path, task, outputs, python_exe=None: failed_result(path),
    )
    monkeypatch.setattr(
        orchestrator,
        "generate_agent_code",
        lambda spec, upstream=None, feedback=None, usage=None, task="": SOURCE,
    )
    recorder = Recorder()
    with pytest.raises(RuntimeError, match="every agent failed"):
        orchestrator.handle_task("do a thing", events=recorder)
    assert "merge" not in recorder.kinds()


def test_on_agent_created_sees_every_written_file(collaborators):
    seen: list[Path] = []
    orchestrator.handle_task("do a thing", on_agent_created=seen.append)
    assert [path.stem for path in seen] == ["research_agent", "summary_agent"]


def test_an_agent_that_failed_is_never_offered_for_the_library(collaborators, monkeypatch):
    """A file that is known not to run must not become a free library hit."""

    def one_fails(path, task, outputs, python_exe=None):
        return failed_result(path) if path.stem == "summary_agent" else ok_result(path)

    monkeypatch.setattr(orchestrator, "execute_agent", one_fails)
    # Repair is not the point here; let it stay broken.
    monkeypatch.setattr(
        orchestrator,
        "generate_agent_code",
        lambda spec, upstream=None, feedback=None, usage=None, task="": SOURCE,
    )

    result = orchestrator.handle_task("do a thing", events=Recorder())

    assert "summary_agent" in result.failures
    assert "summary_agent" not in result.pending
    assert "research_agent" in result.pending


def test_the_task_is_carried_on_the_result(collaborators):
    """The caller records it against each kept agent, for later re-checking."""
    result = orchestrator.handle_task("write me something", events=Recorder())
    assert result.task == "write me something"


def test_a_library_agent_that_hardcoded_its_task_is_retired_and_rebuilt(
    collaborators, monkeypatch
):
    monkeypatch.setattr(orchestrator, "lookup", lambda name: SOURCE)
    monkeypatch.setattr(orchestrator, "reusable", lambda name: name != "summary_agent")
    dropped: list[str] = []
    monkeypatch.setattr(orchestrator, "forget", lambda name: dropped.append(name) or True)

    recorder = Recorder()
    result = orchestrator.handle_task("do a thing", events=recorder)

    assert dropped == ["summary_agent"]
    assert "summary_agent" in result.built
    assert "summary_agent" not in result.reused
    assert "research_agent" in result.reused


# --- the answer is checked, and rebuilt when it does not hold ------------------


def test_a_short_answer_is_rejected_and_the_agents_run_again(collaborators):
    """The point of the check: a run no longer ends wherever the merger stopped."""
    verdicts = iter(
        [Verdict(missing="the 200-word limit was ignored", done=False),
         Verdict(missing="", done=True)]
    )
    collaborators.setattr(
        orchestrator, "judge", lambda task, answer, usage=None: next(verdicts)
    )
    merges = iter(["first draft", "the revised answer"])
    collaborators.setattr(
        orchestrator, "merge_outputs", lambda task, outputs, usage=None: next(merges)
    )

    recorder = Recorder()
    result = orchestrator.handle_task("write 200 words", events=recorder)

    assert result.response == "the revised answer"
    assert result.revisions == 1
    assert recorder.kinds().count("merge") == 2
    assert ("revision", 1) in recorder.log


def test_the_revision_reruns_the_agents_on_the_gap_not_on_the_critique(collaborators):
    """Replacing the task with the complaint is how a second attempt drifts."""
    seen: list[str] = []

    def spy(path, task, outputs, python_exe=None):
        seen.append(task)
        return ok_result(path)

    collaborators.setattr(orchestrator, "execute_agent", spy)
    verdicts = iter(
        [Verdict(missing="no sources were cited", done=False), Verdict(missing="", done=True)]
    )
    collaborators.setattr(
        orchestrator, "judge", lambda task, answer, usage=None: next(verdicts)
    )

    orchestrator.handle_task("write about logging", events=Recorder())

    revised = seen[-1]
    assert "write about logging" in revised  # the original request survives
    assert "no sources were cited" in revised  # and the gap is added to it


def test_a_revision_that_produces_nothing_keeps_the_first_answer(collaborators):
    """A worse second attempt must not overwrite a usable first one."""
    collaborators.setattr(
        orchestrator,
        "judge",
        lambda task, answer, usage=None: Verdict(missing="not enough", done=False),
    )
    calls = {"n": 0}

    def sometimes(path, task, outputs, python_exe=None):
        calls["n"] += 1
        # first round succeeds, every rerun fails
        return ok_result(path) if calls["n"] <= 2 else failed_result(path)

    collaborators.setattr(orchestrator, "execute_agent", sometimes)

    result = orchestrator.handle_task("do a thing", events=Recorder())

    assert result.response == "the answer"
    assert result.revisions == 0


def test_the_check_can_be_turned_off_entirely(collaborators):
    """TASK_REVISIONS=0 means the judge is never called, so it is never billed."""
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("judge must not be called when revisions are off")

    collaborators.setattr(orchestrator, "judge", explode)
    collaborators.setattr(orchestrator, "TASK_REVISIONS", 0)

    recorder = Recorder()
    result = orchestrator.handle_task("do a thing", events=recorder)

    assert result.response == "the answer"
    assert "judged" not in recorder.kinds()


def test_a_library_agent_from_an_older_runtime_is_retired_and_rebuilt(collaborators):
    """It would run perfectly and silently do less than the plan just promised."""
    collaborators.setattr(orchestrator, "lookup", lambda name: SOURCE)
    collaborators.setattr(
        orchestrator, "up_to_date", lambda name: name != "research_agent"
    )
    dropped: list[str] = []
    collaborators.setattr(orchestrator, "forget", lambda name: dropped.append(name) or True)

    recorder = Recorder()
    result = orchestrator.handle_task("do a thing", events=recorder)

    assert dropped == ["research_agent"]
    assert result.built == ["research_agent"]      # rewritten on the new runtime
    assert result.reused == ["summary_agent"]      # still free
    assert ("retired", "research_agent") in recorder.log
