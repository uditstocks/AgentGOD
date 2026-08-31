"""Executor tests. No API key needed: agents here are fake scripts."""

from __future__ import annotations

import sys

import pytest

import executor
from config import estimate_cost
from executor import (
    AgentResult,
    execute_agent,
    execute_all,
    install_dependencies,
    requirement_name,
    save_agent_file,
)
from planner import AgentSpec

PYTHON = sys.executable


def write_agent(folder, name: str, body: str):
    path = folder / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return path


def run(path, task="t", previous=None) -> AgentResult:
    return execute_agent(path, task, previous or {}, python_exe=PYTHON)


# --- C1/C2: encoding must survive both directions ------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Result - with an em dash",  # in cp1252, not ASCII
        "smart “quotes” and €",  # in cp1252
        "check ✓ and ☃",  # NOT in cp1252
        "日本語 مرحبا",  # CJK + Arabic
    ],
)
def test_non_ascii_output_round_trips(tmp_path, text):
    agent = write_agent(
        tmp_path,
        "unicode_agent",
        f"import json, sys\njson.loads(sys.stdin.read())\nprint({text!r})\n",
    )
    result = run(agent)
    assert result.ok, result.error
    assert result.output == text


def test_non_ascii_task_reaches_the_agent(tmp_path):
    agent = write_agent(
        tmp_path,
        "echo_agent",
        "import json, sys\npayload = json.loads(sys.stdin.read())\nprint(payload['task'])\n",
    )
    task = "Résumé du marché – 2026 € ✓"  # noqa: RUF001 - non-ASCII is the point
    result = run(agent, task=task)
    assert result.ok, result.error
    assert result.output == task


# --- H1: a hanging agent is soft-failed, never raised --------------------------


def test_hanging_agent_soft_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "AGENT_TIMEOUT_SECONDS", 1)
    agent = write_agent(tmp_path, "hang_agent", "import time\ntime.sleep(60)\n")
    result = run(agent)
    assert not result.ok
    assert "timed out" in result.error


def test_missing_agent_file_soft_fails(tmp_path):
    result = run(tmp_path / "does_not_exist.py")
    assert not result.ok
    assert result.error


def test_silent_agent_is_a_failure(tmp_path):
    agent = write_agent(tmp_path, "quiet_agent", "import json, sys\njson.loads(sys.stdin.read())\n")
    result = run(agent)
    assert not result.ok
    assert "no output" in result.error


# --- H2: failures never travel downstream as results ---------------------------


def test_failed_agent_output_is_not_forwarded(tmp_path):
    broken = write_agent(tmp_path, "broken_agent", "raise ValueError('boom')\n")
    downstream = write_agent(
        tmp_path,
        "downstream_agent",
        "import json, sys\n"
        "payload = json.loads(sys.stdin.read())\n"
        "print(json.dumps(payload['previous_outputs']))\n",
    )
    results = execute_all([broken, downstream], "t")

    assert results[0].ok is False
    assert results[1].ok is True
    assert results[1].output == "{}"  # the traceback was not passed on
    assert "boom" not in results[1].output


def test_error_text_hides_absolute_paths(tmp_path):
    agent = write_agent(tmp_path, "boom_agent", "raise ValueError('nope')\n")
    result = run(agent)
    assert not result.ok
    assert str(executor.PROJECT_DIR) not in result.error


# --- M8: agents report token usage on stderr, not stdout -----------------------


def test_usage_is_parsed_off_stderr(tmp_path):
    agent = write_agent(
        tmp_path,
        "usage_agent",
        "import json, sys\n"
        "json.loads(sys.stdin.read())\n"
        "print('__AGENT_USAGE__ ' + json.dumps("
        "{'input_tokens': 11, 'output_tokens': 5}), file=sys.stderr)\n"
        "print('the answer')\n",
    )
    result = run(agent)
    assert result.ok
    assert result.output == "the answer"  # marker line stayed out of stdout
    assert (result.input_tokens, result.output_tokens) == (11, 5)
    # The API bills tokens, not money, so the cost is priced locally.
    assert result.cost_usd == pytest.approx(estimate_cost(11, 5))


def test_usage_marker_is_stripped_from_error_text(tmp_path):
    agent = write_agent(
        tmp_path,
        "usage_fail_agent",
        "import json, sys\n"
        "json.loads(sys.stdin.read())\n"
        "print('__AGENT_USAGE__ {\"input_tokens\": 3}', file=sys.stderr)\n"
        "sys.exit(2)\n",
    )
    result = run(agent)
    assert not result.ok
    assert "__AGENT_USAGE__" not in result.error


# --- C4: generated files cannot escape the scratch directory -------------------


def test_save_agent_file_stays_inside_generated_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "GENERATED_DIR", tmp_path / "generated")
    spec = AgentSpec(name="research_agent", role="r", instructions="i")
    path = save_agent_file(spec, "print(1)\n")
    assert path.parent == (tmp_path / "generated").resolve()


def test_traversal_name_is_neutralised_before_it_reaches_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "GENERATED_DIR", tmp_path / "generated")
    spec = AgentSpec(name="../../../pwned", role="r", instructions="i")
    assert spec.name == "pwned"
    path = save_agent_file(spec, "print(1)\n")
    assert path.parent == (tmp_path / "generated").resolve()


# --- H4/C5: dependency handling -----------------------------------------------


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("requests", "requests"),
        ("requests>=2.31.0", "requests"),
        ("pandas[extra]", "pandas"),
        ("python_dateutil", "python-dateutil"),
        ("PyYAML == 6.0", "pyyaml"),
        ("", ""),
    ],
)
def test_requirement_name_strips_pins(requirement, expected):
    assert requirement_name(requirement) == expected


def test_version_pin_is_stripped_before_install(monkeypatch):
    """A pin like 'requests>=2.31' used to raise ModuleNotFoundError from the probe."""
    calls = []

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def record(command, *args, **kwargs):
        calls.append(command)
        return _Completed()

    monkeypatch.setattr(executor, "_ensure_venv", lambda: "python")
    monkeypatch.setattr(executor, "_installed_packages", lambda *a: set())
    monkeypatch.setattr(executor.subprocess, "run", record)

    spec = AgentSpec(name="pin_agent", role="r", instructions="i", dependencies=["requests>=2.31"])
    report = install_dependencies([spec])  # must not raise

    assert report.refused == []
    assert report.installed == ["requests"]
    assert calls and calls[0][-1] == "requests"  # the pin never reached pip


def test_unknown_package_is_refused_not_installed(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("pip must not be invoked for an unvetted package")

    monkeypatch.setattr(executor.subprocess, "run", explode)
    spec = AgentSpec(
        name="sketchy_agent", role="r", instructions="i", dependencies=["totally-not-real-pkg"]
    )
    report = install_dependencies([spec])
    assert report.refused == ["totally-not-real-pkg"]
    assert report.installed == []


def test_no_dependencies_touches_nothing(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("no subprocess should run when nothing is requested")

    monkeypatch.setattr(executor.subprocess, "run", explode)
    spec = AgentSpec(name="plain_agent", role="r", instructions="i")
    report = install_dependencies([spec])
    assert (report.installed, report.refused, report.failed) == ([], [], [])


# --- the effort dial reaches the generated agents ------------------------------


def test_the_runs_effort_grade_reaches_the_agent_environment(tmp_path):
    agent = write_agent(
        tmp_path,
        "effort_agent",
        "import os\nprint(os.environ.get('LLM_EFFORT', 'unset'))\n",
    )
    result = execute_agent(agent, "t", {}, python_exe=PYTHON, effort="high")
    assert result.ok
    assert result.output == "high"


def test_no_grade_leaves_the_environment_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_EFFORT", raising=False)
    agent = write_agent(
        tmp_path,
        "effort_agent",
        "import os\nprint(os.environ.get('LLM_EFFORT', 'unset'))\n",
    )
    result = execute_agent(agent, "t", {}, python_exe=PYTHON)
    assert result.ok
    assert result.output == "unset"
