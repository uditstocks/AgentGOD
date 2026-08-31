"""Steps 3-5: Save agent files, install missing dependencies, and execute the agents.

No AI here - only files, processes and strings - so everything in this module
is unit-testable without an API key.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# The vetted pip packages, defined once in codeguard next to the import check
# that lets them through. Installing a package the generated code may not
# import is a wasted install; refusing an import for a package that installed
# fine is a wasted run. One list makes both impossible.
from codeguard import ALLOWED_PACKAGES
from config import (
    AGENT_TIMEOUT_SECONDS,
    AGENT_VENV_DIR,
    GENERATED_DIR,
    PROJECT_DIR,
    USAGE_MARKER,
    estimate_cost,
)
from planner import AgentSpec

# Splits "requests>=2.31.0" / "pandas[extra]" down to "requests" / "pandas".
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9._-]+")

_PROBE_INSTALLED = (
    "import json, sys\n"
    "from importlib.metadata import PackageNotFoundError, distribution\n"
    "found = []\n"
    "for name in json.loads(sys.argv[1]):\n"
    "    try:\n"
    "        distribution(name)\n"
    "        found.append(name)\n"
    "    except PackageNotFoundError:\n"
    "        pass\n"
    "print(json.dumps(found))\n"
)


@dataclass(frozen=True)
class AgentResult:
    """Outcome of running one generated agent."""

    name: str
    path: Path
    ok: bool
    output: str = ""
    error: str = ""
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class DependencyReport:
    """What happened when the planner asked for extra pip packages."""

    already_present: list[str] = field(default_factory=list)
    installed: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def problems(self) -> list[str]:
        notes = []
        if self.refused:
            notes.append(f"refused (not on the allowlist): {', '.join(self.refused)}")
        if self.failed:
            notes.append(f"failed to install: {', '.join(self.failed)}")
        return notes


def requirement_name(dependency: str) -> str:
    """Canonical pip name for a requirement string, ignoring version pins."""
    match = _REQUIREMENT_NAME.match(dependency.strip())
    return match.group(0).lower().replace("_", "-") if match else ""


def save_agent_file(spec: AgentSpec, code: str) -> Path:
    """Write one generated agent to its own file in generated_agents/.

    The resolved path is checked for containment: agent names come from an
    LLM, and a name like "../../evil" must never escape the scratch directory.
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    root = GENERATED_DIR.resolve()
    path = (root / f"{spec.name}.py").resolve()
    if path.parent != root:
        raise ValueError(f"agent path escapes {GENERATED_DIR.name}/: {path}")
    path.write_text(code, encoding="utf-8")
    return path


def _venv_python() -> Path:
    """Interpreter inside the isolated agent venv (it may not exist yet)."""
    if os.name == "nt":
        return AGENT_VENV_DIR / "Scripts" / "python.exe"
    return AGENT_VENV_DIR / "bin" / "python"


def agent_python() -> str:
    """Interpreter used to run generated agents.

    The isolated venv when one exists (because a task needed extra packages),
    otherwise this interpreter - generated agents are standard library only,
    so the common case needs no venv at all.
    """
    venv_python = _venv_python()
    return str(venv_python) if venv_python.is_file() else sys.executable


def _ensure_venv() -> str:
    """Create the isolated agent venv on first use and return its interpreter."""
    venv_python = _venv_python()
    if not venv_python.is_file():
        subprocess.run(
            [sys.executable, "-m", "venv", str(AGENT_VENV_DIR)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    return str(venv_python)


def _installed_packages(python_exe: str, names: list[str]) -> set[str]:
    """Ask the target interpreter which distributions it already has.

    Queries installed *distributions*, not importable module names: pip names
    and import names differ often enough (beautifulsoup4 -> bs4) that probing
    imports reinstalls the same packages on every run.
    """
    if not names:
        return set()
    try:
        completed = subprocess.run(
            [python_exe, "-c", _PROBE_INSTALLED, json.dumps(names)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        return set(json.loads(completed.stdout or "[]"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return set()


def install_dependencies(specs: list[AgentSpec]) -> DependencyReport:
    """Install the vetted pip packages the generated agents asked for.

    Anything outside ALLOWED_PACKAGES is refused, and everything else goes
    into an isolated venv rather than the interpreter running this program.
    """
    requested: list[str] = []
    for spec in specs:
        for dependency in spec.dependencies:
            name = requirement_name(dependency)
            if name and name not in requested:
                requested.append(name)

    report = DependencyReport()
    if not requested:
        return report

    allowed = [name for name in requested if name in ALLOWED_PACKAGES]
    report.refused = [name for name in requested if name not in ALLOWED_PACKAGES]
    if not allowed:
        return report

    try:
        python_exe = _ensure_venv()
    except (OSError, subprocess.SubprocessError) as error:
        report.failed = allowed
        report.refused.append(f"venv unavailable ({error})")
        return report

    present = _installed_packages(python_exe, allowed)
    report.already_present = [name for name in allowed if name in present]
    missing = [name for name in allowed if name not in present]
    if not missing:
        return report

    completed = subprocess.run(
        [python_exe, "-m", "pip", "install", "--disable-pip-version-check", *missing],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode == 0:
        report.installed = missing
    else:
        report.failed = missing
    return report


def _child_env() -> dict[str, str]:
    """Environment for an agent subprocess.

    UTF-8 is forced both ways: without it a piped child on Windows writes
    cp1252, the parent decodes UTF-8, and the agent's output is lost.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _sanitise(text: str) -> str:
    """Strip absolute project paths out of anything shown to the user."""
    project = str(PROJECT_DIR)
    return text.replace(project, "<project>").replace(project.replace("\\", "/"), "<project>")


def _extract_usage(stderr: str) -> tuple[dict[str, float], str]:
    """Pull the agent's token report off stderr, returning the remaining text.

    The API bills tokens and says nothing about money, so the cost is priced
    here from the same table the main agent's own calls are priced from.
    """
    totals = {"input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0}
    remaining: list[str] = []
    for line in stderr.splitlines():
        if not line.startswith(USAGE_MARKER):
            remaining.append(line)
            continue
        try:
            usage = json.loads(line[len(USAGE_MARKER) :].strip())
        except ValueError:
            continue
        totals["input_tokens"] += float(usage.get("input_tokens") or 0)
        totals["output_tokens"] += float(usage.get("output_tokens") or 0)
    totals["cost_usd"] = estimate_cost(totals["input_tokens"], totals["output_tokens"]) or 0.0
    return totals, "\n".join(remaining).strip()


def execute_agent(
    path: Path,
    task: str,
    previous_outputs: dict[str, str],
    python_exe: str | None = None,
) -> AgentResult:
    """Run one agent file as a subprocess.

    Agents communicate through a well-defined interface:
    JSON {"task", "previous_outputs"} on stdin -> result text on stdout.
    Every failure mode - crash, hang, unstartable, silent - becomes an
    AgentResult rather than an exception, so one bad agent cannot take the
    run down with it.
    """
    # ensure_ascii keeps the payload pure ASCII, so stdin cannot be mangled
    # by whatever encoding the child happens to pick.
    payload = json.dumps({"task": task, "previous_outputs": previous_outputs}, ensure_ascii=True)
    started = time.perf_counter()

    def failure(reason: str) -> AgentResult:
        return AgentResult(
            name=path.stem,
            path=path,
            ok=False,
            error=_sanitise(reason),
            duration_seconds=time.perf_counter() - started,
        )

    try:
        completed = subprocess.run(
            [python_exe or agent_python(), str(path)],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=AGENT_TIMEOUT_SECONDS,
            env=_child_env(),
        )
    except subprocess.TimeoutExpired:
        return failure(f"timed out after {AGENT_TIMEOUT_SECONDS}s")
    except OSError as error:
        return failure(f"could not start agent: {error}")

    usage, stderr = _extract_usage(completed.stderr or "")
    stdout = (completed.stdout or "").strip()

    if completed.returncode != 0:
        return failure(stderr or f"exited with code {completed.returncode}")
    if not stdout:
        return failure(stderr or "agent produced no output")

    return AgentResult(
        name=path.stem,
        path=path,
        ok=True,
        output=stdout,
        duration_seconds=time.perf_counter() - started,
        input_tokens=int(usage["input_tokens"]),
        output_tokens=int(usage["output_tokens"]),
        cost_usd=usage["cost_usd"],
    )


def execute_all(agent_paths: list[Path], task: str) -> list[AgentResult]:
    """Run agents in order; each one sees the outputs of the agents before it.

    Only successful outputs are forwarded. A failed agent's stderr is never
    passed downstream as if it were a result.
    """
    results: list[AgentResult] = []
    outputs: dict[str, str] = {}
    for path in agent_paths:
        result = execute_agent(path, task, outputs)
        if result.ok:
            outputs[result.name] = result.output
        results.append(result)
    return results
