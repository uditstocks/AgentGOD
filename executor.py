"""Steps 3-5: Save agent files, install missing dependencies, and execute the agents."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from config import GENERATED_DIR
from planner import AgentSpec

AGENT_TIMEOUT_SECONDS = 300


def save_agent_file(spec: AgentSpec, code: str) -> Path:
    """Write one generated agent to its own file in generated_agents/."""
    GENERATED_DIR.mkdir(exist_ok=True)
    path = GENERATED_DIR / f"{spec.name}.py"
    path.write_text(code, encoding="utf-8")
    return path


def install_dependencies(specs: list[AgentSpec]) -> None:
    """pip-install any packages the generated agents need but aren't installed."""
    needed = {dep for spec in specs for dep in spec.dependencies}
    missing = [dep for dep in needed if importlib.util.find_spec(dep.replace("-", "_")) is None]
    if missing:
        print(f"  Installing missing dependencies: {', '.join(missing)}")
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)


def execute_agent(path: Path, task: str, previous_outputs: dict[str, str]) -> str:
    """Run one agent file as a subprocess.

    Agents communicate through a well-defined interface:
    JSON {"task", "previous_outputs"} on stdin -> result text on stdout.
    """
    payload = json.dumps({"task": task, "previous_outputs": previous_outputs})
    result = subprocess.run(
        [sys.executable, str(path)],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=AGENT_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        return f"[{path.stem} failed]\n{result.stderr.strip()}"
    return result.stdout.strip()


def execute_all(agent_paths: list[Path], task: str) -> dict[str, str]:
    """Run agents in order; each one sees the outputs of the agents before it."""
    outputs: dict[str, str] = {}
    for path in agent_paths:
        print(f"  Running {path.stem}...")
        outputs[path.stem] = execute_agent(path, task, outputs)
    return outputs
