"""The one permanent Main Agent.

It never solves the user's task itself. It engineers other agents:
analyze -> plan -> generate code -> save files -> install deps -> execute -> merge.

It sequences the modules and owns retry policy; the work itself lives in
planner / generator / executor / merger.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from config import AGENT_REPAIR_ATTEMPTS, Usage
from events import TaskEvents
from executor import (
    AgentResult,
    DependencyReport,
    execute_agent,
    install_dependencies,
    save_agent_file,
)
from generator import generate_agent_code
from library import lookup, record_use, remember
from merger import merge_outputs
from planner import AgentSpec, Plan, plan_agents, upstream_names

PHASES = (
    "Planning agents",
    "Generating agent code",
    "Checking dependencies",
    "Executing agents",
    "Merging outputs",
)


@dataclass
class TaskResult:
    """Everything one task produced, for the caller to present and clean up."""

    response: str
    plan: Plan | None = None
    agent_paths: list[Path] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    agent_calls: int = 0
    agent_input_tokens: int = 0
    agent_output_tokens: int = 0
    agent_cost_usd: float = 0.0
    reused: list[str] = field(default_factory=list)
    built: list[str] = field(default_factory=list)
    pending: dict[str, tuple[str, str]] = field(default_factory=dict)
    dependencies: DependencyReport = field(default_factory=DependencyReport)
    duration_seconds: float = 0.0

    def cost_summary(self) -> str:
        """Total spend across the main agent and every generated agent."""
        main_cost = self.usage.cost_usd
        total_cost = self.agent_cost_usd + (main_cost or 0.0)
        calls = self.usage.calls + self.agent_calls
        tokens_in = self.usage.input_tokens + self.agent_input_tokens
        tokens_out = self.usage.output_tokens + self.agent_output_tokens
        money = f" · ~${total_cost:.4f}" if total_cost else ""
        return f"{calls} LLM calls · {tokens_in:,} in / {tokens_out:,} out tokens{money}"


def _run_with_repair(
    spec: AgentSpec,
    path: Path,
    task: str,
    outputs: dict[str, str],
    upstream: list[str],
    usage: Usage,
    events: TaskEvents,
) -> tuple[AgentResult, str | None]:
    """Execute one agent, regenerating it from its own error output on failure.

    The generator already has the code and the traceback, so a crashed agent
    is a repairable event rather than lost work.
    """
    repaired: str | None = None
    result = execute_agent(path, task, outputs)
    for attempt in range(1, AGENT_REPAIR_ATTEMPTS + 1):
        if result.ok:
            break
        events.agent_repairing(spec.name, attempt, AGENT_REPAIR_ATTEMPTS, result.error)
        try:
            code = generate_agent_code(spec, upstream, feedback=result.error, usage=usage)
        except (ValueError, RuntimeError) as error:
            events.agent_unrepairable(spec.name, str(error))
            break
        path = save_agent_file(spec, code)
        result = execute_agent(path, task, outputs)
        if result.ok:
            repaired = code
    return result, repaired


def handle_task(
    task: str,
    on_agent_created: Callable[[Path], None] | None = None,
    events: TaskEvents | None = None,
) -> TaskResult:
    """Run the full lifecycle for one user task.

    `on_agent_created` is called as each agent file is written, so the caller
    can still clean up the files if a later phase raises. `events` receives
    every notable moment of the run; the default TaskEvents shows nothing,
    so this function stays silent unless the caller wants otherwise.
    """
    started = time.perf_counter()
    usage = Usage()
    events = events or TaskEvents()

    events.phase_started(1, len(PHASES), PHASES[0])
    plan: Plan = plan_agents(task, usage=usage)
    events.plan_ready(plan)

    events.phase_started(2, len(PHASES), PHASES[1])
    agent_paths: list[Path] = []
    reused: list[str] = []
    built: list[str] = []
    # name -> (role, source) for agents the user has not yet chosen to keep
    pending: dict[str, tuple[str, str]] = {}
    for index, spec in enumerate(plan.agents):
        # A remembered agent is free; generating one is the most expensive
        # call in the run. Always look before building.
        code = lookup(spec.name)
        if code is not None:
            reused.append(spec.name)
            record_use(spec.name)
        else:
            events.agent_build_started(spec.name)
            code = generate_agent_code(spec, upstream_names(plan.agents, index), usage=usage)
            built.append(spec.name)
            pending[spec.name] = (spec.role, code)

        path = save_agent_file(spec, code)
        agent_paths.append(path)
        if on_agent_created is not None:
            on_agent_created(path)
        events.agent_ready(spec.name, path.name, reused=spec.name in reused)

    events.phase_started(3, len(PHASES), PHASES[2])
    dependencies = install_dependencies(plan.agents)
    events.deps_checked(dependencies)

    events.phase_started(4, len(PHASES), PHASES[3])
    outputs: dict[str, str] = {}
    failures: dict[str, str] = {}
    results: list[AgentResult] = []
    for index, (spec, path) in enumerate(zip(plan.agents, agent_paths, strict=True)):
        events.agent_started(spec.name, index + 1, len(plan.agents))
        result, repaired = _run_with_repair(
            spec, path, task, outputs, upstream_names(plan.agents, index), usage, events
        )
        if repaired is not None:
            if spec.name in reused:
                # Fixing an agent the user already chose to keep.
                remember(spec.name, spec.role, repaired)
            else:
                pending[spec.name] = (spec.role, repaired)
        results.append(result)
        if result.ok:
            outputs[result.name] = result.output
        else:
            failures[result.name] = result.error
        events.agent_finished(result)

    events.phase_started(5, len(PHASES), PHASES[4])
    if not outputs:
        raise RuntimeError(
            "every agent failed:\n"
            + "\n".join(f"  - {name}: {error}" for name, error in failures.items())
        )
    events.merge_started(len(outputs))
    response = merge_outputs(task, outputs, usage=usage)

    return TaskResult(
        response=response,
        plan=plan,
        reused=reused,
        built=built,
        pending=pending,
        agent_paths=agent_paths,
        failures=failures,
        usage=usage,
        agent_calls=sum(1 for result in results if result.input_tokens or result.output_tokens),
        agent_input_tokens=sum(result.input_tokens for result in results),
        agent_output_tokens=sum(result.output_tokens for result in results),
        agent_cost_usd=sum(result.cost_usd for result in results),
        dependencies=dependencies,
        duration_seconds=time.perf_counter() - started,
    )
