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

from config import AGENT_REPAIR_ATTEMPTS, TASK_REVISIONS, Usage
from events import TaskEvents
from executor import (
    AgentResult,
    DependencyReport,
    execute_agent,
    install_dependencies,
    save_agent_file,
)
from generator import generate_agent_code
from judgment import judge, revision_task
from library import forget, lookup, record_use, remember, reusable, up_to_date
from merger import merge_outputs
from planner import AgentSpec, Plan, canonical_role, plan_agents, upstream_names

PHASES = (
    "Planning agents",
    "Generating agent code",
    "Checking dependencies",
    "Executing agents",
    "Merging outputs",
    "Checking the answer",
)


@dataclass
class TaskResult:
    """Everything one task produced, for the caller to present and clean up."""

    response: str
    plan: Plan | None = None
    # The task exactly as the pipeline received it. Carried so the caller can
    # record what each kept agent was built for, which is what makes a later
    # reusability check possible.
    task: str = ""
    agent_paths: list[Path] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    agent_calls: int = 0
    agent_input_tokens: int = 0
    agent_output_tokens: int = 0
    agent_cost_usd: float = 0.0
    reused: list[str] = field(default_factory=list)
    built: list[str] = field(default_factory=list)
    # How many times the answer was rejected by the main agent and rebuilt.
    revisions: int = 0
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
    subject: str,
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
            code = generate_agent_code(
                spec, upstream, feedback=result.error, usage=usage, task=subject
            )
        except (ValueError, RuntimeError) as error:
            events.agent_unrepairable(spec.name, str(error))
            break
        path = save_agent_file(spec, code)
        result = execute_agent(path, task, outputs)
        if result.ok:
            repaired = code
    return result, repaired


def _rerun_agents(
    plan: Plan,
    agent_paths: list[Path],
    task: str,
    events: TaskEvents,
) -> tuple[dict[str, str], list[AgentResult]]:
    """Run every agent again on a revised task, in the same order as before.

    No repair pass here, unlike the first round: these agents have already run
    once, so a crash now is not the failure mode a revision exists to fix, and
    regenerating them would throw away the working code that produced the
    first answer.
    """
    outputs: dict[str, str] = {}
    results: list[AgentResult] = []
    for index, (spec, path) in enumerate(zip(plan.agents, agent_paths, strict=True)):
        events.agent_started(spec.name, index + 1, len(plan.agents))
        result = execute_agent(path, task, outputs)
        results.append(result)
        if result.ok:
            outputs[result.name] = result.output
        events.agent_finished(result)
    return outputs, results


def _finish_answer(
    task: str,
    response: str,
    plan: Plan,
    agent_paths: list[Path],
    usage: Usage,
    events: TaskEvents,
) -> tuple[str, list[AgentResult], int]:
    """Read the answer back against the request, and try again if it falls short.

    This is the only place the system judges its own output. Without it a run
    ends wherever the merger happened to stop - a 200-word brief answered in
    600 words was simply delivered, because nothing ever compared the two.

    Bounded by TASK_REVISIONS because each attempt re-runs every agent and is
    billed like the first one. A revision that produces nothing usable leaves
    the original answer standing rather than replacing it with worse.
    """
    extra_results: list[AgentResult] = []
    revisions = 0
    for attempt in range(1, TASK_REVISIONS + 1):
        verdict = judge(task, response, usage=usage)
        events.answer_judged(verdict.done, verdict.missing)
        if verdict.done:
            break

        events.revision_started(attempt, TASK_REVISIONS, verdict.missing)
        outputs, results = _rerun_agents(
            plan, agent_paths, revision_task(task, verdict.missing), events
        )
        extra_results.extend(results)
        if not outputs:
            break
        events.merge_started(len(outputs))
        response = merge_outputs(task, outputs, usage=usage)
        revisions = attempt

    return response, extra_results, revisions


def handle_task(
    task: str,
    on_agent_created: Callable[[Path], None] | None = None,
    events: TaskEvents | None = None,
    subject: str | None = None,
) -> TaskResult:
    """Run the full lifecycle for one user task.

    `on_agent_created` is called as each agent file is written, so the caller
    can still clean up the files if a later phase raises. `events` receives
    every notable moment of the run; the default TaskEvents shows nothing,
    so this function stays silent unless the caller wants otherwise.

    `subject` is what the reusability guard measures a generated agent
    against, and defaults to the task. They differ when the caller has folded
    material into the task - the contents of a file, or an earlier exchange.
    That material is not the user's request, and treating a 16 KB README as
    the subject makes every word in it forbidden, which rejects every agent
    that can be written.
    """
    started = time.perf_counter()
    usage = Usage()
    events = events or TaskEvents()
    subject = task if subject is None else subject

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
        # call in the run. Always look before building - but never hand back
        # one that turned out to have hardcoded the task it was built for,
        # because that agent is wrong for every task except that one.
        code = lookup(spec.name)
        if code is not None and not up_to_date(spec.name):
            # Cheapest check first, and the one nothing else can catch: an
            # agent written against an older runtime runs perfectly and
            # silently does less than the plan just promised.
            events.agent_retired(spec.name, "it was built against an older agent runtime")
            forget(spec.name)
            code = None
        if code is not None and not reusable(spec.name):
            events.agent_retired(spec.name, "it hardcoded the task it was built for")
            forget(spec.name)
            code = None

        if code is not None:
            reused.append(spec.name)
            record_use(spec.name)
        else:
            events.agent_build_started(spec.name)
            code = generate_agent_code(
                spec, upstream_names(plan.agents, index), usage=usage, task=subject
            )
            built.append(spec.name)
            pending[spec.name] = (canonical_role(spec.name, spec.role), code)

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
            spec, path, task, outputs, upstream_names(plan.agents, index), usage,
            events, subject,
        )
        if repaired is not None:
            role = canonical_role(spec.name, spec.role)
            if spec.name in reused:
                # Fixing an agent the user already chose to keep.
                remember(spec.name, role, repaired, task=subject)
            else:
                pending[spec.name] = (role, repaired)
        results.append(result)
        if result.ok:
            outputs[result.name] = result.output
        else:
            failures[result.name] = result.error
        events.agent_finished(result)

    # An agent that could not finish is not a capability worth remembering.
    # Offering it would put a file in the library that is known not to run,
    # and the library's whole value is that a hit is free and safe.
    for name in failures:
        pending.pop(name, None)

    events.phase_started(5, len(PHASES), PHASES[4])
    if not outputs:
        raise RuntimeError(
            "every agent failed:\n"
            + "\n".join(f"  - {name}: {error}" for name, error in failures.items())
        )
    events.merge_started(len(outputs))
    response = merge_outputs(task, outputs, usage=usage)

    events.phase_started(6, len(PHASES), PHASES[5])
    response, extra_results, revisions = _finish_answer(
        task, response, plan, agent_paths, usage, events
    )
    results.extend(extra_results)

    return TaskResult(
        response=response,
        plan=plan,
        task=subject,
        revisions=revisions,
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
