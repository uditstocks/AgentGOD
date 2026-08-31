"""The one permanent Main Agent.

It never solves the user's task itself. It engineers other agents:
analyze -> plan -> generate code -> save files -> install deps -> execute -> merge.

It sequences the modules and owns retry policy; the work itself lives in
planner / generator / executor / merger. Two structural facts of a run are
decided here and nowhere else:

- **Parallelism only where the graph proves it.** The planner declares which
  agents feed which; taskgraph turns that into waves. Everything in one wave
  has every input it needs before the wave starts, so running a wave's agents
  at the same time is exactly as correct as running them one by one - and a
  plan whose agents genuinely form a chain still runs as a chain.
- **Effort follows the grade.** The planner sizes the task once (simple /
  standard / deep), and every LLM call after that - generation, merging,
  judging, and the generated agents' own calls - runs at the matching effort.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

from config import (
    AGENT_REPAIR_ATTEMPTS,
    MAX_PARALLEL_AGENTS,
    TASK_REVISIONS,
    Usage,
    effort_for,
)
from council import deliberate, should_convene
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
from library import (
    forget,
    lookup,
    record_outcome,
    record_use,
    reliable,
    remember,
    reusable,
    up_to_date,
)
from merger import merge_outputs
from planner import AgentSpec, Plan, canonical_role, plan_agents
from taskgraph import dependency_closure, waves

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
    # The planner's grade for this task, which set the effort of every call.
    complexity: str = "standard"
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
    # Whether the council found real faults and the answer was refined.
    council_improved: bool = False
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


class _SharedEvents:
    """A TaskEvents facade that serialises emission across worker threads.

    Independent agents run - and are repaired - in parallel, and two threads
    interleaving inside one renderer would corrupt what the user sees. Every
    hook is delegated unchanged; only one fires at a time.
    """

    def __init__(self, events: TaskEvents) -> None:
        self._events = events
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._events, name)
        if not callable(attribute):
            return attribute

        def emit(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attribute(*args, **kwargs)

        return emit


_Outcome = TypeVar("_Outcome")


def _for_each_in_waves(
    agents: list[AgentSpec],
    events: TaskEvents,
    run_one: Callable[[AgentSpec], _Outcome],
    absorb: Callable[[AgentSpec, _Outcome], None],
) -> None:
    """Drive every agent through `run_one`, one dependency wave at a time.

    Agents in the same wave share no data path, so a wave with several agents
    runs them on a thread pool; a wave of one runs inline. `absorb` is always
    called on the caller's thread, as each agent finishes, so shared state is
    mutated from exactly one thread and the interface stays live.
    """
    rounds = waves(agents)
    for wave_number, wave_specs in enumerate(rounds, start=1):
        events.wave_started(wave_number, len(rounds), [spec.name for spec in wave_specs])
        if len(wave_specs) == 1 or MAX_PARALLEL_AGENTS == 1:
            for spec in wave_specs:
                absorb(spec, run_one(spec))
            continue
        workers = min(len(wave_specs), MAX_PARALLEL_AGENTS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, spec): spec for spec in wave_specs}
            for future in as_completed(futures):
                absorb(futures[future], future.result())


def _run_with_repair(
    spec: AgentSpec,
    path: Path,
    task: str,
    outputs: dict[str, str],
    upstream: list[str],
    usage: Usage,
    events: TaskEvents,
    subject: str,
    effort: str | None,
) -> tuple[AgentResult, str | None]:
    """Execute one agent, regenerating it from its own error output on failure.

    The generator already has the code and the traceback, so a crashed agent
    is a repairable event rather than lost work.
    """
    repaired: str | None = None
    result = execute_agent(path, task, outputs, effort=effort)
    for attempt in range(1, AGENT_REPAIR_ATTEMPTS + 1):
        if result.ok:
            break
        events.agent_repairing(spec.name, attempt, AGENT_REPAIR_ATTEMPTS, result.error)
        try:
            code = generate_agent_code(
                spec, upstream, feedback=result.error, usage=usage, task=subject,
                effort=effort,
            )
        except (ValueError, RuntimeError) as error:
            events.agent_unrepairable(spec.name, str(error))
            break
        path = save_agent_file(spec, code)
        result = execute_agent(path, task, outputs, effort=effort)
        if result.ok:
            repaired = code
    return result, repaired


def _rerun_agents(
    plan: Plan,
    agent_paths: list[Path],
    task: str,
    events: TaskEvents,
    effort: str | None,
) -> tuple[dict[str, str], list[AgentResult]]:
    """Run every agent again on a revised task, in the same waves as before.

    No repair pass here, unlike the first round: these agents have already run
    once, so a crash now is not the failure mode a revision exists to fix, and
    regenerating them would throw away the working code that produced the
    first answer.
    """
    path_by_name = {
        spec.name: path for spec, path in zip(plan.agents, agent_paths, strict=True)
    }
    index_of = {spec.name: i + 1 for i, spec in enumerate(plan.agents)}
    total = len(plan.agents)
    outputs: dict[str, str] = {}
    results: list[AgentResult] = []

    def run_one(spec: AgentSpec) -> AgentResult:
        events.agent_started(spec.name, index_of[spec.name], total)
        upstream = dependency_closure(plan.agents, spec.name)
        visible = {name: outputs[name] for name in upstream if name in outputs}
        return execute_agent(path_by_name[spec.name], task, visible, effort=effort)

    def absorb(spec: AgentSpec, result: AgentResult) -> None:
        results.append(result)
        if result.ok:
            outputs[result.name] = result.output
        events.agent_finished(result)

    _for_each_in_waves(plan.agents, events, run_one, absorb)
    return outputs, results


def _finish_answer(
    task: str,
    response: str,
    plan: Plan,
    agent_paths: list[Path],
    usage: Usage,
    events: TaskEvents,
    effort: str | None,
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
        verdict = judge(task, response, usage=usage, effort=effort)
        events.answer_judged(verdict.done, verdict.missing)
        if verdict.done:
            break

        events.revision_started(attempt, TASK_REVISIONS, verdict.missing)
        outputs, results = _rerun_agents(
            plan, agent_paths, revision_task(task, verdict.missing), events, effort
        )
        extra_results.extend(results)
        if not outputs:
            break
        events.merge_started(len(outputs))
        response = merge_outputs(task, outputs, usage=usage, effort=effort)
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
    events = cast(TaskEvents, _SharedEvents(events or TaskEvents()))
    subject = task if subject is None else subject

    results: list[AgentResult] = []

    def spend() -> None:
        """Tell the interface what the run has cost so far."""
        calls = usage.calls + sum(
            1 for result in results if result.input_tokens or result.output_tokens
        )
        main_cost = usage.cost_usd
        agent_cost = sum(result.cost_usd for result in results)
        events.spend_updated(
            calls, None if main_cost is None else main_cost + agent_cost
        )

    events.phase_started(1, len(PHASES), PHASES[0])
    plan: Plan = plan_agents(task, usage=usage)
    events.plan_ready(plan)
    # The grade the planner just gave sets the effort of every call below -
    # the generated agents' own calls included, via their environment.
    effort = effort_for(plan.complexity)
    spend()

    events.phase_started(2, len(PHASES), PHASES[1])
    agent_paths: list[Path] = []
    reused: list[str] = []
    built: list[str] = []
    # name -> (role, source) for agents the user has not yet chosen to keep
    pending: dict[str, tuple[str, str]] = {}
    sources: dict[str, str] = {}
    to_build: list[AgentSpec] = []
    for spec in plan.agents:
        # A remembered agent is free; generating one is the most expensive
        # call in the run. Always look before building - but never hand back
        # one that turned out to have hardcoded the task it was built for,
        # or one whose own record says it fails more than it works.
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
        if code is not None and not reliable(spec.name):
            events.agent_retired(spec.name, "it has failed more tasks than it finished")
            forget(spec.name)
            code = None

        if code is not None:
            reused.append(spec.name)
            record_use(spec.name)
            sources[spec.name] = code
        else:
            to_build.append(spec)

    if to_build:
        for spec in to_build:
            events.agent_build_started(spec.name)

        def generate(spec: AgentSpec) -> str:
            return generate_agent_code(
                spec,
                dependency_closure(plan.agents, spec.name),
                usage=usage,
                task=subject,
                effort=effort,
            )

        # Each agent's code depends on nothing another generation call will
        # produce, so all of them are written at the same time. This is the
        # most expensive phase of a run; independence makes it the fastest.
        if len(to_build) == 1 or MAX_PARALLEL_AGENTS == 1:
            for spec in to_build:
                sources[spec.name] = generate(spec)
        else:
            workers = min(len(to_build), MAX_PARALLEL_AGENTS)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {spec.name: pool.submit(generate, spec) for spec in to_build}
            for spec in to_build:
                sources[spec.name] = futures[spec.name].result()

        for spec in to_build:
            built.append(spec.name)
            pending[spec.name] = (canonical_role(spec.name, spec.role), sources[spec.name])

    # Files land in plan order however the generation calls finished, so the
    # scratch directory - and the cleanup callback - stay deterministic.
    for spec in plan.agents:
        path = save_agent_file(spec, sources[spec.name])
        agent_paths.append(path)
        if on_agent_created is not None:
            on_agent_created(path)
        events.agent_ready(spec.name, path.name, reused=spec.name in reused)
    spend()

    events.phase_started(3, len(PHASES), PHASES[2])
    dependencies = install_dependencies(plan.agents)
    events.deps_checked(dependencies)

    events.phase_started(4, len(PHASES), PHASES[3])
    path_by_name = {
        spec.name: path for spec, path in zip(plan.agents, agent_paths, strict=True)
    }
    index_of = {spec.name: i + 1 for i, spec in enumerate(plan.agents)}
    total = len(plan.agents)
    outputs: dict[str, str] = {}
    failures: dict[str, str] = {}

    def run_one(spec: AgentSpec) -> tuple[AgentResult, str | None]:
        events.agent_started(spec.name, index_of[spec.name], total)
        upstream = dependency_closure(plan.agents, spec.name)
        # Exactly the outputs this agent was promised - its dependencies and
        # theirs - never "whatever happened to finish first".
        visible = {name: outputs[name] for name in upstream if name in outputs}
        return _run_with_repair(
            spec, path_by_name[spec.name], task, visible, upstream, usage,
            events, subject, effort,
        )

    def absorb(spec: AgentSpec, outcome: tuple[AgentResult, str | None]) -> None:
        result, repaired = outcome
        if repaired is not None:
            role = canonical_role(spec.name, spec.role)
            if spec.name in reused:
                # Fixing an agent the user already chose to keep: the rewrite
                # is a new generation, and it starts with a clean record.
                remember(spec.name, role, repaired, task=subject, evolved=True)
            else:
                pending[spec.name] = (role, repaired)
        results.append(result)
        if result.ok:
            outputs[result.name] = result.output
        else:
            failures[result.name] = result.error
        if spec.name in reused:
            # The library's long memory: enough losses and this agent is
            # retired on the next lookup instead of billed for again.
            record_outcome(spec.name, result.ok)
        events.agent_finished(result)
        spend()

    _for_each_in_waves(plan.agents, events, run_one, absorb)

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
    response = merge_outputs(task, outputs, usage=usage, effort=effort)
    spend()

    events.phase_started(6, len(PHASES), PHASES[5])
    council_improved = False
    if should_convene(plan.complexity):
        # The adversarial reading, before compliance is even checked: the
        # judge catches a missed demand, the council catches a weak answer.
        events.council_convened()
        response, council_improved, weaknesses = deliberate(
            task, response, usage=usage, effort=effort
        )
        events.council_ruled(council_improved, weaknesses)
        spend()

    response, extra_results, revisions = _finish_answer(
        task, response, plan, agent_paths, usage, events, effort
    )
    results.extend(extra_results)
    spend()

    return TaskResult(
        response=response,
        plan=plan,
        task=subject,
        complexity=plan.complexity,
        revisions=revisions,
        council_improved=council_improved,
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
