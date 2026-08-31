"""The shape of a plan: which agents feed which, and what may run side by side.

A plan used to be a list, and the list order was the wiring: every agent saw
everything before it. That is safe and slow - a fact-gatherer and a web
checker with nothing to say to each other still queued up single file.

This module reads the dependencies the planner declares and turns them into
two facts the orchestrator needs:

- `waves()`   - which agents may run AT THE SAME TIME. An agent joins a wave
  only when every agent it depends on has already finished, so parallelism
  exists exactly where the graph proves it changes nothing about the data.
- `dependency_closure()` - which upstream outputs one agent actually receives.
  Its declared dependencies and theirs, transitively - never "whatever
  happened to finish first", which would make runs unrepeatable.

Declared dependencies are untrusted input like everything else an LLM writes,
so they are sanitised first: self-references and unknown names are dropped,
a cycle is broken rather than obeyed, and a plan that declares nothing at all
falls back to the old sequential chain - the one wiring that is always safe.

Pure functions over anything with `.name` and `.depends_on`. No AI, no I/O,
so every rule here is unit-testable.
"""

from __future__ import annotations

from typing import Protocol, TypeVar


class _Spec(Protocol):
    """The two fields this module reads. planner.AgentSpec satisfies it."""

    name: str
    depends_on: list[str]


# Generic over the concrete spec type, so a list of AgentSpecs goes in and a
# list of AgentSpecs comes out - not a list of the protocol it happens to fit.
SpecT = TypeVar("SpecT", bound=_Spec)


def sanitise_dependencies(agents: list[SpecT]) -> None:
    """Reduce every declared dependency list to names that can be honoured.

    Self-references, duplicates and names not in the plan are dropped in
    place. A dependency on a misremembered agent is not an error worth
    failing a run for - it is a wire that cannot carry anything, so it is
    removed rather than left to dangle.
    """
    known = {spec.name for spec in agents}
    for spec in agents:
        seen: list[str] = []
        for name in spec.depends_on:
            if name != spec.name and name in known and name not in seen:
                seen.append(name)
        spec.depends_on = seen


def wire_sequential_fallback(agents: list[SpecT]) -> None:
    """Give a plan that declared no dependencies the old sequential wiring.

    An empty graph is ambiguous: it could mean "these agents are independent"
    or it could mean the planner never thought about it. Treating it as
    independence would run everything at once with no data flowing between
    agents - the silent-loss failure this module exists to prevent. So a plan
    with no declared edges at all gets the wiring every plan had before
    dependencies existed: each agent depends on all of the agents before it.
    """
    if any(spec.depends_on for spec in agents):
        return
    for index, spec in enumerate(agents):
        spec.depends_on = [previous.name for previous in agents[:index]]


def topological_order(agents: list[SpecT]) -> list[SpecT]:
    """The plan re-ordered so every agent comes after its dependencies.

    A stable Kahn's sort: among the agents that are ready, the current list
    order decides, so a plan that was already consistent comes back unchanged.
    A cycle cannot be honoured at all - the agents trapped in it have their
    remaining dependencies dropped and run in list order on whatever did
    finish, which loses an edge rather than the whole run.
    """
    remaining = {spec.name: set(spec.depends_on) for spec in agents}
    ordered: list[SpecT] = []
    pending = list(agents)
    while pending:
        ready = [spec for spec in pending if not remaining[spec.name]]
        if not ready:
            # A cycle. Break it: the first pending agent forgets the
            # dependencies that have not run, and proceeds without them.
            trapped = pending[0]
            unmet = remaining[trapped.name]
            trapped.depends_on = [name for name in trapped.depends_on if name not in unmet]
            remaining[trapped.name].clear()
            ready = [trapped]
        for spec in ready:
            ordered.append(spec)
            pending.remove(spec)
            for other in remaining.values():
                other.discard(spec.name)
    return ordered


def dependency_closure(agents: list[SpecT], name: str) -> list[str]:
    """Every agent whose output `name` receives: its dependencies, and theirs.

    Transitive on purpose. An analysis agent that depends on a research agent
    is really standing on everything the research agent stood on, and the
    generator promises each agent the exact keys it will see - a promise that
    only holds if this list is computed, not guessed.

    Returned in plan order, so the contract reads the way the plan does.
    """
    by_name = {spec.name: spec for spec in agents}
    spec = by_name.get(name)
    if spec is None:
        return []
    included: set[str] = set()
    frontier = list(spec.depends_on)
    while frontier:
        upstream = frontier.pop()
        if upstream in included or upstream not in by_name:
            continue
        included.add(upstream)
        frontier.extend(by_name[upstream].depends_on)
    return [candidate.name for candidate in agents if candidate.name in included]


def waves(agents: list[SpecT]) -> list[list[SpecT]]:
    """The plan grouped into rounds that may each run in parallel.

    An agent's wave is one deeper than the deepest agent it depends on, so
    everything in one wave has every input it needs before the wave starts.
    Agents in the same wave share no data path - that is what makes running
    them at the same time exactly as correct as running them one by one.
    """
    level: dict[str, int] = {}
    for spec in agents:  # agents arrive topologically ordered
        met = [level[name] for name in spec.depends_on if name in level]
        level[spec.name] = 1 + max(met, default=-1)
    grouped: dict[int, list[SpecT]] = {}
    for spec in agents:
        grouped.setdefault(level[spec.name], []).append(spec)
    return [grouped[depth] for depth in sorted(grouped)]
