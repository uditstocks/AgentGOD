"""The plan's shape: dependency hygiene, ordering, waves and closures.

Pure functions over anything with `.name` and `.depends_on`, so these tests
use a minimal stand-in rather than full AgentSpecs - the graph rules must
hold for anything shaped like a spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from taskgraph import (
    dependency_closure,
    sanitise_dependencies,
    topological_order,
    waves,
    wire_sequential_fallback,
)


@dataclass
class Node:
    name: str
    depends_on: list[str] = field(default_factory=list)


def names(specs) -> list[str]:
    return [spec.name for spec in specs]


# --- sanitising declared dependencies ------------------------------------------


def test_unknown_and_self_references_are_dropped():
    plan = [Node("a"), Node("b", ["a", "b", "ghost_agent", "a"])]
    sanitise_dependencies(plan)
    assert plan[1].depends_on == ["a"]


def test_a_clean_declaration_is_untouched():
    plan = [Node("a"), Node("b", ["a"]), Node("c", ["a", "b"])]
    sanitise_dependencies(plan)
    assert plan[1].depends_on == ["a"]
    assert plan[2].depends_on == ["a", "b"]


# --- the sequential fallback ----------------------------------------------------


def test_a_plan_that_declared_nothing_gets_the_old_chain():
    """An empty graph means the planner never thought about it, not independence."""
    plan = [Node("a"), Node("b"), Node("c")]
    wire_sequential_fallback(plan)
    assert plan[0].depends_on == []
    assert plan[1].depends_on == ["a"]
    assert plan[2].depends_on == ["a", "b"]


def test_any_declared_edge_means_the_graph_is_trusted():
    plan = [Node("a"), Node("b"), Node("c", ["a"])]
    wire_sequential_fallback(plan)
    assert plan[1].depends_on == []  # genuinely independent, left that way


# --- topological order ----------------------------------------------------------


def test_a_consistent_plan_comes_back_unchanged():
    plan = [Node("a"), Node("b", ["a"]), Node("c", ["b"])]
    assert names(topological_order(plan)) == ["a", "b", "c"]


def test_a_consumer_listed_first_is_moved_after_its_producer():
    plan = [Node("summary", ["research"]), Node("research")]
    assert names(topological_order(plan)) == ["research", "summary"]


def test_a_cycle_is_broken_rather_than_obeyed():
    plan = [Node("a", ["b"]), Node("b", ["a"])]
    ordered = topological_order(plan)
    assert sorted(names(ordered)) == ["a", "b"]
    # The agent that had to break the cycle lost exactly the unmet edge.
    first = ordered[0]
    assert first.depends_on == []


def test_independent_agents_keep_their_list_order():
    plan = [Node("a"), Node("b"), Node("c")]
    assert names(topological_order(plan)) == ["a", "b", "c"]


# --- the closure: what one agent actually receives -----------------------------


def test_closure_is_transitive():
    plan = [Node("a"), Node("b", ["a"]), Node("c", ["b"])]
    assert dependency_closure(plan, "c") == ["a", "b"]


def test_closure_excludes_unrelated_agents():
    plan = [Node("a"), Node("b"), Node("c", ["b"])]
    assert dependency_closure(plan, "c") == ["b"]
    assert dependency_closure(plan, "a") == []


def test_closure_of_an_unknown_agent_is_empty():
    assert dependency_closure([Node("a")], "ghost") == []


def test_closure_returns_plan_order_not_traversal_order():
    plan = [Node("a"), Node("b"), Node("c", ["b", "a"])]
    assert dependency_closure(plan, "c") == ["a", "b"]


# --- waves: what may run at the same time --------------------------------------


def test_a_chain_is_one_agent_per_wave():
    plan = [Node("a"), Node("b", ["a"]), Node("c", ["b"])]
    assert [names(wave) for wave in waves(plan)] == [["a"], ["b"], ["c"]]


def test_independent_agents_share_the_first_wave():
    plan = [Node("a"), Node("b"), Node("c", ["a", "b"])]
    assert [names(wave) for wave in waves(plan)] == [["a", "b"], ["c"]]


def test_the_diamond_runs_in_three_waves():
    plan = [
        Node("source"),
        Node("left", ["source"]),
        Node("right", ["source"]),
        Node("sink", ["left", "right"]),
    ]
    assert [names(wave) for wave in waves(plan)] == [
        ["source"], ["left", "right"], ["sink"],
    ]
