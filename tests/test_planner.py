"""Planner schema tests: the plan is the system's trust boundary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import MAX_AGENTS
from planner import AgentSpec, Plan, safe_agent_name, upstream_names


def spec(name: str = "research_agent", **kwargs) -> AgentSpec:
    return AgentSpec(name=name, role="r", instructions="i", **kwargs)


# --- C4: names become filenames, so they must be safe --------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("research_agent", "research_agent"),
        ("Research Agent", "research_agent"),
        ("../../../pwned", "pwned"),
        ("..\\..\\pwned2", "pwned2"),
        ("agent;rm -rf /", "agent_rm_rf"),
        ("CON", "con_agent"),
        ("nul", "nul_agent"),
        ("123agent", "agent_123agent"),
        ("a/b/c", "a_b_c"),
    ],
)
def test_names_are_sanitised(raw, expected):
    assert safe_agent_name(raw) == expected


def test_overlong_name_is_truncated():
    assert len(safe_agent_name("a" * 200)) == 40


@pytest.mark.parametrize("raw", ["", "   ", "!!!", "../.."])
def test_unsalvageable_names_are_rejected(raw):
    with pytest.raises(ValueError):
        safe_agent_name(raw)


def test_spec_rejects_empty_role():
    with pytest.raises(ValidationError):
        AgentSpec(name="research_agent", role="   ", instructions="i")


# --- M1: the 1..MAX_AGENTS rule is enforced by the schema, not just the prompt --


def test_empty_plan_is_rejected():
    with pytest.raises(ValidationError):
        Plan(agents=[], reasoning="r")


def test_oversized_plan_is_rejected():
    with pytest.raises(ValidationError):
        Plan(agents=[spec(f"agent_{i}") for i in range(MAX_AGENTS + 1)], reasoning="r")


def test_plan_at_the_limit_is_accepted():
    plan = Plan(agents=[spec(f"agent_{i}") for i in range(MAX_AGENTS)], reasoning="r")
    assert len(plan.agents) == MAX_AGENTS


# --- M2: names are dict keys and filenames, so they must be unique --------------


def test_duplicate_names_are_disambiguated():
    plan = Plan(agents=[spec("writer"), spec("writer"), spec("writer")], reasoning="r")
    names = [agent.name for agent in plan.agents]
    assert names == ["writer", "writer_2", "writer_3"]
    assert len(set(names)) == len(names)


def test_duplicates_created_by_sanitising_are_also_split():
    plan = Plan(agents=[spec("Writer Agent"), spec("writer agent")], reasoning="r")
    assert [agent.name for agent in plan.agents] == ["writer_agent", "writer_agent_2"]


# --- M3: agents are generated before the prose that describes them --------------


def test_agents_field_precedes_reasoning():
    assert list(Plan.model_fields) == ["agents", "reasoning"]


# --- C3: upstream names are exactly the preceding agents ------------------------


def test_upstream_names_are_the_preceding_agents():
    agents = [spec("a_agent"), spec("b_agent"), spec("c_agent")]
    assert upstream_names(agents, 0) == []
    assert upstream_names(agents, 1) == ["a_agent"]
    assert upstream_names(agents, 2) == ["a_agent", "b_agent"]
