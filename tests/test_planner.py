"""Planner schema tests: the plan is the system's trust boundary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import planner
from codeguard import ALLOWED_PACKAGES
from config import MAX_AGENTS
from planner import AgentSpec, Plan, safe_agent_name, upstream_names


def spec(name: str = "research_agent", **kwargs) -> AgentSpec:
    fields = {"role": "r", "instructions": "i"}
    fields.update(kwargs)
    return AgentSpec(name=name, **fields)


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


def test_a_repeated_agent_with_the_same_job_is_dropped():
    """The same name and the same role twice is a planning slip, not a team.

    Renaming it to `writer_2` would generate, run and bill a second copy of an
    agent already in the plan, and leave a near-identical twin in the library
    for good. One real run did exactly that and ended up with both
    `summary_agent` and `summary_agent_2` doing the same thing.
    """
    plan = Plan(agents=[spec("writer"), spec("writer"), spec("writer")], reasoning="r")
    assert [agent.name for agent in plan.agents] == ["writer"]


def test_a_repeated_name_with_a_different_job_is_kept_and_renamed():
    plan = Plan(
        agents=[
            spec("writer", role="write the introduction"),
            spec("writer", role="write the conclusion"),
        ],
        reasoning="r",
    )
    names = [agent.name for agent in plan.agents]
    assert names == ["writer", "writer_2"]
    assert len(set(names)) == len(names)


def test_duplicates_created_by_sanitising_are_also_resolved():
    plan = Plan(agents=[spec("Writer Agent"), spec("writer agent")], reasoning="r")
    assert [agent.name for agent in plan.agents] == ["writer_agent"]


def test_wording_alone_is_not_a_different_job():
    """'Condense the findings' and 'condense the findings!' are one agent."""
    plan = Plan(
        agents=[
            spec("summary_agent", role="Condense the findings"),
            spec("summary_agent", role="condense the findings!"),
        ],
        reasoning="r",
    )
    assert [agent.name for agent in plan.agents] == ["summary_agent"]


# --- agents must run in an order where their inputs already exist ---------------


def test_a_reducer_listed_first_is_moved_after_its_producer():
    """The real defect: summary_agent ran before research_agent and summarised nothing."""
    plan = Plan(
        agents=[spec("summary_agent", role="condense"), spec("research_agent", role="gather")],
        reasoning="research feeds summary",
    )
    assert [agent.name for agent in plan.agents] == ["research_agent", "summary_agent"]


def test_a_sensible_order_is_left_alone():
    ordered = ["research_agent", "analysis_agent", "writer_agent", "editor_agent"]
    plan = Plan(agents=[spec(name) for name in ordered], reasoning="r")
    assert [agent.name for agent in plan.agents] == ordered


def test_an_unrecognised_agent_sorts_with_the_work_not_the_review():
    plan = Plan(
        agents=[
            spec("summary_agent", role="condense"),
            spec("bespoke_thing", role="do something unusual"),
            spec("research_agent", role="gather"),
        ],
        reasoning="r",
    )
    assert [agent.name for agent in plan.agents] == [
        "research_agent",
        "bespoke_thing",
        "summary_agent",
    ]


def test_stage_rank_falls_back_to_the_role_when_the_name_is_invented():
    from planner import stage_rank

    assert stage_rank("zeta_agent", "gather facts about the subject") == stage_rank("research_agent")
    assert stage_rank("omega_agent", "condense it all down") == stage_rank("summary_agent")


def test_canonical_role_replaces_a_task_specific_description():
    """The library index must describe a capability, not one old task."""
    from planner import canonical_role

    assert canonical_role("summary_agent", "condense the code review findings") == (
        "condense supplied material to a requested length"
    )
    assert canonical_role("bespoke_agent", "its own description") == "its own description"


# --- M3: agents are generated before the prose that describes them --------------


def test_agents_field_precedes_reasoning():
    assert list(Plan.model_fields) == ["agents", "reasoning"]


# --- C3: upstream names are exactly the preceding agents ------------------------


def test_upstream_names_are_the_preceding_agents():
    agents = [spec("a_agent"), spec("b_agent"), spec("c_agent")]
    assert upstream_names(agents, 0) == []
    assert upstream_names(agents, 1) == ["a_agent"]
    assert upstream_names(agents, 2) == ["a_agent", "b_agent"]


# --- the planner is shown the packages it may ask for --------------------------


def test_the_prompt_lists_every_vetted_package():
    """A name it cannot see is a name it invents, and an invented name is refused."""
    rendered = planner.PLANNER_PROMPT.format(
        task="t",
        max_agents=4,
        library="",
        standard="",
        packages=planner._wrap("  ", sorted(ALLOWED_PACKAGES)),
    )
    for name in ALLOWED_PACKAGES:
        assert name in rendered


def test_the_package_list_wraps_instead_of_running_off_the_line():
    lines = planner._wrap("  ", sorted(ALLOWED_PACKAGES)).splitlines()
    assert len(lines) > 1
    assert all(line.startswith("  ") and len(line) <= 78 for line in lines)
