"""Planner schema tests: the plan is the system's trust boundary."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

import planner
from codeguard import ALLOWED_PACKAGES
from config import MAX_AGENTS
from planner import STANDARD_AGENTS, AgentSpec, Plan, safe_agent_name, upstream_names


def spec(name: str = "research_agent", **kwargs: Any) -> AgentSpec:
    fields: dict[str, Any] = {"role": "r", "instructions": "i"}
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


def test_field_order_is_grade_then_team_then_reasoning():
    """Structured output is generated in field order: the task is sized first,
    the team is emitted next, and the prose describes what actually exists."""
    assert list(Plan.model_fields) == ["complexity", "agents", "reasoning"]


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


# --- the dependency graph arrives untrusted and leaves honest -------------------


def test_declared_dependencies_are_sanitised_like_names():
    plan = Plan(
        agents=[
            spec("research_agent"),
            spec("summary_agent", depends_on=["Research Agent", "ghost_agent"]),
        ],
        reasoning="r",
    )
    by_name = {s.name: s for s in plan.agents}
    assert by_name["summary_agent"].depends_on == ["research_agent"]


def test_a_plan_with_no_declared_edges_falls_back_to_the_chain():
    plan = Plan(
        agents=[spec("research_agent"), spec("analysis_agent"), spec("summary_agent")],
        reasoning="r",
    )
    deps = [s.depends_on for s in plan.agents]
    assert deps == [[], ["research_agent"], ["research_agent", "analysis_agent"]]


def test_declared_independence_is_honoured():
    plan = Plan(
        agents=[
            spec("research_agent"),
            spec("comparison_agent"),
            spec("summary_agent", depends_on=["research_agent", "comparison_agent"]),
        ],
        reasoning="r",
    )
    by_name = {s.name: s for s in plan.agents}
    assert by_name["research_agent"].depends_on == []
    assert by_name["comparison_agent"].depends_on == []


def test_the_plan_is_reordered_so_producers_precede_consumers():
    plan = Plan(
        agents=[
            spec("omega_writer", role="write the piece", depends_on=["fact_finder"]),
            spec("fact_finder", role="gather facts"),
        ],
        reasoning="r",
    )
    assert [s.name for s in plan.agents] == ["fact_finder", "omega_writer"]


def test_complexity_defaults_to_standard():
    plan = Plan(agents=[spec()], reasoning="r")
    assert plan.complexity == "standard"


def test_complexity_accepts_the_three_grades():
    for grade in ("simple", "standard", "deep"):
        assert Plan(agents=[spec()], reasoning="r", complexity=grade).complexity == grade


# --- the cached prefix must hold nothing that changes between runs -------------


def test_the_cached_policy_holds_no_volatile_text():
    """The library catalogue is re-ranked by use count on almost every run.

    Holding it in the cached prefix invalidated the whole ~1,400-token block
    each time - paying the cache-write premium and never once collecting the
    discount. Volatile text belongs in the message.
    """
    from codeguard import ALLOWED_PACKAGES
    from config import MAX_AGENTS
    from planner import PLANNER_PROMPT, STANDARD_AGENTS, TASK_PROMPT, _wrap

    policy = PLANNER_PROMPT.format(
        max_agents=MAX_AGENTS,
        standard="\n".join(f"  - {n} - {r}" for n, r in STANDARD_AGENTS.items()),
        packages=_wrap("  ", sorted(ALLOWED_PACKAGES)),
    )
    # Rendering twice must give byte-identical text, or the cache cannot hit.
    again = PLANNER_PROMPT.format(
        max_agents=MAX_AGENTS,
        standard="\n".join(f"  - {n} - {r}" for n, r in STANDARD_AGENTS.items()),
        packages=_wrap("  ", sorted(ALLOWED_PACKAGES)),
    )
    assert policy == again
    assert "{library}" not in policy  # the catalogue is not a policy field
    assert "{library}" in TASK_PROMPT  # it travels with the task instead


def test_the_task_message_still_shows_the_library():
    """Moving it out of the cache must not hide it from the planner."""
    from planner import TASK_PROMPT

    rendered = TASK_PROMPT.format(task="write a haiku", library="  - writer_agent: write prose")
    assert "writer_agent" in rendered
    assert "write a haiku" in rendered


# --- the capability is the only thing the generator sees, so it must be clean ---


def test_a_capability_that_names_the_task_is_replaced_outright():
    """Not warned about, not regenerated - replaced. Trusting it is the bug."""
    from planner import scrub_capabilities

    plan = Plan(
        agents=[
            spec(
                "code_agent",
                role="Implement the natural-language-to-SQL flow",
                capability="Write a SQL translator with a schema validator",
            )
        ],
        reasoning="r",
    )
    replaced = scrub_capabilities(plan, "build a flow that turns natural language into SQL")
    assert replaced == ["code_agent"]
    assert plan.agents[0].capability == STANDARD_AGENTS["code_agent"]
    assert "sql" not in plan.agents[0].capability.lower()


def test_a_clean_capability_is_left_exactly_as_written():
    from planner import scrub_capabilities

    written = "Write correct, runnable Python for whatever the task asks for."
    plan = Plan(agents=[spec("code_agent", capability=written)], reasoning="r")
    assert scrub_capabilities(plan, "build a flow that turns text into qr codes") == []
    assert plan.agents[0].capability == written


def test_an_empty_capability_is_filled_in_silently():
    """A planner that omitted it is a slip, not something to announce."""
    from planner import scrub_capabilities

    plan = Plan(agents=[spec("summary_agent", capability="")], reasoning="r")
    assert scrub_capabilities(plan, "condense the qr code report") == []
    assert plan.agents[0].capability == STANDARD_AGENTS["summary_agent"]


def test_an_invented_agent_gets_a_generic_brief_not_the_task():
    from planner import scrub_capabilities

    plan = Plan(
        agents=[spec("schema_agent", role="design the SQL schema", capability="design the SQL schema")],
        reasoning="r",
    )
    scrub_capabilities(plan, "design the SQL schema for an orders database")
    capability = plan.agents[0].capability
    assert "sql" not in capability.lower()
    assert "schema" in capability.lower()  # the agent's own name survives
    assert "task" in capability.lower()


def test_the_role_is_untouched_because_it_is_only_ever_displayed():
    from planner import scrub_capabilities

    plan = Plan(
        agents=[spec("code_agent", role="Implement the SQL flow", capability="write SQL")],
        reasoning="r",
    )
    scrub_capabilities(plan, "implement the SQL flow")
    assert plan.agents[0].role == "Implement the SQL flow"
