"""Step 1: Analyze the user's task and decide which specialized agents are needed."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator

from config import MAX_AGENTS, Usage, get_llm

# An agent name becomes a filename and a dict key, so it must be a plain
# snake_case identifier. Anything else is a path-traversal risk.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")

# Reserved device names on Windows: creating "con.py" is a trap.
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def safe_agent_name(raw: str) -> str:
    """Reduce a model-chosen name to a safe snake_case identifier.

    Sanitising rather than rejecting keeps cosmetic wobble ("Research Agent")
    from failing a whole run, while still neutralising traversal attempts
    ("../../../pwned" -> "pwned").
    """
    slug = re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower())
    slug = re.sub(r"_{2,}", "_", slug).strip("_")
    if slug and not slug[0].isalpha():
        slug = f"agent_{slug}"
    if len(slug) > 40:
        slug = slug[:40].rstrip("_")
    if slug in _RESERVED_NAMES:
        slug = f"{slug}_agent"
    if not _NAME_RE.match(slug):
        raise ValueError(f"agent name cannot be made safe: {raw!r}")
    return slug


class AgentSpec(BaseModel):
    """Blueprint for one specialized agent."""

    name: str = Field(description="Short snake_case name, e.g. 'research_agent'")
    role: str = Field(description="One sentence: what this agent is responsible for")
    instructions: str = Field(
        description="Detailed instructions this agent must follow to do its single job"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Extra pip packages this agent needs (usually none; the agent "
        "runtime is standard library only)",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return safe_agent_name(value)

    @field_validator("role", "instructions")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must not be empty")
        return text


class Plan(BaseModel):
    """The full team of agents required for the task.

    `agents` is declared before `reasoning` on purpose: structured output is
    generated in field order, so the model describes the team it actually
    emitted instead of committing in prose to agents it then omits.
    """

    agents: list[AgentSpec] = Field(
        min_length=1,
        max_length=MAX_AGENTS,
        description=f"Agents in execution order (1 to {MAX_AGENTS} agents)",
    )
    reasoning: str = Field(
        description="One or two sentences describing the split you just listed above"
    )

    @model_validator(mode="after")
    def _deduplicate_names(self) -> Plan:
        """Agent names are filenames and dict keys, so they must be unique."""
        seen: set[str] = set()
        for spec in self.agents:
            if spec.name not in seen:
                seen.add(spec.name)
                continue
            suffix = 2
            while f"{spec.name}_{suffix}" in seen:
                suffix += 1
            spec.name = f"{spec.name}_{suffix}"
            seen.add(spec.name)
        return self


# Preferred names for the responsibilities that recur across almost every task.
# A stable vocabulary is what makes the library hit instead of building a new
# near-duplicate agent under a slightly different name every run.
STANDARD_AGENTS = (
    "research_agent - gather facts about whatever subject the task names",
    "analysis_agent - analyse supplied material and draw out the key points",
    "summary_agent - condense supplied material to a requested length",
    "writer_agent - write prose in whatever form the task asks for",
    "editor_agent - revise supplied text for clarity and correctness",
    "outline_agent - produce a structured outline of the requested piece",
    "comparison_agent - compare options against stated criteria",
    "critique_agent - find weaknesses and risks in supplied material",
    "code_agent - write or explain code",
    "translation_agent - translate supplied text",
)

PLANNER_PROMPT = """You are the planner of a multi-agent system.
You never solve tasks yourself. You decide which specialized agents are needed.

Rules:
- Use the FEWEST agents possible (1 for simple tasks, up to {max_agents} for complex ones).
- Each agent must have exactly ONE clear responsibility.
- Agents run in order; each agent receives the outputs of the agents before it.
- Name each agent in snake_case, e.g. 'research_agent'.
- Generated agents run on the Python standard library and need no pip packages.
  Leave 'dependencies' empty unless a package is genuinely unavoidable.
- Your 'reasoning' must describe exactly the agents you listed - no more, no fewer.

REUSE COMES FIRST. Agents are written once and kept. An agent whose name already
exists is free; a new name costs a full code-generation call. So:
- These agents are ALREADY BUILT and cost nothing to use. Prefer them whenever one
  can do the job, and use the name EXACTLY as written:
{library}
- If none fits, prefer one of these standard names before inventing your own:
{standard}
- Describe every agent by its FUNCTION, never by this task's subject.
  Write "gather facts about the subject of the task", not "research electric
  scooters". A subject-specific agent can never be reused and costs full price
  every time.

User task:
{task}
"""


def plan_agents(task: str, usage: Usage | None = None) -> Plan:
    """Ask the main LLM to break the task into a team of agent specs.

    include_raw keeps the underlying message reachable, so the planning call's
    token usage is accounted for like every other call.
    """
    from library import describe_for_planner

    llm = get_llm().with_structured_output(Plan, include_raw=True)
    result = llm.invoke(
        PLANNER_PROMPT.format(
            task=task,
            max_agents=MAX_AGENTS,
            library=describe_for_planner(),
            standard="\n".join(f"  - {line}" for line in STANDARD_AGENTS),
        )
    )

    if isinstance(result, dict):
        if usage is not None and result.get("raw") is not None:
            usage.record(result["raw"])
        if result.get("parsing_error"):
            raise ValueError(f"planner returned an invalid plan: {result['parsing_error']}")
        result = result.get("parsed")

    if isinstance(result, Plan):
        return result
    return Plan.model_validate(result)


def upstream_names(agents: list[AgentSpec], index: int) -> list[str]:
    """Names whose outputs agent `index` will find in `previous_outputs`."""
    return [spec.name for spec in agents[:index]]
