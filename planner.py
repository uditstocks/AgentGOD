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

User task:
{task}
"""


def plan_agents(task: str, usage: Usage | None = None) -> Plan:
    """Ask the main LLM to break the task into a team of agent specs.

    include_raw keeps the underlying message reachable, so the planning call's
    token usage is accounted for like every other call.
    """
    llm = get_llm().with_structured_output(Plan, include_raw=True)
    result = llm.invoke(PLANNER_PROMPT.format(task=task, max_agents=MAX_AGENTS))

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
