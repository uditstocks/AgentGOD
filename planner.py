"""Step 1: Analyze the user's task and decide which specialized agents are needed."""

from __future__ import annotations

import re
import textwrap
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from codeguard import ALLOWED_PACKAGES
from config import MAX_AGENTS, Usage, complete_structured
from taskgraph import (
    sanitise_dependencies,
    topological_order,
    wire_sequential_fallback,
)

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
    depends_on: list[str] = Field(
        default_factory=list,
        description="Names of agents in THIS plan whose outputs this agent needs, "
        "and only those. Agents whose dependencies are all met run at the same "
        "time, so leave this empty for an agent that works from the task alone.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Extra pip packages this agent needs (usually none - the whole "
        "Python standard library is already available). Any name outside the vetted "
        "list in the prompt is refused, so never invent one.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return safe_agent_name(value)

    @field_validator("depends_on")
    @classmethod
    def _validate_depends_on(cls, value: list[str]) -> list[str]:
        """Reduce each declared dependency to the same safe form as a name.

        Agent names are sanitised, so the references to them must be run
        through the identical reduction or "Research Agent" would never match
        the `research_agent` it plainly means. A name that cannot be made
        safe is dropped rather than fatal: it could never match any agent,
        so it is a wire to nowhere, not a reason to fail the plan.
        """
        cleaned: list[str] = []
        for raw in value:
            try:
                cleaned.append(safe_agent_name(raw))
            except ValueError:
                continue
        return cleaned

    @field_validator("role", "instructions")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must not be empty")
        return text


class Plan(BaseModel):
    """The full team of agents required for the task.

    Field order is deliberate throughout, because structured output is
    generated in that order: `complexity` comes first so the task is sized
    before the team is designed, and `agents` comes before `reasoning` so the
    model describes the team it actually emitted instead of committing in
    prose to agents it then omits.
    """

    complexity: Literal["simple", "standard", "deep"] = Field(
        default="standard",
        description="How much thinking this task deserves. 'simple': one obvious "
        "step with a short answer (a translation, a one-liner, a lookup). "
        "'deep': high stakes or real intellectual difficulty - research, "
        "analysis, strategy, long-form writing, non-trivial code - where a "
        "wrong or shallow answer costs the user something. Everything else "
        "is 'standard'. The whole run spends effort and money accordingly.",
    )
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
        """Agent names are filenames and dict keys, so they must be unique.

        A repeated name with a repeated job is a planning slip, not a second
        agent: renaming it to `summary_agent_2` would build, run and bill a
        duplicate of the agent already in the plan, and leave a near-identical
        twin in the library forever. Those are dropped. A repeated name with a
        genuinely different job is kept, and renamed as before.
        """
        kept: list[AgentSpec] = []
        seen: dict[str, str] = {}
        for spec in self.agents:
            existing = seen.get(spec.name)
            if existing is not None and existing == _normalise_role(spec.role):
                continue
            if existing is not None:
                suffix = 2
                while f"{spec.name}_{suffix}" in seen:
                    suffix += 1
                spec.name = f"{spec.name}_{suffix}"
            seen[spec.name] = _normalise_role(spec.role)
            kept.append(spec)
        self.agents = kept
        return self

    @model_validator(mode="after")
    def _order_by_stage(self) -> Plan:
        """Put producers before the agents that consume what they produce.

        Agents run in list order and each one only sees the agents before it,
        so the order is the wiring. The model gets it wrong: one plan listed
        `summary_agent` first and `research_agent` second while its own
        reasoning described the opposite, and the summary ran against an empty
        `previous_outputs`.

        The sort is stable, so a plan that was already sensible is untouched.
        """
        self.agents = sorted(self.agents, key=lambda spec: stage_rank(spec.name, spec.role))
        return self

    @model_validator(mode="after")
    def _wire_the_graph(self) -> Plan:
        """Make the declared dependencies honest, then order the plan by them.

        Runs after deduplication and the stage sort, so it works on final
        names in a sensible baseline order. Unknown and self-references are
        dropped, a plan that declared nothing falls back to the sequential
        chain every plan had before dependencies existed, and a cycle is
        broken rather than obeyed. After this, list order and graph order
        agree: every agent appears after everything it depends on.
        """
        sanitise_dependencies(self.agents)
        wire_sequential_fallback(self.agents)
        self.agents = list(topological_order(self.agents))
        return self


def _normalise_role(role: str) -> str:
    """Role text reduced for comparison, so wording alone is not a difference."""
    return re.sub(r"[^a-z0-9 ]+", "", role.lower()).strip()


# Preferred names for the responsibilities that recur across almost every task.
# A stable vocabulary is what makes the library hit instead of building a new
# near-duplicate agent under a slightly different name every run.
STANDARD_AGENTS: dict[str, str] = {
    "research_agent": "gather facts about whatever subject the task names",
    "analysis_agent": "analyse supplied material and draw out the key points",
    "summary_agent": "condense supplied material to a requested length",
    "writer_agent": "write prose in whatever form the task asks for",
    "editor_agent": "revise supplied text for clarity and correctness",
    "outline_agent": "produce a structured outline of the requested piece",
    "comparison_agent": "compare options against stated criteria",
    "critique_agent": "find weaknesses and risks in supplied material",
    "code_agent": "write or explain code",
    "translation_agent": "translate supplied text",
}

# Where each kind of work sits in a pipeline. Lower runs earlier.
#
# The tiers are the only judgement here: something has to be gathered before
# it can be analysed, written before it can be edited, and finished before it
# can be condensed. Anything unrecognised is treated as the substantive middle
# of the run, which is what a custom agent almost always is.
GATHER_RANK = 10
INTERPRET_RANK = 20
PRODUCE_RANK = 30
REVIEW_RANK = 40
POLISH_RANK = 50
REDUCE_RANK = 60

_STAGE_KEYWORDS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (GATHER_RANK, ("research", "gather", "collect", "fetch", "retrieve", "source", "discover")),
    (INTERPRET_RANK, ("analysis", "analyse", "analyze", "examine", "evaluate", "assess", "interpret")),
    (REDUCE_RANK, ("summary", "summarise", "summarize", "condense", "digest", "abstract")),
    (POLISH_RANK, ("editor", "edit", "polish", "proofread", "refine", "format", "rewrite")),
    (REVIEW_RANK, ("critique", "criticise", "criticize", "review", "verify", "check", "validate")),
    (PRODUCE_RANK, ("outline", "structure", "plan")),
    (PRODUCE_RANK, ("writer", "write", "draft", "compose", "code", "translation", "translate",
                    "comparison", "compare", "generate", "create", "build", "design")),
)


def stage_rank(name: str, role: str = "") -> int:
    """Which pipeline stage an agent belongs to, from its name then its role.

    The name is checked first because it is the part the planner is steered to
    keep stable; the role is only consulted for an agent with an invented name.
    """
    for haystack in (name.lower(), role.lower()):
        if not haystack:
            continue
        for rank, keywords in _STAGE_KEYWORDS:
            if any(keyword in haystack for keyword in keywords):
                return rank
    return PRODUCE_RANK


def _wrap(indent: str, names: list[str], width: int = 78) -> str:
    """Lay a long list of names out over several indented lines.

    Eighty package names on one line is a wall the model skims past; wrapped,
    it reads them, which is the whole point of showing the list at all.
    """
    return textwrap.fill(
        ", ".join(names), width=width, initial_indent=indent, subsequent_indent=indent
    )


def canonical_role(name: str, fallback: str = "") -> str:
    """The reusable description of a standard agent, for the library index.

    The planner writes roles for today's task ("gather facts about the benefits
    of code review"), and that text is what a later run is shown when deciding
    whether the agent fits. Recording the standard wording instead keeps the
    catalogue describing capabilities rather than one old task.
    """
    return STANDARD_AGENTS.get(name, fallback)

PLANNER_PROMPT = """You are the planner of a multi-agent system.
You never solve tasks yourself. You decide which specialized agents are needed.

Rules:
- First grade the task's complexity honestly. 'simple' work gets a fast,
  cheap run; 'deep' work gets the most careful one. Most tasks are 'standard'.
- Use the FEWEST agents possible (1 for simple tasks, up to {max_agents} for complex ones).
- Each agent must have exactly ONE clear responsibility.
- Declare each agent's 'depends_on': the names of the agents whose OUTPUTS it
  genuinely needs, and only those. Agents whose dependencies are all met run
  AT THE SAME TIME - so two agents that each work from the task alone should
  both declare no dependencies, not an invented ordering. An agent that
  synthesises must depend on every agent it synthesises from.
- Name each agent in snake_case, e.g. 'research_agent'.
- Agents can search the web. If the task turns on anything that changed after
  training - prices, news, releases, listings, "current", "latest", "today" -
  give an agent the job of looking it up and say so in its instructions. Do
  not refuse the task and do not let an agent answer from memory instead.
- Generated agents get the whole Python standard library, which covers most
  work. Leave 'dependencies' empty unless a package is genuinely unavoidable.
- If one is unavoidable, it must be named EXACTLY from this list. Any other
  name is refused, so never invent one - if what you want is not here, plan
  the agent around the standard library instead:
{packages}
- Your 'reasoning' must describe exactly the agents you listed - no more, no fewer.

REUSE COMES FIRST. Agents are written once and kept. An agent whose name already
exists is free; a new name costs a full code-generation call. So:
- Prefer an agent that ALREADY EXISTS (the message lists them) whenever one can
  do the job, and use its name EXACTLY as written.
- If none fits, prefer one of these standard names before inventing your own:
{standard}
- Describe every agent by its FUNCTION, never by this task's subject.
  Write "gather facts about the subject of the task", not "research electric
  scooters". A subject-specific agent can never be reused and costs full price
  every time.
"""

# What changes between runs, and therefore may NOT live in the cached block.
#
# The library catalogue is ranked by use count, so it is rewritten by almost
# every run. Holding it in the cached prefix meant the whole 1,400-token
# policy was invalidated each time - paying the cache-write premium and never
# once collecting the discount. Volatile text goes in the message; only text
# that is byte-identical between runs earns a cache breakpoint.
TASK_PROMPT = """Agents already built and free to use (prefer these):
{library}

User task:
{task}
"""


def plan_agents(task: str, usage: Usage | None = None) -> Plan:
    """Ask the main LLM to break the task into a team of agent specs.

    The reply is constrained to the Plan schema by the API itself, so a
    malformed plan is a request error rather than something to salvage here.
    """
    from config import cached_system
    from library import describe_for_planner

    # The rules, the vetted package list and the standard vocabulary are
    # byte-identical on every run, so they travel as one cached system block:
    # roughly 1,400 tokens re-read at cache price instead of full price.
    policy = PLANNER_PROMPT.format(
        max_agents=MAX_AGENTS,
        standard="\n".join(f"  - {name} - {role}" for name, role in STANDARD_AGENTS.items()),
        packages=_wrap("  ", sorted(ALLOWED_PACKAGES)),
    )
    return complete_structured(
        TASK_PROMPT.format(task=task, library=describe_for_planner()),
        Plan,
        system=cached_system(policy),
        usage=usage,
    )


def upstream_names(agents: list[AgentSpec], index: int) -> list[str]:
    """Names whose outputs agent `index` will find in `previous_outputs`."""
    return [spec.name for spec in agents[:index]]
