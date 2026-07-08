"""Step 1: Analyze the user's task and decide which specialized agents are needed."""

from pydantic import BaseModel, Field

from config import get_llm


class AgentSpec(BaseModel):
    """Blueprint for one specialized agent."""

    name: str = Field(description="Short snake_case name, e.g. 'research_agent'")
    role: str = Field(description="One sentence: what this agent is responsible for")
    instructions: str = Field(
        description="Detailed instructions this agent must follow to do its single job"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Extra pip packages this agent needs (beyond langchain/langchain-anthropic)",
    )


class Plan(BaseModel):
    """The full team of agents required for the task."""

    reasoning: str = Field(description="Brief reasoning about how the task was split")
    agents: list[AgentSpec] = Field(description="Agents in execution order (1 to 4 agents)")


PLANNER_PROMPT = """You are the planner of a multi-agent system.
You never solve tasks yourself. You decide which specialized agents are needed.

Rules:
- Use the FEWEST agents possible (1 for simple tasks, up to 4 for complex ones).
- Each agent must have exactly ONE clear responsibility.
- Agents run in order; each agent receives the outputs of the agents before it.
- Only list pip dependencies that are truly required (agents already have langchain).

User task:
{task}
"""


def plan_agents(task: str) -> Plan:
    """Ask the main LLM to break the task into a team of agent specs."""
    llm = get_llm().with_structured_output(Plan)
    return llm.invoke(PLANNER_PROMPT.format(task=task))
