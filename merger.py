"""Step 6: Merge all agent outputs into one final response for the user."""

from config import get_llm

MERGER_PROMPT = """You are the coordinator of a multi-agent system.
Several specialized agents each completed one part of the user's task.
Merge their outputs into ONE clear, complete final answer for the user.
Do not mention the agents or the process — just deliver the answer.

User task:
{task}

Agent outputs:
{outputs}
"""


def merge_outputs(task: str, outputs: dict[str, str]) -> str:
    """Combine every agent's output into a single final response."""
    if len(outputs) == 1:
        # One agent means nothing to merge — return its output directly.
        return next(iter(outputs.values()))

    formatted = "\n\n".join(f"--- {name} ---\n{output}" for name, output in outputs.items())
    response = get_llm().invoke(MERGER_PROMPT.format(task=task, outputs=formatted))
    return response.content
