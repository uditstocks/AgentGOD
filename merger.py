"""Step 6: Merge all agent outputs into one final response for the user."""

from __future__ import annotations

from config import MAX_CHARS_PER_INPUT, Usage, complete

MERGER_PROMPT = """You are the coordinator of a multi-agent system.
Several specialized agents each completed one part of the user's task.
Merge their outputs into ONE clear, complete final answer for the user.

Rules:
- Obey every explicit constraint in the user's task exactly: length limits,
  word counts, format, structure and tone.
- Do not mention the agents or the process - just deliver the answer.
- Add nothing the agent outputs do not support.

User task:
{task}

Agent outputs:
{outputs}
"""


def _format_outputs(outputs: dict[str, str]) -> str:
    """Label each agent's result and cap it, so the prompt cannot run away."""
    sections = []
    for name, output in outputs.items():
        text = output.strip()
        if len(text) > MAX_CHARS_PER_INPUT:
            text = text[:MAX_CHARS_PER_INPUT] + "\n[...truncated...]"
        sections.append(f"--- {name} ---\n{text}")
    return "\n\n".join(sections)


def merge_outputs(task: str, outputs: dict[str, str], usage: Usage | None = None) -> str:
    """Combine every agent's output into a single final response.

    The merger always runs, including for a single agent: it is the only
    stage that still holds the user's original wording, so it is what
    enforces the task's own formatting constraints.
    """
    if not outputs:
        raise ValueError("no agent produced an output to merge")

    return complete(
        MERGER_PROMPT.format(task=task, outputs=_format_outputs(outputs)),
        usage=usage,
    ).strip()
