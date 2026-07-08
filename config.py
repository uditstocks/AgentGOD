"""Shared configuration for the Dynamic Agent Creator.

Uses the OpenRouter API (OpenAI-compatible) so any model on OpenRouter works.
Set your key:  OPENROUTER_API_KEY=sk-or-...
"""

import os
from pathlib import Path

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Best GPT model on OpenRouter (override with the MODEL env var if you like).
MODEL = os.getenv("MODEL", "openai/gpt-5.1")

# Where generated agent files live while they run.
GENERATED_DIR = Path(__file__).parent / "generated_agents"

# Where agents go if the user chooses to keep them.
INVENTORY_DIR = Path(__file__).parent / "inventory"


def get_llm(max_tokens: int = 8192) -> ChatOpenAI:
    """One place to build the LLM. Used by the main agent and all generated agents."""
    return ChatOpenAI(
        model=MODEL,
        max_tokens=max_tokens,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )
