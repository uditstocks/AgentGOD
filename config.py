"""Shared configuration for the Dynamic Agent Creator.

Uses the OpenRouter API (OpenAI-compatible) so any model on OpenRouter works.
Set your key in .env:  OPENROUTER_API_KEY=sk-or-...
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env next to this file, so it works no matter where you run from.
# Real environment variables always win over .env values.
load_dotenv(Path(__file__).parent / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Best GPT model on OpenRouter (override with the MODEL env var if you like).
MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")

# Where generated agent files live while they run.
GENERATED_DIR = Path(__file__).parent / "generated_agents"

# Where agents go if the user chooses to keep them.
INVENTORY_DIR = Path(__file__).parent / "inventory"


def require_api_key() -> None:
    """Fail fast with a helpful message if the key is missing."""
    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")


def get_llm(max_tokens: int = 8192) -> ChatOpenAI:
    """One place to build the LLM. Used by the main agent and all generated agents."""
    return ChatOpenAI(
        model=MODEL,
        max_tokens=max_tokens,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )
