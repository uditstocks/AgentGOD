"""Shared configuration for the Dynamic Agent Creator.

Uses the OpenRouter API (OpenAI-compatible) so any model on OpenRouter works.
Set your key in .env:  OPENROUTER_API_KEY=sk-or-...

This is the only provider-aware module: model choice, client construction,
response normalisation and cost accounting all live here.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

PROJECT_DIR = Path(__file__).parent

# Load .env next to this file, so it works no matter where you run from.
# Real environment variables always win over .env values.
load_dotenv(PROJECT_DIR / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

# Default model (override with the MODEL env var).
MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")

# Where generated agent files live while they run.
GENERATED_DIR = PROJECT_DIR / "generated_agents"

# Where agents go if the user chooses to keep them.
INVENTORY_DIR = PROJECT_DIR / "inventory"

# Where the answer to every completed task is archived.
RUNS_DIR = PROJECT_DIR / "runs"

# Isolated interpreter used when a generated agent needs extra pip packages.
AGENT_VENV_DIR = PROJECT_DIR / ".agent_venv"


def _int_env(name: str, default: int) -> int:
    """Read an int from the environment, falling back on anything unparseable."""
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


# How many specialised agents a single plan may contain.
MAX_AGENTS = _int_env("MAX_AGENTS", 4)

# Hard wall-clock limit for one generated agent subprocess.
AGENT_TIMEOUT_SECONDS = _int_env("AGENT_TIMEOUT_SECONDS", 300)

# How many times a failing agent is regenerated from its own error output.
AGENT_REPAIR_ATTEMPTS = _int_env("AGENT_REPAIR_ATTEMPTS", 2)

# How many times malformed generated code is regenerated before giving up.
CODEGEN_ATTEMPTS = _int_env("CODEGEN_ATTEMPTS", 3)

# Per-request LLM limits for the main agent.
LLM_TIMEOUT_SECONDS = _int_env("LLM_TIMEOUT_SECONDS", 60)
LLM_MAX_RETRIES = _int_env("LLM_MAX_RETRIES", 3)

# Upper bound on how much of one upstream result is forwarded to the next stage.
MAX_CHARS_PER_INPUT = _int_env("MAX_CHARS_PER_INPUT", 6000)

# Line prefix a generated agent uses to report token usage on stderr.
USAGE_MARKER = "__AGENT_USAGE__"

# USD per 1M tokens, used only for a rough run-cost estimate.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "anthropic/claude-3.5-haiku": (0.80, 4.00),
}


def require_api_key() -> None:
    """Fail fast with a helpful message if the key is missing."""
    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")


def get_llm(max_tokens: int = 8192, temperature: float = 0.0) -> ChatOpenAI:
    """One place to build the LLM for the main agent.

    temperature defaults to 0: planning and code generation are structural
    decisions and must be reproducible run to run.
    """
    return ChatOpenAI(
        model=MODEL,
        # Field alias: pydantic exposes `max_tokens` under this name.
        max_completion_tokens=max_tokens,
        temperature=temperature,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url=OPENROUTER_BASE_URL,
    )


def response_text(response: Any) -> str:
    """Normalise an LLM reply to plain text.

    langchain-core 1.x may return either a string or a list of content blocks,
    so never touch `.content` directly.
    """
    accessor = getattr(response, "text", None)
    if accessor is not None:
        return str(accessor() if callable(accessor) and not isinstance(accessor, str) else accessor)

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, (str, dict))
        ]
        return "".join(parts)
    return str(content)


@dataclass
class Usage:
    """Running token/cost total for one task."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def record(self, response: Any) -> None:
        """Accumulate usage from a LangChain response, if it reported any."""
        metadata = getattr(response, "usage_metadata", None) or {}
        self.add(int(metadata.get("input_tokens") or 0), int(metadata.get("output_tokens") or 0))

    @property
    def cost_usd(self) -> float | None:
        """Rough cost estimate, or None when the model's price is unknown."""
        price = PRICING_PER_MTOK.get(MODEL)
        if price is None:
            return None
        return (self.input_tokens * price[0] + self.output_tokens * price[1]) / 1_000_000

    def summary(self) -> str:
        cost = self.cost_usd
        money = f" · ~${cost:.4f}" if cost is not None else ""
        return (
            f"{self.calls} LLM calls · {self.input_tokens:,} in / "
            f"{self.output_tokens:,} out tokens{money}"
        )
