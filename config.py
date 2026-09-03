"""Shared configuration for the Dynamic Agent Creator.

Uses the Anthropic Messages API through the official `anthropic` SDK.
Set your key in .env:  ANTHROPIC_API_KEY=sk-ant-...

This is the only provider-aware module: model choice, client construction,
response normalisation and cost accounting all live here.
"""

from __future__ import annotations

import functools
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_DIR = Path(__file__).parent

# Load .env next to this file, so it works no matter where you run from.
# Real environment variables always win over .env values.
load_dotenv(PROJECT_DIR / ".env")

# Generated agents are standard library only, so they cannot import the SDK.
# They POST to this endpoint themselves, which is why the URL and the wire
# version live here, next to the client the main agent uses.
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Default model (override with the MODEL env var).
MODEL = os.getenv("MODEL", "claude-sonnet-5")

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

# How many agents may run at the same time. Parallelism only ever happens
# between agents the plan's dependency graph proves independent, so this is
# a ceiling on subprocesses, not a correctness knob. 1 disables it entirely.
MAX_PARALLEL_AGENTS = max(1, _int_env("MAX_PARALLEL_AGENTS", 4))

# Whether the council - an adversarial critic that cross-examines the merged
# answer before the judge sees it - convenes. 'auto' convenes it only for
# tasks the planner graded deep; 'always' and 'off' do what they say.
COUNCIL = os.getenv("COUNCIL", "auto").strip().lower()

# Whether the one pre-run clarifying question may be asked at all. 'off'
# skips the call entirely for people who always want the run to just start.
CLARIFY = os.getenv("CLARIFY", "auto").strip().lower()

# Hard wall-clock limit for one generated agent subprocess.
AGENT_TIMEOUT_SECONDS = _int_env("AGENT_TIMEOUT_SECONDS", 300)

# How many times a failing agent is regenerated from its own error output.
AGENT_REPAIR_ATTEMPTS = _int_env("AGENT_REPAIR_ATTEMPTS", 2)

# How many times malformed generated code is regenerated before giving up.
CODEGEN_ATTEMPTS = _int_env("CODEGEN_ATTEMPTS", 3)

# How many times a finished answer that does not meet the request may be
# attempted again. Each revision re-runs the agents, so it costs a full round;
# 0 turns self-checking off entirely.
TASK_REVISIONS = _int_env("TASK_REVISIONS", 1)

# Per-request LLM limits for the main agent. The ceiling is generous because
# the model reasons before it answers: a planning or code-generation call
# spends real time thinking before the first output token appears.
LLM_TIMEOUT_SECONDS = _int_env("LLM_TIMEOUT_SECONDS", 120)
LLM_MAX_RETRIES = _int_env("LLM_MAX_RETRIES", 3)
LLM_MAX_TOKENS = _int_env("LLM_MAX_TOKENS", 8192)

# How hard the model works per call: low | medium | high | xhigh | max.
# This is the knob that replaced temperature, which current models reject.
LLM_EFFORT = os.getenv("LLM_EFFORT", "medium")

# The effort scale, weakest first, for comparing two settings.
_EFFORT_SCALE = ("low", "medium", "high", "xhigh", "max")


def effort_for(complexity: str) -> str:
    """How hard every call in a run works, given the planner's grade.

    'simple' drops to low - a translation does not deserve a deliberation.
    'deep' raises to high, but never *lowers* a stronger LLM_EFFORT the user
    set on purpose. Anything else runs at the configured default, so a user
    who never touches this sees exactly the behaviour they configured.
    """
    if complexity == "simple":
        return "low"
    if complexity == "deep":
        configured = _EFFORT_SCALE.index(LLM_EFFORT) if LLM_EFFORT in _EFFORT_SCALE else 0
        return _EFFORT_SCALE[max(configured, _EFFORT_SCALE.index("high"))]
    return LLM_EFFORT

# Upper bound on how much of one upstream result is forwarded to the next stage.
MAX_CHARS_PER_INPUT = _int_env("MAX_CHARS_PER_INPUT", 6000)

# Web search runs on Anthropic's servers, not here: the model issues the
# queries and reads the results before it ever replies. That is why looking
# something up needs no scraper, no search-provider key, and no new hole in
# codeguard - a generated agent only adds this tool to its own request.
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"

# How many searches one call may run. The ceiling is the cost control: a
# question that needs more than this is a question that needs a better prompt.
WEB_SEARCH_MAX_USES = _int_env("WEB_SEARCH_MAX_USES", 5)

# The server loop pauses after 10 tool iterations and asks to be resumed.
# Resuming is cheap; resuming forever is not.
MAX_CONTINUATIONS = _int_env("MAX_CONTINUATIONS", 4)

# Line prefix a generated agent uses to report token usage on stderr.
USAGE_MARKER = "__AGENT_USAGE__"

# Bumped whenever the trusted runtime in generator.py gains or changes a
# capability. A library agent written against an older runtime is retired and
# rewritten rather than handed back: it cannot call what it never knew about,
# and the planner has no way to tell. Version 2 added web search.
AGENT_RUNTIME_VERSION = 2

# USD per 1M tokens (input, output), used only for a rough run-cost estimate.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
}


def require_api_key() -> None:
    """Fail fast with a helpful message if the key is missing."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")


@functools.lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """The one Anthropic client the main agent uses.

    Cached: the SDK holds a connection pool, and rebuilding it per call throws
    that away. Retries and the timeout are the client's own job, so no caller
    has to reimplement backoff.
    """
    return anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=float(LLM_TIMEOUT_SECONDS),
        max_retries=LLM_MAX_RETRIES,
    )


def web_search_tool() -> dict[str, Any]:
    """The server-side search tool, as a request declares it."""
    return {
        "type": WEB_SEARCH_TOOL_TYPE,
        "name": "web_search",
        "max_uses": WEB_SEARCH_MAX_USES,
    }


def cached_system(text: str) -> list[dict[str, Any]]:
    """A system prompt the API may keep warm between calls.

    The planner's rules and the generator's contract are identical on every
    run - thousands of tokens re-sent and re-billed each time. Marking the
    block as an ephemeral cache breakpoint means the second and later calls
    read it from the provider's cache at a fraction of the price. Nothing
    task-specific may go in here: the cache key is the text itself.
    """
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def complete(
    prompt: str,
    system: str | list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
    usage: Usage | None = None,
    search: bool = False,
    effort: str | None = None,
) -> str:
    """One Messages API call, returned as plain text.

    `system` carries standing instructions; `prompt` carries this call's
    material. Keeping them apart is what stops a long input from diluting the
    rules - the API weighs a system prompt as policy, not as more input.

    With `search=True` the model may look things up before answering. That
    runs server-side, so the reply arrives complete rather than as a tool call
    to service here - except that a long search session pauses partway and
    asks to be resumed, which is what the loop below is for.
    """
    request: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
        "output_config": {"effort": effort or LLM_EFFORT},
    }
    if system:
        request["system"] = system
    if search:
        request["tools"] = [web_search_tool()]

    messages: list[Any] = [{"role": "user", "content": prompt}]
    client = get_client()
    message: Any = None
    for _ in range(MAX_CONTINUATIONS + 1):
        message = client.messages.create(messages=messages, **request)
        if usage is not None:
            usage.record(message)
        if message.stop_reason != "pause_turn":
            break
        # Resume by handing the paused turn straight back. No "continue"
        # message: the server recognises its own trailing tool block and
        # picks up where it stopped.
        messages.append({"role": "assistant", "content": message.content})

    return response_text(message)


_Schema = TypeVar("_Schema", bound=BaseModel)


def complete_structured(
    prompt: str,
    output_format: type[_Schema],
    system: str | list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
    usage: Usage | None = None,
    effort: str | None = None,
) -> _Schema:
    """One call whose reply is constrained to `output_format`, and validated.

    The shape is enforced by the API rather than fished out of prose
    afterwards, so a plan either arrives usable or not at all.
    """
    request: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
        "output_config": {"effort": effort or LLM_EFFORT},
        "output_format": output_format,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        request["system"] = system

    # cast(Any): the typeshed for the SDK lags the live API's parse endpoint.
    message = cast(Any, get_client().messages).parse(**request)
    if usage is not None:
        usage.record(message)

    parsed = getattr(message, "parsed_output", None)
    if parsed is None:
        raise ValueError(f"the model did not return a usable {output_format.__name__}")
    return parsed


def response_text(response: Any) -> str:
    """Normalise a Messages API reply to plain text.

    A reply is a list of content blocks, and on a reasoning model the first of
    them is usually not the answer. Only `text` blocks are, so nothing here
    ever touches `.content[0]` directly.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        elif getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    return "".join(parts)


def estimate_cost(input_tokens: float, output_tokens: float) -> float | None:
    """Rough USD cost for a token count, or None if this model has no price here."""
    price = PRICING_PER_MTOK.get(MODEL)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


@dataclass
class Usage:
    """Running token/cost total for one task.

    Safe to share across threads: independent agents are generated - and
    repaired - in parallel, and a lost update here would silently under-bill
    the run. The lock never leaves this class.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def merge(self, other: Usage) -> None:
        """Fold another total into this one.

        Some calls happen outside the pipeline that owns the run's accounting -
        the clarifying question is asked before `handle_task` exists. Their
        cost is still the user's, so it is added rather than quietly dropped.
        """
        with self._lock:
            self.calls += other.calls
            self.input_tokens += other.input_tokens
            self.output_tokens += other.output_tokens

    def record(self, response: Any) -> None:
        """Accumulate usage from a Messages API reply, if it reported any."""
        reported = getattr(response, "usage", None)
        self.add(
            int(getattr(reported, "input_tokens", 0) or 0),
            int(getattr(reported, "output_tokens", 0) or 0),
        )

    @property
    def cost_usd(self) -> float | None:
        """Rough cost estimate, or None when the model's price is unknown."""
        return estimate_cost(self.input_tokens, self.output_tokens)

    def summary(self) -> str:
        cost = self.cost_usd
        money = f" · ~${cost:.4f}" if cost is not None else ""
        return (
            f"{self.calls} LLM calls · {self.input_tokens:,} in / "
            f"{self.output_tokens:,} out tokens{money}"
        )
