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

# The model for the run's mechanical decisions - the ones that compare, check
# and classify rather than write or reason about the subject. Reading an
# answer back against a stated word count is not the work the architect model
# is for, and it is billed at half the price here.
#
# Set FAST_MODEL to MODEL to turn the split off entirely.
FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-4-5")

# The model for work the planner graded `deep`. Unset, it is MODEL, so nothing
# changes for anyone who does not ask for it; set it to a stronger model and
# the extra money is spent ONLY on the tasks that were judged to deserve it,
# never on a translation or a one-liner.
DEEP_MODEL = os.getenv("DEEP_MODEL", "").strip() or MODEL

# Which model each job in a run is entitled to. One table, so the question
# "why did this cost that?" has exactly one place to look.
#
#   clarify / judge   mechanical checks - compare, classify, answer yes or no
#   plan / generate   the architect's own reasoning and code writing
#   run               the generated agents doing the actual work
#   merge             writing the answer the user reads
#   council           the adversarial reading, which only ever sits on deep work
_MECHANICAL_ROLES = frozenset({"clarify", "judge"})


def model_for(role: str, complexity: str = "standard") -> str:
    """The model this job should run on, given how the task was graded.

    Mechanical checks are always cheap: their job does not get better on a
    bigger model. Everything else runs on MODEL, and rises to DEEP_MODEL only
    for a task the planner itself called deep - which is how a run can spend
    seriously on hard work without spending anything extra on easy work.
    """
    if role in _MECHANICAL_ROLES:
        return FAST_MODEL
    return DEEP_MODEL if complexity == "deep" else MODEL

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

# Not every model takes the effort dial: the small, fast ones reject the
# parameter outright with a 400 rather than ignoring it. Sending it anyway
# would make the cheap half of a run fail every single time, so the capability
# is named here rather than assumed.
_NO_EFFORT_PREFIXES = ("claude-haiku",)


def supports_effort(model: str) -> bool:
    """Whether `model` accepts output_config.effort at all."""
    return not model.startswith(_NO_EFFORT_PREFIXES)


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

# How many searches one call may run. This is the most expensive dial in the
# product: a search carries a per-use fee of its own, on top of the tokens the
# results cost - one search bills more than an entire simple task. Three is
# enough to check a fact from more than one angle; a question needing more is
# a question needing a better prompt.
WEB_SEARCH_MAX_USES = _int_env("WEB_SEARCH_MAX_USES", 3)

# The server loop pauses after 10 tool iterations and asks to be resumed.
# Resuming is cheap; resuming forever is not.
MAX_CONTINUATIONS = _int_env("MAX_CONTINUATIONS", 4)

# Line prefix a generated agent uses to report token usage on stderr.
USAGE_MARKER = "__AGENT_USAGE__"

# Bumped whenever the trusted runtime in generator.py gains or changes a
# capability. A library agent written against an older runtime is retired and
# rewritten rather than handed back: it cannot call what it never knew about,
# and the planner has no way to tell. Version 2 added web search; version 3
# added deep(), which is what stops an agent paying for a refinement call on
# a task that never warranted one.
AGENT_RUNTIME_VERSION = 3

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
    model: str | None = None,
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
    chosen = model or MODEL
    request: dict[str, Any] = {
        "model": chosen,
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
    }
    if supports_effort(chosen):
        request["output_config"] = {"effort": effort or LLM_EFFORT}
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
            usage.record(message, model=chosen)
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
    model: str | None = None,
) -> _Schema:
    """One call whose reply is constrained to `output_format`, and validated.

    The shape is enforced by the API rather than fished out of prose
    afterwards, so a plan either arrives usable or not at all.
    """
    chosen = model or MODEL
    request: dict[str, Any] = {
        "model": chosen,
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
        "output_format": output_format,
        "messages": [{"role": "user", "content": prompt}],
    }
    if supports_effort(chosen):
        request["output_config"] = {"effort": effort or LLM_EFFORT}
    if system:
        request["system"] = system

    # cast(Any): the typeshed for the SDK lags the live API's parse endpoint.
    message = cast(Any, get_client().messages).parse(**request)
    if usage is not None:
        usage.record(message, model=chosen)

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


# What the provider charges for cached input, as multiples of the input rate:
# writing a cache entry costs a premium, reading one is nearly free. Ignoring
# these would make every cost estimate wrong in both directions - a cache read
# is not free, and a cache write is not the plain input price.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


def estimate_cost(
    input_tokens: float,
    output_tokens: float,
    cache_write_tokens: float = 0.0,
    cache_read_tokens: float = 0.0,
    model: str | None = None,
) -> float | None:
    """Rough USD cost for a token count, or None if this model has no price here.

    `model` names the model that actually produced these tokens. A run now
    spends across two of them, and pricing every call at the architect's rate
    would overstate what the cheap ones cost.
    """
    price = PRICING_PER_MTOK.get(model or MODEL)
    if price is None:
        return None
    billed_input = (
        input_tokens
        + cache_write_tokens * CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * CACHE_READ_MULTIPLIER
    )
    return (billed_input * price[0] + output_tokens * price[1]) / 1_000_000


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
    # Cached input is billed on its own terms and is reported separately by
    # the API - it is NOT included in input_tokens, so a run that cached well
    # would otherwise look like a run that sent almost nothing.
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    # Money is accumulated as it is spent, at the rate of the model that
    # actually produced each call. A run spends across two models now, so a
    # single total priced at the architect's rate would simply be wrong.
    cost: float = 0.0
    priced_calls: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def add(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
        model: str | None = None,
    ) -> None:
        spent = estimate_cost(
            input_tokens, output_tokens, cache_write_tokens, cache_read_tokens, model
        )
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cache_write_tokens += cache_write_tokens
            self.cache_read_tokens += cache_read_tokens
            if spent is not None:
                self.cost += spent
                self.priced_calls += 1

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
            self.cache_write_tokens += other.cache_write_tokens
            self.cache_read_tokens += other.cache_read_tokens
            self.cost += other.cost
            self.priced_calls += other.priced_calls

    def record(self, response: Any, model: str | None = None) -> None:
        """Accumulate usage from a Messages API reply, if it reported any."""
        reported = getattr(response, "usage", None)
        self.add(
            int(getattr(reported, "input_tokens", 0) or 0),
            int(getattr(reported, "output_tokens", 0) or 0),
            int(getattr(reported, "cache_creation_input_tokens", 0) or 0),
            int(getattr(reported, "cache_read_input_tokens", 0) or 0),
            model=model,
        )

    @property
    def cost_usd(self) -> float | None:
        """Rough cost estimate, or None when no call could be priced."""
        return self.cost if self.priced_calls else None

    def summary(self) -> str:
        cost = self.cost_usd
        money = f" · ~${cost:.4f}" if cost is not None else ""
        cached = f" · {self.cache_read_tokens:,} cached" if self.cache_read_tokens else ""
        return (
            f"{self.calls} LLM calls · {self.input_tokens:,} in / "
            f"{self.output_tokens:,} out tokens{cached}{money}"
        )
