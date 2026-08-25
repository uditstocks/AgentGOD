"""Step 2: Generate the Python source for each planned agent.

The LLM writes only the agent's own logic (``run()`` plus any helpers it needs).
The surrounding runtime - imports, the OpenRouter call, the stdin/stdout
contract - comes from a fixed, trusted header defined here. That removes a
whole class of failure: the boilerplate can no longer drift, and every
generated agent is standard-library only, so it starts in ~0.05 s instead of
paying a ~6 s LangChain import.
"""

from __future__ import annotations

import re

from codeguard import check_agent_source
from config import (
    CODEGEN_ATTEMPTS,
    MAX_CHARS_PER_INPUT,
    MODEL,
    OPENROUTER_CHAT_URL,
    USAGE_MARKER,
    Usage,
    get_llm,
    response_text,
)
from planner import AgentSpec
from topicguard import check_task_is_used, check_topic_leakage, task_subjects

# Placeholders filled in by _render_header(). Deliberately not str.format():
# the header contains literal braces that .format() would choke on.
_MODEL_MARKER = "@@MODEL@@"
_API_URL_MARKER = "@@API_URL@@"
_MAX_CHARS_MARKER = "@@MAX_CHARS@@"
_USAGE_MARKER_MARKER = "@@USAGE_MARKER@@"
_NAME_MARKER = "@@NAME@@"
_ROLE_MARKER = "@@ROLE@@"

# The docstring at the top of every generated agent. It is the first thing
# anyone opening the file reads, so it says what this specific agent is for,
# what its contract is, and what it costs to start - not "auto-generated".
AGENT_DOCSTRING = '''"""@@NAME@@ - @@ROLE@@

Written by AgentGod for one task and kept only because it is topic-agnostic:
the subject arrives at runtime on stdin, so this same file serves any later
task that needs the same capability.

  contract   stdin   JSON {"task": str, "previous_outputs": dict}
             stdout  this agent's result, as plain text
             stderr  one usage line, prefixed @@USAGE_MARKER@@

  runtime    Python standard library only - no install step, no framework,
             no import of AgentGod itself. Starts in ~0.05s and runs
             standalone anywhere a .env with an OpenRouter key is reachable.

  model      @@MODEL@@ (override with the MODEL environment variable)
"""

# ==========================================================================
#  RUNTIME - fixed and identical in every agent AgentGod writes.
#  Not generated: this half is a trusted header, so the boilerplate cannot
#  drift and the model only ever has to write the part below it.
# =========================================================================='''

AGENT_HEADER = '''
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "@@API_URL@@"
MODEL = os.environ.get("MODEL", "@@MODEL@@")
LLM_TIMEOUT_SECONDS = 120
LLM_ATTEMPTS = 3
MAX_CHARS_PER_INPUT = @@MAX_CHARS@@
USAGE_MARKER = "@@USAGE_MARKER@@"


def api_key() -> str:
    """Read the key from the environment, falling back to a nearby .env file.

    The fallback is what lets an archived agent run standalone.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    for folder in Path(__file__).resolve().parents:
        env_file = folder / ".env"
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY is not set.")


def call_llm(prompt, system=None, temperature=None, max_tokens=None):
    """One chat completion against OpenRouter, with retries on transient errors.

    `system` carries the agent's standing identity and rules; `prompt` carries
    this run's material. Separating them is what stops a long upstream result
    from diluting the instructions - the model weighs a system message as
    policy rather than as more input to summarise.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(prompt)})

    payload_out = {"model": MODEL, "messages": messages}
    if temperature is not None:
        payload_out["temperature"] = float(temperature)
    if max_tokens is not None:
        payload_out["max_tokens"] = int(max_tokens)

    body = json.dumps(payload_out).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": "Bearer " + api_key(),
            "Content-Type": "application/json",
        },
    )

    payload = None
    last_error = None
    for attempt in range(LLM_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            last_error = "HTTP %s %s" % (error.code, error.reason)
            if error.code < 500 and error.code != 429:
                raise SystemExit("OpenRouter returned " + last_error)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = "%s: %s" % (type(error).__name__, error)
        if attempt < LLM_ATTEMPTS - 1:
            time.sleep(2 ** attempt)

    if payload is None:
        raise SystemExit("OpenRouter request failed: %s" % last_error)
    if "choices" not in payload:
        raise SystemExit("Unexpected OpenRouter response: %s" % json.dumps(payload)[:400])

    # Token usage goes to stderr; stdout is reserved for the result.
    print(USAGE_MARKER + " " + json.dumps(payload.get("usage") or {}), file=sys.stderr)
    return payload["choices"][0]["message"]["content"] or ""


def format_previous(previous_outputs):
    """Render upstream results as labelled, length-capped sections."""
    if not previous_outputs:
        return "(none)"
    sections = []
    for name, output in previous_outputs.items():
        text = str(output)
        if len(text) > MAX_CHARS_PER_INPUT:
            text = text[:MAX_CHARS_PER_INPUT] + "\\n[...truncated...]"
        sections.append("### " + str(name) + "\\n" + text)
    return "\\n\\n".join(sections)


def upstream(previous_outputs, name, default=""):
    """One upstream agent's result, or `default` if that agent did not finish.

    Reading previous_outputs[name] directly is the most common way a generated
    agent dies: an upstream agent that failed leaves no key at all.
    """
    value = (previous_outputs or {}).get(name)
    return str(value) if value else default


def chunk(text, size=MAX_CHARS_PER_INPUT):
    """Split long material on paragraph boundaries, for map-then-reduce work.

    Truncating loses the end of a document silently. An agent handed more
    material than one call can hold should process it in pieces instead.
    """
    text = str(text)
    if len(text) <= size:
        return [text] if text else []
    pieces = []
    current = ""
    for paragraph in text.split("\\n\\n"):
        if current and len(current) + len(paragraph) + 2 > size:
            pieces.append(current)
            current = paragraph
        else:
            current = current + "\\n\\n" + paragraph if current else paragraph
    if current:
        pieces.append(current)
    return pieces


def constraints(task):
    """The explicit, checkable demands in the task text.

    Length limits and formats are the constraints a merged answer most often
    breaks, so they are pulled out and restated to the model as rules rather
    than left buried in a sentence.
    """
    found = []
    limits = re.findall(r"(?:in |under |max(?:imum)? |no more than |about |~)?"
                        r"(\\d{1,5})[- ](word|words|character|characters|sentence|"
                        r"sentences|line|lines|paragraph|paragraphs|bullet|bullets|"
                        r"point|points|item|items)", task.lower())
    for amount, unit in limits:
        found.append("about " + amount + " " + unit)
    for shape in ("bullet", "table", "json", "markdown", "outline", "numbered list",
                  "step by step", "haiku", "email", "memo"):
        if shape in task.lower():
            found.append("output shape: " + shape)
    return found


def word_count(text):
    """How many words a draft actually has, for checking a stated limit."""
    return len(str(text).split())
'''

# The divider that separates the trusted runtime from the generated logic, so
# a reader can see at a glance which half a machine wrote.
AGENT_BODY_BANNER = """
# ==========================================================================
#  AGENT - the specialist itself.
#  Everything below this line was written for this one capability. It may
#  describe its role, and nothing about any particular subject: the subject
#  arrives in `task`, which is what lets this file be reused.
# ==========================================================================
"""

AGENT_FOOTER = """
# ==========================================================================
#  ENTRY POINT - the stdin/stdout contract, honoured identically everywhere.
# ==========================================================================

if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())
    print(run(payload["task"], payload.get("previous_outputs") or {}))
"""

GENERATOR_PROMPT = """You are a code generator for a multi-agent system.
Write the logic for ONE specialized agent - a professional-grade one.

Agent name: {name}
Agent role: {role}
Agent instructions:
{instructions}

Write ONLY a top-level function with this exact signature, plus any small
helper functions it needs:

    def run(task: str, previous_outputs: dict) -> str:
        ...

Already defined in the module - use them, never redefine them, and do NOT
write any import statements or an "if __name__" block:

    call_llm(prompt, system=None, temperature=None, max_tokens=None) -> str
        One LLM call. Put the agent's standing identity and rules in `system`,
        and this run's material in `prompt`. Use temperature 0.0-0.2 for
        analysis, extraction and code; 0.7-0.9 for creative writing.
    format_previous(previous_outputs) -> str      # all upstream results, labelled
    upstream(previous_outputs, name, default="") -> str   # one result, safely
    chunk(text, size=...) -> list[str]            # split long material
    constraints(task) -> list[str]                # length/format demands in the task
    word_count(text) -> int                       # for checking a stated limit

{upstream_contract}

BUILD A REAL SPECIALIST, NOT A ONE-LINE WRAPPER.

1. Give it a system message that establishes an expert identity, its single
   responsibility, its standards, and what it must never do (invent facts,
   pad, editorialise, mention the process).
2. Build the user prompt with labelled sections, not one run-on sentence:
       OBJECTIVE / MATERIAL / CONSTRAINTS / OUTPUT FORMAT
   The OBJECTIVE section MUST contain the `task` argument itself. Not a
   description of it, not a paraphrase you wrote - the variable. An agent
   whose prompt does not contain `task` cannot see what the user asked and
   returns the same thing every time; it is rejected.
3. Honour the task's own demands. Call constraints(task) and state whatever it
   returns in the CONSTRAINTS section, so a word limit or a required format
   reaches the model as a rule instead of being buried in prose.
4. Handle missing input. If an upstream agent failed, its key is absent - say
   so in the prompt and work from what is there rather than pretending.
5. Handle long material. If the material is longer than one call can hold,
   use chunk() and process the pieces, then combine - never silently truncate.
6. Where quality genuinely benefits (writing, analysis, code), draft and then
   make ONE improvement pass over the draft against the task's constraints.
   Do not do this for simple extraction or translation; two calls cost twice.
7. Return clean final text: no preamble, no "Here is", no meta-commentary.

REUSABLE - the rule this agent is rejected for breaking:
This agent is kept and run again on completely different subjects. Describe
only its ROLE in every string literal you write. Never hardcode the current
task's subject, topic, names, numbers, language or domain - every one of those
must arrive through the `task` argument at runtime.
    Write:  "You are a research agent. Gather key facts for the task below."
    Not:    "You are researching the electric scooter industry."
    Write:  "Translate the supplied text into the language the task names."
    Not:    "Translate the phrase into French."

Other rules:
- Standard library only. No imports at all unless truly unavoidable.
- No file writing, no subprocess, no eval.
- Readable and self-contained. Every line must earn its place.
- Output ONLY Python code. No explanations, no markdown fences.
"""

_FIRST_AGENT_CONTRACT = """This is the FIRST agent, so `previous_outputs` is an empty dict.
Work from the task alone and do not read any key from it."""

_UPSTREAM_CONTRACT = """`previous_outputs` is a dict whose keys are EXACTLY these upstream agent names:
{names}
Never invent a key. Read one specific result with
previous_outputs.get("<exact name>", ""), or include them all with
format_previous(previous_outputs). Reading a key that is not in the list above
silently loses that agent's work, so use the exact names. Use .get() rather than
[...]: an upstream agent that failed leaves its key absent."""

_REPAIR_PROMPT = """
The previous attempt was rejected. Fix these problems and return the corrected code:
{problems}
"""

# The planner names the agent's role for today's task ("gather facts about the
# benefits of logging"), and the generator is asked to encode that role - so
# the model is, in effect, being instructed to hardcode the subject. Listing
# the offending words outright is what breaks that: a rule the model kept
# failing becomes a list it can check itself against before answering.
_SUBJECT_BAN = """
FORBIDDEN WORDS - these come from today's task, and this agent outlives today.
None of them may appear in ANY string literal in your code:
{words}
Where you would have written one, refer to the `task` argument instead. The
role description above may mention them; your code may not.
"""

_FENCED_BLOCK = re.compile(r"`{3}[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)`{3}", re.DOTALL)
_FENCE_LINE = "`" * 3


def _fill(template: str) -> str:
    """Fill a template's placeholders with the live configuration."""
    return (
        template.replace(_API_URL_MARKER, OPENROUTER_CHAT_URL)
        .replace(_MODEL_MARKER, MODEL)
        .replace(_MAX_CHARS_MARKER, str(MAX_CHARS_PER_INPUT))
        .replace(_USAGE_MARKER_MARKER, USAGE_MARKER)
    )


def _render_header() -> str:
    """Fill the trusted header's placeholders with the live configuration."""
    return _fill(AGENT_HEADER)


def _render_docstring(spec: AgentSpec | None) -> str:
    """The per-agent docstring, or a neutral one when there is no spec."""
    name = spec.name if spec is not None else "agent"
    role = (spec.role.strip().rstrip(".") if spec is not None and spec.role else "a generated specialist")
    return _fill(AGENT_DOCSTRING).replace(_NAME_MARKER, name).replace(_ROLE_MARKER, role)


def _strip_code_fences(text: str) -> str:
    """Remove markdown fences, keeping every block the model emitted."""
    blocks = _FENCED_BLOCK.findall(text)
    if blocks:
        return "\n\n".join(block.strip("\r\n") for block in blocks)
    if _FENCE_LINE in text:
        # Unterminated fence: drop the fence lines rather than write them to a .py file.
        return "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith(_FENCE_LINE)
        )
    return text


def assemble_agent(body: str, spec: AgentSpec | None = None) -> str:
    """Wrap a generated ``run()`` body in the trusted runtime.

    The result is one file and only ever one file: docstring, runtime,
    the generated specialist, entry point. It imports nothing from AgentGod,
    so it can be copied out and run on its own.
    """
    return (
        f"{_render_docstring(spec)}\n"
        f"{_render_header()}\n"
        f"{AGENT_BODY_BANNER}\n"
        f"{body.strip()}\n"
        f"{AGENT_FOOTER}"
    )


def _upstream_contract(upstream: list[str]) -> str:
    if not upstream:
        return _FIRST_AGENT_CONTRACT
    return _UPSTREAM_CONTRACT.format(names="\n".join(f'  - "{name}"' for name in upstream))


def generate_agent_code(
    spec: AgentSpec,
    upstream: list[str] | None = None,
    feedback: str | None = None,
    usage: Usage | None = None,
    task: str = "",
) -> str:
    """Return validated, ready-to-run source for one agent.

    Regenerates up to CODEGEN_ATTEMPTS times, feeding the validators'
    complaints back to the model, so malformed code never reaches disk.
    ``feedback`` carries a runtime error from an earlier execution attempt.

    ``task`` is the text this agent is being written for. It is never put in
    the prompt - it is what the agent must NOT contain. topicguard compares
    the generated source against it, because an agent that hardcodes today's
    subject is wrong on every task after this one.
    """
    prompt = GENERATOR_PROMPT.format(
        name=spec.name,
        role=spec.role,
        instructions=spec.instructions,
        upstream_contract=_upstream_contract(upstream or []),
    )
    forbidden = task_subjects(task)
    if forbidden:
        prompt += _SUBJECT_BAN.format(words=", ".join(repr(word) for word in forbidden))
    if feedback:
        prompt += _REPAIR_PROMPT.format(problems=feedback)

    llm = get_llm(max_tokens=4000)
    problems: list[str] = []
    for _ in range(CODEGEN_ATTEMPTS):
        attempt_prompt = prompt
        if problems:
            attempt_prompt += _REPAIR_PROMPT.format(
                problems="\n".join(f"- {problem}" for problem in problems)
            )

        response = llm.invoke(attempt_prompt)
        if usage is not None:
            usage.record(response)
        body = _strip_code_fences(response_text(response)).strip()
        source = assemble_agent(body, spec)
        # Safety first, then reusability: there is no point telling the model
        # its prompt is too specific if the code will not even parse.
        # Safety, then "does it do the job at all", then reusability. There is
        # no point telling the model its prompt is too specific if the code
        # will not parse, or never looks at the task in the first place.
        problems = (
            check_agent_source(source)
            or check_task_is_used(source)
            or check_topic_leakage(source, task)
        )
        if not problems:
            return source

    raise ValueError(
        f"could not generate valid code for {spec.name!r} after {CODEGEN_ATTEMPTS} attempts: "
        + "; ".join(problems)
    )
