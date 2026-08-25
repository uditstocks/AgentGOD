"""What AgentGod actually is, answered from live state instead of guessed.

A question about this system used to be routed to the system: the planner
built a research agent, the research agent asked the model "what can a
multi-agent system do", and the answer came back describing a program that
does not exist. It listed live data retrieval and dataset analysis - two
things this project specifically cannot do.

So none of these answers come from the model. Every number below is read at
the moment it is asked (the configured model, the agents actually in the
library, the runs actually on disk), and every claim is one the code in this
repository can back. The limits are listed as plainly as the strengths,
because a wrong "yes" costs the user a real run to discover.
"""

from __future__ import annotations

from dataclasses import dataclass

# What the generated agents are, in practice, able to specialise into. This
# tracks planner.STANDARD_AGENTS - the vocabulary the planner is steered
# toward - so the promise made here is the one the planner can keep.
SKILLS = (
    ("Research & explain", "gather and lay out what the model knows about a subject"),
    ("Analyse", "pull the key points, themes and implications out of material you supply"),
    ("Summarise", "condense anything to a length or shape you name"),
    ("Write", "reports, memos, posts, emails, descriptions, creative pieces"),
    ("Outline", "structure a piece before it is written"),
    ("Compare", "weigh options against criteria you set"),
    ("Critique", "find the weaknesses, risks and gaps in a piece of work"),
    ("Translate", "move text between languages"),
    ("Code", "write, explain or review code"),
)

# The honest other half. Each of these is enforced somewhere in the code:
# the first by the agent runtime having no tool but one LLM call, the second
# by codeguard, the third by the stdin/stdout text contract, the fourth by
# the executor running agents in a plain loop.
LIMITS = (
    "No live internet. No prices, weather, news, scores, or anything that changed "
    "today - answers come from model knowledge, which has a training cutoff.",
    "No writing to your files, no running your code, no shell commands. Generated "
    "agents are read-only by design and are checked before they are allowed to run.",
    "Text in, text out. No images, audio, video or spreadsheets.",
    "Agents run one after another, not in parallel, so a four-agent task takes "
    "roughly four times as long as a one-agent task.",
)

# Shown when someone asks for a number only the internet has.
LIVE_DATA_ALTERNATIVES = (
    "explain how that number is arrived at, and what moves it",
    "compare the options or players involved",
    "draft the analysis around it, and you paste the current figure in",
)


@dataclass(frozen=True)
class Snapshot:
    """The facts about this installation, as they are right now."""

    model: str = ""
    max_agents: int = 0
    timeout_seconds: int = 0
    library: tuple[tuple[str, str, int], ...] = ()
    runs: int = 0

    @property
    def library_count(self) -> int:
        return len(self.library)

    def status_line(self) -> str:
        kept = f"{self.library_count} agent{'s' if self.library_count != 1 else ''} kept"
        archived = f"{self.runs} run{'s' if self.runs != 1 else ''} archived"
        return (
            f"model {self.model} · up to {self.max_agents} agents per task · "
            f"{kept} · {archived}"
        )


def snapshot() -> Snapshot:
    """Read the live configuration and library.

    Every lookup is defended: a question about the system must still get an
    answer on an installation whose library or runs directory is unreadable.
    """
    from config import AGENT_TIMEOUT_SECONDS, MAX_AGENTS, MODEL, RUNS_DIR

    try:
        from library import catalogue

        entries = tuple((entry.name, entry.role, entry.uses) for entry in catalogue())
    except Exception:
        entries = ()

    try:
        runs = sum(1 for _ in RUNS_DIR.glob("*.md")) if RUNS_DIR.is_dir() else 0
    except OSError:
        runs = 0

    return Snapshot(
        model=MODEL,
        max_agents=MAX_AGENTS,
        timeout_seconds=AGENT_TIMEOUT_SECONDS,
        library=entries,
        runs=runs,
    )


def _bullets(pairs: tuple[tuple[str, str], ...]) -> str:
    return "\n".join(f"- **{title}** - {detail}" for title, detail in pairs)


def _numbered(lines: tuple[str, ...]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def describe_capabilities(state: Snapshot | None = None) -> str:
    """Answer 'what can you do?' - the strengths, the limits, and the numbers."""
    state = state or snapshot()
    return f"""**AgentGod** does not answer you itself. It works out which specialists your
task needs, writes each one as a real Python program, runs them in order, and
merges what they produce into a single answer.

**What it is good at** - multi-step text and knowledge work:

{_bullets(SKILLS)}

Those combine. "Research this, then critique it, then write me 300 words" is
one task, and it becomes three agents that hand work to each other.

**What it cannot do:**

{_numbered(LIMITS)}

**Getting a better answer** - a task with a shape in it builds a better team:

    write a 200-word investor memo on X, cover the main risks, end with a recommendation

gets you further than `tell me about X`.

**This installation:** {state.status_line()}.

Type `/help` for commands, or just describe what you need."""


def describe_identity(state: Snapshot | None = None) -> str:
    """Answer 'who are you?' - what this program is, and how it runs."""
    state = state or snapshot()
    library = (
        ", ".join(f"`{name}`" for name, _role, _uses in state.library)
        if state.library
        else "nothing yet - the first task will start it"
    )
    return f"""I am **AgentGod**: one permanent process that never does your work itself.

Give me a task and I plan the team it needs, write each agent as a real Python
file, check that code before it is allowed to run, execute the agents in order -
each one seeing what the agents before it produced - and merge the results into
one answer. Then I ask whether to keep the agents or let them go.

    plan  ▸  forge  ▸  deps  ▸  run  ▸  merge

Nothing is hard-coded. There is no researcher or writer sitting here waiting for
work; the team for your task is written seconds after you ask for it, and most of
it is deleted a minute later. Agents that earn their keep go into a library and
come back free next time.

**Running on:** {state.model}, through OpenRouter, with up to {state.max_agents}
agents per task and a {state.timeout_seconds}s ceiling on any one of them.

**In the library:** {library}.

Ask me `what can you do?` for the list of things I am actually good at."""


def describe_live_data_limit(question: str = "") -> str:
    """Answer a request for a number only the internet has - immediately.

    This used to cost a full pipeline run: two agents, four LLM calls and
    twenty seconds, all to arrive at "I cannot provide real-time data".
    Saying so up front is both faster and more honest.
    """
    asked = question.strip()
    about = f' — "{asked}"' if asked and len(asked) <= 90 else ""
    alternatives = "\n".join(f"- I can {line}" for line in LIVE_DATA_ALTERNATIVES)
    return f"""I cannot look that up{about}.

I have no internet access, so live prices, weather, news, scores and today's
date are all outside what I can see. Everything I produce comes from model
knowledge, which has a training cutoff - if I guessed, the number would be
confidently wrong, which is worse than no answer.

For the live figure, use a source that actually reads it. Then:

{alternatives}

Paste the current numbers into your task and I will work with them properly."""


def describe_greeting(state: Snapshot | None = None) -> str:
    """A greeting costs nothing and should end with the user knowing what to type."""
    state = state or snapshot()
    return f"""Hello. I am **AgentGod** - describe a task and I will build a small team of
agents to do it, then merge their work into one answer.

Try something with a shape to it:

    write a 200-word brief comparing electric and petrol scooters for a city commuter

`what can you do?` for the full list · `/help` for commands · `quit` to leave.

{state.status_line()}."""


def describe_thanks() -> str:
    return "Any time. Give me another task whenever you are ready."


def describe_farewell() -> str:
    return "Goodbye."
