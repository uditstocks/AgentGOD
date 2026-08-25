<div align="center">

<img src="assets/hero.jpg" alt="AgentGod" width="400">

# AgentGod
https://agent-god-landing.vercel.app/
### One agent that writes other agents - runs them, merges their answers, and deletes them.

![python](https://img.shields.io/badge/python-3.10%2B-334155?style=flat-square&labelColor=0d1117)
![tests](https://img.shields.io/badge/tests-183%20passed-15803d?style=flat-square&labelColor=0d1117)
![ruff](https://img.shields.io/badge/ruff-clean-15803d?style=flat-square&labelColor=0d1117)
![pyright](https://img.shields.io/badge/pyright-0%20errors-15803d?style=flat-square&labelColor=0d1117)
![model](https://img.shields.io/badge/any%20model-OpenRouter-334155?style=flat-square&labelColor=0d1117)

**[Quickstart](#quickstart)** · **[How it works](#how-it-works)** · **[See it run](#see-it-run)** · **[Safety](#containment)** · **[Config](#calibration)** · **[Architecture](ARCHITECTURE.md)**

</div>

---

## The idea, in ten seconds

You give it a task. **It does not answer you.**

It works out which specialists the task actually needs, writes each one as a
real Python file, runs them in order, feeds each the results of the last, and
merges everything into a single answer.

Then it asks whether to delete them.

```
  task ──▶ architect ──▶ writes 1–4 agents ──▶ runs them ──▶ one answer ──▶ released
```

No agent exists before you ask for it. Most no longer exist a minute later.

---

## Quickstart

```bash
git clone https://github.com/uditstocks/AgentGOD.git
cd AgentGOD
pip install -r requirements.txt
```

Get a key from **[openrouter.ai/keys](https://openrouter.ai/keys)**, then:

```bash
cp .env.example .env        # Windows:  Copy-Item .env.example .env
```

```ini
# .env
OPENROUTER_API_KEY=sk-or-...
```

```bash
python main.py
```

That is the whole setup. One key, one command. A real environment variable
outranks the file, so CI can inject the key without one.

---

## See it run

While a task runs, one live board shows the whole machine: which phase is
active, every agent's state, and what is happening *right now* - written,
reused, running, repairing itself, done, failed. It animates while the
system works and disappears when it is finished, leaving only the answer
and a compact transcript.

```
  ────────────────────────────────────────────────────── 22:14 ──

  PLAN  ▸  FORGE  ▸  DEPS  ▸  RUN  ▸  MERGE                00:08
  ⠸ run  translation_agent is working  (1/2)

   ⠼  translation_agent  Translate the phrase into the      3s
   ○  summary_agent      Condense the findings              …

  ╭─  A N S W E R  ──────────────────────────────────────────╮
  │                                                          │
  │   'Good morning' is 'Bonjour' in French and              │
  │   'Guten Morgen' in German.                              │
  │                                                          │
  ╰──────────────────────────────────────────────────────────╯
  team   ●  translation_agent  built   2.8s · 61 tok
         ●  summary_agent      reused  1.5s · 85 tok
  run    10.4s · 5 LLM calls · 1,395 in / 226 out · ~$0.0003
  saved  runs/20260825_221204_translate-the-phrase.md

  ❯ Keep the 1 new agent (translation_agent) for reuse? [keep/discard]:
```

Pipe it, redirect it, run it in CI, or run it without `rich` installed, and
the same run degrades to clean, aligned plain text - same information, no
color, no animation, nothing that fights a log file:

```
$ python main.py --plain --task "In one line, name one benefit of static typing."

[1/5] Planning agents...
  - research_agent: gather facts about the benefits of static typing
  - summary_agent: condense the findings into a single line

[2/5] Generating agent code...
  reused research_agent.py (from library, free)
  reused summary_agent.py (from library, free)

[4/5] Executing agents...
  [1/2] research_agent running...
    done in 7.2s

============================================================
ANSWER
============================================================
Static typing helps catch type-related errors early, improving reliability.

22.1s · 4 LLM calls · 1,031 in / 198 out tokens · ~$0.0003
```

The pipeline itself never prints. It emits **events** (`events.py`), and the
interface (`ui.py` / `richui.py`) decides what they look like - so the same
run is a living board in a terminal and a clean transcript in a pipe.

This is `memo_writer.py` - written during that run, by the machine, verbatim:

```python
def run(task: str, previous_outputs: dict) -> str:
    analysis = previous_outputs["analysis_agent"]
    formatted_previous = format_previous(previous_outputs)

    prompt = (
        f"As a memo writer, your task is to draft a concise 200-word investor memo. "
        f"Here is the user task: {task}. "
        f"Based on the analysis provided: {analysis}, "
        f"and the previous outputs: {formatted_previous}, "
        f"please summarize the findings, highlight the main risks, "
        f"and provide a clear investment recommendation."
    )

    return call_llm(prompt)
```

No human wrote it. It ran for 6.3 seconds and was deleted.

---

## Why disposable

Most systems that call themselves *multi-agent* ship a fixed roster - a
researcher, a writer, a critic - hard-coded and permanently resident, waiting
for work whether or not the work ever arrives.

This one ships nobody.

|  | Conventional framework | AgentGod |
|---|---|---|
| **Roster** | researcher · writer · critic | empty |
| **Defined** | at install time | at the moment you ask |
| **Lifespan** | forever | one task |
| **Idle cost** | permanent | zero |
| **Author** | a human, months ago | the architect, seconds ago |

A single permanent process - the *architect* - decides the team and writes it
from nothing. It never does the work itself.

---

## How it works

```
        task
         │
         ▼
   ┌───────────┐
   │  PLANNER  │   decides how many agents this needs, and what each is for
   └─────┬─────┘
         │  1–4 specifications
         ▼
   ┌───────────┐
   │ GENERATOR │   writes the one function each agent will execute
   └─────┬─────┘
         │  source, per agent
         ▼
   ┌───────────┐
   │ CODEGUARD │   reads that function before it is allowed to run
   └─────┬─────┘
         │  cleared
         ▼
   ┌───────────┐
   │  EXECUTOR │   runs it in isolation, against a clock
   └─────┬─────┘
         │  output - or a reason it failed
         ▼
   ┌───────────┐
   │   MERGER  │   collapses every voice into one answer
   └─────┬─────┘
         ▼
    final response
         │
         ▼
   keep it, or delete it
```

Each stage does one thing and knows nothing about the others. The planner has
never seen a line of Python. The executor has never seen a prompt. A factory
line, not one mind holding the whole problem at once.

---

## Containment

A system that writes its own workers, in a real language, and then runs them
has to answer one question before anything else: **what stops the thing it just
wrote?**

- **Names are not trusted.** Every agent identifier is reduced to a safe token
  before it goes near the filesystem. `../../../pwned` becomes `pwned`.
- **Code is not trusted.** Every generated file is parsed and inspected -
  import by import, call by call - before it is allowed to become a process.
  No `eval`. No `exec`. No shelling out. No writing to disk. Anything outside a
  vetted set of imports is refused, not warned about.
- **Dependencies are not trusted.** A package is installed only if it is on an
  explicit allowlist, and only into an isolated environment - never yours.
- **Time is not unlimited.** Every agent runs against a hard deadline. If it
  fails, its own error becomes the instruction for rewriting it.

The full life of one agent:

```mermaid
stateDiagram-v2
    [*] --> Written
    Written --> Inspected : codeguard reads it
    Inspected --> Written : rejected · rewrite, max 3
    Inspected --> Cleared : passes
    Cleared --> Running
    Running --> Delivered : exit 0
    Running --> Failed : crash · timeout · silence
    Failed --> Written : rewrite from its own error, max 2
    Failed --> Excluded : attempts exhausted
    Delivered --> [*]
    Excluded --> [*]

    classDef ok fill:#0b3d24,stroke:#22c55e,stroke-width:2px,color:#4ade80
    classDef gone fill:#2b1414,stroke:#7f1d1d,stroke-width:1px,color:#fca5a5

    class Cleared ok
    class Delivered ok
    class Failed gone
    class Excluded gone
```

An agent that cannot be repaired is excused, named in the report, and its error
is never passed downstream as though it were a result.

> This is static validation, **not a sandbox**. Treat the task string as a trust
> boundary - don't paste untrusted text into it. Docker-per-agent is on the roadmap.

---

## The contract

Every generated agent, whatever it was built to do, obeys the same interface:

```
stdin   →  {"task": "...", "previous_outputs": {"agent_name": "...", ...}}
stdout  →  plain-text result, nothing else
stderr  →  diagnostics, plus one line of token usage
exit 0  →  success        exit ≠ 0  →  failure, with a reason
```

The keys in `previous_outputs` are never guessed at. Each agent is told, as it
is written, exactly which upstream results it will receive and under what name -
so a summarizer never reaches for data that was never going to arrive.

```mermaid
sequenceDiagram
    autonumber
    participant X as executor
    participant R as research_agent
    participant A as analysis_agent
    participant W as memo_writer
    participant M as merger

    X->>R: task · no upstream
    R-->>X: findings
    X->>A: task · research_agent
    A-->>X: risks
    X->>W: task · research_agent + analysis_agent
    W-->>X: memo

    rect rgb(11, 61, 36)
    X->>M: every output that survived
    M-->>X: one answer
    end
```

No shared memory. No message bus. Nothing travels between agents except what
the one before it actually returned.

### Standard library only

Generated agents import nothing. They speak to the model directly over plain
HTTPS, which is why they start instantly instead of paying a framework import
on every single run:

```diff
  COLD START · measured · Python 3.12 · Windows 11

+ stdlib agent, as shipped ....... 0.07 s   ▏
- framework import, as removed ... 5.70 s   ███████████████████████████████
                                            └── once per agent, in sequence
```

It also means an agent saved to `inventory/` still runs months later, on its
own, with nothing installed.

---

## It gets cheaper every run

An agent is written once and kept. The next task that needs the same
capability gets it back for free - no planning guess, no code generation, no
tokens. Only genuinely new capabilities cost anything.

```diff
  THREE REPORTS, THREE DIFFERENT SUBJECTS

- run 1  solar panels      6 LLM calls   3,665 in / 1,978 out   built research_agent + summary_agent
+ run 2  Brazilian coffee  4 LLM calls   2,472 in / 1,552 out   both reused, free
+ run 3  European e-bikes  4 LLM calls   2,702 in / 1,639 out   both reused, free
```

This works because generated agents are **topic-agnostic by construction**. The
generator is forbidden from writing the current subject into the agent's prompt;
the subject arrives at runtime on stdin. The `research_agent` built for solar
panels contains the word "solar" exactly zero times:

```python
def run(task: str, previous_outputs: dict) -> str:
    prompt = (
        "You are a research agent. Gather key facts for the task below.
"
        f"Task: {task}
"
        f"Previous outputs: {format_previous(previous_outputs)}"
    )
    return call_llm(prompt)
```

**You decide what gets kept.** After every run that had to build something new,
AgentGod shows you what it built and asks `keep/discard`. Nothing enters the
library without your say-so, and reused agents are never re-asked about.

The planner is shown your library before it plans, and is told to prefer an
existing name over inventing a new one. Agents live in `inventory/agents/`,
ranked by how often they have actually been used.

---

## Proof

```diff
  $ pip install pytest ruff pyright

  $ pytest
+ 183 passed

  $ ruff check .
+ All checks passed!

  $ pyright
+ 0 errors, 0 warnings
```

No test needs an API key, a network connection, or the model to be in a good
mood. Every safety claim above is asserted against a real subprocess, a real
syntax tree, and a real filesystem boundary.

---

## Calibration

Everything below has a working default. Only the key is required.

| Variable | Default | Governs |
|---|---|---|
| `OPENROUTER_API_KEY` | - *(required)* | Access to the model |
| `MODEL` | `openai/gpt-4o-mini` | Any chat model OpenRouter can reach |
| `MAX_AGENTS` | `4` | Ceiling on team size |
| `AGENT_TIMEOUT_SECONDS` | `300` | Hard deadline per running agent |
| `AGENT_REPAIR_ATTEMPTS` | `2` | Rewrites allowed for a crashing agent |
| `CODEGEN_ATTEMPTS` | `3` | Rewrites allowed for invalid generated code |
| `LLM_TIMEOUT_SECONDS` | `60` | Deadline for the architect's own calls |
| `LLM_MAX_RETRIES` | `3` | Retries on a transient failure |
| `MAX_CHARS_PER_INPUT` | `6000` | Cap on text forwarded to the next agent |
| `AGENTGOD_PLAIN` | - | Force plain output (same as `--plain`) |
| `NO_COLOR` | - | Keep the interface, strip the color |

The interface needs `rich`, but only wants it: if it is missing, every run
still works in plain text. Pipes, redirects and CI are detected and get
plain text automatically.

---

## The structure

```diff
  main.py             the only file that speaks to a human
  ui.py               the presentation surface - and its plain-text fallback
  richui.py           the live interface: phase rail, agent board, panels
  events.py           the seam: the pipeline emits, the interface draws
  orchestrator.py     the architect - sequences everything, owns retries
  planner.py          task → team specification (and the security boundary)
  generator.py        specification → source code
  codeguard.py        reads that source before it is trusted
  executor.py         files, subprocesses, timeouts - no model calls
  merger.py           every output → one voice
  library.py          remembers every agent, hands it back free
  runlog.py           archives the answer to runs/
  inventory.py        clears the scratch copies

- generated_agents/   where an agent lives while it works       DISPOSABLE
+ inventory/          where an agent goes if you keep it        YOURS
  .agent_venv/        where a borrowed dependency lives         ISOLATED
```

One file, one responsibility. The reasoning behind every decision is in
**[ARCHITECTURE.md](ARCHITECTURE.md)**.

<div align="center">

Every agent this system builds will eventually stop existing.

What it produces before then is yours to keep.

</div>
