# Changelog

## It thinks in graphs, spends effort where it matters, and argues with itself - 2026-08-31

The plan is no longer a queue, the effort is no longer flat, and the answer
no longer leaves the building unchallenged.

### The plan is a graph, and independence runs in parallel

- New `taskgraph.py`. The planner now declares **`depends_on`** per agent, and
  the plan becomes a real dependency DAG: sanitised (self and unknown
  references dropped), cycle-proof (a cycle is broken, never obeyed), and
  topologically ordered so list order and graph order always agree. A plan
  that declares nothing falls back to the old sequential chain - the one
  wiring that is always safe.
- The executor runs the graph in **waves**: everything in a wave has all of
  its inputs before the wave starts, so a wave's agents run at the same time
  on a thread pool - parallelism exists exactly where the graph proves it
  changes nothing about the data each agent sees. Chains still run as chains.
  `MAX_PARALLEL_AGENTS` is the ceiling; `1` disables it.
- Each agent receives **exactly its dependency closure** - its declared
  dependencies and theirs, never "whatever happened to finish first" - and
  the generator writes that exact contract into the agent's prompt.
- **Code generation is parallel too.** Newly planned agents are all written at
  once; their generation calls share nothing. This was the most expensive
  phase of a run, and independence makes it the fastest.
- `Usage` is now thread-safe, so parallel calls cannot under-bill the run.

### The effort dial

- The planner grades every task first: **`simple` / `standard` / `deep`**
  (`Plan.complexity`, graded before the team is designed - field order is the
  prompt). `effort_for()` maps the grade onto every call that follows:
  generation, merging, judging, and the generated agents' own runtime calls,
  which inherit it through `LLM_EFFORT` in their environment. A translation
  no longer deliberates; an analysis no longer rushes. A stronger
  `LLM_EFFORT` the user set on purpose is never lowered.

### The council

- New `council.py`. For tasks graded deep, an **adversarial critic reads the
  merged answer before the judge does** - hunting for unsupported claims,
  reasoning that does not carry its conclusion, and the counter-case a
  competent reviewer would demand. Real faults drive one refinement pass that
  fixes what was named and preserves everything else; a sound answer stands,
  unbilled. Two calls at most, no agent reruns, biased toward acquittal like
  every self-check here. `COUNCIL=auto|always|off` (default `auto`).

### The library curates itself

- Every reused agent now carries a **reliability record**: wins and losses,
  written by the orchestrator after every run it was handed back for. An
  agent that has failed three or more times and lost more than it won is
  **retired automatically** at the next lookup and rebuilt fresh - repair
  fixes a run; this is the longer memory.
- Repairing an already-kept agent is recorded as an **evolution**: the
  generation counter advances and the record resets, because the code that
  earned those losses no longer exists.

### The interface keeps up

- The live board's phase rail shows all **six** phases again
  (`PLAN ▸ FORGE ▸ DEPS ▸ RUN ▸ MERGE ▸ CHECK` - the sixth had been missing,
  so the rail silently degraded to bare numbers).
- The board shows the task's **grade**, a **live spend meter** (calls and
  estimated cost accruing while the run works), **wave lines**
  (`wave 2/3 · research_agent + web_agent in parallel`), a `∥` marker on
  agents genuinely running at once, and the council's cross-examination as a
  visible activity. The final summary names the grade and whether the council
  refined the answer. Plain mode prints the same facts as plain lines.
- New **`/stats`** - the lifetime dashboard from disk: runs archived, agents
  kept, free reuses, evolutions, and a reliability leaderboard. `/library`
  now shows each agent's record (`3W/1L · gen 2`) beside its uses.

### Proof

- 526 tests (was 463), ruff clean, pyright 0 errors (was 6). New suites for
  the graph, the council, the effort dial and the reliability record; the
  parallel paths are pinned by tests that assert each agent saw exactly its
  closure and nothing else.

## It looks things up, and it checks its own work - 2026-08-26

Two things AgentGod could not do: find out anything that happened after the
model's training cutoff, and notice when its own answer did not answer the
question. Both are now part of the loop.

### Web search

- The agent runtime declares Anthropic's **server-side `web_search` tool**.
  The search runs on the API's servers, so a generated agent needs no scraper,
  no second API key, no new dependency and no new hole in `codeguard` - it is
  still standard library only, still ~0.05 s to start. `call_llm(..., search=True)`
  is the whole surface.
- A long search session pauses partway (`stop_reason: "pause_turn"`) and is
  resumed by handing the paused turn straight back, capped by
  `MAX_CONTINUATIONS`. `WEB_SEARCH_MAX_USES` is the per-call cost ceiling.
- **The refusal path is gone.** `Intent.LIVE_DATA`, its regex battery and
  `describe_live_data_limit` are deleted - roughly sixty lines whose only job
  was to say "I have no internet access" before the pipeline ever started.
  Those questions are now work, and the planner is told to give them to an
  agent that looks them up rather than one that answers from memory.

### Judgement

- New `judgment.py`. Before planning, `clarifying_question` decides whether one
  question is worth asking; after merging, `judge` reads the answer back
  against the request and returns either "done" or an instruction the next
  attempt can act on. A rejection with nothing actionable in it is treated as
  done, because retrying on it just bills for the same answer twice.
- A run no longer ends wherever the merger happened to stop. When the answer
  misses the request the agents run again on `revision_task(task, missing)` -
  the original wording with the gap appended, never the critique in place of
  the request. Bounded by `TASK_REVISIONS` (default 1, `0` disables it), and a
  revision that produces nothing usable leaves the first answer standing.
- The clarifying question is asked in `main.py`, before the live display goes
  up and only when stdin is a person: a question nobody can answer is a
  stalled run, not a careful one.

### The library learned about versions

Found by running the thing rather than by reading it. A research task planned
correctly, said "searching the web" in its own reasoning, reused
`research_agent` from the library for free - and answered from memory, because
that agent was written before `search` existed. It ran perfectly and silently
did less than the plan promised.

- `config.AGENT_RUNTIME_VERSION` is stamped onto every agent as it is
  remembered, and `library.up_to_date()` is the second promise the library
  makes, alongside `reusable()`. An agent from an older runtime is retired and
  rewritten instead of handed back.
- Every agent already on disk carries `runtime: 0` and is rebuilt on first use.

### Proof

- 23 new tests: the verdict rules and the revised task's shape, the clarifying
  question's silence cases, the search tool's wire shape (declared only when
  asked for, `max_uses` present, a paused turn resumed with no "continue"
  message, a bounded continuation loop, search results never mistaken for the
  answer), and the runtime-version retirement.
- 463 total, still no API key or network needed.

## Claude, directly - 2026-08-26

OpenRouter is gone. Both halves of the system now talk to the Anthropic
Messages API, and the default model is `claude-sonnet-5`.

### The main agent

- `config.py` drops LangChain and OpenRouter for the official `anthropic` SDK.
  `get_llm()` is replaced by `get_client()` plus two call sites - `complete()`
  for text and `complete_structured()` for a schema-constrained reply - so the
  planner no longer routes structured output through a framework adapter.
- `plan_agents()` uses `output_format=Plan`: the API enforces the schema, so a
  malformed plan is a request error instead of something to salvage.
- **`temperature` is gone.** Current models reject it. `LLM_EFFORT`
  (`low` - `max`, default `medium`) is the knob that replaced it, and
  `LLM_TIMEOUT_SECONDS` rises to 120s because the model reasons before it
  answers.
- Two dependencies removed (`langchain-openai`, `langchain-core`), one added.

### Generated agents

- The trusted runtime POSTs to `/v1/messages` with `x-api-key` and
  `anthropic-version` - still standard library only, still no SDK import, still
  ~0.05 s to start.
- It returns the reply's **text** blocks. A reasoning model puts a thinking
  block first, and returning that instead of the answer would be a silent
  whole-run failure.
- `call_llm(..., temperature=...)` is still accepted and now ignored, so the
  four agents already in the library kept working across the switch. All four
  were re-emitted on the new runtime.
- Token usage on stderr is now `input_tokens` / `output_tokens`, and the
  per-agent cost is priced locally from `config.PRICING_PER_MTOK`: the API
  bills tokens and says nothing about money.

### What an agent is allowed to import

The curated import allowlist was refusing ordinary work. A task that needed a
QR code was refused for `qrcode`; one that needed a CSV was refused for `csv`,
which is in the standard library and was simply absent from a hand-written set
of twenty-four names.

- **The standard library is now allowed wholesale**, minus `BLOCKED_STDLIB` -
  the dozen modules that would undo a check `codeguard` already makes
  elsewhere in the same file (`subprocess`, `multiprocessing`, `ctypes`, `pty`,
  `socket` for shelling out; `importlib`, `runpy`, `pickle`, `marshal` for
  running code chosen at runtime; `shutil` for the filesystem). 199 modules
  available, up from 24. Banning `subprocess` while banning `eval()` is one
  rule, not two.
- **The vetted package list grew from 9 names to 81**, covering web, data,
  documents, images, dates, validation, crypto and databases.
- `ALLOWED_PACKAGES` moved into `codeguard.py` and is now the **single source
  of truth**: `executor.py` installs from it, and the import check derives from
  it. A package that installs but may not be imported can no longer exist.
- **The planner is shown the list.** The refusal was never really an allowlist
  problem - the model named a package it could not see and had therefore
  guessed. The planner prompt now carries all 81 names, and the generator
  prompt tells each agent exactly which of them were installed *for it*.
- The refusal message says why (`shells out`, `not a vetted package`) instead
  of reciting the allowlist, which is now three hundred names and would be
  three hundred names of noise in a repair prompt.

An invented package name is still refused rather than installed. That guard is
the point: a hallucinated name is a supply-chain vector, not a typo to be
helpfully resolved.

### Proof

- 4 new tests exec the trusted runtime with the network stubbed and assert the
  wire shape: auth headers, the required `max_tokens`, no `temperature` on the
  request, thinking blocks skipped, usage line on stderr.
- 13 more cover the allowlist change: ordinary stdlib admitted, blocked stdlib
  still refused, every vetted package importable under its import name, the
  planner prompt carrying every name, and an agent never being offered a
  package that was not installed for it.
- 440 total, still no API key or network needed.

## The interface - 2026-08-25

The pipeline is unchanged; how it *feels* is not. The terminal experience is
redesigned end to end, behind a strict logic/presentation seam.

### One surface, two renderers

- New `events.py`: `handle_task` now emits `TaskEvents` (phase started, plan
  ready, agent building / running / repairing / finished, merge started)
  instead of printing. The pipeline is now completely silent on its own.
- New `ui.py`: `PlainUI`, the reference renderer - every visual as an aligned
  ASCII line - plus `make_ui()`, which picks the renderer once per session.
- New `richui.py`: the full interface on `rich` - a startup wordmark with the
  session's model, ceiling, library and archive counts; a live board with a
  phase rail (`PLAN ▸ FORGE ▸ DEPS ▸ RUN ▸ MERGE`), per-agent status rows
  (queued / writing / reused·free / running / repairing / done / failed, with
  spinners, per-agent timing and token counts); the answer in a panel; a
  compact team/cost/archive transcript; styled error and cancellation states.
  The board is transient: animation during the run, clean scrollback after.

### Degradation is a feature

- `rich` is a soft dependency: missing, the whole product still runs in plain
  text. Pipes, redirects and CI are detected (`Console.is_terminal`) and get
  plain text automatically; `--plain` / `AGENTGOD_PLAIN=1` force it;
  `NO_COLOR` keeps the interface but strips color; the legacy Windows console
  gets an ASCII glyph set.
- Piped answers on Windows arrive with a UTF-8 BOM (PowerShell adds one);
  `ask()` now strips it, so `echo discard | python main.py --task ...` is
  understood instead of silently falling back to the default.
- `inventory.delete_agents` no longer prints; scratch cleanup is bookkeeping,
  and what the user hears about it is the interface's decision.

### Proof

- 64 new tests: the orchestrator's event stream and retry sequencing with
  every collaborator faked, PlainUI's transcript asserted line by line, and
  RichUI driven through a full run (repair and failure included) into a
  capture buffer. 183 total, still no API key or network needed.
- Verified live on Windows 11: rich and plain, interactive and piped,
  success, failure (bad key), cancellation, empty library, and reuse paths.

## Hardening pass - 2026-08-25

Closes all 30 findings from the end-to-end audit of commit `78e6d79`.
Verified on Windows 11, Python 3.12, langchain-core 1.4.9, `openai/gpt-4o-mini`.

### Measured before / after

Same three tasks, same model:

| Task | Before | After | Change |
|---|---|---|---|
| `17 * 23` | 22.5 s | **9.7 s** | −57 % |
| 3-sentence pros/cons | 34.4 s | **15.6 s** | −55 % |
| 200-word investor memo | 22.9 s | 34.1 s | correct output, 3 agents instead of 1 |

Per-agent subprocess time fell from **9.2–13.2 s** to **2.0–6.3 s**: importing
LangChain inside every agent cost ~5.7 s and is gone. Accuracy changed more than
speed did - the memo task now produces **208 words in memo form** where it
previously produced 253 words of bullet-point research, because the agent that
was supposed to write it is no longer silently dropped.

### Correctness

- **Agents received nothing from upstream agents.** The generator prompt never
  said what the `previous_outputs` keys were, so the model invented them
  (`previous_outputs.get('pros_cons', '')` against a real key of
  `research_agent`) and the agent silently ran on the task string alone. The
  prompt now lists the exact upstream names, and agents are told to use `.get()`
  because a failed upstream leaves its key absent.
- **Non-ASCII output crashed the run.** A piped child on Windows wrote cp1252
  while the parent decoded UTF-8, so an em dash produced `stdout=None` and an
  `AttributeError`. UTF-8 is now forced in the child (`PYTHONUTF8`,
  `PYTHONIOENCODING`), the parent decodes with `errors="replace"`, and both
  streams are `None`-guarded.
- **Hangs killed the pipeline.** `subprocess.TimeoutExpired` was uncaught
  despite the docs claiming otherwise; every failure mode is now an
  `AgentResult`.
- **Failed agents poisoned the results.** A crashed agent's traceback was passed
  downstream and into the merger as if it were output. Failures are tracked
  separately, excluded from the merge, and reported to the user with absolute
  paths stripped.
- **The merger was skipped for single-agent plans**, so the task's own
  constraints (word counts, format) went unenforced. It now always runs.
- **The planner promised agents it didn't emit.** `agents` is now declared
  before `reasoning`, so the prose describes the list that was actually
  generated rather than committing to one first.

### Security

- Agent names are sanitised in the `AgentSpec` validator (`"../../../pwned"` →
  `"pwned"`, Windows device names avoided), and `save_agent_file` asserts the
  resolved path stays inside `generated_agents/`.
- Generated code is `ast`-validated by the new `codeguard.py` before it reaches
  disk: allowlisted imports only, no `eval`/`exec`/`compile`/`__import__`, no
  process or filesystem-mutating calls, no `open()` in a write mode.
- pip packages must be on an allowlist and install into an isolated venv
  (`.agent_venv/`) - never the interpreter running the program. Unvetted names
  are refused rather than installed.

### Reliability

- A crashing agent is regenerated from its own stderr (`AGENT_REPAIR_ATTEMPTS`,
  default 2); malformed generated code is regenerated from the validator's
  complaints (`CODEGEN_ATTEMPTS`, default 3).
- One failed task no longer ends the REPL, and cleanup runs in a `finally` so a
  mid-run failure still offers to remove the files it wrote. EOF/Ctrl-C exit
  cleanly.
- LLM calls have an explicit `timeout` and `max_retries`.
- Dependency probing uses `importlib.metadata` distributions instead of import
  names, so `beautifulsoup4`/`pillow`/`scikit-learn`/`pyyaml` are no longer
  reinstalled on every run, and version pins like `requests>=2.31` no longer
  raise `ModuleNotFoundError`.

### Performance & observability

- Generated agents are **standard library only**, calling OpenRouter over plain
  HTTPS. Startup dropped from ~5.7 s to ~0.05 s per agent.
- Token usage and cost are tracked across the planner, generator, merger *and*
  the agent subprocesses (which report usage on stderr), and printed per run.
- Upstream results are forwarded as labelled, length-capped sections instead of
  a raw Python dict repr.

### Maintainability

- New `tests/` suite: **119 tests**, no API key or network required.
- `pyproject.toml` configures pytest and ruff; `ruff check`, `ruff format
  --check` and `pyright` are all clean.
- `requirements.txt` pinned to major ranges; the unused top-level `langchain`
  dependency removed.
- `temperature=0` for planning and code generation, so runs are reproducible.
- `response_text()` normalises replies - no module reads `.content` directly,
  which langchain-core 1.x may return as a list of content blocks.
- README and ARCHITECTURE corrected: they previously claimed timeouts were
  soft-failed, described a `{{ }}` escape that didn't exist, and disagreed with
  the code about the default model and the phase count.
