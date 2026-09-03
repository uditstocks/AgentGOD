# Architecture - Dynamic Agent Creator

This document explains how the system is designed, why it is designed that way,
and everything you need to know to extend or rebuild it by hand.

---

## 1. The Core Idea

There is **one permanent agent** (the "Main Agent"). It never answers the user's
task itself. Its only job is to **engineer other agents**:

- decide what team of agents a task needs,
- write their Python code,
- run them,
- combine their results,
- read the answer back against the request, and try again if it falls short,
- then delete or archive them.

Think of it as a *factory*, not a *worker*. Generated agents are disposable,
single-purpose programs that exist only for the lifetime of one task
(unless the user saves them to inventory).

---

## 2. High-Level Flow

```
                            USER
                             │  task (string)
                             ▼
                    ┌─────────────────┐
                    │   main.py       │  CLI loop, cleanup prompt
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ orchestrator.py │  THE permanent Main Agent
                    └────────┬────────┘
       ┌──────────┬──────────┼──────────┬────────────┬────────────┐
       ▼          ▼          ▼          ▼            ▼            ▼
  planner.py  generator.py executor.py  merger.py  judgment.py inventory.py
  (1 LLM call) (1+ calls    (0 LLM calls,(1 LLM call)(1 LLM call,(0 LLM calls,
   task →       per agent,   subprocess   outputs →   + a rerun   file ops)
   Plan)        spec →       per agent)   final       if the
                .py file)                 answer      answer is
                    │                                 short)
                    ▼
              codeguard.py   (0 LLM calls - validates generated code)
```

Numbered lifecycle for one task:

| Step | Module | What happens | LLM used? |
|---|---|---|---|
| 1 | `planner.py` | Task → `Plan`: a complexity grade and a dependency DAG of `AgentSpec`s | ✅ structured output |
| 1b | `taskgraph.py` | Sanitise the declared graph; order it; derive waves and closures | ❌ |
| 2 | `generator.py` | Each `AgentSpec` → Python source, **all new agents generated in parallel** | ✅ one call per agent (more if rejected) |
| 2b | `codeguard.py` | Reject unsafe/malformed source before it lands | ❌ |
| 3 | `executor.py` | Source → file in `generated_agents/` | ❌ |
| 4 | `executor.py` | Install vetted pip packages into an isolated venv | ❌ |
| 5 | `executor.py` | Run the agents wave by wave - independent agents in parallel, dependent ones in sequence | ❌ (the *agents* call the LLM) |
| 6 | `merger.py` | All outputs → one final response | ✅ always |
| 6b | `council.py` | Deep tasks only: an adversarial critic challenges the answer; real faults drive one refinement | ✅ one call, +1 when flawed |
| 6c | `judgment.py` | Answer checked against the request; agents rerun if short | ✅ one call, plus a full round per revision |
| 7 | `library.py` | Remember each new agent; record each reused agent's win or loss | ❌ |
| 8 | `runlog.py` | Archive the answer to `runs/` | ❌ |
| 9 | `inventory.py` | Clear the scratch copies | ❌ |

Every LLM call in the run - and every generated agent's own calls, through
`LLM_EFFORT` in its environment - runs at the effort the planner's grade
selected: `simple` drops to `low`, `deep` raises to at least `high`, and
`standard` runs at whatever the user configured (`config.effort_for`).

---

## 3. Module Responsibilities

One module = one responsibility. If you're adding code and can't decide where
it goes, that usually means it deserves a new module.

### `config.py` - shared setup
- Loads `.env`, then owns the model name, the Anthropic endpoint, the
  directories, and every tunable limit (agent timeout, retry counts, plan size).
- `get_client()` is the **only** place an SDK client is constructed for the main
  agent, and `complete()` / `complete_structured()` are the only two ways to
  reach the model. There is no temperature: current models reject it, so
  `LLM_EFFORT` (`low` - `max`) is what paces a call instead.
- `response_text()` normalises a reply to `str`. A reply is a list of content
  blocks and on a reasoning model the first one is the thinking, not the
  answer, so **never touch `.content[0]` directly**.
- `Usage` accumulates tokens and estimates cost.

### `planner.py` - task analysis
- Defines the two Pydantic models that are the system's most important data
  structures (see §4).
- Grades the task first (`Plan.complexity`: simple / standard / deep) -
  declared before `agents` so the model sizes the work before designing the
  team - and declares each agent's `depends_on`, which is what makes selective
  parallelism possible.
- `plan_agents(task)` makes one call through `complete_structured(...)`, which
  constrains the reply to the `Plan` schema with `output_format`. The shape is
  enforced by the API, so a malformed plan is a request error rather than
  something to salvage afterwards.
- **The plan is a trust boundary.** An `AgentSpec.name` becomes a filename and a
  dict key, so it is sanitised to a safe snake_case identifier
  (`"../../../pwned"` → `"pwned"`), Windows device names are avoided, duplicates
  are disambiguated, and the agent count is bounded by the schema - not merely
  requested in the prompt.

### `taskgraph.py` - the shape of the plan (no AI, no I/O)
- Pure functions over anything with `.name` and `.depends_on`.
- Declared dependencies are untrusted like everything else an LLM writes:
  self and unknown references are dropped, a cycle is broken rather than
  obeyed, and a plan that declares nothing falls back to the sequential
  chain every plan had before dependencies existed - an empty graph means
  the planner never thought about it, not independence.
- `topological_order` makes list order and graph order agree; `waves` groups
  the plan into rounds that may each run in parallel; `dependency_closure`
  computes exactly which upstream outputs one agent receives - its declared
  dependencies and theirs, never "whatever finished first", which would make
  runs unrepeatable.

### `generator.py` - code generation
- `AGENT_HEADER` is a **fixed, trusted runtime** - imports, the Messages API call,
  the stdin/stdout contract. The LLM never writes it, so it cannot drift.
- The LLM writes only `run()` (plus small helpers). `assemble_agent()` splices
  that into the header. This is why generated agents are *always* structurally
  correct and *always* standard-library only.
- The generator prompt states the **exact `previous_outputs` keys** this agent
  will receive. Without that the model invents key names, `.get()` returns the
  default, and the agent silently runs on the task string alone.
- `generate_agent_code()` retries on rejection, feeding `codeguard`'s complaints
  back to the model. Malformed code never reaches disk.

### `codeguard.py` - static validation (no AI, no I/O)
- `ast.parse` the source; confirm `run(task, previous_outputs)` exists with the
  right arity and is not async.
- Imports are an **allowlist**, not a denylist - an unknown module is refused.
  The standard library is allowed wholesale minus `BLOCKED_STDLIB`, the dozen
  modules that would undo a check made elsewhere in the same file (`subprocess`
  and friends for shelling out, `importlib`/`pickle` for running code chosen at
  runtime, `shutil` for the filesystem). A curated subset was the wrong shape:
  it refused `csv`, `sqlite3` and `zipfile` for no reason but absence.
- `ALLOWED_PACKAGES` (pip name → import name) is the **single source of truth**
  for third-party code. `executor.py` installs from it and the import check
  derives from it, so an installable-but-unimportable package cannot exist.
- Refuses `eval`/`exec`/`compile`/`__import__`, process and filesystem-mutating
  attribute calls (`os.system`, `os.remove`, …), and `open()` in any write mode.
- Fully unit-testable without an API key.

### `executor.py` - filesystem + processes (no AI here)
- `save_agent_file` - writes `generated_agents/<name>.py`, asserting the
  resolved path is still inside that directory.
- `install_dependencies` - only packages in `codeguard.ALLOWED_PACKAGES` are
  installed, and they go into an isolated venv (`.agent_venv/`), never the
  interpreter running this program. Anything else is refused and reported.
  Installed *distributions* are probed via `importlib.metadata`, because pip
  names and import names differ (`beautifulsoup4` → `bs4`).
- `execute_agent` - runs ONE agent via `subprocess.run`, feeding JSON on stdin.
  **UTF-8 is forced in both directions** (`PYTHONUTF8`/`PYTHONIOENCODING` in the
  child, `errors="replace"` in the parent); without it a piped child on Windows
  writes cp1252, the parent decodes UTF-8, and the output is lost. Every failure
  mode - crash, hang, unstartable, silent - becomes an `AgentResult`, never an
  exception.
- `execute_all` - the pipeline loop: agent N gets the outputs of agents 1..N-1.
  **Only successful outputs are forwarded.**

### `judgment.py` - the main agent judging itself
- `clarifying_question(task)` - the one question worth asking before anything is
  spent, or None. Called from `main.py` before the live display goes up, and
  only when stdin is a person: a question nobody can answer is a stalled run,
  not a careful one. Biased hard toward silence - an agent that asks about
  every task is worse than one that never asks.
- `judge(task, answer)` - a `Verdict` of `done` plus, when it is not, `missing`:
  an instruction the next attempt can act on. `missing` is declared before
  `done` so the model examines the answer before it rules on it.
- `revision_task(task, missing)` - the original wording with the gap appended.
  Replacing the task with the critique is how a second attempt drifts onto the
  complaint instead of onto what the user actually asked for.

### `council.py` - the adversarial reading
- The judge checks *compliance*; the council checks *quality*, and only for
  tasks graded deep - exactly the tasks where "meets every stated demand and
  is still shallow" is the failure that matters.
- One critic call names concrete faults (unsupported claims, reasoning that
  does not carry its conclusion, the missing counter-case); if any are real,
  one refinement call fixes what was named and preserves everything else.
  Two calls at most, no agent reruns, and biased toward acquittal: a critic
  that always objects bills every deep run twice and teaches the user to
  ignore it. An empty refinement never replaces the answer it reviewed.
- `COUNCIL=auto|always|off`; `Challenge.weaknesses` is declared before
  `Challenge.flawed` for the same reason `missing` precedes `done` in the
  judge - the model must state what it found before it rules.

### `merger.py` - synthesis
- One LLM call that receives the task + all labelled outputs and writes the
  final answer. It **always runs**, including for a single agent: it is the only
  stage that still holds the user's original wording, so it is what enforces the
  task's own constraints (word counts, format, tone).

### `library.py` - the reusable agent library
- `lookup(name)` returns stored source, or None. A hit skips the generator
  entirely, which is the single largest cost in a run.
- `remember(name, role, source)` stores an agent under `inventory/agents/`.
  The orchestrator never calls it for a newly built agent: those are held in
  `TaskResult.pending` until `main.ask_keep` gets a decision from the user.
  Repairing an already-kept agent is the one case that writes without asking.
- `describe_for_planner()` is injected into the planner prompt, so the model
  prefers an agent that already exists over inventing a near-duplicate.
- **The library curates itself.** `record_outcome` writes a win or a loss
  against every agent handed back from the library; `reliable()` retires one
  that has failed at least three times and lost more than it won, so the next
  task rebuilds it fresh instead of billing for the same crash again. A
  repair that replaces a kept agent's source is an *evolution*
  (`remember(..., evolved=True)`): the generation counter advances and the
  record resets, because the code that earned those losses is gone.
- The `.py` files are the source of truth; `index.json` is a rebuildable
  convenience, so a corrupt index never costs the user their library.
- Reuse is only sound because the generator is forbidden from baking the
  task's subject into an agent. See `generator.GENERATOR_PROMPT`.

### `runlog.py` - the answer outlives the terminal
- Every completed task is written to `runs/<timestamp>_<slug>.md`.
- A failed write returns None rather than raising: the answer is already
  printed and paid for.

### `inventory.py` - scratch cleanup
- `delete_agents` - unlink the working copies, tolerating ones already gone.
  Keeping an agent is `library.remember`'s job, not this module's.

### `orchestrator.py` - the conductor
- `handle_task(task)` calls the six phases in order, reports each notable
  moment to a `TaskEvents` object, and returns a `TaskResult`. It never
  prints: presentation is the caller's problem.
- It owns **retry policy** (`_run_with_repair`): a failed agent is regenerated
  from its own stderr, up to `AGENT_REPAIR_ATTEMPTS` times. The generator
  already has the code and the traceback, so a crash is a repairable event
  rather than lost work.
- It owns **scheduling**: `_for_each_in_waves` drives generation-independent
  work and each execution wave through a thread pool (ceiling:
  `MAX_PARALLEL_AGENTS`), while results are absorbed on the caller's thread
  so shared state is mutated from exactly one place. `_SharedEvents`
  serialises event emission, so two workers can never interleave inside one
  renderer. An agent's `previous_outputs` is always its dependency closure -
  parallel or not, a run is repeatable.
- No other business logic of its own - it should read like the flow diagram.

### `events.py` - the presentation seam
- `TaskEvents` names every notable moment of a run (phase started, plan ready,
  agent building / running / repairing / finished, merge started) as a no-op
  hook. The pipeline emits; whoever is watching overrides.
- The default instance shows nothing, so `handle_task` runs headless - in
  tests, in scripts, inside another program - at zero presentation cost.

### `ui.py` / `richui.py` - the interface itself
- All presentation lives behind one surface. `PlainUI` (in `ui.py`) is the
  reference implementation: every visual the product can show exists as an
  aligned, colorless ASCII line, safe for pipes, CI and redirected output.
- **Three streams, one rule.** The answer is stdout, always. Narration is
  stdout only when a person is watching; into a pipe it moves to stderr.
  Errors are stderr always, and survive `--quiet` - suppressing an answer's
  decoration must never suppress the reason a run died.
- `RichUI` (in `richui.py`) subclasses it and repaints the same surface with
  `rich`: the startup wordmark, a live phase rail, a per-agent status board
  with spinners, the answer panel, and the run transcript. Anything it does
  not override degrades to a plain line instead of an AttributeError.
- `make_ui()` chooses once per session: `AGENTGOD_PLAIN`/`--plain`, a missing
  `rich`, or a non-terminal stdout all select `PlainUI`. `rich` is a *soft*
  dependency on purpose - the product must run before its wardrobe arrives.
- Neither module imports the rest of the project: every fact they display
  (model name, limits, results) arrives as an argument or through an event.

### `cli.py` - the command-line contract (no project imports)
- One argparse surface: the positional task, `--task` (REMAINDER, so task
  text may contain flag-shaped words), `-` for stdin, the output flags
  (`--plain`, `--quiet`, `--json`), the per-invocation overrides
  (`--model`, `--effort`, `--council`) and the keep policy.
- Exit codes are the contract: 0 succeeded, 1 failed, 2 usage, 130 interrupt.
- `apply()` writes overrides into the environment *before* `config` is
  imported, so a flag and a `.env` value take the same path - there is no
  second configuration system to keep in sync.
- Imports nothing from the project, so `--version` and usage errors work
  even before dependencies are installed. `__version__` lives here and
  `pyproject.toml` reads it, so one number is the truth.

### `problems.py` - failures, translated
- `explain(error) -> Problem(headline, advice, technical)`. Exceptions are
  matched by class name across the MRO rather than imported types, so the
  translator never fails on an import and survives an SDK version change.
- Every branch names the thing the user can change: the key, the model, the
  connection, the timeout. `_explain_every_agent_failed` reads the shared
  cause out of the orchestrator's report - four identical timeouts are one
  story, not four.
- The raw text is kept as `technical` so renderers can dim it. The headline
  is never a class name.

### `main.py` - the conversation
- Validates `ANTHROPIC_API_KEY`, loops on the task prompt, hands each run's
  events to the active UI, then asks *keep or discard?*.
- One failed task must not end the session: `handle_task` is wrapped, and
  cleanup runs in a `finally` so a mid-run failure still offers to remove the
  files it wrote. EOF/Ctrl-C end the session cleanly instead of raising.
- The only module that talks to a human - and it talks through `ui`, so it
  stays about the conversation, not the paint.

---

## 4. Key Data Structures

These models are the "wire format" between planning and everything downstream.

```python
class AgentSpec(BaseModel):
    name: str  # sanitised snake_case; becomes <name>.py and a dict key
    role: str  # one sentence, used in prompts and logs
    instructions: str  # detailed brief the generated agent must follow
    dependencies: list[str]  # extra pip packages (usually empty; allowlist-gated)


class Plan(BaseModel):
    agents: list[AgentSpec]  # IN EXECUTION ORDER, 1..MAX_AGENTS (schema-enforced)
    reasoning: str  # describes the list above
```

`agents` is declared **before** `reasoning` on purpose: structured output is
generated in field order, so the model describes the team it actually emitted
instead of committing in prose to agents it then omits.

Why Pydantic + `with_structured_output`?
- The LLM's reply is validated against the schema; malformed output raises
  instead of silently corrupting the pipeline.
- Field descriptions (`Field(description=...)`) are sent to the model as part
  of the schema - they are *prompt engineering*, not just docs.
- Validators are where untrusted model output is made safe, once, for everyone
  downstream.

`handle_task` returns a `TaskResult`: the merged `response`, the `agent_paths`
(so the caller decides about cleanup), any `failures`, and token/cost totals.

---

## 5. The Generated-Agent Contract

This is the most important design decision in the project. Every generated
agent - no matter what it does - obeys the same tiny interface:

```
stdin  ──►  JSON {"task": str, "previous_outputs": {agent_name: output, ...}}
stdout ──►  plain-text result
stderr ──►  diagnostics + a "__AGENT_USAGE__ {...}" token report
exit 0 ──►  success        exit != 0 ──►  failure (stderr = reason)
```

The payload is written with `json.dumps(..., ensure_ascii=True)`, so stdin is
pure ASCII on the wire and cannot be mangled by the child's encoding.

Structurally, every generated file is `AGENT_HEADER` + the model's `run()` +
`AGENT_FOOTER`. The header is **standard library only** (`json`, `os`, `sys`,
`time`, `urllib`, `pathlib`) and POSTs to the Messages API over plain HTTPS -
it does not even import the Anthropic SDK. Two consequences worth knowing:

- **Startup is ~0.05 s, not ~6 s.** Importing a framework inside every agent
  subprocess used to dominate wall-clock time.
- **An archived agent runs on its own.** `api_key()` falls back to a `.env` file
  up the directory tree, so an agent in `inventory/` works with nothing
  installed.

**Why a subprocess contract instead of `import`-ing the generated module?**

| Concern | Subprocess (chosen) | Dynamic import |
|---|---|---|
| A buggy agent crashes the app | ❌ isolated, becomes an `AgentResult` | ✅ can take the process down |
| Hangs / infinite loops | killed by `timeout=`, caught | needs threads to interrupt |
| Dependency conflicts | contained per run | pollute the main process |
| Simplicity | one `subprocess.run` call | `importlib` gymnastics |

The contract also means agents are **language-agnostic in principle** - a
future version could generate a Node.js agent and the executor wouldn't care,
as long as stdin/stdout behave the same.

**Communication model:** a sequential pipeline. Agent 3 sees
`{"research_agent": "...", "analysis_agent": "..."}` in `previous_outputs`, and
the generator was told those exact key names. There is no shared memory and no
message bus - just data passed forward.

---

## 6. Where the LLM Is Called (cost/latency map)

For a task planned with N agents:

```
1 call   planner        (structured output; also grades the task)
N calls  generator      (one per agent; +1 per rejected or repaired attempt;
                         new agents are generated in parallel)
N calls  inside agents  (each generated agent calls the LLM itself at runtime;
                         independent agents run in parallel waves)
1 call   merger         (always - it enforces the task's own constraints)
1 call   judge          (unless TASK_REVISIONS=0; + a full round per revision)
+1..2    council        (deep tasks only: challenge, + refine when flawed)
─────────────────────────
total: 2N + 3 calls, plus retries, revisions and the council's sitting
```

Every call is accounted for. The main agent's tokens come from
`response.usage_metadata`; each generated agent reports its own usage on stderr,
which the executor parses off. `TaskResult.cost_summary()` totals both.

---

## 7. Error-Handling Philosophy

- **Planner/codegen failures** (bad key, network, schema mismatch, code that
  cannot be made valid) → raise. `main.py` catches, reports, and keeps the REPL
  alive; the files written so far are still offered for cleanup.
- **Agent runtime failures** → soft-fail into an `AgentResult`, then **repair**:
  the code and stderr go back to the generator for up to
  `AGENT_REPAIR_ATTEMPTS` regenerations. If it still fails, that agent is
  recorded in `TaskResult.failures` and excluded - its stderr is **never**
  passed downstream or into the merger as if it were a result.
- **Every agent failed** → raise, rather than merge nothing into a confident
  hallucination.
- **Dependency problems** → never fatal. Refused and failed packages are
  reported; the agent that needed them will fail and be repaired or excluded.

---

## 8. Directory Layout & Conventions

```
AgentGOD/
├── main.py               # entry point - the conversation, not the paint
├── ui.py                 # presentation surface + PlainUI + renderer choice
├── richui.py             # the rich renderer (loaded only when rich exists)
├── events.py             # TaskEvents - the pipeline/presentation seam
├── orchestrator.py       # sequences the pipeline + retry policy + scheduling
├── planner.py            # LLM: task → Plan (grade + DAG; the trust boundary)
├── taskgraph.py          # the plan's shape: waves and closures (no LLM)
├── generator.py          # LLM: AgentSpec → source code (trusted header + run())
├── codeguard.py          # static validation of generated code
├── executor.py           # files, pip, subprocesses (no LLM)
├── merger.py             # LLM: outputs → final answer
├── council.py            # LLM: adversarial review of deep answers
├── library.py            # remembers agents for reuse (no LLM)
├── runlog.py             # archives each answer to runs/ (no LLM)
├── inventory.py          # clears scratch copies (no LLM)
├── config.py             # model, key, paths, limits - the only provider-aware file
├── tests/                # pytest suite; needs no API key
├── generated_agents/     # scratch space, contents are disposable
├── inventory/            # user-kept teams
├── .agent_venv/          # isolated interpreter, created only if a task needs pip
├── .env                  # your real key - gitignored, never committed
├── .env.example          # committed template
├── pyproject.toml        # pytest + ruff configuration
├── requirements.txt
├── README.md
└── ARCHITECTURE.md       # this file
```

Conventions to keep while contributing:

1. **Agent name == filename == `previous_outputs` key.** Sanitise it in
   `planner.safe_agent_name`, nowhere else.
2. **`config.complete()` and `config.complete_structured()` are the only two
   ways to reach the model.** Never construct an `anthropic.Anthropic` client
   anywhere else in the main codebase.
3. **Executor and codeguard stay AI-free.** They deal in files, processes,
   strings and syntax trees only. That separation is what makes the test suite
   runnable without an API key.
4. **Prompts live as module-level constants** (`PLANNER_PROMPT`,
   `GENERATOR_PROMPT`, `MERGER_PROMPT`) - easy to find, easy to diff.
5. **Never read `response.content` directly** - use `config.response_text()`.
6. **`orchestrator.handle_task` returns a `TaskResult`** and lets the caller
   decide about cleanup. Don't make the orchestrator ask questions; user
   interaction belongs in `main.py`.
7. **Anything an LLM produces is untrusted input** until a validator has seen
   it. Names go through `safe_agent_name`, code through `codeguard`, packages
   through `ALLOWED_PACKAGES`.

---

## 9. Security Model

Generated code is executed on your machine. The defences, in order:

1. **Names** are sanitised, so a plan cannot write outside `generated_agents/`.
2. **Code** is `ast`-validated: allowlisted imports only, no `eval`/`exec`, no
   shelling out, no writing files.
3. **Packages** are allowlisted and installed into an isolated venv.
4. **Execution** is a subprocess with a hard timeout.

What this does **not** do: it is static validation, not a sandbox. A determined
prompt-injection payload that stays within the allowlist can still make network
calls (agents need HTTPS to reach the API). **Treat the task string as a
trust boundary** - the roadmap's Docker-per-agent item is the real fix.

---

## 10. Ideas / Roadmap

**Small (1 commit each)**
- [ ] `--task "..."` CLI argument so it runs non-interactively (`argparse`).
- [ ] Log each run (task, plan, timings, cost) to a `runs.log` or JSON file.
- [ ] Colored terminal output for the five phases.

**Medium (a few commits each)**
- [ ] **Inventory reuse**: before planning, search `inventory/*/TASK.txt` for a
      similar past task and offer to rerun that saved team.
- [x] **Parallel execution**: the planner marks dependencies (`depends_on`),
      `taskgraph.py` derives waves, and independent agents run on a thread
      pool while dependent ones stay sequential. Silent data loss is designed
      out: each agent receives its dependency closure, and a plan that
      declares nothing falls back to the sequential chain.
- [ ] Stream agent output instead of buffering it to completion.

**Large (multi-day)**
- [x] Give generated agents **tools** (web search) instead of a single bare
      LLM call. File reading still pending.
- [ ] Sandbox execution (Docker container per agent) so generated code can't
      touch your filesystem or network freely.
- [x] A planner that outputs a DAG instead of a list, with a topological-sort
      executor (`taskgraph.py`).
- [ ] Simple web UI (FastAPI + one HTML page) replacing the CLI.

---

## 11. Concepts You'll Practice Here

- **LLM orchestration**: prompt design, structured output, multi-step chains.
- **Pydantic**: schemas as validation, prompt engineering, *and* a security
  boundary.
- **Metaprogramming**: a program that writes, validates, saves and runs other
  programs.
- **`ast`**: treating generated code as data before treating it as code.
- **`subprocess`**: stdin/stdout piping, exit codes, timeouts, encodings.
- **Filesystem hygiene**: `pathlib`, containment checks, scratch vs. archive.
- **Separation of concerns**: UI / orchestration / AI calls / IO in separate
  modules - only `main.py` talks to the user, only `config.py` knows the
  provider, and only `executor.py` touches the filesystem.
