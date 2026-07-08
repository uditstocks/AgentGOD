# Architecture — Dynamic Agent Creator

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
          ┌──────────┬───────┼──────────┬─────────────┐
          ▼          ▼       ▼          ▼             ▼
     planner.py  generator.py  executor.py   merger.py   inventory.py
     (1 LLM call) (1 LLM call   (0 LLM calls, (1 LLM call) (0 LLM calls,
      task →       per agent,    subprocess    outputs →    file ops)
      Plan         spec →        per agent)    final
      object)      .py file)                   answer)
```

Numbered lifecycle for one task:

| Step | Module | What happens | LLM used? |
|---|---|---|---|
| 1 | `planner.py` | Task → `Plan` (list of `AgentSpec`) | ✅ structured output |
| 2 | `generator.py` | Each `AgentSpec` → Python source code | ✅ one call per agent |
| 3 | `executor.py` | Source → file in `generated_agents/` | ❌ |
| 4 | `executor.py` | pip-install missing dependencies | ❌ |
| 5 | `executor.py` | Run each agent as a subprocess, in order | ❌ (the *agents* call the LLM) |
| 6 | `merger.py` | All outputs → one final response | ✅ (skipped if 1 agent) |
| 7 | `inventory.py` | Delete files, or move to `inventory/` | ❌ |

---

## 3. Module Responsibilities

One module = one responsibility. If you're adding code and can't decide where
it goes, that usually means it deserves a new module.

### `config.py` — shared setup
- Owns the model name, the OpenRouter base URL, and the two directories
  (`generated_agents/`, `inventory/`).
- `get_llm()` is the **only** place an LLM client is constructed for the main
  agent. Every other module imports it. If you ever switch providers
  (OpenRouter → Anthropic → local Ollama), you change **one function**.

### `planner.py` — task analysis
- Defines the two Pydantic models that are the system's most important data
  structures (see §4).
- `plan_agents(task)` makes one LLM call with
  `llm.with_structured_output(Plan)`, which forces the model to return a
  validated `Plan` object instead of free text. No JSON parsing by hand.
- Design rule enforced by the prompt: *fewest agents possible, one
  responsibility each, execution order matters*.

### `generator.py` — code generation
- `AGENT_TEMPLATE` is the **contract skeleton** every generated agent must
  follow (see §5). It's embedded into the prompt so the LLM copies its shape.
- `generate_agent_code(spec)` = prompt + one LLM call + strip markdown fences.
- Note the double braces `{{ }}` inside `AGENT_TEMPLATE` — that's how you
  escape literal braces in a Python `.format()` string. Classic gotcha.

### `executor.py` — filesystem + processes (no AI here)
- `save_agent_file` — writes `generated_agents/<name>.py`.
- `install_dependencies` — collects deps from all specs, checks each with
  `importlib.util.find_spec`, pip-installs only what's missing.
- `execute_agent` — runs ONE agent via `subprocess.run([sys.executable, file])`,
  feeding JSON on stdin, capturing stdout/stderr, with a 300 s timeout.
  A non-zero exit code becomes an `[agent failed]` string instead of crashing
  the whole pipeline — downstream agents and the merger still run.
- `execute_all` — the pipeline loop: agent N gets a dict of outputs from
  agents 1..N-1.

### `merger.py` — synthesis
- One LLM call that receives the task + all labeled outputs and writes the
  final answer. Short-circuit: with a single agent there's nothing to merge,
  so its output is returned directly (saves a call).

### `inventory.py` — lifecycle end
- `delete_agents` — unlink the files.
- `save_to_inventory` — move files into `inventory/<timestamp>/` and write a
  `TASK.txt` describing what the team was built for.

### `orchestrator.py` — the conductor
- `handle_task(task)` calls the five phases in order and prints progress.
- Contains **no business logic of its own** — it only sequences the modules.
  Keep it that way; it should read like the flow diagram above.

### `main.py` — the user interface
- Validates `OPENROUTER_API_KEY` exists, loops on `input()`, prints the final
  response, then asks *delete or save?*.
- The only module that talks to a human. Everything else is importable and
  testable without a terminal.

---

## 4. Key Data Structures

These two Pydantic models are the "wire format" between planning and
everything downstream. Change them and you change the whole system.

```python
class AgentSpec(BaseModel):
    name: str                 # snake_case, becomes the filename: <name>.py
    role: str                 # one sentence, used in prompts and logs
    instructions: str         # detailed brief the generated agent must follow
    dependencies: list[str]   # extra pip packages (usually empty)

class Plan(BaseModel):
    reasoning: str            # why the task was split this way
    agents: list[AgentSpec]   # IN EXECUTION ORDER (1–4 agents)
```

Why Pydantic + `with_structured_output`?
- The LLM's reply is validated against the schema; malformed output raises
  instead of silently corrupting the pipeline.
- Field descriptions (`Field(description=...)`) are sent to the model as part
  of the schema — they are *prompt engineering*, not just docs.

---

## 5. The Generated-Agent Contract

This is the most important design decision in the project. Every generated
agent — no matter what it does — obeys the same tiny interface:

```
stdin  ──►  JSON {"task": str, "previous_outputs": {agent_name: output, ...}}
stdout ──►  plain-text result
exit 0 ──►  success        exit != 0 ──►  failure (stderr = reason)
```

And structurally, every generated file must contain:

```python
def run(task: str, previous_outputs: dict) -> str: ...

if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())
    print(run(payload["task"], payload["previous_outputs"]))
```

**Why a subprocess contract instead of `import`-ing the generated module?**

| Concern | Subprocess (chosen) | Dynamic import |
|---|---|---|
| A buggy agent crashes the app | ❌ isolated, becomes an error string | ✅ can take the process down |
| Hangs / infinite loops | killed by `timeout=` | needs threads to interrupt |
| Dependency conflicts | contained per run | pollute the main process |
| Simplicity | one `subprocess.run` call | `importlib` gymnastics |

The contract also means agents are **language-agnostic in principle** — a
future version could generate a Node.js agent and the executor wouldn't care,
as long as stdin/stdout behave the same.

**Communication model:** a sequential pipeline. Agent 3 sees
`{"research_agent": "...", "analysis_agent": "..."}` in `previous_outputs`.
There is no shared memory, no message bus — just data passed forward. Simple,
debuggable, and enough for V0.

---

## 6. Where the LLM Is Called (cost/latency map)

For a task planned with N agents:

```
1 call   planner        (structured output)
N calls  generator      (one per agent, biggest token spend)
N calls  inside agents  (each generated agent calls the LLM itself at runtime)
0–1 call merger         (skipped when N == 1)
─────────────────────────
total: 2N + 1 or 2N + 2 calls
```

Keep this in mind when tuning: the generator calls are the largest prompts
(they embed the full template + instructions), and agent runtime calls are
invisible to the main process — they happen inside subprocesses.

---

## 7. Error-Handling Philosophy

- **Planner/generator failures** (bad API key, network, schema mismatch) →
  raise and crash loudly. If planning fails there is nothing sensible to do.
- **Agent runtime failures** → soft-fail. The executor converts a non-zero
  exit into `"[name failed]\n<stderr>"` and the pipeline continues; the merger
  then works with whatever succeeded. Rationale: one broken generated agent
  shouldn't discard the work of three good ones.
- **Dependency install failures** → `subprocess.run(..., check=True)` raises.
  Better to stop than run agents that will import-error anyway.

---

## 8. Directory Layout & Conventions

```
AgentGOD/
├── main.py               # entry point (only module with input()/print UI)
├── orchestrator.py       # sequences the pipeline, no logic of its own
├── planner.py            # LLM: task → Plan
├── generator.py          # LLM: AgentSpec → source code
├── executor.py           # files, pip, subprocesses (no LLM)
├── merger.py             # LLM: outputs → final answer
├── inventory.py          # delete/archive (no LLM)
├── config.py             # model, key, paths — the only provider-aware file
├── generated_agents/     # scratch space, contents are disposable
│   └── <agent_name>.py
├── inventory/            # user-kept teams
│   └── 20260709_183000/
│       ├── research_agent.py
│       └── TASK.txt
├── requirements.txt
├── README.md
└── ARCHITECTURE.md       # this file
```

Conventions to keep while contributing:

1. **Agent name == filename.** `AgentSpec.name` must stay a valid Python
   identifier because it becomes `<name>.py` and a key in `previous_outputs`.
2. **`config.get_llm()` is the single LLM factory.** Never construct
   `ChatOpenAI` anywhere else in the main codebase.
3. **Executor stays AI-free.** It deals in files, processes, and strings only.
   That separation is what makes it unit-testable without an API key.
4. **Prompts live as module-level constants** (`PLANNER_PROMPT`,
   `GENERATOR_PROMPT`, `MERGER_PROMPT`) — easy to find, easy to diff.
5. **`orchestrator.handle_task` returns `(response, paths)`** and lets the
   caller decide about cleanup. Don't make the orchestrator ask questions;
   user interaction belongs in `main.py`.

---

## 9. Ideas / Roadmap (good daily-commit material)

Roughly ordered from easiest to hardest — each one is a self-contained,
committable improvement:

**Small (1 commit each)**
- [ ] `--task "..."` CLI argument so it runs non-interactively (`argparse`).
- [ ] Log each run (task, plan, timings) to a `runs.log` or JSON file.
- [ ] Colored terminal output for the 5 phases.
- [ ] Configurable timeout / model via a `.env` file (`python-dotenv`).
- [ ] Show token/cost estimates per run (OpenRouter returns usage info).

**Medium (a few commits each)**
- [ ] **Retry loop for broken agents**: if an agent exits non-zero, feed the
      code + stderr back to the generator LLM and ask for a fix (max 2 retries).
      This is the single highest-value improvement.
- [ ] Validate generated code before running: `ast.parse()` it, check that
      `run(` exists, regenerate if not.
- [ ] **Inventory reuse**: before planning, search `inventory/*/TASK.txt` for a
      similar past task and offer to rerun that saved team instead of
      generating a new one.
- [ ] Parallel execution: let the planner mark agents as independent, run
      those with `concurrent.futures`, keep dependent ones sequential.
- [ ] Unit tests: `executor.py` and `inventory.py` are pure-Python — test them
      with a fake agent file (a script that echoes stdin). No API key needed.

**Large (multi-day)**
- [ ] Give generated agents **tools** (web search, file reading) instead of a
      single bare LLM call — real LangChain tool-calling agents.
- [ ] Sandbox execution (Docker container per agent) so generated code can't
      touch your filesystem.
- [ ] A planner that outputs a DAG (graph of dependencies) instead of a list,
      with a topological-sort executor.
- [ ] Simple web UI (FastAPI + one HTML page) replacing the CLI.

---

## 10. Concepts You'll Practice Here

For learning purposes, this project touches:

- **LLM orchestration**: prompt design, structured output, multi-step chains.
- **Pydantic**: schemas as both validation *and* prompt engineering.
- **Metaprogramming**: a program that writes, saves, and runs other programs.
- **`subprocess`**: stdin/stdout piping, exit codes, timeouts, isolation.
- **Filesystem hygiene**: `pathlib`, atomic-ish moves, scratch vs. archive dirs.
- **Separation of concerns**: UI / orchestration / AI calls / IO in separate
  modules — notice how only `main.py` talks to the user and only `config.py`
  knows which provider is used.
