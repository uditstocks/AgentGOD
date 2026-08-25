# Changelog

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
