# Dynamic Agent Creator (V0)

One permanent AI agent that **builds other agents** instead of doing the work itself.

Give it a task, and the main agent will:

1. **Analyze** the task
2. **Decide** how many specialized agents are needed (1–4)
3. **Generate** real Python code for each agent (LangChain + OpenRouter)
4. **Save** every agent as its own file in `generated_agents/`
5. **Install** any missing pip dependencies
6. **Execute** the agents in order — each agent sees the outputs of the ones before it
7. **Merge** all outputs into one final response
8. **Ask you**: delete the agents, or save them to `inventory/` for later reuse

```
User → Main Agent → Plan → Generate Code → Save Files → Install Deps
     → Execute Agents → Collect Outputs → Merge → Final Response
     → "Delete or Save?"
```

## Why

Most multi-agent frameworks ship a fixed set of agents. Here, the agents
**don't exist until you ask for something** — they are written, executed, and
retired (or archived) per task. The main agent is an agent *engineer*, not a worker.

## Setup

```bash
pip install -r requirements.txt
```

Get an API key from [openrouter.ai](https://openrouter.ai/keys) and set it:

```bash
# Windows (PowerShell)
$env:OPENROUTER_API_KEY = "sk-or-..."

# macOS / Linux
export OPENROUTER_API_KEY="sk-or-..."
```

## Run

```bash
python main.py
```

Example:

```
What do you need done?
> Write a short market summary of the electric-scooter industry with pros and cons

[1/5] Analyzing task and planning agents...
  - research_agent: Gathers key facts about the industry
  - analysis_agent: Turns facts into pros and cons
  - writer_agent: Writes the final summary
[2/5] Generating agent code...
[3/5] Checking dependencies...
[4/5] Executing agents...
[5/5] Merging outputs...

FINAL RESPONSE
...

Delete the generated agents or save them to inventory? [delete/save]:
```

## Project layout

| File | Responsibility |
|---|---|
| `main.py` | CLI loop: get task → show answer → ask delete/save |
| `orchestrator.py` | The permanent main agent; drives the whole lifecycle |
| `planner.py` | Task analysis → list of agent blueprints (structured output) |
| `generator.py` | Writes the Python source for each agent |
| `executor.py` | Saves files, installs deps, runs agents as subprocesses |
| `merger.py` | Combines all agent outputs into one answer |
| `inventory.py` | Deletes agents or archives them into `inventory/` |
| `config.py` | Model + OpenRouter setup, shared paths |

## How generated agents communicate

Every generated agent follows the same tiny contract, so the executor can run
any of them identically:

- **Input**: JSON `{"task": ..., "previous_outputs": {...}}` on stdin
- **Output**: plain text result on stdout

Agents run sequentially; agent N receives the outputs of agents 1..N-1.

## Model

Defaults to `openai/gpt-5.1` via the OpenRouter API. Override with an env var:

```bash
MODEL="openai/gpt-5.1-codex" python main.py
```

## V0 limitations (on purpose)

- Sequential execution only (no parallel agents)
- Generated code isn't sandboxed — it runs with your Python interpreter
- Saved inventory agents aren't auto-reused yet (that's the next version)
