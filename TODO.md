# AgentGOD — Flagship Upgrade TODO

Working list for the current upgrade pass. Items are worked strictly in order
and ticked as they land. Baseline before this pass: 463 tests passing, ruff
clean, pyright clean.

## Features

- [x] **1. Task-graph planning** — the planner declares `depends_on` per agent,
      producing a real dependency DAG instead of a flat list. Sanitised
      (self/unknown refs dropped), cycle-proof, and backward compatible: a plan
      that declares nothing falls back to the old sequential chain.
      - [x] `taskgraph.py` — pure graph module (sanitise, topological order,
            waves, dependency closure)
      - [x] `planner.py` — `depends_on` + `complexity` fields, graph validator,
            prompt rules
      - [x] orchestrator + generator wired to the closure instead of
            "everything before me"

- [x] **2. Selective parallel execution** — independent agents run concurrently
      in waves; dependent ones stay strictly sequential. Parallelism only where
      the graph proves it changes nothing about the data flow.

- [x] **3. Parallel code generation** — all newly-built agents are written
      concurrently (independent LLM calls), collapsing the most expensive phase.

- [x] **4. Adaptive intellect dial** — the planner grades each task
      `simple / standard / deep`; every downstream LLM call **and every
      generated agent** runs at matched reasoning effort. Trivial tasks get
      fast and cheap; deep tasks get maximum rigour.

- [x] **5. The Council** — for deep tasks, an adversarial critic
      cross-examines the merged answer and a refine pass repairs only proven
      faults, before the judge ever sees it. A built-in devil's advocate.

- [x] **6. Self-curating library** — per-agent reliability records
      (wins / losses / generation); an agent that fails more than it works is
      retired automatically; a repair evolves it one generation and resets its
      record.

- [x] **7. Thread-safe accounting + new UI events** — `Usage` safe under
      parallel calls; wave / council moments rendered by both the rich board
      and the plain renderer.

## Flagship visual / UX features

- [x] **V1. Wave-aware live board** — the rich board shows the plan as a
      dependency graph, not a queue: agents grouped by execution wave,
      side-by-side spinners for agents genuinely running in parallel, and a
      `∥ parallel` marker on concurrent waves. Plain renderer prints the wave
      layout too.

- [x] **V2. Complexity badge + council on the board** — the task's grade
      (`simple / standard / deep`) is shown the moment the plan lands, and the
      council's cross-examination is a visible activity ("the council is
      cross-examining the answer"), not silence before the verdict.

- [x] **V3. Live spend ticker** — running total of LLM calls and estimated
      cost on the live board while the run works, so cost is watched, not
      discovered at the end.

- [x] **V4. `/stats` session dashboard** — one command that renders lifetime
      numbers from disk: runs archived, agents kept, reliability leaderboard
      (wins/losses/generation), and how much the library's free reuses saved.

## CLI visual polish

- [x] **C1. Fix the phase rail** — `RAIL` only names 5 phases while the
      pipeline has 6, so the live rail has been silently degrading to bare
      numbers (`1 ▸ 2 ▸ ...`). Add the missing `CHECK` segment so the rail
      reads `PLAN ▸ FORGE ▸ DEPS ▸ RUN ▸ MERGE ▸ CHECK` again.

- [x] **C2. Richer run summary** — the final summary names the task's grade,
      shows a `council` row when the answer was refined. (Generation markers
      landed in `/library` and `/stats` rather than the transcript.)

- [x] **C3. Parallel glyphs** — a `∥` marker (ASCII `||` fallback) on agents
      running concurrently, and wave activity lines
      (`wave 2/3 · research_agent + web_agent in parallel`).

- [x] **C4. `/library` upgrade** — the library listing shows each agent's
      reliability record and generation, not just use counts.

- [x] **8. Tests** — new modules covered, existing suite updated and green,
      ruff + pyright clean.

- [x] **9. Docs** — README, ARCHITECTURE, CHANGELOG, `.env.example` updated
      for everything above.

---

## Outcome

All items complete. Final state: **526 tests passing** (was 463), ruff clean,
**pyright 0 errors** (was 6 at baseline). New modules: `taskgraph.py`,
`council.py`, `tests/test_taskgraph.py`, `tests/test_council.py`,
`tests/test_config.py`. Docs (README, ARCHITECTURE, CHANGELOG, .env.example)
updated to match.
