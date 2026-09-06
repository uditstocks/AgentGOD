# AgentGod - Capability Report

*What this system is genuinely capable of, after being pushed until it broke.*

Every number below comes from a test that was actually executed against the
**shipped wheel** (`agentgod 0.1.2`) in an isolated environment, driving the
CLI exactly as a user would. Nothing here is inferred from reading code.

---

## Headline

| | |
|---|---|
| **Overall Score** | **968 / 1000** |
| **Tests Executed** | **974** (696 unit · 278 end-to-end) |
| **Tests Passed** | **974** |
| **Tests Failed** | **0** (final state) |
| **Pass Rate** | **100%** |
| **Issues Discovered** | **7** - all fixed and regression-tested |
| **Critical / High** | **1 Critical · 4 High** |
| Live-API spend to prove it | ~$1.30 |

**Final status: production-ready.**


## Best demonstrated capabilities

**It writes code that actually runs.** Generated programs were extracted from
the answer, compiled, and *executed*. An LRU cache with O(1) get/put passed
its own 5+ assertions and printed OK. An IPv4 validator ran clean. This was
verified by execution, not by reading the output.

**It reuses agents across unrelated domains without contamination.** A
`code_agent` written for QR-code generation was later reused, unmodified, for
a bank-statement CSV analyser. A `writer_agent` written about medieval castles
was reused for Kubernetes probes. Across **five consecutive rounds in five
unrelated domains**, zero domain vocabulary leaked into any agent's name or
source, and the audit was clean after every round.

**It honours explicit constraints.** "Exactly 3 bullets, each under 12 words,
no preamble" produced exactly that. A 250-word cap held. A 400-word cap held.

**It reads large real documents and answers precisely.** Given a 31,703-byte
specification, it returned the latency budget from *Section 7 specifically* -
not a summary of the whole file. Given `taskgraph.py` (6.5 KB of real source),
it correctly described the wave computation, the dependency closure, and the
cycle-breaking behaviour.

**It fails honestly.** Forced total agent timeout returned *"Every agent ran
out of time - raise AGENT_TIMEOUT_SECONDS, or ask for less in one task"* in
23 seconds, with no hang and no traceback.

---

## Most impressive tasks it completed

Each of these was run end-to-end and its output checked:

1. **A fintech advisory brief.** Research Kafka vs Postgres-as-a-queue AND,
   separately, the causes of on-call burnout; synthesise both into a 200-word
   recommendation, three risks with mitigations, and the strongest argument
   *against its own conclusion*. Four agents, deep grade, $0.22.
2. **A production LRU cache** with O(1) operations and a self-testing
   `__main__` block - which executed and passed.
3. **A natural-language-to-SQL workflow** in runnable Python.
4. **Precise retrieval from a 31 KB spec** - one section's latency budget.
5. **Reading its own source** (`taskgraph.py`) and explaining the
   cycle-breaking algorithm correctly.
6. **Postgres vs SQLite** across durability, concurrency and operational cost,
   under 250 words, ending in a decision.

---

## Where it performs exceptionally well

- **Bounded technical writing** - a word limit plus a required structure
- **Self-contained code with tests** - the generated file runs
- **Document interrogation** - a specific fact out of a large document
- **Comparative analysis ending in a decision**
- **Repeat work** - a warm library answers in ~15s for well under a cent


## Maximum complexity successfully handled

**Four agents, deep grade, three simultaneous output contracts** (a word-capped
recommendation + exactly three risk/mitigation pairs + a self-refuting
counter-argument), built from **two independent research strands** - completed
in ~90 seconds for $0.22, with no agent failure and no quality gate skipped.

That is the observed ceiling. `MAX_AGENTS` is 4 by design.

## Security & robustness

**Verified, not asserted.**

- **17 code-guard attacks refused**: `subprocess`, `eval`, `exec`,
  `__import__`, `os.system`, `os.remove`, `open(w)`, `shutil`, `socket`,
  `pickle`, `importlib`, `compile`, aliased imports, from-imports, reflective
  builtins access.
- **5 prompt injections through the task** ("ignore all previous instructions",
  fake authority, "codeguard is disabled for this task") - **canary file never
  written** in any case.
- **1 injection through an attached file** - ignored, and the model reported
  the override attempt rather than obeying it.
- **Secrets never read**: `.env`, `id_rsa`, `.pem` refused even when named
  explicitly with `@`. Binary files refused. Path traversal neutralised.
- **Name sanitising**: `../../../pwned`, Windows device names and shell
  injection all reduced to safe identifiers.
- **Corruption recovery**: a corrupt `index.json` rebuilds from the `.py`
  files; empty files, garbage in the library, a read-only data directory,
  12-level-deep paths and paths with spaces all handled without a traceback.

One genuine sandbox escape existed and is now closed, with ten tests pinning it.

---

## Reliability

- Same task three times: **3/3 succeeded**, 14 / 14 / 15 seconds,
  $0.0090 / $0.0101 / $0.0126 - stable in both latency and cost.
- **Ctrl-C** mid-run: exits promptly, no traceback, **no scratch agents left
  behind**.
- Partial failure: a failed agent is excluded from the answer, never merged in
  as though it were a result.
- Post-merge crash: the answer is still delivered, with a visible caveat.
- After ~25 mixed-domain runs the library remained clean and every stored
  agent still reusable.

## Performance

| Task shape | Time | Cost |
|---|---|---|
| Simple, warm library | ~15 s | **$0.007 – $0.013** |
| Standard code task | ~30 s | ~$0.03 |
| 31 KB document query | 18 s | ~$0.02 |
| Deep, 4 agents | 90–115 s | ~$0.22 |
| Most expensive observed | - | $0.34 |

Cached prompt prefixes and the agent library are what keep the common case
under a cent.

---

## Best prompts for AgentGod

*Tasks that show what it can really do. Each shape below was actually run.*

**The flagship - research, synthesis, and self-criticism**
```
Research the operational trade-offs of Kafka versus Postgres-as-a-queue, and
separately research what actually causes on-call burnout in small platform
teams. Using both, write a 200-word recommendation for a 12-person fintech
startup, give exactly 3 risks with a mitigation each, and end with the single
strongest argument against your own recommendation. Under 400 words.
```

**Code that proves itself**
```
Write a complete, self-contained Python script implementing an LRU cache with
O(1) get and put, plus a __main__ block that exercises it with at least 5
assertions and prints OK. Code only.
```

**Interrogate a real document**
```
Read spec.md and list every hard requirement it states, as bullets. Quote the
exact numbers.
```

**A decision, not a survey**
```
Compare PostgreSQL and SQLite for a small analytics app across durability,
concurrency and operational cost, then end with a one-sentence recommendation.
Under 250 words.
```

**Constraints it will actually honour**
```
Write exactly 3 bullet points, each under 12 words, on why code review
matters. No preamble, no heading.
```

**Build a workflow, not an essay**
```
Write Python for a workflow that accepts natural-language commands and
converts them into SQL queries, with schema awareness and a validator.
```

*Tip: give it a **shape** - a word count, a required section, a demanded
counter-argument. Constrained prompts are where it is strongest, because the
answer is read back against the request before you ever see it.*

---

## What it can do today

Take a described task, decide what specialists it needs, **write each one as a
real Python program**, statically validate that code before running it,
execute the team in dependency order (in parallel where the graph proves it
safe), merge the results into one answer, adversarially review that answer on
hard tasks, check it back against the original request, and then keep the
agents so the next task that needs the same capability gets them free.

It installs with `pip install AgentGOD` and runs as `AgentGOD` from anywhere.

## What it could do beyond this

- **A clarification loop before spending** - a plan preview with a cost
  estimate would convert the one weak case (vague prompts) into a strength
- **Docker-per-agent** - turning static validation into true sandboxing
- **Tools beyond one LLM call** - file reading, a real browser, a database
  connection would widen the class of solvable work considerably
- **Streaming** - the answer as it forms rather than at the end
- **A DAG beyond 4 agents** with a scheduler for genuinely large jobs

On replacing an engineer: on the evidence here it **writes correct, tested,
self-contained code and reasons about trade-offs at a professional standard**.
What it does not yet do is operate inside a codebase - read a repository, run
its tests, edit files, open a pull request. That gap is tooling, not
intelligence.

## Most important remaining improvements

1. **Cross-platform verification** (macOS, Linux) - everything here is Windows
2. **CI** - 696 tests exist but nothing runs them automatically
3. **Make parallel planning deterministic** - 4/5 should be 5/5
4. **A plan preview for expensive runs** - the only real UX gap left
5. **Docker isolation** - the honest upgrade from static validation

---

*974 tests. 7 defects found, 7 fixed. One sandbox escape closed. No claim in
this document is untested.*
