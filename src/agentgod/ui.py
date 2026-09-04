"""What AgentGod looks like when the terminal cannot paint.

PlainUI is the reference implementation of the presentation surface:
every visual the project can show exists here as an aligned, colorless
ASCII line, safe for pipes, CI logs, and consoles from another century.
richui.RichUI subclasses it and repaints the same surface with color,
motion and structure - so anything Rich cannot draw degrades to a line
of text instead of an AttributeError.

Three streams, one rule. The ANSWER is the product and goes to stdout,
always - `agentgod "..." | clip` must capture the answer and nothing
else. NARRATION (phases, progress, summaries) goes to stdout only when
stdout is a person; into a pipe it moves to stderr, where progress
belongs. ERRORS and warnings go to stderr, always, and survive --quiet:
suppressing the answer's decoration must never suppress the reason a
run died.

make_ui() chooses the renderer once per session:

- quiet requested -> PlainUI(quiet=True), the answer-only renderer
- AGENTGOD_PLAIN set, `rich` missing, or stdout not a terminal -> PlainUI
- otherwise -> RichUI

The pipeline never chooses. It emits TaskEvents and moves on.

This module imports nothing from the rest of the project except `events`:
every fact it displays (model name, limits, results) arrives as an
argument, so it can be imported before dependencies have been checked.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from typing import TYPE_CHECKING, Any

from events import TaskEvents

if TYPE_CHECKING:
    from pathlib import Path

    from executor import AgentResult, DependencyReport
    from planner import Plan

RULE_WIDTH = 60

HELP_TITLE = "AgentGod - one permanent agent that builds, runs and retires task-specific agents."

# One source of truth for the help screen; each renderer lays it out its own way.
HELP_USAGE = (
    ("agentgod", "interactive session"),
    ('agentgod "write a haiku"', "one task, then exit"),
    ('agentgod --json "..."', "machine-readable result for scripts"),
    ("agentgod library | stats", "the free, offline commands"),
    ("agentgod --plain ...", "no color, no animation"),
)
HELP_ENVIRONMENT = (
    ("AGENTGOD_PLAIN=1", "force plain output (same as --plain)"),
    ("AGENTGOD_KEEP=always", "keep new agents without asking"),
    ("NO_COLOR=1", "keep the interface, strip the color"),
)


def first_line(text: str, limit: int = 120) -> str:
    """The first non-empty line of `text`, capped for one-line display."""
    stripped = text.strip()
    line = stripped.splitlines()[0] if stripped else ""
    return line if len(line) <= limit else line[: max(1, limit - 3)] + "..."


class PlainUI(TaskEvents):
    """Aligned, colorless, animation-free. Safe anywhere text can go."""

    # Class-level default so subclasses that define their own __init__
    # (RichUI does) still satisfy every inherited method.
    quiet = False

    def __init__(self, quiet: bool = False) -> None:
        # Quiet keeps the answer and the errors; everything else is silence.
        self.quiet = quiet

    # -- the three streams -----------------------------------------------

    def _narration_target(self):
        """Where progress belongs right now: the terminal, or stderr.

        Decided per call, not per session, so a test or a caller that swaps
        the streams is honoured immediately.
        """
        try:
            if sys.stdout.isatty():
                return sys.stdout
        except (ValueError, OSError):
            pass
        return sys.stderr

    def _say(self, message: str = "") -> None:
        """Narration: progress, phases, summaries. Suppressed by quiet."""
        if not self.quiet:
            print(message, file=self._narration_target())

    def _tell(self, message: str = "") -> None:
        """The product - answers. Always stdout, so pipes capture it clean."""
        print(message)

    def _alert(self, message: str = "") -> None:
        """Errors and warnings. Always stderr, never silenced by quiet."""
        print(message, file=sys.stderr)

    # -- session ---------------------------------------------------------

    def banner(self, model: str, max_agents: int, library_count: int, run_count: int) -> None:
        library = f"{library_count} kept" if library_count else "empty"
        runs = f"{run_count} archived" if run_count else "none yet"
        self._say("\nAGENTGOD - the creator and destroyer of the Agentworld")
        self._say(
            f"model {model} · up to {max_agents} agents/task · library {library} · runs {runs}"
        )

    def hint(self) -> None:
        self._say(
            "Type a task and press Enter · /help commands · /paste multi-line · 'quit' leaves."
        )

    def first_run_welcome(self) -> None:
        """Shown once, on a session with nothing kept and nothing archived."""
        self._say("\nFirst run - try something with a shape to it:")
        self._say('  write a 150-word brief on why small teams ship faster')
        self._say('  compare Postgres and SQLite for a small web app, end with a pick')

    def help(self) -> None:
        self._tell(HELP_TITLE)
        self._tell("\nUsage:")
        for command, explanation in HELP_USAGE:
            self._tell(f"  {command:<29} {explanation}")
        self._tell("\nEnvironment:")
        for variable, explanation in HELP_ENVIRONMENT:
            self._tell(f"  {variable:<20} {explanation}")

    def input(self, message: str) -> str:
        """Read one reply. EOFError / KeyboardInterrupt pass through to the caller."""
        return input(message)

    def status(self, message: str) -> Any:
        """A heartbeat for a wait with no board on screen yet.

        Returns a context manager (typed Any so renderers may return their
        own kind - rich's Status here, a null context in plain mode). Plain
        mode prints the line once; the rich renderer animates a spinner for
        the duration instead. Either way the first LLM call of a task is no
        longer silent.
        """
        self._say(f"  {message}")
        return contextlib.nullcontext()

    def blank(self) -> None:
        self._say()

    def note(self, message: str) -> None:
        self._say(message)

    def success(self, message: str) -> None:
        self._say(message)

    def warn(self, message: str) -> None:
        self._alert(message)

    def error(self, message: str) -> None:
        self._alert(message)

    def reply(self, text: str) -> None:
        """Answer a conversational turn - no pipeline ran, so there is nothing to report.

        Deliberately not run_succeeded(): there is no team, no cost and no
        archive, and dressing an instant answer up as a completed run would
        misrepresent what happened.
        """
        if not self.quiet:
            self._tell()
        self._tell(text)

    def attachments_read(self, labels: list[str]) -> None:
        """Name every local file that was read into the task.

        This is not decoration. The contents are about to be sent to a model
        provider, and the user is entitled to know that before it happens -
        so it goes to stderr and survives --quiet.
        """
        for label in labels:
            self._alert(f"  read {label}")

    def context_carried(self, previous_task: str) -> None:
        """Say when a follow-up was answered using the exchange before it."""
        self._say(f"  (continuing: {first_line(previous_task, 60)})")

    def farewell(self) -> None:
        self._say("Goodbye.")

    # -- one task, start to finish ---------------------------------------

    def run_started(self, task: str, echo: bool = False) -> None:
        if echo:
            self._say(f"\n> {first_line(task, 200)}")

    def run_succeeded(self, result, saved: Path | None) -> None:
        if self.quiet:
            self._tell(result.response)
            if getattr(result, "caveat", ""):
                self._alert(f"  ! {result.caveat}")
            for name, error in result.failures.items():
                self._alert(f"  ! {name} failed and was excluded: {first_line(error, 120)}")
            return

        self._tell("\n" + "=" * RULE_WIDTH)
        self._tell("ANSWER")
        self._tell("=" * RULE_WIDTH)
        self._tell(result.response)

        if result.failures:
            self._alert("\nAgents that failed (their output was excluded):")
            for name, error in result.failures.items():
                self._alert(f"  - {name}: {first_line(error, 150)}")
        if getattr(result, "caveat", ""):
            self._alert(f"\n! {result.caveat}")

        self._say(f"\n{result.duration_seconds:.1f}s · {result.cost_summary()}")
        if getattr(result, "council_improved", False):
            self._say("The council reviewed the answer and refined it before delivery.")
        if result.reused:
            self._say(f"Reused free from library: {', '.join(result.reused)}")
        if result.built:
            self._say(f"Newly built this run: {', '.join(result.built)}")

        if saved is not None:
            self._say(f"Saved to {display_path(saved)}")
        else:
            self._say("(could not write the run archive)")

    def run_cancelled(self) -> None:
        self._alert(
            "\nCancelled - nothing from this run was kept. Any model call already "
            "in flight will finish and still be billed."
        )

    def run_failed(self, error: BaseException, problem: Any = None) -> None:
        """Say what went wrong in words, with the raw detail dimmed below.

        `problem` is a problems.Problem when the caller translated the
        failure; without one, the old technical line still appears - the
        renderer must never depend on the translation existing.
        """
        if problem is None:
            self._alert(f"\nTask failed: {type(error).__name__}: {error}")
            return
        self._alert(f"\n{problem.headline}")
        self._alert(f"  {problem.advice}")
        if problem.technical:
            self._alert(f"  (detail: {problem.technical})")

    def run_ended(self) -> None:
        """Always called last, even after a crash. Nothing to release here."""

    # -- pipeline events -------------------------------------------------

    def phase_started(self, index: int, total: int, title: str) -> None:
        self._say(f"\n[{index}/{total}] {title}...")

    def plan_ready(self, plan: Plan) -> None:
        self._say(f"  {plan.reasoning}")
        for spec in plan.agents:
            needs = getattr(spec, "depends_on", [])
            wiring = f"  (needs {', '.join(needs)})" if needs else ""
            self._say(f"  - {spec.name}: {spec.role}{wiring}")
        complexity = getattr(plan, "complexity", "standard")
        if complexity != "standard":
            self._say(f"  graded {complexity} - the whole run spends effort accordingly")

    def agent_retired(self, name: str, reason: str) -> None:
        self._say(f"  retired {name} from the library: {reason}")

    def agent_build_started(self, name: str) -> None:
        self._say(f"  writing {name}...")

    def agent_ready(self, name: str, filename: str, reused: bool) -> None:
        if reused:
            self._say(f"  reused {filename} (from library, free)")
        else:
            self._say(f"  wrote {filename}")

    def deps_checked(self, report: DependencyReport) -> None:
        for note in report.problems:
            self._alert(f"  ! {note}")
        if report.installed:
            self._say(f"  installed: {', '.join(report.installed)}")

    def wave_started(self, index: int, total: int, names: list[str]) -> None:
        # A wave of one is just the next agent; only genuine parallelism is
        # worth a line of its own.
        if len(names) > 1:
            self._say(f"  wave {index}/{total}: {' + '.join(names)} in parallel")

    def agent_started(self, name: str, index: int, total: int) -> None:
        self._say(f"  [{index}/{total}] {name} running...")

    def agent_repairing(self, name: str, attempt: int, attempts: int, error: str) -> None:
        self._say(f"    {name} failed ({first_line(error, 80)}); repairing {attempt}/{attempts}")

    def agent_unrepairable(self, name: str, reason: str) -> None:
        self._say(f"    could not regenerate {name}: {first_line(reason, 120)}")

    def agent_finished(self, result: AgentResult) -> None:
        if result.ok:
            self._say(f"    {result.name} done in {result.duration_seconds:.1f}s")
        else:
            self._alert(f"  ! {result.name} failed: {first_line(result.error, 120)}")

    def merge_started(self, survivors: int) -> None:
        noun = "output" if survivors == 1 else "outputs"
        self._say(f"  merging {survivors} {noun}")

    def council_convened(self) -> None:
        self._say("  the council is cross-examining the answer")

    def council_ruled(self, improved: bool, weaknesses: str) -> None:
        if improved:
            self._say(f"  the council found real faults; refined: {first_line(weaknesses, 90)}")
        else:
            self._say("  the council found nothing worth changing")

    def answer_judged(self, done: bool, missing: str) -> None:
        if done:
            self._say("  checked the answer against the request - it holds")
        else:
            self._say(f"  the answer falls short: {first_line(missing, 100)}")

    def revision_started(self, attempt: int, attempts: int, missing: str) -> None:
        self._say(f"  revision {attempt}/{attempts} - closing the gap")


def display_path(path: Path) -> str:
    """A path as a person would type it: relative when nearby, absolute otherwise."""
    try:
        from pathlib import Path as _Path

        return str(path.relative_to(_Path.cwd()))
    except (ValueError, OSError):
        return str(path)


def make_ui(plain: bool = False, quiet: bool = False) -> PlainUI:
    """Pick the best renderer this session can actually sustain."""
    if quiet or os.environ.get("AGENTGOD_QUIET"):
        return PlainUI(quiet=True)
    if plain or os.environ.get("AGENTGOD_PLAIN"):
        return PlainUI()

    try:
        # invalidate_caches so a `pip install rich` from preflight is
        # visible without restarting the process.
        importlib.invalidate_caches()
        importlib.import_module("rich")
        from richui import RichUI, rich_console

        console = rich_console()
    except Exception:
        # A missing rich is expected; an ancient or broken one must degrade
        # identically. The interface is never allowed to take the product down.
        return PlainUI()

    if not console.is_terminal:
        return PlainUI()
    return RichUI(console)
