"""What AgentGod looks like when the terminal cannot paint.

PlainUI is the reference implementation of the presentation surface:
every visual the project can show exists here as an aligned, colorless
ASCII line, safe for pipes, CI logs, and consoles from another century.
richui.RichUI subclasses it and repaints the same surface with color,
motion and structure - so anything Rich cannot draw degrades to a line
of text instead of an AttributeError.

make_ui() chooses the renderer once per session:

- AGENTGOD_PLAIN set, `rich` missing, or stdout not a terminal -> PlainUI
- otherwise -> RichUI

The pipeline never chooses. It emits TaskEvents and moves on.

This module imports nothing from the rest of the project except `events`:
every fact it displays (model name, limits, results) arrives as an
argument, so it can be imported before dependencies have been checked.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

from events import TaskEvents

if TYPE_CHECKING:
    from pathlib import Path

    from executor import AgentResult, DependencyReport
    from planner import Plan

RULE_WIDTH = 60

HELP_TITLE = "AgentGod - one permanent agent that builds, runs and retires task-specific agents."

# One source of truth for the help screen; each renderer lays it out its own way.
HELP_USAGE = (
    ("python main.py", "interactive session"),
    ('python main.py --task "..."', "run one task and exit"),
    ("python main.py --plain ...", "no color, no animation"),
)
HELP_ENVIRONMENT = (
    ("AGENTGOD_PLAIN=1", "force plain output (same as --plain)"),
    ("NO_COLOR=1", "keep the interface, strip the color"),
)


def first_line(text: str, limit: int = 120) -> str:
    """The first non-empty line of `text`, capped for one-line display."""
    stripped = text.strip()
    line = stripped.splitlines()[0] if stripped else ""
    return line if len(line) <= limit else line[: max(1, limit - 3)] + "..."


class PlainUI(TaskEvents):
    """Aligned, colorless, animation-free. Safe anywhere text can go."""

    # -- session ---------------------------------------------------------

    def banner(self, model: str, max_agents: int, library_count: int, run_count: int) -> None:
        library = f"{library_count} kept" if library_count else "empty"
        runs = f"{run_count} archived" if run_count else "none yet"
        print("\nAGENTGOD - the creator and destroyer of the Agentworld")
        print(f"model {model} · up to {max_agents} agents/task · library {library} · runs {runs}")

    def hint(self) -> None:
        print("Type a task and press Enter. 'quit' leaves.")

    def help(self) -> None:
        print(HELP_TITLE)
        print("\nUsage:")
        for command, explanation in HELP_USAGE:
            print(f"  {command:<29} {explanation}")
        print("\nEnvironment:")
        for variable, explanation in HELP_ENVIRONMENT:
            print(f"  {variable:<18} {explanation}")

    def input(self, message: str) -> str:
        """Read one reply. EOFError / KeyboardInterrupt pass through to the caller."""
        return input(message)

    def blank(self) -> None:
        print()

    def note(self, message: str) -> None:
        print(message)

    def success(self, message: str) -> None:
        print(message)

    def warn(self, message: str) -> None:
        print(message)

    def error(self, message: str) -> None:
        print(message)

    def reply(self, text: str) -> None:
        """Answer a conversational turn - no pipeline ran, so there is nothing to report.

        Deliberately not run_succeeded(): there is no team, no cost and no
        archive, and dressing an instant answer up as a completed run would
        misrepresent what happened.
        """
        print()
        print(text)

    def attachments_read(self, labels: list[str]) -> None:
        """Name every local file that was read into the task.

        This is not decoration. The contents are about to be sent to a model
        provider, and the user is entitled to know that before it happens.
        """
        for label in labels:
            print(f"  read {label}")

    def context_carried(self, previous_task: str) -> None:
        """Say when a follow-up was answered using the exchange before it."""
        print(f"  (continuing: {first_line(previous_task, 60)})")

    def farewell(self) -> None:
        print("Goodbye.")

    # -- one task, start to finish ---------------------------------------

    def run_started(self, task: str, echo: bool = False) -> None:
        if echo:
            print(f"\n> {first_line(task, 200)}")

    def run_succeeded(self, result, saved: Path | None) -> None:
        print("\n" + "=" * RULE_WIDTH)
        print("ANSWER")
        print("=" * RULE_WIDTH)
        print(result.response)

        if result.failures:
            print("\nAgents that failed (their output was excluded):")
            for name, error in result.failures.items():
                print(f"  - {name}: {first_line(error, 150)}")

        print(f"\n{result.duration_seconds:.1f}s · {result.cost_summary()}")
        if result.reused:
            print(f"Reused free from library: {', '.join(result.reused)}")
        if result.built:
            print(f"Newly built this run: {', '.join(result.built)}")

        if saved is not None:
            print(f"Saved to {saved.parent.name}/{saved.name}")
        else:
            print("(could not write the run archive)")

    def run_cancelled(self) -> None:
        print("\nCancelled - nothing from this run was kept.")

    def run_failed(self, error: BaseException) -> None:
        print(f"\nTask failed: {type(error).__name__}: {error}")

    def run_ended(self) -> None:
        """Always called last, even after a crash. Nothing to release here."""

    # -- pipeline events -------------------------------------------------

    def phase_started(self, index: int, total: int, title: str) -> None:
        print(f"\n[{index}/{total}] {title}...")

    def plan_ready(self, plan: Plan) -> None:
        print(f"  {plan.reasoning}")
        for spec in plan.agents:
            print(f"  - {spec.name}: {spec.role}")

    def agent_retired(self, name: str, reason: str) -> None:
        print(f"  retired {name} from the library: {reason}")

    def agent_build_started(self, name: str) -> None:
        print(f"  writing {name}...")

    def agent_ready(self, name: str, filename: str, reused: bool) -> None:
        if reused:
            print(f"  reused {filename} (from library, free)")
        else:
            print(f"  wrote {filename}")

    def deps_checked(self, report: DependencyReport) -> None:
        for note in report.problems:
            print(f"  ! {note}")
        if report.installed:
            print(f"  installed: {', '.join(report.installed)}")

    def agent_started(self, name: str, index: int, total: int) -> None:
        print(f"  [{index}/{total}] {name} running...")

    def agent_repairing(self, name: str, attempt: int, attempts: int, error: str) -> None:
        print(f"    {name} failed ({first_line(error, 80)}); repairing {attempt}/{attempts}")

    def agent_unrepairable(self, name: str, reason: str) -> None:
        print(f"    could not regenerate {name}: {first_line(reason, 120)}")

    def agent_finished(self, result: AgentResult) -> None:
        if result.ok:
            print(f"    done in {result.duration_seconds:.1f}s")
        else:
            print(f"  ! {result.name} failed: {first_line(result.error, 120)}")

    def merge_started(self, survivors: int) -> None:
        noun = "output" if survivors == 1 else "outputs"
        print(f"  merging {survivors} {noun}")

    def answer_judged(self, done: bool, missing: str) -> None:
        if done:
            print("  checked the answer against the request - it holds")
        else:
            print(f"  the answer falls short: {first_line(missing, 100)}")

    def revision_started(self, attempt: int, attempts: int, missing: str) -> None:
        print(f"  revision {attempt}/{attempts} - running the agents again")


def make_ui() -> PlainUI:
    """Pick the best renderer this session can actually sustain."""
    if os.environ.get("AGENTGOD_PLAIN"):
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
