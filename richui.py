"""What AgentGod looks like when the terminal can paint.

One visual language for the whole product. The permanent architect is
gold; the agents it builds are steel blue; success is green, failure is
red, and everything else stays out of the way. During a run a single
live board shows the phase rail, the team and what is happening right
now - then disappears, leaving only the answer and a compact transcript.
The animation is the process; the transcript is the product.

This module is imported only by ui.make_ui(), and only after `rich` is
known to exist, so it may import rich freely at the top level. It knows
nothing about the pipeline: like PlainUI, every fact arrives through the
TaskEvents surface or as an argument.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from ui import HELP_ENVIRONMENT, HELP_USAGE, PlainUI, first_line

if TYPE_CHECKING:
    from pathlib import Path

    from executor import AgentResult, DependencyReport
    from planner import Plan

THEME = Theme(
    {
        "brand": "bold #e8b64c",  # the architect: gold
        "brand.dim": "#8a6d2f",
        "agent": "bold #58a6ff",  # the agents it builds: steel
        "ok": "#3fb950",
        "warn": "#d29922",
        "err": "bold #f85149",
        "dim": "grey58",
        "label": "#8a6d2f",
    }
)

# Short names for the five pipeline phases, shown on the rail.
RAIL = ("PLAN", "FORGE", "DEPS", "RUN", "MERGE")

# What the activity line says while each phase has nothing more specific.
PHASE_ACTIVITY = {
    1: ("architect", "deciding what team this task needs"),
    2: ("forge", "writing the team"),
    3: ("deps", "resolving extra packages"),
    4: ("run", "executing the team"),
    5: ("merge", "collapsing every output into one answer"),
}

GLYPHS = {
    "pending": "○",
    "active": "◆",
    "ready": "◇",
    "done": "●",
    "fail": "×",  # noqa: RUF001 - the multiplication sign is the design, not a typo'd x
    "arrow": "▸",
    "prompt": "❯",  # noqa: RUF001 - deliberate prompt ornament, not a greater-than
    "rule": "─",
    "spinner": "dots",
}

# The legacy Windows console cannot draw the set above.
ASCII_GLYPHS = {
    "pending": ".",
    "active": ">",
    "ready": "+",
    "done": "*",
    "fail": "x",
    "arrow": ">",
    "prompt": ">",
    "rule": "-",
    "spinner": "line",
}

WORDMARK = (
    "▄▀█ █▀▀ █▀▀ █▄ █ ▀█▀ █▀▀ █▀█ █▀▄",
    "█▀█ █▄█ ██▄ █ ▀█  █  █▄█ █▄█ █▄▀",
)

TAGLINE = "the creator and destroyer of the Agentworld"


def rich_console() -> Console:
    """The one Console for the session. Styling is deliberate, never automatic."""
    return Console(theme=THEME, highlight=False, emoji=False)


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m {rest:02d}s"


def _tokens(count: int) -> str:
    if count <= 0:
        return ""
    if count < 1000:
        return f"{count} tok"
    return f"{count / 1000:.1f}k tok"


def _finished_meta(row: _Row) -> str:
    """`6.3s · 1.4k tok` - one agent's outcome, board and transcript alike."""
    bits = [_duration(row.duration or 0.0)]
    if row.tokens:
        bits.append(_tokens(row.tokens))
    return " · ".join(bits)


@dataclass
class _Row:
    """One agent's place on the live board."""

    name: str
    role: str = ""
    status: str = "queued"  # queued|writing|ready|reused|running|repair|done|failed
    error: str = ""
    repair_attempt: int = 0
    repair_attempts: int = 0
    started_at: float | None = None
    duration: float | None = None
    tokens: int = 0
    reused: bool = False
    spinner: Spinner | None = None


class _Board:
    """Everything the live display knows about the run in progress.

    Mutated from the pipeline thread, rendered from Live's refresh thread;
    __rich__ therefore snapshots the row list and never removes rows.
    """

    def __init__(self, glyphs: dict[str, str]) -> None:
        self.glyphs = glyphs
        self.phase = 0
        self.total = len(RAIL)
        self.rows: list[_Row] = []
        self.by_name: dict[str, _Row] = {}
        self.notices: list[str] = []
        self.reasoning = ""
        self.activity: tuple[str, str] = ("architect", "reading the task")
        self.started_at = time.monotonic()
        # One spinner for the activity line, created once: a fresh Spinner
        # every refresh would reset its clock and freeze it on frame zero.
        self._activity_spinner = Spinner(glyphs["spinner"], style="brand")

    def row(self, name: str) -> _Row:
        existing = self.by_name.get(name)
        if existing is not None:
            return existing
        created = _Row(name=name)
        self.rows.append(created)
        self.by_name[name] = created
        return created

    def spin(self, row: _Row, style: str) -> Spinner:
        row.spinner = Spinner(self.glyphs["spinner"], style=style)
        return row.spinner

    # -- rendering -------------------------------------------------------

    def __rich__(self) -> RenderableType:
        parts: list[RenderableType] = [self._rail(), self._activity()]
        rows = list(self.rows)
        if rows:
            parts += [Text(""), self._team(rows)]
        if self.notices:
            parts.append(Text(""))
            parts += [Text(f"  ! {notice}", style="warn") for notice in list(self.notices)]
        return Group(*parts)

    def _rail(self) -> RenderableType:
        labels = RAIL if self.total == len(RAIL) else tuple(str(n) for n in range(1, self.total + 1))
        g = self.glyphs
        rail = Text("  ")
        # Each phase carries a state glyph as well as a color, so the rail
        # still reads correctly under NO_COLOR and in a pasted transcript.
        for index, label in enumerate(labels, 1):
            if index < self.phase:
                rail.append(f"{g['done']} ", style="brand.dim")
                rail.append(label, style="brand.dim")
            elif index == self.phase:
                rail.append(f"{g['active']} ", style="brand")
                rail.append(label, style="brand")
            else:
                rail.append(f"{g['pending']} ", style="dim")
                rail.append(label, style="dim")
            if index < len(labels):
                rail.append(f"  {g['arrow']}  ", style="dim")

        elapsed = int(time.monotonic() - self.started_at)
        clock = Text(f"{elapsed // 60:02d}:{elapsed % 60:02d}  ", style="dim")
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column(justify="right")
        grid.add_row(rail, clock)
        return grid

    def _activity(self) -> RenderableType:
        channel, message = self.activity
        self._activity_spinner.update(text=Text.assemble((channel, "label"), "  ", message))
        return Padding(self._activity_spinner, (0, 0, 0, 2))

    def _team(self, rows: list[_Row]) -> RenderableType:
        table = Table(box=None, show_header=False, padding=(0, 1), pad_edge=False)
        table.add_column(width=1)  # indent
        table.add_column(width=2, justify="center")  # state glyph
        table.add_column(no_wrap=True)  # agent name
        table.add_column(overflow="ellipsis", ratio=1)  # what it is / what went wrong
        table.add_column(justify="right", no_wrap=True)  # timing / tokens
        for row in rows:
            glyph, note, meta = self._cells(row)
            table.add_row("", glyph, Text(row.name, style="agent"), note, meta)
        return table

    def _cells(self, row: _Row) -> tuple[RenderableType, Text, Text]:
        g = self.glyphs
        role = Text(row.role, style="dim")
        if row.status == "queued":
            return Text(g["pending"], style="dim"), role, Text("")
        if row.status == "writing":
            spinner = row.spinner or self.spin(row, "brand")
            return spinner, Text("writing its code", style="warn"), Text("")
        if row.status == "ready":
            return Text(g["ready"], style="brand.dim"), role, Text("ready", style="dim")
        if row.status == "reused":
            return Text(g["ready"], style="ok"), role, Text("reused · free", style="ok")
        if row.status == "running":
            spinner = row.spinner or self.spin(row, "agent")
            elapsed = time.monotonic() - (row.started_at or time.monotonic())
            return spinner, role, Text(f"{elapsed:.0f}s", style="dim")
        if row.status == "repair":
            spinner = row.spinner or self.spin(row, "warn")
            note = Text("rewriting itself from its own error", style="warn")
            meta = Text(f"attempt {row.repair_attempt}/{row.repair_attempts}", style="warn")
            return spinner, note, meta
        if row.status == "failed":
            note = Text(first_line(row.error, 90), style="err")
            meta = Text(_duration(row.duration or 0.0), style="dim")
            return Text(g["fail"], style="err"), note, meta
        return Text(g["done"], style="ok"), role, Text(_finished_meta(row), style="dim")

    def team_transcript(self) -> RenderableType:
        """The team's final states, for after the live board is gone."""
        grid = Table.grid(padding=(0, 1), pad_edge=False)
        grid.add_column(width=2, justify="center")
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(overflow="ellipsis", ratio=1)
        g = self.glyphs
        for row in list(self.rows):
            if row.status == "failed":
                glyph = Text(g["fail"], style="err")
                tag = Text("failed", style="err")
                outcome = Text(first_line(row.error, 90), style="err")
            else:
                glyph = Text(g["done"], style="ok")
                tag = Text("reused", style="ok") if row.reused else Text("built", style="dim")
                outcome = Text(_finished_meta(row), style="dim")
            grid.add_row(glyph, Text(row.name, style="agent"), tag, outcome)
        return grid


class RichUI(PlainUI):
    """The full experience: every visual PlainUI defines, repainted.

    Subclassing PlainUI is the safety net - any surface method this class
    does not override degrades to a plain line rather than an AttributeError.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self.glyphs = ASCII_GLYPHS if console.legacy_windows else GLYPHS
        if self.glyphs is GLYPHS and sys.platform == "win32":
            # Classic conhost renders shapes but rarely braille: keep the
            # glyphs, swap the spinner. Windows Terminal and VS Code both
            # announce themselves; conhost is the silence between them.
            if not (os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM")):
                self.glyphs = {**GLYPHS, "spinner": "line"}
        self._board: _Board | None = None
        self._live: Live | None = None

    # -- session ---------------------------------------------------------

    def banner(self, model: str, max_agents: int, library_count: int, run_count: int) -> None:
        c = self.console
        c.print()
        if self.glyphs.get("rule") == "-":
            c.print(Text("  A G E N T G O D", style="brand"))
        else:
            c.print(Text("  " + WORDMARK[0], style="brand"))
            self._beat()
            c.print(Text("  " + WORDMARK[1], style="brand.dim"))
        self._beat()
        rule_width = max(len(WORDMARK[0]), len(TAGLINE))
        c.print(Text("  " + self.glyphs["rule"] * rule_width, style="brand.dim"))
        c.print(Text(f"  {TAGLINE}", style="dim"))
        self._beat()
        c.print()

        library = f"{library_count} agents remembered" if library_count else "empty · first run builds it"
        runs = f"{run_count} archived" if run_count else "none yet"
        meta = Table.grid(padding=(0, 2), pad_edge=False)
        meta.add_column(width=1)
        meta.add_column(style="label")
        meta.add_column(style="bold")
        meta.add_column(style="label")
        meta.add_column(style="dim")
        meta.add_row("", "model", model, "library", library)
        meta.add_row("", "ceiling", f"{max_agents} agents per task", "runs", runs)
        c.print(meta)

    def hint(self) -> None:
        self.console.print(
            Text("  describe a task and press enter · 'quit' leaves", style="dim")
        )

    def help(self) -> None:
        c = self.console
        c.print()
        c.print(Text("  AGENTGOD", style="brand"))
        c.print(
            Text(
                "  one permanent agent that builds, runs and retires task-specific agents",
                style="dim",
            )
        )
        c.print()
        usage = Table.grid(padding=(0, 3), pad_edge=False)
        usage.add_column(width=1)
        usage.add_column(style="bold")
        usage.add_column(style="dim")
        for command, explanation in HELP_USAGE:
            usage.add_row("", command, explanation)
        usage.add_row("", "", "")
        for variable, explanation in HELP_ENVIRONMENT:
            usage.add_row("", variable, explanation)
        c.print(usage)

    def input(self, message: str) -> str:
        """Print all but the last line of `message`, then prompt on the last.

        A last line that is just ">" (the session prompt convention in
        main.TASK_PROMPT) is replaced by the prompt glyph; anything else is
        kept as the question text.
        """
        while message.startswith("\n"):
            self.console.print()
            message = message[1:]
        *head, last = message.split("\n")
        for line in head:
            if line.strip():
                self.console.print(Text(f"  {line.strip()}", style="bold"))
        prompt = Text(f"  {self.glyphs['prompt']} ", style="brand")
        if last.strip() and last.strip() != ">":
            prompt.append(f"{last.strip()} ", style="bold")
        return self.console.input(prompt)

    def blank(self) -> None:
        self.console.print()

    def note(self, message: str) -> None:
        self.console.print(Text(message, style="dim"))

    def success(self, message: str) -> None:
        self.console.print(Text(message, style="ok"))

    def warn(self, message: str) -> None:
        self.console.print(Text(message, style="warn"))

    def error(self, message: str) -> None:
        self.console.print(Text(message, style="err"))

    def reply(self, text: str) -> None:
        """A conversational answer: same panel as an answer, no run summary.

        There is no team, no cost and no archive behind this one, so the
        trailing statistics are simply absent rather than zeroed - a run that
        never happened must not be reported as a run that cost nothing.
        """
        self._stop_live()
        self.console.print()
        self.console.print(
            Panel(
                Markdown(text),
                border_style="brand.dim",
                padding=(1, 3),
            )
        )

    def attachments_read(self, labels: list[str]) -> None:
        board = self._board
        if board is None:
            return super().attachments_read(labels)
        for label in labels:
            board.notices.append(f"read {label}")

    def context_carried(self, previous_task: str) -> None:
        board = self._board
        if board is None:
            return super().context_carried(previous_task)
        board.notices.append(f"continuing: {first_line(previous_task, 60)}")

    def farewell(self) -> None:
        self.console.print(Text("  goodbye.", style="dim"))

    def _beat(self) -> None:
        """A barely-there reveal on startup. Cinematic, never slow."""
        if self.console.is_terminal:
            time.sleep(0.045)

    # -- one task, start to finish ---------------------------------------

    def run_started(self, task: str, echo: bool = False) -> None:
        self._stop_live()
        self._board = _Board(self.glyphs)
        c = self.console
        c.print()
        stamp = datetime.now().strftime("%H:%M:%S")
        c.print(Rule(Text(stamp, style="dim"), style="brand.dim", align="right"))
        if echo:
            c.print(Text.assemble("  ", ("task", "label"), "  ", (first_line(task, 180), "dim")))
        self._live = Live(
            self._board, console=c, refresh_per_second=12, transient=True
        )
        self._live.start()

    def run_succeeded(self, result, saved: Path | None) -> None:
        self._stop_live()
        c = self.console
        board = self._board

        c.print()
        response = result.response.strip() or "(the merger returned an empty response)"
        c.print(
            Panel(
                Markdown(response),
                title=Text(" A N S W E R ", style="brand"),
                title_align="left",
                border_style="brand.dim",
                padding=(1, 3),
            )
        )

        summary = Table.grid(padding=(0, 1), pad_edge=False)
        summary.add_column(width=1)
        summary.add_column(style="label", no_wrap=True)
        summary.add_column(ratio=1)
        if board is not None and board.reasoning:
            summary.add_row("", "plan", Text(board.reasoning, style="dim"))
        if board is not None and board.rows:
            summary.add_row("", "team", board.team_transcript())
            if board.notices:
                summary.add_row("", "notes", Text("\n".join(board.notices), style="warn"))
        if result.failures:
            excluded = Text(
                "failed agents were excluded from the answer", style="warn"
            )
            summary.add_row("", "", excluded)
        stats = f"{_duration(result.duration_seconds)} · {result.cost_summary()}"
        summary.add_row("", "run", Text(stats, style="dim"))
        if saved is not None:
            summary.add_row("", "saved", Text(f"{saved.parent.name}/{saved.name}", style="dim"))
        else:
            summary.add_row("", "saved", Text("the run archive could not be written", style="warn"))
        c.print(summary)

    def run_cancelled(self) -> None:
        self._stop_live()
        self.console.print()
        self.console.print(
            Text("  interrupted - this run is abandoned; nothing was kept", style="warn")
        )

    def run_failed(self, error: BaseException) -> None:
        self._stop_live()
        c = self.console
        message = str(error).strip() or type(error).__name__
        lines = message.splitlines()
        if len(lines) > 14:
            lines = [*lines[:14], f"... {len(lines) - 14} more lines"]
        c.print()
        c.print(
            Panel(
                Text("\n".join(lines)),
                title=Text(f" {type(error).__name__} ", style="err"),
                title_align="left",
                border_style="err",
                padding=(1, 3),
            )
        )
        board = self._board
        if board is not None and board.rows:
            c.print(Padding(board.team_transcript(), (0, 0, 0, 2)))
        c.print(Text("  nothing was kept · fix the cause above, then run the task again", style="dim"))

    def run_ended(self) -> None:
        self._stop_live()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    # -- pipeline events -------------------------------------------------

    def phase_started(self, index: int, total: int, title: str) -> None:
        board = self._board
        if board is None:
            return super().phase_started(index, total, title)
        board.phase = index
        board.total = total
        board.activity = PHASE_ACTIVITY.get(index, ("architect", title.lower()))

    def plan_ready(self, plan: Plan) -> None:
        board = self._board
        if board is None:
            return super().plan_ready(plan)
        board.reasoning = plan.reasoning
        for spec in plan.agents:
            row = board.row(spec.name)
            row.role = spec.role
        noun = "agent" if len(plan.agents) == 1 else "agents"
        board.activity = ("architect", f"team of {len(plan.agents)} {noun} planned")

    def agent_retired(self, name: str, reason: str) -> None:
        board = self._board
        if board is None:
            return super().agent_retired(name, reason)
        board.notices.append(f"retired {name} from the library - {reason}")
        board.activity = ("forge", f"{name} retired · rebuilding it")

    def agent_build_started(self, name: str) -> None:
        board = self._board
        if board is None:
            return super().agent_build_started(name)
        row = board.row(name)
        row.status = "writing"
        row.spinner = None
        board.activity = ("forge", f"writing {name}.py")

    def agent_ready(self, name: str, filename: str, reused: bool) -> None:
        board = self._board
        if board is None:
            return super().agent_ready(name, filename, reused)
        row = board.row(name)
        row.reused = reused
        row.status = "reused" if reused else "ready"
        if reused:
            board.activity = ("forge", f"{name} reclaimed from the library · free")

    def deps_checked(self, report: DependencyReport) -> None:
        board = self._board
        if board is None:
            return super().deps_checked(report)
        board.notices.extend(report.problems)
        if report.installed:
            summary = f"installed {', '.join(report.installed)}"
        elif report.already_present or report.problems:
            summary = "packages resolved"
        else:
            summary = "no extra packages needed"
        board.activity = ("deps", summary)

    def agent_started(self, name: str, index: int, total: int) -> None:
        board = self._board
        if board is None:
            return super().agent_started(name, index, total)
        row = board.row(name)
        row.status = "running"
        row.started_at = time.monotonic()
        row.spinner = None
        board.activity = ("run", f"{name} is working  ({index}/{total})")

    def agent_repairing(self, name: str, attempt: int, attempts: int, error: str) -> None:
        board = self._board
        if board is None:
            return super().agent_repairing(name, attempt, attempts, error)
        row = board.row(name)
        row.status = "repair"
        row.repair_attempt = attempt
        row.repair_attempts = attempts
        row.spinner = None
        board.activity = ("repair", f"rewriting {name} from its own error")

    def agent_unrepairable(self, name: str, reason: str) -> None:
        board = self._board
        if board is None:
            return super().agent_unrepairable(name, reason)
        board.notices.append(f"{name} could not be regenerated: {first_line(reason, 90)}")

    def agent_finished(self, result: AgentResult) -> None:
        board = self._board
        if board is None:
            return super().agent_finished(result)
        row = board.row(result.name)
        row.duration = result.duration_seconds
        row.tokens = result.input_tokens + result.output_tokens
        row.spinner = None
        if result.ok:
            row.status = "done"
        else:
            row.status = "failed"
            row.error = result.error

    def merge_started(self, survivors: int) -> None:
        board = self._board
        if board is None:
            return super().merge_started(survivors)
        noun = "output" if survivors == 1 else "outputs"
        board.activity = ("merge", f"merging {survivors} {noun} into one answer")
