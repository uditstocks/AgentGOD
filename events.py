"""The seam between the pipeline and whatever is watching it run.

orchestrator.handle_task() reports every notable moment of a task's
lifecycle to a TaskEvents object instead of printing. The base class does
nothing, so the pipeline runs headless (tests, scripts, other programs)
at zero presentation cost; ui.py subclasses it to draw the same moments
as a live interface.

Nothing here imports the rest of the project at runtime: this module must
be importable before dependencies have been checked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from executor import AgentResult, DependencyReport
    from planner import Plan


class TaskEvents:
    """Every hook is a no-op; a subclass overrides only what it can show."""

    def phase_started(self, index: int, total: int, title: str) -> None:
        """A pipeline stage began. `index` is 1-based."""

    def plan_ready(self, plan: Plan) -> None:
        """The planner decided the team."""

    def agent_build_started(self, name: str) -> None:
        """Code generation began for one agent (a library miss)."""

    def agent_retired(self, name: str, reason: str) -> None:
        """A library agent was dropped rather than reused, and will be rebuilt."""

    def agent_ready(self, name: str, filename: str, reused: bool) -> None:
        """One agent's file is on disk - reclaimed from the library, or newly written."""

    def deps_checked(self, report: DependencyReport) -> None:
        """Dependency resolution finished, well or badly."""

    def agent_started(self, name: str, index: int, total: int) -> None:
        """One agent subprocess is about to run."""

    def agent_repairing(self, name: str, attempt: int, attempts: int, error: str) -> None:
        """A crashed agent is being regenerated from its own error output."""

    def agent_unrepairable(self, name: str, reason: str) -> None:
        """Regeneration itself failed; the agent keeps its last error."""

    def agent_finished(self, result: AgentResult) -> None:
        """One agent finished, successfully or not."""

    def merge_started(self, survivors: int) -> None:
        """The merger began collapsing `survivors` outputs into one answer."""
