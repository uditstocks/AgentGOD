"""The run-wide knobs: the effort dial and the shared cost meter.

No network and no API key - these rules are about arithmetic and locking,
not about the model.
"""

from __future__ import annotations

import threading

import config
from config import Usage, effort_for

# --- the adaptive effort dial ---------------------------------------------------


def test_simple_drops_to_low(monkeypatch):
    monkeypatch.setattr(config, "LLM_EFFORT", "medium")
    assert effort_for("simple") == "low"


def test_standard_runs_at_the_configured_default(monkeypatch):
    monkeypatch.setattr(config, "LLM_EFFORT", "medium")
    assert effort_for("standard") == "medium"
    monkeypatch.setattr(config, "LLM_EFFORT", "xhigh")
    assert effort_for("standard") == "xhigh"


def test_deep_raises_to_high(monkeypatch):
    monkeypatch.setattr(config, "LLM_EFFORT", "medium")
    assert effort_for("deep") == "high"


def test_deep_never_lowers_a_stronger_user_setting(monkeypatch):
    """A user who forced max effort meant it; the grade may only raise, not cut."""
    monkeypatch.setattr(config, "LLM_EFFORT", "max")
    assert effort_for("deep") == "max"


def test_an_unknown_grade_behaves_like_standard(monkeypatch):
    monkeypatch.setattr(config, "LLM_EFFORT", "medium")
    assert effort_for("weird") == "medium"


def test_an_unrecognised_configured_effort_still_deepens_safely(monkeypatch):
    """A typo in LLM_EFFORT must not crash the dial mid-run."""
    monkeypatch.setattr(config, "LLM_EFFORT", "turbo")
    assert effort_for("deep") == "high"


# --- the shared cost meter ------------------------------------------------------


def test_usage_survives_parallel_recording():
    """Independent agents are generated in parallel; a lost update under-bills."""
    usage = Usage()
    rounds = 500

    def hammer() -> None:
        for _ in range(rounds):
            usage.add(3, 2)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert usage.calls == 8 * rounds
    assert usage.input_tokens == 8 * rounds * 3
    assert usage.output_tokens == 8 * rounds * 2


def test_merge_still_folds_totals_in():
    total, extra = Usage(), Usage()
    extra.add(10, 5)
    total.merge(extra)
    assert (total.calls, total.input_tokens, total.output_tokens) == (1, 10, 5)


# --- prompt caching: the static half travels as a cache breakpoint ---------------


def test_cached_system_marks_an_ephemeral_breakpoint():
    from config import cached_system

    blocks = cached_system("the standing rules")
    assert blocks == [
        {
            "type": "text",
            "text": "the standing rules",
            "cache_control": {"type": "ephemeral"},
        }
    ]
