"""The run-wide knobs: the effort dial and the shared cost meter.

No network and no API key - these rules are about arithmetic and locking,
not about the model.
"""

from __future__ import annotations

import threading

import pytest

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


# --- cached input is billed on its own terms ------------------------------------


def test_cache_tokens_are_priced_apart_from_plain_input(monkeypatch):
    from config import CACHE_READ_MULTIPLIER, CACHE_WRITE_MULTIPLIER, estimate_cost

    monkeypatch.setattr(config, "MODEL", "claude-sonnet-5")
    plain = estimate_cost(1000, 0)
    assert plain is not None

    written = estimate_cost(0, 0, cache_write_tokens=1000)
    read = estimate_cost(0, 0, cache_read_tokens=1000)
    assert written is not None and read is not None
    # A write costs a premium; a read is nearly free. Both are real money.
    assert written == pytest.approx(plain * CACHE_WRITE_MULTIPLIER)
    assert read == pytest.approx(plain * CACHE_READ_MULTIPLIER)
    assert 0 < read < plain < written


def test_usage_records_the_cache_fields_the_api_reports():
    from types import SimpleNamespace

    usage = Usage()
    usage.record(
        SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                cache_creation_input_tokens=1400,
                cache_read_input_tokens=0,
            )
        )
    )
    usage.record(
        SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=1400,
            )
        )
    )
    assert usage.cache_write_tokens == 1400
    assert usage.cache_read_tokens == 1400
    assert usage.input_tokens == 200  # cached input is NOT counted here


def test_a_cached_run_names_what_it_reused():
    usage = Usage()
    usage.add(100, 20, cache_read_tokens=2543)
    assert "2,543 cached" in usage.summary()


# --- one table decides which model does which job -------------------------------


def test_mechanical_checks_always_run_on_the_fast_model(monkeypatch):
    from config import model_for

    monkeypatch.setattr(config, "FAST_MODEL", "claude-haiku-4-5")
    monkeypatch.setattr(config, "MODEL", "claude-sonnet-5")
    monkeypatch.setattr(config, "DEEP_MODEL", "claude-opus-5")
    # A compliance check does not get better on a bigger model, at any grade.
    for grade in ("simple", "standard", "deep"):
        assert model_for("judge", grade) == "claude-haiku-4-5"
        assert model_for("clarify", grade) == "claude-haiku-4-5"


def test_real_work_rises_to_the_deep_model_only_for_deep_tasks(monkeypatch):
    from config import model_for

    monkeypatch.setattr(config, "MODEL", "claude-sonnet-5")
    monkeypatch.setattr(config, "DEEP_MODEL", "claude-opus-5")
    for role in ("plan", "generate", "run", "merge", "council"):
        assert model_for(role, "simple") == "claude-sonnet-5"
        assert model_for(role, "standard") == "claude-sonnet-5"
        assert model_for(role, "deep") == "claude-opus-5"


def test_the_deep_model_is_the_workhorse_unless_asked_for(monkeypatch):
    """Nothing changes for anyone who never sets DEEP_MODEL."""
    from config import model_for

    monkeypatch.setattr(config, "MODEL", "claude-sonnet-5")
    monkeypatch.setattr(config, "DEEP_MODEL", "claude-sonnet-5")
    assert model_for("run", "deep") == "claude-sonnet-5"


def test_cost_is_priced_per_call_at_its_own_models_rate(monkeypatch):
    """A run spends across two models; one blended rate would be wrong."""
    monkeypatch.setattr(config, "MODEL", "claude-sonnet-5")
    usage = Usage()
    usage.add(1_000_000, 0, model="claude-sonnet-5")  # $2.00
    usage.add(1_000_000, 0, model="claude-haiku-4-5")  # $1.00
    assert usage.cost_usd == pytest.approx(3.00)


def test_an_unpriced_model_leaves_the_cost_unknown(monkeypatch):
    usage = Usage()
    usage.add(1000, 100, model="some-unlisted-model")
    assert usage.cost_usd is None


def test_the_effort_dial_is_only_sent_to_models_that_take_it():
    """Haiku rejects output_config.effort with a 400 - verified against the API."""
    from config import supports_effort

    assert supports_effort("claude-sonnet-5") is True
    assert supports_effort("claude-opus-5") is True
    assert supports_effort("claude-haiku-4-5") is False
