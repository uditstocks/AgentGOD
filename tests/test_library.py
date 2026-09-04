"""The reusable agent library: build once, hand back free forever after."""

from __future__ import annotations

import json

import pytest

import library
from library import (
    catalogue,
    describe_for_planner,
    forget,
    lookup,
    record_use,
    remember,
    up_to_date,
)

SOURCE = "def run(task, previous_outputs):\n    return 'ok'\n"


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    """Never touch the real inventory/ during tests."""
    root = tmp_path / "inventory"
    monkeypatch.setattr(library, "INVENTORY_DIR", root)
    monkeypatch.setattr(library, "LIBRARY_DIR", root / "agents")
    monkeypatch.setattr(library, "INDEX_PATH", root / "index.json")
    return root


# --- the core promise: build once, reuse free --------------------------------


def test_unknown_agent_is_a_miss():
    assert lookup("research_agent") is None


def test_remembered_agent_comes_back_verbatim():
    assert remember("research_agent", "Gather facts", SOURCE) is True
    assert lookup("research_agent") == SOURCE


def test_remember_is_idempotent_and_refreshes_source():
    remember("writer_agent", "Write prose", SOURCE)
    updated = SOURCE.replace("ok", "better")
    remember("writer_agent", "Write prose", updated)
    assert lookup("writer_agent") == updated
    assert len(catalogue()) == 1  # refreshed, not duplicated


def test_forget_removes_it():
    remember("temp_agent", "role", SOURCE)
    assert forget("temp_agent") is True
    assert lookup("temp_agent") is None
    assert forget("temp_agent") is False


# --- usage accounting drives what the planner is shown first -----------------


def test_uses_are_counted_and_rank_the_catalogue():
    remember("rare_agent", "Rare", SOURCE)
    remember("common_agent", "Common", SOURCE)
    for _ in range(3):
        record_use("common_agent")

    names = [entry.name for entry in catalogue()]
    assert names == ["common_agent", "rare_agent"]
    assert catalogue()[0].uses == 3


def test_record_use_on_a_missing_agent_is_harmless():
    record_use("never_built")  # must not raise


# --- what the planner actually sees ------------------------------------------


def test_planner_text_is_explicit_when_empty():
    assert "none yet" in describe_for_planner()


def test_planner_text_lists_name_and_role():
    remember("research_agent", "Gather facts about the subject of the task", SOURCE)
    text = describe_for_planner()
    assert "research_agent" in text
    assert "Gather facts about the subject of the task" in text


def test_planner_text_respects_the_limit():
    for index in range(10):
        remember(f"agent_{index}", "role", SOURCE)
    assert len(describe_for_planner(limit=4).splitlines()) == 4


# --- durability: the .py files are the source of truth -----------------------


def test_library_survives_a_corrupt_index(isolated_library):
    remember("research_agent", "Gather facts", SOURCE)
    (isolated_library / "index.json").write_text("{ not json", encoding="utf-8")

    assert lookup("research_agent") == SOURCE  # file still readable
    assert [entry.name for entry in catalogue()] == ["research_agent"]


def test_index_is_rebuilt_from_files_when_missing(isolated_library):
    remember("writer_agent", "Write", SOURCE)
    (isolated_library / "index.json").unlink()
    assert [entry.name for entry in catalogue()] == ["writer_agent"]


def test_index_is_valid_json_with_a_version(isolated_library):
    remember("research_agent", "Gather facts", SOURCE)
    payload = json.loads((isolated_library / "index.json").read_text(encoding="utf-8"))
    assert payload["version"] == library.INDEX_VERSION
    assert payload["agents"]["research_agent"]["role"] == "Gather facts"


# --- names reaching the filesystem stay contained ----------------------------


def test_a_traversing_name_is_refused(isolated_library):
    assert remember("../../escaped", "role", SOURCE) is False
    assert not (isolated_library.parent / "escaped.py").exists()


# --- the second promise: an agent must match the runtime it will run on --------


def test_a_freshly_remembered_agent_is_current():
    remember("research_agent", "gather", SOURCE, task="t")
    assert up_to_date("research_agent")


def test_an_agent_from_before_the_stamp_is_not_current():
    """The four agents already on disk when search shipped had no version."""
    remember("research_agent", "gather", SOURCE, task="t")
    entries = library._read_index()
    entries["research_agent"].runtime = 0
    library._write_index(entries)
    assert not up_to_date("research_agent")


def test_an_agent_built_against_an_older_runtime_is_not_current():
    remember("research_agent", "gather", SOURCE, task="t")
    entries = library._read_index()
    entries["research_agent"].runtime = library.AGENT_RUNTIME_VERSION - 1
    library._write_index(entries)
    assert not up_to_date("research_agent")


def test_an_agent_that_is_not_in_the_library_is_not_current():
    assert not up_to_date("never_built_agent")


# --- the reliability record: the library curates itself ------------------------


def test_outcomes_accumulate_on_the_record():
    remember("research_agent", "Gather facts", SOURCE)
    library.record_outcome("research_agent", True)
    library.record_outcome("research_agent", True)
    library.record_outcome("research_agent", False)
    entry = library.entry_for("research_agent")
    assert entry is not None
    assert (entry.wins, entry.losses) == (2, 1)


def test_an_unknown_agent_takes_no_record():
    library.record_outcome("ghost_agent", True)  # must not raise or create
    assert library.entry_for("ghost_agent") is None


def test_a_fresh_agent_is_trusted():
    remember("research_agent", "Gather facts", SOURCE)
    assert library.reliable("research_agent") is True
    assert library.reliable("never_kept_agent") is True


def test_one_unlucky_crash_never_retires_a_good_agent():
    remember("research_agent", "Gather facts", SOURCE)
    library.record_outcome("research_agent", False)
    assert library.reliable("research_agent") is True


def test_an_agent_that_fails_more_than_it_works_is_unreliable():
    remember("research_agent", "Gather facts", SOURCE)
    for _ in range(3):
        library.record_outcome("research_agent", False)
    assert library.reliable("research_agent") is False


def test_enough_wins_outweigh_the_losses():
    remember("research_agent", "Gather facts", SOURCE)
    for _ in range(3):
        library.record_outcome("research_agent", False)
    for _ in range(3):
        library.record_outcome("research_agent", True)
    assert library.reliable("research_agent") is True


# --- evolution: a repair is a new generation with a clean slate ----------------


def test_a_plain_refresh_keeps_the_generation():
    remember("writer_agent", "Write prose", SOURCE)
    remember("writer_agent", "Write prose", SOURCE.replace("ok", "fine"))
    entry = library.entry_for("writer_agent")
    assert entry is not None
    assert entry.generation == 1


def test_an_evolution_advances_the_generation_and_wipes_the_record():
    remember("writer_agent", "Write prose", SOURCE)
    library.record_outcome("writer_agent", False)
    library.record_outcome("writer_agent", False)
    remember("writer_agent", "Write prose", SOURCE.replace("ok", "fixed"), evolved=True)
    entry = library.entry_for("writer_agent")
    assert entry is not None
    assert entry.generation == 2
    assert (entry.wins, entry.losses) == (0, 0)
    assert library.reliable("writer_agent") is True


# --- a poisoned agent heals itself, with no manual step -------------------------


def test_an_agent_poisoned_under_the_old_guard_is_refused_on_sight():
    """The production guarantee: nobody has to run /audit or /forget by hand.

    Agents kept before the subject extractor understood acronyms are sitting
    in real libraries right now. reusable() is consulted on every lookup, so
    the next task that reaches for one retires it and rebuilds it clean -
    no migration, no manual purge.
    """
    poisoned = (
        "def run(task, previous_outputs):\n"
        "    system = 'You write runnable scripts that generate QR code images.'\n"
        "    return call_llm(task, system=system)\n"
    )
    remember(
        "code_agent",
        "write or explain code",
        poisoned,
        task="write a python code to convert any link or text into qr code",
    )
    # It is on disk and would be handed back for free...
    assert lookup("code_agent") == poisoned
    # ...but the guard refuses it, which is what makes the orchestrator retire it.
    assert library.reusable("code_agent") is False
    assert "code_agent" in library.audit()


def test_a_clean_agent_is_not_swept_up_by_the_stricter_guard():
    """A stricter rule that retires everything would cost more than it saves."""
    clean = (
        "def run(task, previous_outputs):\n"
        "    system = 'You are a code agent. Write the code the task asks for.'\n"
        "    return call_llm(task, system=system)\n"
    )
    remember(
        "code_agent",
        "write or explain code",
        clean,
        task="write a python code to convert any link or text into qr code",
    )
    assert library.reusable("code_agent") is True
    assert library.audit() == {}
