"""Tests for config helpers, the merger's prompt shaping, and inventory."""

from __future__ import annotations

import pytest

import inventory
import merger
from config import MAX_CHARS_PER_INPUT, Usage, response_text

# --- M10: an LLM reply may be a string or a list of content blocks --------------


class _Message:
    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain text", "plain text"),
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "ab"),
        (["a", "b"], "ab"),
        ([], ""),
        ([{"type": "image", "url": "x"}], ""),
    ],
)
def test_response_text_normalises_content(content, expected):
    assert response_text(_Message(content)) == expected


def test_response_text_prefers_the_text_accessor():
    from langchain_core.messages import AIMessage

    assert response_text(AIMessage(content="hello")) == "hello"
    blocks = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert response_text(blocks) == "ab"


# --- M8: usage accounting -------------------------------------------------------


def test_usage_accumulates():
    usage = Usage()
    usage.add(100, 50)
    usage.add(20, 5)
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 120, 55)


def test_usage_records_from_a_response():
    usage = Usage()
    usage.record(_Message("x", {"input_tokens": 7, "output_tokens": 3}))
    assert (usage.input_tokens, usage.output_tokens) == (7, 3)


def test_usage_survives_a_response_without_metadata():
    usage = Usage()
    usage.record(_Message("x"))
    assert usage.calls == 1
    assert usage.input_tokens == 0


def test_usage_summary_is_printable():
    usage = Usage()
    usage.add(1000, 500)
    assert "1,000 in" in usage.summary()


# --- H7/M11: the merger always runs and caps what it forwards -------------------


def test_merging_nothing_is_an_error_not_a_hallucination():
    with pytest.raises(ValueError):
        merger.merge_outputs("task", {})


def test_outputs_are_labelled():
    formatted = merger._format_outputs({"research_agent": "facts", "writer_agent": "prose"})
    assert "--- research_agent ---" in formatted
    assert "--- writer_agent ---" in formatted


def test_long_outputs_are_truncated():
    formatted = merger._format_outputs({"a_agent": "x" * (MAX_CHARS_PER_INPUT * 3)})
    assert "[...truncated...]" in formatted
    assert len(formatted) < MAX_CHARS_PER_INPUT * 2


# --- L3: inventory edge cases ---------------------------------------------------


def test_save_to_inventory_moves_files_and_records_the_task(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path / "inventory")
    agent = tmp_path / "research_agent.py"
    agent.write_text("print(1)\n", encoding="utf-8")

    folder = inventory.save_to_inventory([agent], "find things")
    assert (folder / "research_agent.py").is_file()
    assert (folder / "TASK.txt").read_text(encoding="utf-8") == "find things"
    assert not agent.exists()


def test_saves_in_the_same_second_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path / "inventory")
    monkeypatch.setattr(inventory, "_unique_folder", inventory._unique_folder)

    folders = []
    for index in range(3):
        agent = tmp_path / f"agent_{index}.py"
        agent.write_text("print(1)\n", encoding="utf-8")
        folders.append(inventory.save_to_inventory([agent], f"task {index}"))

    assert len(set(folders)) == 3
    for index, folder in enumerate(folders):
        assert (folder / "TASK.txt").read_text(encoding="utf-8") == f"task {index}"


def test_saving_a_missing_file_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(inventory, "INVENTORY_DIR", tmp_path / "inventory")
    folder = inventory.save_to_inventory([tmp_path / "gone.py"], "t")
    assert (folder / "TASK.txt").is_file()


def test_deleting_a_missing_file_does_not_raise(tmp_path):
    assert inventory.delete_agents([tmp_path / "gone.py"]) == 1
