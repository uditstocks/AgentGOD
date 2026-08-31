"""Tests for config helpers, the merger's prompt shaping, and inventory."""

from __future__ import annotations

import pytest

import inventory
import merger
from config import MAX_CHARS_PER_INPUT, Usage, response_text

# --- M10: an LLM reply may be a string or a list of content blocks --------------


class _Block:
    """A content block as the SDK returns it: an object, not a dict."""

    def __init__(self, type, text=""):
        self.type = type
        self.text = text


class _Usage:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Message:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage


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


def test_response_text_reads_sdk_content_blocks():
    assert response_text(_Message([_Block("text", "hello")])) == "hello"


def test_response_text_skips_the_thinking_block():
    """The answer is the text blocks. Returning content[0] returns the reasoning."""
    reply = _Message([_Block("thinking", "let me see"), _Block("text", "the answer")])
    assert response_text(reply) == "the answer"


# --- M8: usage accounting -------------------------------------------------------


def test_usage_accumulates():
    usage = Usage()
    usage.add(100, 50)
    usage.add(20, 5)
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 120, 55)


def test_usage_records_from_a_response():
    usage = Usage()
    usage.record(_Message("x", _Usage(input_tokens=7, output_tokens=3)))
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


# --- scratch cleanup (keeping an agent is library.remember's job) ------------


def test_deleting_a_missing_file_does_not_raise(tmp_path):
    assert inventory.delete_agents([tmp_path / "gone.py"]) == 1


def test_delete_removes_real_files(tmp_path):
    agent = tmp_path / "research_agent.py"
    agent.write_text("print(1)", encoding="utf-8")
    assert inventory.delete_agents([agent]) == 1
    assert not agent.exists()
