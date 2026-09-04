"""The final answer's shape: it must be valid Markdown before anyone reads it.

Markdown joins a run of unfenced lines into one paragraph, so raw source code
arrives with its newlines collapsed and its indentation stripped - a whole
function on one line, a `#` comment turned into a centred heading. That
happens in the terminal and in the .md archive alike, which is why the answer
is normalised once, where it is produced.
"""

from __future__ import annotations

import pytest

from agentgod.merger import FENCE, as_markdown, looks_like_source

CODE = '''#!/usr/bin/env python3
"""Turn text into a QR code."""

import sys
from pathlib import Path

def encode(text: str) -> bytes:
    if not text:
        raise ValueError("empty input")
    return text.encode("utf-8")
'''

JAVASCRIPT = """const fs = require('fs');

function encode(text) {
  if (!text) {
    throw new Error('empty');
  }
  return Buffer.from(text);
}
"""

PROSE = (
    "Code review catches bugs before they reach production, and it spreads "
    "knowledge of the codebase across the team.\n\n"
    "The second benefit is often the more valuable one: a team that has read "
    "each other's work can cover for each other."
)

PROSE_ABOUT_CODE = (
    "To import the CSV, open it with the standard library first. "
    "Define a helper that returns the parsed rows, then class-based handlers "
    "can consume it. Import order matters less than clarity here."
)


# --- what counts as source code -------------------------------------------------


@pytest.mark.parametrize("source", [CODE, JAVASCRIPT])
def test_real_source_is_recognised(source):
    assert looks_like_source(source) is True


@pytest.mark.parametrize("text", [PROSE, PROSE_ABOUT_CODE, "", "   ", "Hello."])
def test_prose_is_never_mistaken_for_source(text):
    assert looks_like_source(text) is False


def test_a_single_declaration_is_not_enough():
    """One 'import' in a sentence must not turn an answer into a code block."""
    assert looks_like_source("import matters here.\nSo does clarity.") is False


# --- what the normaliser does with it -------------------------------------------


def test_raw_code_is_fenced_so_its_lines_survive():
    result = as_markdown(CODE)
    assert result.startswith(FENCE)
    assert result.endswith(FENCE)
    # The structure Markdown would have destroyed is still there.
    assert "def encode(text: str) -> bytes:" in result
    assert '    if not text:' in result


def test_prose_is_returned_exactly_as_written():
    assert as_markdown(PROSE) == PROSE


def test_an_answer_that_already_fences_its_code_is_left_alone():
    """The model did the job; second-guessing it is how a formatter corrupts."""
    written = f"Here is the script:\n\n{FENCE}python\nimport os\n{FENCE}\n"
    assert as_markdown(written) == written


def test_an_empty_answer_is_untouched():
    assert as_markdown("") == ""
    assert as_markdown("   \n ") == "   \n "


def test_the_result_is_always_renderable_markdown():
    """A fence that is opened must be closed, or the rest of the answer is eaten."""
    assert as_markdown(CODE).count(FENCE) == 2
