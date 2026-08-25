"""Generator tests: fence handling, assembly, and the upstream-key contract."""

from __future__ import annotations

import ast
import json
import subprocess
import sys

import pytest

import generator
from codeguard import check_agent_source
from config import USAGE_MARKER

FENCE = "`" * 3
BODY = (
    "def run(task, previous_outputs):\n"
    "    return call_llm(task + format_previous(previous_outputs))\n"
)


# --- M5: fence stripping must not drop or keep the wrong thing ------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("print(1)\n", "print(1)\n"),
        (f"{FENCE}python\nprint(1)\n{FENCE}", "print(1)"),
        (f"Here:\n{FENCE}python\nprint(1)\n{FENCE}\nHope that helps!", "print(1)"),
        (f"{FENCE}\nprint(1)\n{FENCE}", "print(1)"),
        (f"{FENCE}python\r\nprint(1)\r\n{FENCE}", "print(1)"),
    ],
)
def test_single_block_cases(raw, expected):
    assert generator._strip_code_fences(raw) == expected


def test_every_block_is_kept():
    raw = f"{FENCE}python\nimport os\n{FENCE}\nand then\n{FENCE}python\nprint(2)\n{FENCE}"
    assert generator._strip_code_fences(raw) == "import os\n\nprint(2)"


def test_unterminated_fence_does_not_leak_markdown():
    stripped = generator._strip_code_fences(f"{FENCE}python\nprint(1)\n")
    assert FENCE not in stripped
    ast.parse(stripped)


# --- the trusted header ---------------------------------------------------------


def test_header_parses_and_has_no_placeholders_left():
    header = generator._render_header()
    ast.parse(header)
    assert "@@" not in header


def test_assembled_agent_is_valid_and_self_contained():
    source = generator.assemble_agent(BODY)
    assert check_agent_source(source) == []
    tree = ast.parse(source)
    imported = {
        node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
    }
    assert "langchain_openai" not in imported  # H6: no framework import
    assert imported <= {"json", "os", "re", "sys", "time", "urllib"}


def test_assembled_agent_defines_the_contract():
    functions = {
        node.name
        for node in ast.parse(generator.assemble_agent(BODY)).body
        if isinstance(node, ast.FunctionDef)
    }
    assert {"run", "call_llm", "format_previous", "api_key"} <= functions


# --- C3: the generator is told the exact previous_outputs keys ------------------


def test_first_agent_is_told_the_dict_is_empty():
    contract = generator._upstream_contract([])
    assert "FIRST agent" in contract
    assert "empty dict" in contract


def test_upstream_names_are_listed_verbatim():
    contract = generator._upstream_contract(["research_agent", "risk_agent"])
    assert '"research_agent"' in contract
    assert '"risk_agent"' in contract
    assert "Never invent a key" in contract


def test_prompt_carries_the_upstream_contract():
    prompt = generator.GENERATOR_PROMPT.format(
        name="summary_agent",
        role="r",
        instructions="i",
        upstream_contract=generator._upstream_contract(["research_agent"]),
    )
    assert '"research_agent"' in prompt


# --- the generated runtime actually behaves (no network involved) ---------------


def _run_agent(tmp_path, body, payload, extra_env=None):
    path = tmp_path / "agent_under_test.py"
    path.write_text(generator.assemble_agent(body), encoding="utf-8")
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PATH": ""}
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, str(path)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=60,
    )


def test_format_previous_labels_and_caps(tmp_path):
    body = "def run(task, previous_outputs):\n    return format_previous(previous_outputs)\n"
    completed = _run_agent(
        tmp_path, body, {"task": "t", "previous_outputs": {"research_agent": "FINDINGS"}}
    )
    assert completed.returncode == 0, completed.stderr
    assert "### research_agent" in completed.stdout
    assert "FINDINGS" in completed.stdout


def test_format_previous_handles_no_upstream(tmp_path):
    body = "def run(task, previous_outputs):\n    return format_previous(previous_outputs)\n"
    completed = _run_agent(tmp_path, body, {"task": "t", "previous_outputs": {}})
    assert completed.stdout.strip() == "(none)"


def test_missing_previous_outputs_key_is_tolerated(tmp_path):
    body = "def run(task, previous_outputs):\n    return str(len(previous_outputs))\n"
    completed = _run_agent(tmp_path, body, {"task": "t"})
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "0"


def test_missing_api_key_fails_loudly(tmp_path, monkeypatch):
    body = "def run(task, previous_outputs):\n    return call_llm('hi')\n"
    # HOME/USERPROFILE are cleared so no stray .env up the tree is found.
    completed = _run_agent(
        tmp_path, body, {"task": "t", "previous_outputs": {}}, {"HOME": str(tmp_path)}
    )
    assert completed.returncode != 0
    assert "OPENROUTER_API_KEY" in completed.stderr


def test_usage_marker_constant_matches_config():
    assert USAGE_MARKER in generator._render_header()
