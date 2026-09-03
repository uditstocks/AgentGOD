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
from planner import AgentSpec

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
    assert "anthropic" not in imported  # H6: no SDK import
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
    prompt = generator.GENERATOR_BRIEF.format(
        name="summary_agent",
        role="r",
        instructions="i",
        upstream_contract=generator._upstream_contract(["research_agent"]),
        packages=generator._package_rule(AgentSpec(name="summary_agent", role="r", instructions="i")),
    )
    assert '"research_agent"' in prompt


def test_the_static_policy_is_separate_from_the_per_agent_brief():
    """The cached half must hold no per-agent text, or the cache never hits."""
    policy = generator.GENERATOR_POLICY
    assert "call_llm(" in policy  # the contract lives in the cached half
    assert "REUSABLE" in policy
    assert "{name}" not in policy and "{role}" not in policy
    assert "{instructions}" not in policy and "{packages}" not in policy


# --- an agent may import only what was actually installed for it ----------------


def test_an_agent_with_no_dependencies_is_told_not_to_import_one():
    spec = AgentSpec(name="summary_agent", role="r", instructions="i")
    assert "Do not import one" in generator._package_rule(spec)


def test_declared_packages_are_named_by_their_import_name():
    """The planner declares 'pillow'; the code has to write 'import PIL'."""
    spec = AgentSpec(
        name="chart_agent", role="r", instructions="i", dependencies=["pillow", "qrcode>=7.0"]
    )
    rule = generator._package_rule(spec)
    assert "PIL" in rule and "qrcode" in rule
    assert "pillow" not in rule


def test_a_package_that_would_be_refused_is_never_offered():
    """It is not installed, so telling the model it exists guarantees a crash."""
    spec = AgentSpec(name="odd_agent", role="r", instructions="i", dependencies=["leftpad"])
    assert "Do not import one" in generator._package_rule(spec)


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
    assert "ANTHROPIC_API_KEY" in completed.stderr


def test_usage_marker_constant_matches_config():
    assert USAGE_MARKER in generator._render_header()


# --- the wire shape of the agent runtime's Messages API call --------------------


def _load_runtime(monkeypatch, reply, captured):
    """Exec the trusted header in isolation, with the network stubbed out.

    `reply` may be one payload or a list of them, so a paused turn and its
    resumption can be scripted.
    """
    import urllib.request

    namespace: dict = {}
    exec(compile(generator._render_header(), "<agent-runtime>", "exec"), namespace)

    replies = list(reply) if isinstance(reply, list) else [reply]
    captured["bodies"] = []

    class _Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def _fake_urlopen(request, timeout=None):
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["bodies"].append(captured["body"])
        return _Response(replies.pop(0) if len(replies) > 1 else replies[0])

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    return namespace


def test_call_llm_sends_a_messages_api_request(monkeypatch):
    captured: dict = {}
    reply = {"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 4}}
    runtime = _load_runtime(monkeypatch, reply, captured)

    runtime["call_llm"]("material", system="rules", max_tokens=99)

    # Header names arrive title-cased through urllib's own normalisation.
    headers = {name.lower(): value for name, value in captured["headers"].items()}
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"]

    body = captured["body"]
    assert body["system"] == "rules"
    assert body["messages"] == [{"role": "user", "content": "material"}]
    assert body["max_tokens"] == 99  # required by the API, so never omitted
    assert "temperature" not in body  # current models reject it


def test_call_llm_ignores_temperature_from_older_agents(monkeypatch):
    """Agents written before the switch still pass it; sending it would 400."""
    captured: dict = {}
    reply = {"content": [{"type": "text", "text": "hi"}], "usage": {}}
    runtime = _load_runtime(monkeypatch, reply, captured)

    runtime["call_llm"]("material", temperature=0.7)

    assert "temperature" not in captured["body"]


def test_call_llm_returns_the_answer_not_the_reasoning(monkeypatch):
    captured: dict = {}
    reply = {
        "content": [
            {"type": "thinking", "thinking": "weighing it up"},
            {"type": "text", "text": "the answer"},
        ],
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }
    runtime = _load_runtime(monkeypatch, reply, captured)

    assert runtime["call_llm"]("material") == "the answer"


def test_call_llm_reports_usage_on_stderr(monkeypatch, capsys):
    captured: dict = {}
    reply = {"content": [{"type": "text", "text": "hi"}],
             "usage": {"input_tokens": 11, "output_tokens": 5}}
    runtime = _load_runtime(monkeypatch, reply, captured)

    runtime["call_llm"]("material")

    line = capsys.readouterr().err.strip()
    assert line.startswith(USAGE_MARKER)
    assert json.loads(line[len(USAGE_MARKER) :]) == {"input_tokens": 11, "output_tokens": 5}


# --- looking things up: the tool is declared, and a paused search resumes -------


def test_search_is_off_unless_the_agent_asks_for_it(monkeypatch):
    """A search costs time and money on every call, so it is never the default."""
    captured: dict = {}
    reply = {"content": [{"type": "text", "text": "hi"}], "usage": {}}
    runtime = _load_runtime(monkeypatch, reply, captured)

    runtime["call_llm"]("material")

    assert "tools" not in captured["body"]


def test_search_declares_the_server_side_tool(monkeypatch):
    captured: dict = {}
    reply = {"content": [{"type": "text", "text": "hi"}], "usage": {}}
    runtime = _load_runtime(monkeypatch, reply, captured)

    runtime["call_llm"]("material", search=True)

    tools = captured["body"]["tools"]
    assert tools[0]["name"] == "web_search"
    assert tools[0]["type"].startswith("web_search_")
    assert tools[0]["max_uses"] >= 1  # the cost ceiling, not decoration


def test_a_paused_search_is_resumed_and_the_answer_still_arrives(monkeypatch):
    """The server pauses a long search and expects the turn handed straight back."""
    captured: dict = {}
    paused = {
        "content": [{"type": "server_tool_use", "name": "web_search"}],
        "stop_reason": "pause_turn",
        "usage": {"input_tokens": 100},
    }
    finished = {
        "content": [{"type": "text", "text": "the researched answer"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 200},
    }
    runtime = _load_runtime(monkeypatch, [paused, finished], captured)

    assert runtime["call_llm"]("material", search=True) == "the researched answer"

    # The resumed request carries the paused turn back, with no "continue" text.
    resumed = captured["bodies"][-1]["messages"]
    assert len(resumed) == 2
    assert resumed[-1]["role"] == "assistant"


def test_a_search_that_never_finishes_does_not_loop_forever(monkeypatch):
    captured: dict = {}
    paused = {
        "content": [{"type": "server_tool_use", "name": "web_search"}],
        "stop_reason": "pause_turn",
        "usage": {},
    }
    runtime = _load_runtime(monkeypatch, paused, captured)

    runtime["call_llm"]("material", search=True)

    assert len(captured["bodies"]) <= generator.MAX_CONTINUATIONS + 1


def test_search_results_are_not_mistaken_for_the_answer(monkeypatch):
    captured: dict = {}
    reply = {
        "content": [
            {"type": "server_tool_use", "name": "web_search"},
            {"type": "web_search_tool_result", "content": [{"title": "a page"}]},
            {"type": "text", "text": "the answer"},
        ],
        "usage": {},
    }
    runtime = _load_runtime(monkeypatch, reply, captured)

    assert runtime["call_llm"]("material", search=True) == "the answer"
