import json

import click
import httpx
import pytest
from click.testing import CliRunner

from connic import cli

CONTROL_TEXT = "Grüße ✓\x1b[2J\x1b]52;c;YQ==\x07\r\b\t\n\x00\x7f\x9b2J\x9dtitle\x9c"
ESCAPED_TEXT = r"Grüße ✓\x1b[2J\x1b]52;c;YQ==\x07\x0d\x08\x09\x0a\x00\x7f\x9b2J\x9dtitle\x9c"


@pytest.fixture(autouse=True)
def disable_update_checks(monkeypatch):
    monkeypatch.setenv("CONNIC_NO_UPDATE_CHECK", "1")


@pytest.mark.parametrize("command", [["lint"], ["lint", "--verbose"], ["test", "--coverage"]])
def test_project_terminal_output_escapes_controls(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        'name: "support\\n"\n'
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
        f"description: {json.dumps(CONTROL_TEXT)}\n"
        f"output_schema: {json.dumps(CONTROL_TEXT)}\n"
    )
    if command[0] == "lint":
        for directory in ("middleware", "hooks"):
            (tmp_path / directory).mkdir()
            (tmp_path / directory / "invalid\x1b[2J.py").write_text("def broken(:\n")

    result = CliRunner().invoke(cli.main, command, color=True)

    assert result.exit_code == 0, repr(result.output)
    assert f"Could not load output schema '{ESCAPED_TEXT}'" in result.output
    assert "\x1b[2J" not in result.output
    assert "\x1b[36m" in result.output
    if command[0] == "lint":
        assert f"Description: {ESCAPED_TEXT}" in result.output
        assert "middleware for invalid\\x1b[2J" in result.output
        assert "tool hooks for invalid\\x1b[2J" in result.output
    else:
        assert "│ support\\x0a " in result.output
        table_lines = [line for line in click.unstyle(result.output).splitlines() if line.startswith("    │")]
        assert len({len(line) for line in table_lines}) == 1


@pytest.mark.parametrize("as_json", [False, True])
def test_api_test_output_escapes_controls_and_preserves_json(tmp_path, monkeypatch, as_json):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "support.yaml").write_text("tests: []\n")
    case = {
        "agent_name": "support" + CONTROL_TEXT,
        "test_name": "refund" + CONTROL_TEXT,
        "passed": False,
        "successes": 0,
        "runs": 1,
        "success_threshold": 100,
        "failure_reason": "Failure: " + CONTROL_TEXT,
        "agent_run_ids": ["run_123"],
        "agent_run_passed": [False],
    }
    run = {
        "input": "Input: " + CONTROL_TEXT,
        "output": "Output: " + CONTROL_TEXT,
        "error": "Error: " + CONTROL_TEXT,
        "traces": [
            {"name": "Agent: " + CONTROL_TEXT, "status": "ok"},
            {"name": "tool", "status": "error", "metadata_json": json.dumps({"tool_name": "Tool: " + CONTROL_TEXT, "error": CONTROL_TEXT})},
        ],
    }

    def respond(request):
        if request.method == "POST":
            return httpx.Response(202, json={"id": "test_123"})
        if request.url.path.endswith("/test-runs/test_123"):
            return httpx.Response(200, json={"status": "failed", "phase": "Phase: " + CONTROL_TEXT, "cases": [case]})
        assert request.url.path.endswith("/runs/run_123")
        return httpx.Response(200, json=run)

    client_type = httpx.Client
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs))
    args = ["test", "--env", "env_test", "--api-key", "cnc_test", "--project-id", "project_123"]
    if as_json:
        args.append("--json")

    result = CliRunner().invoke(cli.main, args, color=True)

    assert result.exit_code == 1, repr(result.output)
    if as_json:
        parsed_case = json.loads(result.output)["cases"][0]
        assert {key: parsed_case[key] for key in case} == case
        parsed_run = parsed_case["failed_runs"]["shown"][0]
        assert parsed_run["input"] == run["input"]
        assert parsed_run["traces"][1]["metadata_json"] == run["traces"][1]["metadata_json"]
    else:
        for label in ("support", "refund", "Failure: ", "Input: ", "Output: ", "Error: ", "Agent: ", "Tool: ", "Phase: "):
            assert label + ESCAPED_TEXT in result.output
        assert "\x1b[2J" not in result.output
        assert "\x1b[31m\x1b[1m FAIL " in result.output
        table_lines = [line for line in click.unstyle(result.output).splitlines() if line.startswith("    │")]
        assert len({len(line) for line in table_lines}) == 1
