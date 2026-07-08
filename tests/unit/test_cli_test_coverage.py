"""Tests for `connic test --coverage` (offline static coverage report)."""

import json
from pathlib import Path
from textwrap import dedent

from click.testing import CliRunner

from connic import cli

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content.endswith("\n") else content + "\n")


def _write_calculator_tool(project: Path) -> None:
    _write(
        project / "tools" / "calculator.py",
        dedent(
            '''
            def add(a: int, b: int) -> int:
                """Add two numbers."""
                return a + b

            def subtract(a: int, b: int) -> int:
                """Subtract b from a."""
                return a - b
            '''
        ).strip()
        + "\n",
    )


def _write_llm_agent(project: Path, name: str, tools: list[str], discoverable: list[str] | None = None) -> None:
    lines = [
        'version: "1.0"',
        f"name: {name}",
        "type: llm",
        "model: openai/gpt-5.2",
        'description: "Test agent"',
        'system_prompt: "Do the thing."',
        "tools:",
    ]
    lines.extend(f"  - {t}" for t in tools)
    if discoverable:
        lines.append("discoverable_tools:")
        lines.extend(f"  - {t}" for t in discoverable)
    _write(project / "agents" / f"{name}.yaml", "\n".join(lines) + "\n")


def _write_test_file(project: Path, agent_name: str, expected_calls_per_case: list[list]) -> None:
    lines = [
        'version: "1.0"',
        f"agent: {agent_name}",
        "tests:",
    ]
    for i, calls in enumerate(expected_calls_per_case):
        lines.append(f"  - name: case_{i}")
        lines.append('    payload: \'{"x": 1}\'')
        lines.append('    expected_result: status == "completed"')
        if calls:
            lines.append("    expected_tool_calls:")
            for c in calls:
                lines.append(f"      - {c}")
        else:
            lines.append("    expected_tool_calls: []")
    _write(project / "tests" / f"{agent_name}.yaml", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# _compute_local_coverage
# ---------------------------------------------------------------------------

def test_coverage_is_zero_for_agent_without_test_file(tmp_path):
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add", "calculator.subtract"])

    report = cli._compute_local_coverage(tmp_path)

    [agent] = report["agents"]
    assert agent["name"] == "math-agent"
    assert agent["has_tests"] is False
    assert agent["tools_total"] == 2
    assert agent["tools_covered"] == 0
    assert agent["percent"] == 0.0
    assert sorted(agent["uncovered_tools"]) == ["calculator.add", "calculator.subtract"]
    assert report["overall"] == 0.0


def test_coverage_is_full_when_every_tool_appears_in_expected_tool_calls(tmp_path):
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add", "calculator.subtract"])
    _write_test_file(
        tmp_path,
        "math-agent",
        [["calculator.add", "calculator.subtract"]],
    )

    report = cli._compute_local_coverage(tmp_path)

    [agent] = report["agents"]
    assert agent["has_tests"] is True
    assert agent["tools_covered"] == 2
    assert agent["tools_total"] == 2
    assert agent["percent"] == 100.0
    assert agent["uncovered_tools"] == []
    assert report["overall"] == 100.0


def test_partial_coverage_reports_only_missing_tools(tmp_path):
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add", "calculator.subtract"])
    _write_test_file(tmp_path, "math-agent", [["calculator.add"]])

    report = cli._compute_local_coverage(tmp_path)

    [agent] = report["agents"]
    assert agent["tools_covered"] == 1
    assert agent["tools_total"] == 2
    assert agent["percent"] == 50.0
    assert agent["uncovered_tools"] == ["calculator.subtract"]


def test_coverage_combines_split_suites_targeting_the_same_agent(tmp_path):
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add", "calculator.subtract"])
    _write_test_file(tmp_path, "math-agent", [["calculator.add"]])
    _write(tmp_path / "tests" / "files" / "subtraction.json", '{"a": 5, "b": 3}')
    _write(
        tmp_path / "tests" / "math-agent-subtraction.yml",
        """
        version: "1.0"
        agent: math-agent
        tests:
          - name: subtracts_numbers
            payload: '{"a": 5, "b": 3}'
            expected_result: status == "completed"
            expected_tool_calls:
              - calculator.subtract
        """,
    )

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    assert agent["has_tests"] is True
    assert agent["tools_covered"] == 2
    assert agent["percent"] == 100.0
    assert agent["uncovered_tools"] == []


def test_mapping_form_of_expected_tool_calls_counts_as_covered(tmp_path):
    """`- tool: expression` mapping form counts the same as bare strings."""
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add"])
    _write_test_file(
        tmp_path,
        "math-agent",
        [["{calculator.add: invocations >= 1}"]],
    )

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    assert agent["percent"] == 100.0


def test_short_name_in_expected_tool_calls_matches_full_ref(tmp_path):
    """`- add` (bare function name) matches a tool whose ref is `calculator.add`."""
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add"])
    _write_test_file(tmp_path, "math-agent", [["add"]])

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    assert agent["percent"] == 100.0


def test_expected_tool_call_order_counts_as_covered(tmp_path):
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add", "calculator.subtract"])
    _write(
        tmp_path / "tests" / "math-agent.yaml",
        """
        version: "1.0"
        agent: math-agent
        tests:
          - name: ordered_math
            payload: '{"x": 1}'
            expected_result: status == "completed"
            expected_tool_call_order:
              - calculator.add
              - calculator.subtract
        """,
    )

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    assert agent["percent"] == 100.0


def test_each_agent_contributes_equally_to_overall_percentage(tmp_path):
    """Two agents, one fully covered and one untested → overall is 50%."""
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "covered", tools=["calculator.add"])
    _write_llm_agent(tmp_path, "uncovered", tools=["calculator.subtract"])
    _write_test_file(tmp_path, "covered", [["calculator.add"]])

    report = cli._compute_local_coverage(tmp_path)

    by_name = {r["name"]: r for r in report["agents"]}
    assert by_name["covered"]["percent"] == 100.0
    assert by_name["uncovered"]["percent"] == 0.0
    assert report["overall"] == 50.0


def test_discoverable_tools_count_toward_coverage(tmp_path):
    _write_calculator_tool(tmp_path)
    _write_llm_agent(
        tmp_path,
        "math-agent",
        tools=["calculator.add"],
        discoverable=["calculator.subtract"],
    )
    _write_test_file(tmp_path, "math-agent", [["calculator.add"]])

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    # add is covered, subtract (discoverable) is not — but both are in the denominator.
    assert agent["tools_total"] == 2
    assert agent["tools_covered"] == 1
    assert agent["percent"] == 50.0
    assert agent["uncovered_tools"] == ["calculator.subtract"]


def test_auto_injected_search_tools_and_use_tool_markers_are_excluded(tmp_path):
    """When `discoverable_tools` is set, the loader injects `search_tools`/`use_tool`.
    Those are plumbing the user didn't author and must not appear in the denominator."""
    _write_calculator_tool(tmp_path)
    _write_llm_agent(
        tmp_path,
        "math-agent",
        tools=["calculator.add"],
        discoverable=["calculator.subtract"],
    )
    _write_test_file(tmp_path, "math-agent", [["calculator.add", "calculator.subtract"]])

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    assert agent["tools_total"] == 2  # not 4
    assert agent["percent"] == 100.0
    assert "search_tools" not in agent["uncovered_tools"]
    assert "use_tool" not in agent["uncovered_tools"]


def test_agent_with_no_tools_but_a_test_file_is_full_coverage(tmp_path):
    """Sequential / orchestrator-style agents with zero tools shouldn't be penalised
    if the user did write a test for them."""
    _write(
        tmp_path / "agents" / "noop.yaml",
        """
        version: "1.0"
        name: noop
        type: llm
        model: openai/gpt-5.2
        description: ""
        system_prompt: "Just respond."
        """,
    )
    _write_test_file(tmp_path, "noop", [[]])

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    assert agent["tools_total"] == 0
    assert agent["percent"] == 100.0


def test_agent_with_no_tools_and_no_test_file_is_zero_coverage(tmp_path):
    _write(
        tmp_path / "agents" / "noop.yaml",
        """
        version: "1.0"
        name: noop
        type: llm
        model: openai/gpt-5.2
        description: ""
        system_prompt: "Just respond."
        """,
    )

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    assert agent["tools_total"] == 0
    assert agent["has_tests"] is False
    assert agent["percent"] == 0.0


def test_ab_test_variant_agents_are_excluded_from_coverage(tmp_path):
    """Variants like `support-test-fast` share the base agent's tools and shouldn't
    inflate the denominator."""
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "support", tools=["calculator.add"])
    _write_llm_agent(tmp_path, "support-test-fast", tools=["calculator.add"])
    _write_test_file(tmp_path, "support", [["calculator.add"]])

    report = cli._compute_local_coverage(tmp_path)
    names = [r["name"] for r in report["agents"]]
    assert names == ["support"]
    assert report["overall"] == 100.0


def test_tool_agent_body_tool_is_covered_when_test_file_exists(tmp_path):
    """A `type: tool` agent IS its body tool — any test run invokes it,
    so the body tool counts as covered without needing expected_tool_calls."""
    _write_calculator_tool(tmp_path)
    _write(
        tmp_path / "agents" / "adder.yaml",
        dedent(
            """
            version: "1.0"
            name: adder
            type: tool
            description: "Adds two numbers directly."
            tool_name: calculator.add
            """
        ).strip()
        + "\n",
    )
    # No expected_tool_calls — just invoking the agent runs the tool.
    _write_test_file(tmp_path, "adder", [[]])

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    assert agent["tools_total"] == 1
    assert agent["tools_covered"] == 1
    assert agent["percent"] == 100.0
    assert agent["uncovered_tools"] == []


def test_tool_agent_without_test_file_is_zero_coverage(tmp_path):
    _write_calculator_tool(tmp_path)
    _write(
        tmp_path / "agents" / "adder.yaml",
        dedent(
            """
            version: "1.0"
            name: adder
            type: tool
            description: "Adds two numbers directly."
            tool_name: calculator.add
            """
        ).strip()
        + "\n",
    )

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    assert agent["has_tests"] is False
    assert agent["tools_total"] == 1
    assert agent["tools_covered"] == 0
    assert agent["percent"] == 0.0


def test_unparseable_test_file_is_recorded_without_counting_in_overall(tmp_path):
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "covered-agent", tools=["calculator.subtract"])
    _write_test_file(tmp_path, "covered-agent", [["calculator.subtract"]])
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add"])
    _write(tmp_path / "tests" / "math-agent.yaml", "this: is: not: valid: yaml: at all")

    report = cli._compute_local_coverage(tmp_path)
    by_name = {r["name"]: r for r in report["agents"]}
    assert by_name["math-agent"]["has_tests"] is True
    assert by_name["math-agent"]["percent"] is None
    assert by_name["math-agent"]["parse_error"] is not None
    assert report["overall"] == 100.0


def test_unparseable_test_file_for_agent_without_tools_is_not_counted(tmp_path):
    _write(
        tmp_path / "agents" / "noop.yaml",
        """
        version: "1.0"
        name: noop
        type: llm
        model: openai/gpt-5.2
        description: ""
        system_prompt: "Just respond."
        """,
    )
    _write(tmp_path / "tests" / "noop.yaml", "this: is: not: valid: yaml: at all")

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    assert agent["has_tests"] is True
    assert agent["tools_total"] == 0
    assert agent["percent"] is None
    assert agent["parse_error"] is not None


def test_invalid_split_suite_is_reported_for_its_explicit_agent(tmp_path):
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add"])
    _write_test_file(tmp_path, "math-agent", [["calculator.add"]])
    _write(
        tmp_path / "tests" / "math-agent-invalid.yaml",
        """
        version: "1.0"
        agent: math-agent
        tests: []
        """,
    )

    [agent] = cli._compute_local_coverage(tmp_path)["agents"]
    assert agent["has_tests"] is True
    assert agent["percent"] is None
    assert "at least 1 item" in agent["parse_error"]


def test_compute_local_coverage_returns_error_when_agents_dir_is_missing(tmp_path):
    report = cli._compute_local_coverage(tmp_path)
    assert report["agents"] == []
    assert "Agents directory not found" in report["error"]


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_test_command_with_coverage_flag_runs_offline_without_credentials(tmp_path, monkeypatch):
    """--coverage must NOT require API key/project ID and must NOT call the backend."""
    monkeypatch.chdir(tmp_path)
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add"])
    _write_test_file(tmp_path, "math-agent", [["calculator.add"]])

    # No CONNIC_* env vars, no .connic file: a real run would fail with "Run `connic login`".
    monkeypatch.delenv("CONNIC_API_KEY", raising=False)
    monkeypatch.delenv("CONNIC_PROJECT_ID", raising=False)

    result = CliRunner().invoke(cli.main, ["test", "--coverage"])

    assert result.exit_code == 0, result.output
    assert "Test Coverage" in result.output
    assert "math-agent" in result.output
    assert "100.0%" in result.output
    assert "Overall coverage" in result.output


def test_test_command_with_coverage_lists_uncovered_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add", "calculator.subtract"])
    _write_test_file(tmp_path, "math-agent", [["calculator.add"]])
    monkeypatch.delenv("CONNIC_API_KEY", raising=False)
    monkeypatch.delenv("CONNIC_PROJECT_ID", raising=False)

    result = CliRunner().invoke(cli.main, ["test", "--coverage"])

    assert result.exit_code == 0, result.output
    assert "math-agent" in result.output
    assert "1/2" in result.output
    assert "50.0%" in result.output
    assert "Uncovered tools" in result.output
    assert "math-agent: calculator.subtract" in result.output


def test_test_command_with_coverage_reports_agent_without_tests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "untested-agent", tools=["calculator.add"])
    monkeypatch.delenv("CONNIC_API_KEY", raising=False)
    monkeypatch.delenv("CONNIC_PROJECT_ID", raising=False)

    result = CliRunner().invoke(cli.main, ["test", "--coverage"])

    assert result.exit_code == 0, result.output
    assert "untested-agent" in result.output
    assert "no tests" in result.output
    assert "0.0%" in result.output
    assert "Overall coverage: 0.0%" in result.output


def test_test_command_with_coverage_reports_empty_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agents").mkdir()
    monkeypatch.delenv("CONNIC_API_KEY", raising=False)
    monkeypatch.delenv("CONNIC_PROJECT_ID", raising=False)

    result = CliRunner().invoke(cli.main, ["test", "--coverage"])

    assert result.exit_code == 0, result.output
    assert "Test Coverage" in result.output
    assert "No agents found." in result.output
    assert "API key and project ID required" not in result.output


def test_test_command_with_coverage_reports_missing_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONNIC_API_KEY", raising=False)
    monkeypatch.delenv("CONNIC_PROJECT_ID", raising=False)

    result = CliRunner().invoke(cli.main, ["test", "--coverage"])

    assert result.exit_code != 0
    assert "Agents directory not found" in result.output
    assert "API key and project ID required" not in result.output


def test_test_command_with_coverage_stops_on_unparseable_test_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add"])
    _write(tmp_path / "tests" / "math-agent.yaml", "this: is: not: valid: yaml: at all")
    monkeypatch.delenv("CONNIC_API_KEY", raising=False)
    monkeypatch.delenv("CONNIC_PROJECT_ID", raising=False)

    result = CliRunner().invoke(cli.main, ["test", "--coverage"])

    assert result.exit_code != 0
    assert "Test files failed to parse" in result.output
    assert "math-agent" in result.output
    assert "Overall coverage" not in result.output


def test_test_command_with_coverage_and_json_emits_machine_readable_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_calculator_tool(tmp_path)
    _write_llm_agent(tmp_path, "math-agent", tools=["calculator.add", "calculator.subtract"])
    _write_test_file(tmp_path, "math-agent", [["calculator.add"]])
    monkeypatch.delenv("CONNIC_API_KEY", raising=False)
    monkeypatch.delenv("CONNIC_PROJECT_ID", raising=False)

    result = CliRunner().invoke(cli.main, ["test", "--coverage", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["overall"] == 50.0
    [agent] = payload["agents"]
    assert agent["name"] == "math-agent"
    assert agent["tools_covered"] == 1
    assert agent["tools_total"] == 2
    assert agent["uncovered_tools"] == ["calculator.subtract"]
