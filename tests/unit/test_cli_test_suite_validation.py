"""Tests for local test-suite validation in `connic lint` and `connic test`."""

import json
from pathlib import Path
from textwrap import dedent

import pytest
from click.testing import CliRunner

from connic import cli


@pytest.fixture(autouse=True)
def disable_update_checks(monkeypatch):
    monkeypatch.setenv("CONNIC_NO_UPDATE_CHECK", "1")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n")


def _write_agent(project: Path, name: str) -> None:
    _write(
        project / "agents" / f"{name}.yaml",
        f"""
        version: "1.0"
        name: {name}
        type: llm
        model: openai/gpt-5.2
        description: "Test agent"
        system_prompt: "Help the user."
        """,
    )


def _write_suite(path: Path, agent: str | None = None) -> None:
    lines = ['version: "1.0"']
    if agent:
        lines.append(f"agent: {agent}")
    lines += ["tests:", "  - name: answers", '    payload: "hello"']
    _write(path, "\n".join(lines))


def test_lint_passes_when_every_suite_targets_a_loaded_agent(tmp_path, capsys):
    _write_agent(tmp_path, "order-agent")
    _write_suite(tmp_path / "tests" / "order-agent.yaml")
    _write_suite(tmp_path / "tests" / "nested" / "refunds.yaml", agent="order-agent")

    assert cli._run_lint(project_root=str(tmp_path)) is True
    assert "2 test suite(s) found" in capsys.readouterr().out


def test_lint_reports_suite_whose_file_name_matches_no_agent(tmp_path, capsys):
    _write_agent(tmp_path, "order-agent")
    _write_suite(tmp_path / "tests" / "order-agnet.yaml")

    assert cli._run_lint(project_root=str(tmp_path)) is False
    err = capsys.readouterr().err
    assert "tests/order-agnet.yaml: unknown agent 'order-agnet' (from the file name" in err
    assert "Did you mean 'order-agent'?" in err


def test_lint_reports_suite_whose_agent_field_matches_no_agent(tmp_path, capsys):
    _write_agent(tmp_path, "order-agent")
    _write_suite(tmp_path / "tests" / "nested" / "refunds.yaml", agent="order_agent")

    assert cli._run_lint(quiet=True, project_root=str(tmp_path)) is False
    err = capsys.readouterr().err
    assert "tests/nested/refunds.yaml: unknown agent 'order_agent'. Did you mean 'order-agent'?" in err
    assert "from the file name" not in err


def test_lint_reports_suite_schema_errors(tmp_path, capsys):
    _write_agent(tmp_path, "order-agent")
    _write(
        tmp_path / "tests" / "order-agent.yaml",
        """
        version: "1.0"
        tests: []
        """,
    )

    assert cli._run_lint(project_root=str(tmp_path)) is False
    err = capsys.readouterr().err
    assert "tests/order-agent.yaml:" in err
    assert "at least 1 item" in err


def test_lint_skips_files_the_test_runner_never_reads(tmp_path):
    _write_agent(tmp_path, "order-agent")
    _write_suite(tmp_path / "tests" / "unknown.yml")
    _write_suite(tmp_path / "tests" / ".drafts" / "unknown.yaml")
    (tmp_path / "tests" / "empty.yaml").write_text("")

    assert cli._run_lint(quiet=True, project_root=str(tmp_path)) is True


def test_lint_rejects_suite_linked_outside_tests(tmp_path, capsys):
    project = tmp_path / "project"
    _write_agent(project, "order-agent")
    outside = tmp_path / "outside.yaml"
    _write_suite(outside, agent="order-agent")
    (project / "tests").mkdir()
    (project / "tests" / "order-agent.yaml").symlink_to(outside)

    assert cli._run_lint(quiet=True, project_root=str(project)) is False
    assert "is outside" in capsys.readouterr().err


@pytest.mark.parametrize("as_json", [False, True])
def test_test_command_stops_on_unknown_agent_before_upload(tmp_path, monkeypatch, as_json):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_test_secret", "project_id": "proj_123"}))
    _write_agent(tmp_path, "order-agent")
    _write_suite(tmp_path / "tests" / "orders.yaml")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("test should not package or upload a project with an invalid suite")

    monkeypatch.setattr(cli, "_package_project_for_tests", fail_if_called)
    monkeypatch.setattr(cli.httpx, "Client", fail_if_called)

    result = CliRunner().invoke(cli.main, ["test"] + (["--json"] if as_json else []))

    assert result.exit_code == 1, result.output
    assert "tests/orders.yaml: unknown agent 'orders'" in result.stderr
    assert "`connic lint` checks the whole project" in result.stderr
