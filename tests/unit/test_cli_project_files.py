import base64
import io
import json
import os
import signal
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from connic import cli, update_check


@pytest.fixture(autouse=True)
def hide_installed_plugin_clients(monkeypatch):
    monkeypatch.setattr(update_check.shutil, "which", lambda executable: None)


def _write_minimal_support_agent(project: Path) -> None:
    (project / "agents").mkdir(exist_ok=True)
    (project / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support\n"
        "description: Support agent\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
    )


def test_validate_project_files_accepts_real_project_tree_and_skips_generated_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agents" / "support").mkdir(parents=True)
    (tmp_path / "agents" / "support" / "agent.yaml").write_text("name: support\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "support.py").write_text("def lookup_ticket(ticket_id: str) -> dict:\n    return {}\n")
    (tmp_path / "tools" / "__pycache__").mkdir()
    (tmp_path / "tools" / "__pycache__" / "support.cpython-312.pyc").write_bytes(b"cached")
    (tmp_path / "tools" / ".secret.py").write_text("TOKEN = 'hidden'\n")
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "reply.json").write_text('{"type":"object"}\n')
    (tmp_path / "requirements.txt").write_text("httpx>=0.25.0\n")

    is_valid, error, files = cli._validate_project_files()

    assert is_valid is True
    assert error == ""
    assert {path.as_posix() for path in files} == {
        "agents/support/agent.yaml",
        "tools/support.py",
        "schemas/reply.json",
        "requirements.txt",
    }


@pytest.mark.parametrize("project_path", [Path("tools/outside.py"), Path("requirements.txt")])
def test_package_project_for_tests_rejects_symlinked_files(tmp_path, monkeypatch, project_path):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("SECRET = 'outside project'\n")
    link = tmp_path / project_path
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="Symbolic links are not allowed"):
        cli._package_project_for_tests(quiet=True)


def test_validate_project_files_rejects_symlinked_project_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside_tools = tmp_path.parent / f"{tmp_path.name}-tools"
    outside_tools.mkdir()
    (outside_tools / "lookup.py").write_text("def lookup():\n    return 'outside project'\n")
    (tmp_path / "tools").symlink_to(outside_tools, target_is_directory=True)

    is_valid, error, files = cli._validate_project_files()

    assert is_valid is False
    assert error == "Symbolic links are not allowed: tools"
    assert files == []


def test_package_project_for_tests_rejects_hard_linked_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "tools" / "lookup.py"
    source.parent.mkdir()
    source.write_text("def lookup():\n    return {'status': 'ok'}\n")
    alias = tmp_path / "tools" / "lookup_alias.py"
    alias.hardlink_to(source)

    with pytest.raises(ValueError, match="Hard links are not allowed"):
        cli._package_project_for_tests(quiet=True)


def test_package_project_for_tests_rejects_requirements_hard_link(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dependency_list = tmp_path / "tools" / "dependencies.txt"
    dependency_list.parent.mkdir()
    dependency_list.write_text("httpx>=0.25\n")
    (tmp_path / "requirements.txt").hardlink_to(dependency_list)

    with pytest.raises(ValueError, match="Hard links are not allowed"):
        cli._package_project_for_tests(quiet=True)


def test_package_project_for_tests_allows_file_hard_linked_outside_upload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shared_source = tmp_path / "shared_lookup.py"
    shared_source.write_text("def lookup():\n    return {'status': 'ok'}\n")
    tool = tmp_path / "tools" / "lookup.py"
    tool.parent.mkdir()
    tool.hardlink_to(shared_source)

    tar_data, files, _ = cli._package_project_for_tests(quiet=True)

    assert files == [Path("tools/lookup.py")]
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as archive:
        assert archive.getmember("tools/lookup.py").isreg()


def test_validate_project_files_reports_file_removed_during_hard_link_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tool = Path("tools/lookup.py")
    tool.parent.mkdir()
    tool.write_text("def lookup():\n    return {'status': 'ok'}\n")
    original_lstat = Path.lstat
    tool_lstat_calls = 0

    def lstat(path):
        nonlocal tool_lstat_calls
        if path == tool:
            tool_lstat_calls += 1
            if tool_lstat_calls == 2:
                raise OSError("file was removed")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", lstat)

    is_valid, error, files = cli._validate_project_files()

    assert is_valid is False
    assert error == "Could not inspect tools/lookup.py: file was removed"
    assert files == []


def test_validate_project_files_allows_zero_inode_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path("tools/lookup.py")
    source.parent.mkdir()
    source.write_text("def lookup():\n    return {'status': 'ok'}\n")
    alias = Path("tools/lookup_alias.py")
    alias.hardlink_to(source)
    original_lstat = Path.lstat

    def lstat(path):
        result = original_lstat(path)
        if path in {source, alias}:
            values = list(result)
            values[1] = 0
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "lstat", lstat)

    is_valid, error, files = cli._validate_project_files()

    assert is_valid is True
    assert error == ""
    assert set(files) == {source, alias}


def test_validate_project_files_rejects_unsupported_files_before_upload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "malware.exe").write_bytes(b"not an agent")

    is_valid, error, files = cli._validate_project_files()

    assert is_valid is False
    assert "File type '.exe' not allowed" in error
    assert files == []


def test_validate_project_files_rejects_projects_over_upload_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "large.py").write_bytes(b"x" * (cli.MAX_UPLOAD_SIZE + 1))

    is_valid, error, files = cli._validate_project_files()

    assert is_valid is False
    assert f"Total file size exceeds {cli.MAX_UPLOAD_SIZE:,} byte limit" in error
    assert files == []


def test_validate_project_files_counts_requirements_toward_code_size_cap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "requirements.txt").write_text(
        "\n".join(["httpx>=0.27"] * (cli.MAX_CODE_SIZE // 11 + 100))
    )

    is_valid, error, files = cli._validate_project_files()

    assert is_valid is False
    assert "Code/config size exceeds" in error
    assert "Move large fixtures into tests/files/" in error
    assert files == []


def test_validate_project_files_allows_binary_fixtures_under_tests_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fixture = tmp_path / "tests" / "files" / "invoice.pdf"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"%PDF-1.7\n" + b"x" * cli.MAX_CODE_SIZE)

    is_valid, error, files = cli._validate_project_files()

    assert is_valid is True
    assert error == ""
    assert [path.as_posix() for path in files] == ["tests/files/invoice.pdf"]


def test_validate_project_files_rejects_non_python_builders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    builder = tmp_path / "tests" / "builders" / "invoice.json"
    builder.parent.mkdir(parents=True)
    builder.write_text('{"builder": true}\n')

    is_valid, error, files = cli._validate_project_files()

    assert is_valid is False
    assert "only .py files are allowed under tests/builders/" in error
    assert files == []


def test_write_essential_files_creates_minimal_scaffold_without_overwriting_existing_files(tmp_path):
    (tmp_path / "README.md").write_text("# Existing\n")

    cli._write_essential_files(tmp_path, quiet=True)

    assert (tmp_path / ".gitignore").read_text().startswith("# Connic")
    assert "Add your tool dependencies" in (tmp_path / "requirements.txt").read_text()
    assert (tmp_path / "README.md").read_text() == "# Existing\n"


def test_merge_template_copies_agent_under_template_namespace_and_merges_requirements(tmp_path):
    template = tmp_path / "templates" / "invoice"
    (template / "agents").mkdir(parents=True)
    (template / "tools").mkdir()
    (template / "tests" / "mocks").mkdir(parents=True)
    (template / "agents" / "extractor.yaml").write_text("name: invoice-extractor\n")
    (template / "agents" / "_draft.yaml").write_text("name: draft\n")
    (template / "tools" / "invoice_tools.py").write_text("def parse_invoice():\n    return {}\n")
    (template / "tests" / "invoice-extractor.yaml").write_text("agent: invoice-extractor\n")
    (template / "tests" / "mocks" / "invoice_mocks.py").write_text("def mock_lookup():\n    return {}\n")
    (template / "requirements.txt").write_text("pypdf>=4\nhttpx>=0.25\n")
    (template / "README.md").write_text("# Invoice Agent\n\nExtract invoices.\n")
    project = tmp_path / "project"
    project.mkdir()

    requirements = []
    readme = cli._merge_template_into_project(template, project, requirements, "invoice")

    assert (project / "agents" / "invoice" / "extractor.yaml").read_text() == "name: invoice-extractor\n"
    assert not (project / "agents" / "invoice" / "_draft.yaml").exists()
    assert (project / "tools" / "invoice_tools.py").exists()
    assert (project / "tests" / "invoice-extractor.yaml").read_text() == "agent: invoice-extractor\n"
    assert (project / "tests" / "mocks" / "invoice_mocks.py").exists()
    assert requirements == ["pypdf>=4", "httpx>=0.25"]
    assert readme == "# Invoice Agent\n\nExtract invoices."


def test_write_merged_requirements_deduplicates_by_package_name(tmp_path):
    cli._write_merged_requirements(
        tmp_path,
        [
            "# base",
            "httpx>=0.25.0",
            "httpx==0.27.0",
            "pydantic>=2",
            "# base",
            "",
        ],
    )

    assert (tmp_path / "requirements.txt").read_text().splitlines() == [
        "# base",
        "httpx>=0.25.0",
        "pydantic>=2",
    ]


def test_install_skill_replaces_existing_project_skill(tmp_path):
    source = tmp_path / "source"
    (source / "references").mkdir(parents=True)
    (source / "SKILL.md").write_text("# Connic\n")
    (source / "references" / "agent-yaml.md").write_text("agent docs\n")
    destination = tmp_path / ".agents" / "skills" / "connic"
    destination.mkdir(parents=True)
    (destination / "stale.md").write_text("remove me\n")

    cli._install_skill(source, destination)

    assert (destination / "SKILL.md").read_text() == "# Connic\n"
    assert (destination / "references" / "agent-yaml.md").read_text() == "agent docs\n"
    assert not (destination / "stale.md").exists()


def test_install_skill_replaces_existing_skill_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("# Current\n")
    legacy = tmp_path / "legacy-skill"
    legacy.mkdir()
    (legacy / "SKILL.md").write_text("# Legacy\n")
    destination = tmp_path / ".claude" / "skills" / "connic"
    destination.parent.mkdir(parents=True)
    destination.symlink_to(legacy, target_is_directory=True)

    cli._install_skill(source, destination)

    assert not destination.is_symlink()
    assert (destination / "SKILL.md").read_text() == "# Current\n"


def test_skill_command_installs_fetched_skill_into_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    source = tmp_path / "fetched-skill"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text("# Connic\n")
    monkeypatch.setattr(cli, "_fetch_skill_from_github", lambda: source)
    plugin_offers = []
    monkeypatch.setattr(cli, "_offer_plugin_installs", lambda: plugin_offers.append(True))

    result = CliRunner().invoke(cli.main, ["skill"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".agents" / "skills" / "connic" / "SKILL.md").read_text() == "# Connic\n"
    assert (tmp_path / ".claude" / "skills" / "connic" / "SKILL.md").read_text() == "# Connic\n"
    assert "Fetched" in result.output
    assert "Connic skill is ready" in result.output
    assert plugin_offers == [True]


def test_plugin_offer_prompts_each_detected_client_and_installs_accepted_client(monkeypatch):
    monkeypatch.setattr(cli, "_can_prompt_for_plugins", lambda: True)
    monkeypatch.setattr(cli.shutil, "which", lambda executable: f"/usr/bin/{executable}")
    monkeypatch.setattr(cli, "_plugin_is_installed", lambda executable: False)

    prompts = []
    answers = iter([True, False])

    def confirm(prompt, default):
        prompts.append((prompt, default))
        return next(answers)

    commands = []

    def run(command, **kwargs):
        commands.append(tuple(command))
        return cli.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.click, "confirm", confirm)
    monkeypatch.setattr(cli.subprocess, "run", run)

    cli._offer_plugin_installs()

    assert [prompt for prompt, _ in prompts] == [
        "  Codex detected. Install the Connic plugin (skill + MCP)?",
        "  Claude Code detected. Install the Connic plugin (skill + MCP)?",
    ]
    assert [default for _, default in prompts] == [False, False]
    assert commands == [
        ("codex", "plugin", "marketplace", "add", "connic-org/connic-skill"),
        ("codex", "plugin", "add", "connic@connic"),
    ]


@pytest.mark.parametrize(
    ("executable", "payload", "expected"),
    [
        ("claude", [{"id": "connic@connic", "scope": "user", "enabled": True}], True),
        ("claude", [{"id": "connic@connic", "scope": "project", "enabled": True}], False),
        ("claude", [{"id": "connic@connic", "scope": "user", "enabled": False}], False),
        (
            "claude",
            [
                {"id": "connic@connic", "scope": "project", "enabled": True},
                {"id": "connic@connic", "scope": "user", "enabled": True},
            ],
            True,
        ),
        ("codex", {"installed": [{"pluginId": "connic@connic"}]}, True),
    ],
)
def test_plugin_installed_check_supports_claude_and_codex_json(
    monkeypatch,
    executable,
    payload,
    expected,
):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: cli.subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(payload),
        ),
    )

    assert cli._plugin_is_installed(executable) is expected


def test_noninteractive_plugin_offer_only_prints_setup_link(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "fetched-skill"
    source.mkdir()
    (source / "SKILL.md").write_text("# Connic\n")
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_fetch_skill_from_github", lambda: source)
    monkeypatch.setattr(cli, "_can_prompt_for_plugins", lambda: False)
    monkeypatch.setattr(cli.shutil, "which", lambda executable: f"/usr/bin/{executable}")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("noninteractive setup must not run client commands"),
    )

    result = CliRunner().invoke(cli.main, ["skill"])

    assert result.exit_code == 0, result.output
    assert cli.AI_AGENT_SETUP_URL in result.output


def test_plugin_install_attempts_install_when_marketplace_add_fails(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(tuple(command))
        return cli.subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(cli.subprocess, "run", run)

    cli._install_plugin_for_client("Codex", cli.PLUGIN_INSTALLS[0][2])

    assert commands == list(cli.PLUGIN_INSTALLS[0][2])


def test_fetch_skill_from_github_extracts_main_branch_archive(monkeypatch):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as archive:
        archive.writestr("connic-skill-main/plugins/connic/skills/connic/SKILL.md", "# Connic\n")
        archive.writestr("connic-skill-main/plugins/connic/skills/connic/references/agent-yaml.md", "agent docs\n")

    class Response:
        content = zip_buffer.getvalue()

        @staticmethod
        def raise_for_status():
            return None

    class FakeClient:
        def __init__(self, timeout, follow_redirects):
            self.timeout = timeout
            self.follow_redirects = follow_redirects

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            assert url == cli.SKILL_ZIP_URL
            return Response()

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    extracted = cli._fetch_skill_from_github()

    assert extracted is not None
    assert (extracted / "SKILL.md").read_text() == "# Connic\n"
    assert (extracted / "references" / "agent-yaml.md").read_text() == "agent docs\n"


def test_parse_login_token_requires_project_and_api_key():
    assert cli._parse_login_token("proj_123:key_456") == ("proj_123", "key_456")

    with pytest.raises(SystemExit):
        cli._parse_login_token("missing-separator")

    with pytest.raises(SystemExit):
        cli._parse_login_token("proj_123:")


def test_login_command_opens_dashboard_and_saves_interactive_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    opened_urls = []

    def fake_open(url):
        opened_urls.append(url)
        return True

    monkeypatch.setattr("webbrowser.open", fake_open)

    result = CliRunner().invoke(
        cli.main,
        ["login", "--base-url", "https://app.connic.test"],
        input="proj_123:cnc_live_secret\n",
    )

    assert result.exit_code == 0, result.output
    assert opened_urls == ["https://app.connic.test/projects?to=/settings/cli?add=1"]
    assert json.loads((tmp_path / ".connic").read_text()) == {
        "api_key": "cnc_live_secret",
        "project_id": "proj_123",
    }
    assert "Credentials saved to .connic" in result.output
    assert "API key:  cnc_live_sec..." in result.output
    assert "Project:  proj_123" in result.output


def test_init_command_creates_documented_minimal_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)

    result = CliRunner().invoke(cli.main, ["init", "my-agents"])

    assert result.exit_code == 0, result.output
    project = tmp_path / "my-agents"
    assert (project / "agents").is_dir()
    assert (project / "tools").is_dir()
    assert (project / "middleware").is_dir()
    assert (project / "schemas").is_dir()
    assert (project / ".gitignore").read_text().startswith("# Connic")
    assert "Add your tool dependencies" in (project / "requirements.txt").read_text()
    readme = (project / "README.md").read_text()
    assert "Connic Agent Project" in readme
    assert "connic lint" in readme
    assert "Created directory: my-agents" in result.output


def test_init_command_with_skill_installs_skill_into_new_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    source = tmp_path / "fetched-skill"
    source.mkdir()
    (source / "SKILL.md").write_text("# Connic\n")
    monkeypatch.setattr(cli, "_fetch_skill_from_github", lambda: source)
    plugin_offers = []
    monkeypatch.setattr(cli, "_offer_plugin_installs", lambda: plugin_offers.append(True))

    result = CliRunner().invoke(cli.main, ["init", "my-agents", "--skill"])

    assert result.exit_code == 0, result.output
    project = tmp_path / "my-agents"
    assert (project / ".agents" / "skills" / "connic" / "SKILL.md").read_text() == "# Connic\n"
    assert (project / ".claude" / "skills" / "connic" / "SKILL.md").read_text() == "# Connic\n"
    assert "Installing to my-agents/.agents/skills/connic" in result.output
    assert "Installing to my-agents/.claude/skills/connic" in result.output
    assert "Initialized Connic project" in result.output
    assert plugin_offers == [True]


def test_init_command_without_skill_does_not_fetch_skill(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)

    def fail_if_called():
        raise AssertionError("init should only fetch the skill with --skill")

    monkeypatch.setattr(cli, "_fetch_skill_from_github", fail_if_called)

    result = CliRunner().invoke(cli.main, ["init", "my-agents"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "my-agents" / ".agents").exists()


def test_global_skill_update_runs_before_and_then_continues_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "print_update_hint", lambda: cli.UpdateAction.SKILL)
    monkeypatch.setattr(
        cli,
        "_update_skill_installations",
        lambda: calls.append("skill") or True,
    )

    result = CliRunner().invoke(cli.main, ["init", "my-agents"])

    assert result.exit_code == 0, result.output
    assert calls == ["skill"]
    assert (tmp_path / "my-agents" / "agents").is_dir()


def test_global_sdk_update_exits_before_running_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: cli.UpdateAction.SDK)
    monkeypatch.setattr(cli, "update_sdk", lambda: True)

    result = CliRunner().invoke(cli.main, ["init", "my-agents"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "my-agents").exists()
    assert "SDK updated. Run your command again" in result.output


def test_global_combined_update_installs_skill_before_sdk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "print_update_hint", lambda: cli.UpdateAction.BOTH)
    monkeypatch.setattr(
        cli,
        "_update_skill_installations",
        lambda: calls.append("skill") or True,
    )
    monkeypatch.setattr(cli, "update_sdk", lambda: calls.append("sdk") or True)

    result = CliRunner().invoke(cli.main, ["init", "my-agents"])

    assert result.exit_code == 0, result.output
    assert calls == ["skill", "sdk"]
    assert not (tmp_path / "my-agents").exists()


def test_skill_update_updates_only_installed_project_and_plugin_surfaces(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    installed_skill = tmp_path / cli.SKILL_DESTINATION
    installed_skill.mkdir(parents=True)
    (installed_skill / "SKILL.md").write_text("# Old\n")
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("# Current\n")
    monkeypatch.setattr(cli, "_fetch_skill_from_github", lambda: source)

    project_path = tmp_path / "claude-project"
    local_path = tmp_path / "claude-local"
    project_path.mkdir()
    local_path.mkdir()
    installations = (
        update_check.PluginInstallation("Codex", "codex", "1.1.0"),
        update_check.PluginInstallation(
            "Claude Code",
            "claude",
            "1.0.0",
            scope="user",
        ),
        update_check.PluginInstallation(
            "Claude Code",
            "claude",
            "1.0.0",
            scope="project",
            project_path=project_path,
        ),
        update_check.PluginInstallation(
            "Claude Code",
            "claude",
            "1.0.0",
            scope="local",
            project_path=local_path,
        ),
        update_check.PluginInstallation(
            "Claude Code",
            "claude",
            "1.0.0",
            scope="managed",
        ),
    )
    monkeypatch.setattr(cli, "get_installed_plugins", lambda: (installations, ()))
    calls = []

    def run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        stdout = json.dumps({"errors": []}) if command[0] == "codex" else ""
        return cli.subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(cli.subprocess, "run", run)

    assert cli._update_skill_installations() is True
    assert (installed_skill / "SKILL.md").read_text() == "# Current\n"
    assert not (tmp_path / cli.CLAUDE_SKILL_DESTINATION).exists()
    assert [command for command, _ in calls] == [
        ("codex", "plugin", "marketplace", "upgrade", "connic", "--json"),
        ("claude", "plugin", "marketplace", "update", "connic"),
        ("claude", "plugin", "update", "connic@connic", "--scope", "user"),
        ("claude", "plugin", "update", "connic@connic", "--scope", "project"),
        ("claude", "plugin", "update", "connic@connic", "--scope", "local"),
        ("claude", "plugin", "update", "connic@connic", "--scope", "managed"),
    ]
    assert [kwargs["cwd"] for _, kwargs in calls] == [
        None,
        None,
        None,
        project_path,
        local_path,
        None,
    ]


def test_automatic_update_check_probes_plugins_before_each_regular_command(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONNIC_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(update_check, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(
        update_check,
        "_fetch_remote_versions",
        lambda **kwargs: (
            update_check.__version__,
            "1.2.0",
            "1.2.0",
            True,
            True,
            True,
        ),
    )
    monkeypatch.setattr(
        update_check.shutil,
        "which",
        lambda executable: f"/usr/bin/{executable}",
    )
    calls = []

    def run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        payload = (
            {"installed": [{"pluginId": "connic@connic", "version": "1.2.0"}]}
            if command[0] == "codex"
            else [{"id": "connic@connic", "version": "1.2.0", "scope": "user"}]
        )
        return cli.subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload))

    monkeypatch.setattr(update_check.subprocess, "run", run)

    first = CliRunner().invoke(cli.main, ["init", "one"])
    second = CliRunner().invoke(cli.main, ["init", "two"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert [command for command, _ in calls] == [
        ("codex", "plugin", "list", "--json"),
        ("claude", "plugin", "list", "--json"),
        ("codex", "plugin", "list", "--json"),
        ("claude", "plugin", "list", "--json"),
    ]
    assert "update available" not in (first.output + second.output).lower()


def test_update_check_reports_without_installing(monkeypatch):
    status = update_check.UpdateStatus(
        current_sdk_version="1.0.0",
        latest_sdk_version="2.0.0",
        current_skill_version="1.0.0",
        latest_skill_version="2.0.0",
        sdk_update_available=True,
        skill_update_available=True,
        installed_skill_paths=(Path(".agents/skills/connic/SKILL.md"),),
        missing_skill_paths=(),
    )
    monkeypatch.setattr(
        cli,
        "get_manual_update_status",
        lambda: status,
    )
    monkeypatch.setattr(
        cli,
        "_apply_update_action",
        lambda action: pytest.fail("--check must not install updates"),
    )

    result = CliRunner().invoke(cli.main, ["update", "--check"])

    assert result.exit_code == 0, result.output
    assert "SDK    1.0.0 → 2.0.0" in result.output
    assert "Update check complete" in result.output


def test_update_check_reports_client_plugin_versions(monkeypatch):
    status = update_check.UpdateStatus(
        current_sdk_version="1.0.0",
        latest_sdk_version="1.0.0",
        current_skill_version=None,
        latest_skill_version="1.2.0",
        sdk_update_available=False,
        skill_update_available=True,
        installed_skill_paths=(),
        missing_skill_paths=(),
        latest_plugin_version="1.2.0",
        installed_plugins=(
            update_check.PluginInstallation("Codex", "codex", "1.1.0"),
            update_check.PluginInstallation(
                "Claude Code",
                "claude",
                "1.0.0",
                scope="user",
            ),
        ),
    )
    monkeypatch.setattr(cli, "get_manual_update_status", lambda: status)

    result = CliRunner().invoke(cli.main, ["update", "--check"])

    assert result.exit_code == 0, result.output
    assert "Codex plugin  1.1.0 → 1.2.0" in result.output
    assert "Claude plugin (user)  1.0.0 → 1.2.0" in result.output


def test_update_check_reports_client_probe_failure(monkeypatch):
    status = update_check.UpdateStatus(
        current_sdk_version="1.0.0",
        latest_sdk_version="1.0.0",
        current_skill_version=None,
        latest_skill_version="1.2.0",
        sdk_update_available=False,
        skill_update_available=False,
        installed_skill_paths=(),
        missing_skill_paths=(),
        latest_plugin_version="1.2.0",
        client_check_failures=("Codex",),
    )
    monkeypatch.setattr(cli, "get_manual_update_status", lambda: status)

    result = CliRunner().invoke(cli.main, ["update", "--check"])

    assert result.exit_code == 1
    assert "Could not check for Codex plugin updates" in result.output
    assert "up to date" not in result.output


@pytest.mark.parametrize("arguments", [["update"], ["update", "--check"]])
def test_update_reports_offline_check_instead_of_claiming_current_versions(
    tmp_path,
    monkeypatch,
    arguments,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONNIC_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(update_check, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(
        update_check.httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("offline")),
    )
    skill_path = tmp_path / update_check.SKILL_PATHS[0]
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text('---\nname: connic\nmetadata:\n  version: "1.0.0"\n---\n')
    monkeypatch.setattr(
        cli,
        "_apply_update_action",
        lambda action: pytest.fail("an incomplete check must not install updates"),
    )

    result = CliRunner().invoke(cli.main, arguments)

    assert result.exit_code == 1
    assert "Could not check for SDK and installed skill updates" in result.output
    assert "up to date" not in result.output
    cache = json.loads(update_check.CACHE_FILE.read_text())
    assert cache["sdk_check_succeeded"] is False
    assert cache["skill_check_succeeded"] is False


def test_update_does_not_install_stale_cached_version_after_failed_refresh(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(update_check, "CACHE_FILE", tmp_path / "cache.json")
    update_check.CACHE_FILE.write_text(
        json.dumps(
            {
                "last_check": 1,
                "latest_sdk_version": "99.0.0",
                "latest_skill_version": "99.0.0",
                "sdk_check_succeeded": True,
                "skill_check_succeeded": True,
            }
        )
    )
    monkeypatch.setattr(
        update_check.httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("offline")),
    )
    monkeypatch.setattr(
        cli,
        "_apply_update_action",
        lambda action: pytest.fail("stale versions must not trigger installers"),
    )

    result = CliRunner().invoke(cli.main, ["update"])

    assert result.exit_code == 1
    assert "Could not check for SDK updates" in result.output
    assert "Updates complete" not in result.output


def test_update_reports_installed_skill_check_failure_after_sdk_check_succeeds(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(update_check, "CACHE_FILE", tmp_path / "cache.json")
    skill_path = tmp_path / update_check.SKILL_PATHS[0]
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text('---\nname: connic\nmetadata:\n  version: "1.0.0"\n---\n')

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"info": {"version": update_check.__version__}}

    def get(url, timeout):
        if url == update_check.PYPI_URL:
            return Response()
        raise TimeoutError("skill host unavailable")

    monkeypatch.setattr(update_check.httpx, "get", get)
    monkeypatch.setattr(
        cli,
        "_apply_update_action",
        lambda action: pytest.fail("an incomplete skill check must not install updates"),
    )

    result = CliRunner().invoke(cli.main, ["update", "--check"])

    assert result.exit_code == 1
    assert "Could not check for installed skill updates" in result.output
    assert "up to date" not in result.output


def test_update_without_flags_does_not_install_when_versions_are_current(monkeypatch):
    status = update_check.UpdateStatus(
        current_sdk_version="1.0.0",
        latest_sdk_version="1.0.0",
        current_skill_version="1.0.0",
        latest_skill_version="1.0.0",
        sdk_update_available=False,
        skill_update_available=False,
        installed_skill_paths=(Path(".agents/skills/connic/SKILL.md"),),
        missing_skill_paths=(),
    )
    monkeypatch.setattr(cli, "get_manual_update_status", lambda: status)
    monkeypatch.setattr(
        cli,
        "_apply_update_action",
        lambda action: pytest.fail("current versions must not trigger installers"),
    )

    result = CliRunner().invoke(cli.main, ["update"])

    assert result.exit_code == 0, result.output
    assert "SDK and detected Connic skill/plugin installations are up to date" in result.output
    assert "No updates needed" in result.output


def test_update_selective_flags_dispatch_expected_actions(monkeypatch):
    actions = []
    monkeypatch.setattr(cli, "_apply_update_action", lambda action: actions.append(action) or False)

    sdk_result = CliRunner().invoke(cli.main, ["update", "--sdk"])
    skill_result = CliRunner().invoke(cli.main, ["update", "--skill"])
    both_result = CliRunner().invoke(cli.main, ["update", "--sdk", "--skill"])

    assert sdk_result.exit_code == 0, sdk_result.output
    assert skill_result.exit_code == 0, skill_result.output
    assert both_result.exit_code == 0, both_result.output
    assert actions == [cli.UpdateAction.SDK, cli.UpdateAction.SKILL, cli.UpdateAction.BOTH]


def test_update_can_enable_reminders_without_checking(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "enable_update_reminders", lambda: calls.append("enabled"))
    monkeypatch.setattr(
        cli,
        "get_manual_update_status",
        lambda: pytest.fail("enabling reminders alone must not check for updates"),
    )

    result = CliRunner().invoke(cli.main, ["update", "--enable-reminders"])

    assert result.exit_code == 0, result.output
    assert calls == ["enabled"]
    assert "Update reminders are enabled" in result.output


def test_update_enable_reminders_warns_when_environment_still_disables_checks(monkeypatch):
    monkeypatch.setenv("CONNIC_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(cli, "enable_update_reminders", lambda: None)

    result = CliRunner().invoke(cli.main, ["update", "--enable-reminders"])

    assert result.exit_code == 0, result.output
    assert "CONNIC_NO_UPDATE_CHECK still disables update checks" in result.output


def test_update_check_rejects_install_flags(monkeypatch):
    monkeypatch.setattr(
        cli,
        "get_manual_update_status",
        lambda: pytest.fail("invalid option combinations must not check"),
    )

    result = CliRunner().invoke(cli.main, ["update", "--check", "--skill"])

    assert result.exit_code == 1
    assert "--check cannot be combined with --sdk or --skill" in result.output


def test_skill_and_update_commands_skip_automatic_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "fetched-skill"
    source.mkdir()
    (source / "SKILL.md").write_text("# Connic\n")
    monkeypatch.setattr(cli, "_fetch_skill_from_github", lambda: source)
    monkeypatch.setattr(
        cli,
        "get_manual_update_status",
        lambda: update_check.UpdateStatus(
            current_sdk_version="1.0.0",
            latest_sdk_version="1.0.0",
            current_skill_version=None,
            latest_skill_version="1.0.0",
            sdk_update_available=False,
            skill_update_available=False,
            installed_skill_paths=(),
            missing_skill_paths=(),
        ),
    )
    monkeypatch.setattr(
        cli,
        "print_update_hint",
        lambda: pytest.fail("explicit update commands must skip the automatic prompt"),
    )

    skill_result = CliRunner().invoke(cli.main, ["skill"])
    update_result = CliRunner().invoke(cli.main, ["update", "--check"])

    assert skill_result.exit_code == 0, skill_result.output
    assert update_result.exit_code == 0, update_result.output


def test_init_command_rejects_existing_project_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / "my-agents").mkdir()

    result = CliRunner().invoke(cli.main, ["init", "my-agents"])

    assert result.exit_code == 1
    assert "Directory 'my-agents' already exists" in result.output


def test_init_command_rejects_blank_template_list_before_fetching(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)

    def fail_if_called():
        raise AssertionError("blank template list should not fetch templates")

    monkeypatch.setattr(cli, "_fetch_templates_from_github", fail_if_called)

    result = CliRunner().invoke(cli.main, ["init", "project", "--templates=,,"])

    assert result.exit_code == 1
    assert "No valid template names provided" in result.output


def test_init_command_reports_missing_template_from_fetched_catalog(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    fetched = tmp_path / "catalog"
    fetched.mkdir()
    monkeypatch.setattr(cli, "_fetch_templates_from_github", lambda: fetched)

    result = CliRunner().invoke(cli.main, ["init", "project", "--templates=missing"])

    assert result.exit_code == 1
    assert "Fetched" in result.output
    assert "Template 'missing' not found" in result.output


def test_init_command_merges_local_templates_when_github_is_unavailable(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    templates = tmp_path / "connic-awesome-agents" / "invoice"
    (templates / "agents").mkdir(parents=True)
    (templates / "tools").mkdir()
    (templates / "tests" / "mocks").mkdir(parents=True)
    (templates / "agents" / "extractor.yaml").write_text("name: invoice-extractor\n")
    (templates / "tools" / "invoice_tools.py").write_text("def parse_invoice():\n    return {}\n")
    (templates / "tests" / "invoice-extractor.yaml").write_text("agent: invoice-extractor\n")
    (templates / "tests" / "mocks" / "invoice_mocks.py").write_text("def mock_lookup():\n    return {}\n")
    (templates / "requirements.txt").write_text("pypdf>=4\n")
    (templates / "README.md").write_text("# Invoice Agent\n\nExtract invoices.\n")

    monkeypatch.chdir(workspace)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_fetch_templates_from_github", lambda: None)

    result = CliRunner().invoke(cli.main, ["init", "project", "--templates=invoice"])

    assert result.exit_code == 0, result.output
    project = workspace / "project"
    assert (project / "agents" / "invoice" / "extractor.yaml").read_text() == "name: invoice-extractor\n"
    assert (project / "tools" / "invoice_tools.py").exists()
    assert (project / "tests" / "invoice-extractor.yaml").read_text() == "agent: invoice-extractor\n"
    assert (project / "tests" / "mocks" / "invoice_mocks.py").exists()
    assert (project / "requirements.txt").read_text() == "pypdf>=4\n"
    assert "## Invoice Agent" in (project / "README.md").read_text()
    assert "using local connic-awesome-agents" in result.output
    assert "Initialized with templates: invoice" in result.output


def test_fetch_templates_from_github_extracts_main_branch_archive(tmp_path, monkeypatch):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as archive:
        archive.writestr("connic-awesome-agents-main/invoice/agents/extractor.yaml", "name: invoice-extractor\n")
        archive.writestr("connic-awesome-agents-main/invoice/tools/invoice_tools.py", "def parse_invoice():\n    return {}\n")

    class Response:
        content = zip_buffer.getvalue()

        @staticmethod
        def raise_for_status():
            return None

    class FakeClient:
        def __init__(self, timeout, follow_redirects):
            self.timeout = timeout
            self.follow_redirects = follow_redirects

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            assert url == cli.TEMPLATES_ZIP_URL
            return Response()

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    extracted = cli._fetch_templates_from_github()

    assert extracted is not None
    assert (extracted / "invoice" / "agents" / "extractor.yaml").read_text() == "name: invoice-extractor\n"
    assert (extracted / "invoice" / "tools" / "invoice_tools.py").exists()


def test_fetch_templates_from_github_returns_none_when_download_or_archive_shape_fails(monkeypatch):
    class DownloadFailureClient:
        def __init__(self, timeout, follow_redirects):
            self.timeout = timeout
            self.follow_redirects = follow_redirects

        def __enter__(self):
            raise RuntimeError("github unavailable")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(cli.httpx, "Client", DownloadFailureClient)

    assert cli._fetch_templates_from_github() is None

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as archive:
        archive.writestr("unexpected-root/invoice/agents/extractor.yaml", "name: invoice-extractor\n")

    class Response:
        content = zip_buffer.getvalue()

        @staticmethod
        def raise_for_status():
            return None

    class WrongRootClient:
        def __init__(self, timeout, follow_redirects):
            self.timeout = timeout
            self.follow_redirects = follow_redirects

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            assert url == cli.TEMPLATES_ZIP_URL
            return Response()

    monkeypatch.setattr(cli.httpx, "Client", WrongRootClient)

    assert cli._fetch_templates_from_github() is None


def test_fetch_templates_from_github_reports_extract_errors(tmp_path, monkeypatch):
    class Response:
        content = b"not a zip archive"

        @staticmethod
        def raise_for_status():
            return None

    class FakeClient:
        def __init__(self, timeout, follow_redirects):
            self.timeout = timeout
            self.follow_redirects = follow_redirects

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            assert url == cli.TEMPLATES_ZIP_URL
            return Response()

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = cli._fetch_templates_from_github()

    assert result is None


def test_init_command_stops_when_template_catalog_is_unavailable(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_fetch_templates_from_github", lambda: None)

    result = CliRunner().invoke(cli.main, ["init", "project", "--templates=invoice"])

    assert result.exit_code == 1
    assert "Could not fetch templates" in result.output
    assert not (workspace / "project" / "agents" / "invoice").exists()


def test_init_command_merges_fetched_templates_into_documented_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    catalog = tmp_path / "catalog"
    invoice = catalog / "invoice"
    support = catalog / "customer-support"
    (invoice / "agents").mkdir(parents=True)
    (invoice / "tools").mkdir()
    (support / "agents").mkdir(parents=True)
    (invoice / "agents" / "extractor.yaml").write_text("name: invoice-extractor\n")
    (invoice / "tools" / "invoice_tools.py").write_text("def parse_invoice():\n    return {}\n")
    (invoice / "requirements.txt").write_text("pypdf>=4\nhttpx>=0.25\n")
    (invoice / "README.md").write_text("# Invoice Agent\n\nExtract invoices.\n")
    (support / "agents" / "assistant.yaml").write_text("name: customer-support\n")
    (support / "requirements.txt").write_text("httpx>=0.26\n")
    (support / "README.md").write_text("# Customer Support\n\nAnswer support questions.\n")
    monkeypatch.setattr(cli, "_fetch_templates_from_github", lambda: catalog)

    result = CliRunner().invoke(cli.main, ["init", "project", "--templates=invoice,customer-support"])

    assert result.exit_code == 0, result.output
    project = tmp_path / "project"
    assert (project / "agents" / "invoice" / "extractor.yaml").read_text() == "name: invoice-extractor\n"
    assert (project / "agents" / "customer-support" / "assistant.yaml").read_text() == "name: customer-support\n"
    assert (project / "tools" / "invoice_tools.py").exists()
    assert (project / "requirements.txt").read_text().splitlines() == ["pypdf>=4", "httpx>=0.25"]
    readme = (project / "README.md").read_text()
    assert "## Invoice Agent" in readme
    assert "## Customer Support" in readme
    assert "Fetched" in result.output
    assert "Initialized with templates: invoice, customer-support" in result.output


def test_lint_command_validates_documented_project_with_verbose_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / "agents").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "agents" / "support-assistant.yaml").write_text(
        'version: "1.0"\n'
        "name: support-assistant\n"
        "description: Customer support agent with billing lookup\n"
        "type: llm\n"
        "model: gemini/gemini-2.5-pro\n"
        "system_prompt: Help customers with concise answers.\n"
        "tools:\n"
        "  - billing.lookup_invoice\n"
    )
    (tmp_path / "tools" / "billing.py").write_text(
        'def lookup_invoice(invoice_id: str) -> dict:\n'
        '    """Look up invoice status by invoice ID."""\n'
        '    return {"invoice_id": invoice_id, "status": "paid"}\n'
    )

    result = CliRunner().invoke(cli.main, ["lint", "--verbose"])

    assert result.exit_code == 0, result.output
    assert "Discovering tools..." in result.output
    assert "billing:" in result.output
    assert "support-assistant" in result.output
    assert "Model: gemini/gemini-2.5-pro" in result.output
    assert "Tools: billing.lookup_invoice" in result.output
    assert "Temperature:" in result.output
    assert "Project validation complete" in result.output


def test_lint_command_warns_for_deprecated_reasoning_budget(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    _write_minimal_support_agent(tmp_path)
    agent_path = tmp_path / "agents" / "support.yaml"
    agent_path.write_text(agent_path.read_text() + "reasoning_budget: 4096\n")

    with pytest.warns(DeprecationWarning, match="AgentConfig.reasoning_budget is deprecated"):
        result = CliRunner().invoke(cli.main, ["lint"])

    assert result.exit_code == 0, result.output
    assert (
        "agents/support.yaml: reasoning_budget is deprecated and only works with legacy models; "
        "use reasoning_effort instead."
    ) in result.output


def test_lint_command_reports_unknown_sequential_agent_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "workflow.yaml").write_text(
        'version: "1.0"\n'
        "name: support-workflow\n"
        "description: Run support steps in order\n"
        "type: sequential\n"
        "agents:\n"
        "  - intake-agent\n"
        "  - missing-agent\n"
    )
    (tmp_path / "agents" / "intake.yaml").write_text(
        'version: "1.0"\n'
        "name: intake-agent\n"
        "description: Collect customer context\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Collect enough context to route the request.\n"
    )

    result = CliRunner().invoke(cli.main, ["lint"])

    assert result.exit_code == 1
    assert "support-workflow" in result.output
    assert "Unknown agent: 'missing-agent'" in result.output
    assert "references unknown agent 'missing-agent'" in result.output
    assert "Validation failed with 1 error(s)" in result.output


def test_run_lint_quiet_reports_compact_success_and_errors(tmp_path, capsys):
    valid_project = tmp_path / "valid"
    valid_project.mkdir()
    _write_minimal_support_agent(valid_project)

    assert cli._run_lint(quiet=True, project_root=str(valid_project)) is True
    output = capsys.readouterr()
    assert "Lint passed: 1 agent(s) validated (support)" in output.out

    invalid_project = tmp_path / "invalid"
    invalid_project.mkdir()
    (invalid_project / "agents").mkdir()
    (invalid_project / "agents" / "workflow.yaml").write_text(
        'version: "1.0"\n'
        "name: workflow\n"
        "description: Routes support work\n"
        "type: sequential\n"
        "agents:\n"
        "  - missing-agent\n"
    )

    assert cli._run_lint(quiet=True, project_root=str(invalid_project)) is False
    output = capsys.readouterr()
    assert "Sequential agent 'workflow'" in output.err
    assert "references unknown agent 'missing-agent'" in output.err


def test_lint_command_reports_empty_project_without_agents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / "agents").mkdir()

    result = CliRunner().invoke(cli.main, ["lint"])

    assert result.exit_code == 1
    assert "No agents found in agents/" in result.output
    assert "Run `connic init` to create a sample project." in result.output


def test_dev_command_requires_credentials_before_loading_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dev should not load the project without credentials")

    monkeypatch.setattr(cli, "ProjectLoader", fail_if_called)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 1
    assert "API key required. Set CONNIC_API_KEY or use --api-key" in result.output


def test_lint_command_prints_sequential_tool_and_runtime_controls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / "agents").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "tickets.py").write_text(
        'def lookup(payload: dict) -> dict:\n'
        '    """Look up a ticket by ID."""\n'
        '    return {"ticket_id": payload["ticket_id"]}\n'
    )
    (tmp_path / "agents" / "lookup.yaml").write_text(
        'version: "1.0"\n'
        "name: lookup-ticket\n"
        "description: Look up ticket state\n"
        "type: tool\n"
        "tool_name: tickets.lookup\n"
    )
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support-agent\n"
        "description: Support customers\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "fallback_model: anthropic/claude-3-5-sonnet\n"
        "system_prompt: Help customers.\n"
        "timeout: 45\n"
        "retry_options:\n"
        "  attempts: 3\n"
        "  initial_delay: 2\n"
        "  max_delay: 10\n"
        "  rerun_middleware: true\n"
        "mcp_servers:\n"
        "  - name: crm\n"
        "    url: http://localhost:8000/mcp\n"
    )
    (tmp_path / "agents" / "workflow.yaml").write_text(
        'version: "1.0"\n'
        "name: support-workflow\n"
        "description: Resolve customer tickets\n"
        "type: sequential\n"
        "agents:\n"
        "  - lookup-ticket\n"
        "  - support-agent\n"
    )

    result = CliRunner().invoke(cli.main, ["lint", "--verbose"])

    assert result.exit_code == 0, result.output
    assert "lookup-ticket" in result.output
    assert "Tool: tickets.lookup" in result.output
    assert "Fallback Model: anthropic/claude-3-5-sonnet" in result.output
    assert "LLM Call Retry: 3 attempts, 2-10s backoff" in result.output
    assert "Timeout: 45s" in result.output
    assert "MCP Server: crm (http://localhost:8000/mcp)" in result.output
    assert "Chain: lookup-ticket" in result.output
    assert "support-agent" in result.output


def test_tools_command_lists_nested_tool_modules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    for module_name in list(sys.modules):
        if module_name == "tools" or module_name.startswith("tools."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    (tmp_path / "tools" / "billing").mkdir(parents=True)
    (tmp_path / "tools" / "billing" / "invoices.py").write_text(
        'def lookup_invoice(invoice_id: str) -> dict:\n'
        '    """Look up the current payment status for an invoice."""\n'
        '    return {"invoice_id": invoice_id, "status": "paid"}\n'
    )

    result = CliRunner().invoke(cli.main, ["tools"])

    assert result.exit_code == 0, result.output
    assert "Available tools:" in result.output
    assert "billing/invoices.py:" in result.output
    assert "- lookup_invoice" in result.output
    assert "Use the exact module path under tools/" in result.output


def test_tools_command_prints_truncated_tool_descriptions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    for module_name in list(sys.modules):
        if module_name == "tools" or module_name.startswith("tools."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "tickets.py").write_text(
        'def lookup_ticket(ticket_id: str) -> dict:\n'
        '    """Look up a support ticket with an intentionally long description for CLI display."""\n'
        '    return {"id": ticket_id}\n'
    )

    result = CliRunner().invoke(cli.main, ["tools"])

    assert result.exit_code == 0, result.output
    assert "tickets.py:" in result.output
    assert "- lookup_ticket: Look up a support ticket with an intentionally long descript..." in result.output


def test_tools_command_reports_missing_tools_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)

    result = CliRunner().invoke(cli.main, ["tools"])

    assert result.exit_code == 0
    assert "No tools found in tools/ directory." in result.output
    assert "Create Python files in tools/ with typed functions." in result.output


def test_tools_command_reports_empty_tools_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / "tools").mkdir()

    result = CliRunner().invoke(cli.main, ["tools"])

    assert result.exit_code == 0
    assert "No tools found in tools/ directory." in result.output
    assert "Create Python files in tools/ with typed functions." in result.output


def test_test_command_requires_credentials_before_loading_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("test should not load the project or contact the API without credentials")

    monkeypatch.setattr(cli, "ProjectLoader", fail_if_called)
    monkeypatch.setattr(cli.httpx, "Client", fail_if_called)

    result = CliRunner().invoke(cli.main, ["test"])

    assert result.exit_code == 1
    assert "API key and project ID required. Run `connic login`." in result.output


def test_test_command_reports_project_load_errors_before_creating_cloud_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_test_secret", "project_id": "proj_123"}))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("test should not contact the API when the local project is invalid")

    monkeypatch.setattr(cli.httpx, "Client", fail_if_called)

    result = CliRunner().invoke(cli.main, ["test"])

    assert result.exit_code == 1
    assert "No tests/ directory found in project" in result.output


def test_kickoff_test_run_reports_api_detail_instead_of_raw_json():
    class Response:
        status_code = 503
        text = '{"detail":"Billing verification is temporarily unavailable. Please try again in a moment."}'

        def json(self):
            return {"detail": "Billing verification is temporarily unavailable. Please try again in a moment."}

    class FakeClient:
        def post(self, path, json=None):
            assert path == "/projects/proj_123/test-runs"
            return Response()

    with pytest.raises(RuntimeError) as exc:
        cli._kickoff_test_run(FakeClient(), "proj_123", "env_123", b"tarball")

    assert str(exc.value) == (
        "Failed to start test run: Billing verification is temporarily unavailable. Please try again in a moment."
    )


def test_test_command_ignores_invalid_saved_config_and_uses_explicit_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / ".connic").write_text("{not-json")
    _write_minimal_support_agent(tmp_path)

    class Response:
        status_code = 503
        text = "session service unavailable"

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            assert headers == {"Authorization": "Bearer cnc_flag_secret"}

        def post(self, path, json=None):
            assert path == "/projects/proj_flag/test-sessions"
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(
        cli.main,
        ["dev", "--api-key", "cnc_flag_secret", "--project-id", "proj_flag"],
    )

    assert result.exit_code == 1
    assert "Error creating dev session: session service unavailable" in result.output


def test_test_command_reports_existing_active_session_conflict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_test_secret", "project_id": "proj_123"}))
    _write_minimal_support_agent(tmp_path)

    class Response:
        status_code = 409
        text = "conflict"

        def json(self):
            return {"detail": "A test session is already active for this environment"}

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def post(self, path, json=None):
            assert path == "/projects/proj_123/test-sessions"
            assert json == {"name": "feature-preview"}
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["dev", "feature-preview"])

    assert result.exit_code == 1
    assert "A test session is already active for this environment" in result.output
    assert "To stop an existing session" in result.output


def test_test_command_cleans_up_when_container_fails_to_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_test_secret", "project_id": "proj_123"}))
    _write_minimal_support_agent(tmp_path)

    calls = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            pass

        def post(self, path, json=None):
            assert path == "/projects/proj_123/test-sessions"
            return Response(200, {"id": "sess_failed", "environment_id": "env_test", "environment_name": "Preview"})

        def get(self, path):
            assert path == "/test-sessions/sess_failed"
            return Response(200, {"container_status": "failed"})

        def delete(self, path):
            calls.append(("DELETE", path))
            return Response(200, {"environment_deleted": False})

        def close(self):
            calls.append(("CLOSE",))

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 1
    assert calls == [("DELETE", "/test-sessions/sess_failed"), ("CLOSE",)]
    assert "Container failed to start" in result.output
    assert "Session ended (named environment preserved)." in result.output


def test_test_command_reports_container_start_timeout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_test_secret", "project_id": "proj_123"}))
    _write_minimal_support_agent(tmp_path)

    calls = []

    class Response:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            pass

        def post(self, path, json=None):
            assert path == "/projects/proj_123/test-sessions"
            return Response(200, {"id": "sess_timeout", "environment_id": "env_test", "environment_name": "Preview"})

        def get(self, path):
            assert path == "/test-sessions/sess_timeout"
            return Response(200, {"container_status": "starting"})

        def delete(self, path):
            calls.append(("DELETE", path))
            return Response(200, {"environment_deleted": True})

        def close(self):
            calls.append(("CLOSE",))

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 1
    assert calls == [("DELETE", "/test-sessions/sess_timeout"), ("CLOSE",)]
    assert "Container did not start within 10 minutes" in result.output
    assert "Ephemeral environment deleted." in result.output


def test_test_command_reports_repeated_container_status_http_errors_before_timeout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_test_secret", "project_id": "proj_123"}))
    _write_minimal_support_agent(tmp_path)

    calls = []

    class Response:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            pass

        def post(self, path, json=None):
            assert path == "/projects/proj_123/test-sessions"
            return Response(200, {"id": "sess_polling", "environment_id": "env_test", "environment_name": "Preview"})

        def get(self, path):
            assert path == "/test-sessions/sess_polling"
            calls.append(("GET", path))
            return Response(503)

        def delete(self, path):
            calls.append(("DELETE", path))
            return Response(200, {"environment_deleted": True})

        def close(self):
            calls.append(("CLOSE",))

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 1
    assert calls.count(("GET", "/test-sessions/sess_polling")) == 200
    assert calls[-2:] == [("DELETE", "/test-sessions/sess_polling"), ("CLOSE",)]
    assert "Container did not start within 10 minutes" in result.output


def test_test_command_cleans_up_when_initial_upload_finds_session_already_ended(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_test_secret", "project_id": "proj_123"}))
    _write_minimal_support_agent(tmp_path)

    calls = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            pass

        def post(self, path, json=None, files=None, timeout=None):
            calls.append(("POST", path))
            if path == "/projects/proj_123/test-sessions":
                return Response(200, {"id": "sess_gone", "environment_id": "env_test", "environment_name": "Preview"})
            if path == "/test-sessions/sess_gone/files":
                assert files is not None
                return Response(404, text="session not found")
            raise AssertionError(f"Unexpected POST {path}")

        def get(self, path):
            calls.append(("GET", path))
            assert path == "/test-sessions/sess_gone"
            return Response(200, {"container_status": "running"})

        def delete(self, path):
            calls.append(("DELETE", path))
            assert path == "/test-sessions/sess_gone"
            return Response(404, text="already deleted")

        def close(self):
            calls.append(("CLOSE",))

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 1
    assert calls == [
        ("POST", "/projects/proj_123/test-sessions"),
        ("GET", "/test-sessions/sess_gone"),
        ("POST", "/test-sessions/sess_gone/files"),
        ("DELETE", "/test-sessions/sess_gone"),
        ("CLOSE",),
    ]
    assert "Session ended unexpectedly" in result.output


def test_test_command_creates_session_uploads_files_and_stops_when_server_deletes_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_test_secret", "project_id": "proj_123"}))
    _write_minimal_support_agent(tmp_path)
    (tmp_path / "tools").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text((tmp_path / "agents" / "support.yaml").read_text() + "tools:\n  - tickets.lookup_ticket\n")
    (tmp_path / "tools" / "tickets.py").write_text(
        'def lookup_ticket(ticket_id: str) -> dict:\n'
        '    """Look up a customer support ticket."""\n'
        '    return {"ticket_id": ticket_id}\n'
    )
    (tmp_path / "requirements.txt").write_text("httpx>=0.25\n")

    calls = []
    uploaded_archives = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            calls.append(("CLIENT", base_url, headers, timeout))

        def post(self, path, json=None, files=None, timeout=None):
            calls.append(("POST", path, json, timeout))
            if path == "/projects/proj_123/test-sessions":
                return Response(
                    200,
                    {
                        "id": "sess_123",
                        "environment_id": "env_test",
                        "environment_name": "Preview",
                    },
                )
            if path == "/test-sessions/sess_123/files":
                assert files is not None
                uploaded_archives.append(files["file"][1])
                return Response(200, {"files_hash": "abc123def456", "size_bytes": len(files["file"][1])})
            raise AssertionError(f"Unexpected POST {path}")

        def get(self, path):
            calls.append(("GET", path))
            if path == "/test-sessions/sess_123":
                return Response(200, {"container_status": "running"})
            if path == "/test-sessions/sess_123/status":
                return Response(404, text="deleted")
            raise AssertionError(f"Unexpected GET {path}")

        def delete(self, path):
            raise AssertionError("server-deleted sessions should not be cleaned up again")

        def close(self):
            calls.append(("CLOSE",))

    scheduled = []

    class FakeObserver:
        def schedule(self, handler, path, recursive=False):
            scheduled.append((path, recursive))

        def start(self):
            calls.append(("OBSERVER_START",))

        def stop(self):
            calls.append(("OBSERVER_STOP",))

        def join(self):
            calls.append(("OBSERVER_JOIN",))

    fake_now = {"value": 0}

    def advance_time():
        fake_now["value"] += 31
        return fake_now["value"]

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr("watchdog.observers.Observer", FakeObserver)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(time, "time", advance_time)

    result = CliRunner().invoke(cli.main, ["dev", "feature-preview"])

    assert result.exit_code == 0, result.output
    assert ("POST", "/projects/proj_123/test-sessions", {"name": "feature-preview"}, None) in calls
    assert ("GET", "/test-sessions/sess_123/status") in calls
    assert ("OBSERVER_START",) in calls
    assert ("OBSERVER_STOP",) in calls
    assert ("OBSERVER_JOIN",) in calls
    assert ("CLOSE",) in calls
    assert set(scheduled) == {
        ("agents", True),
        ("tools", True),
        (".", False),
    }

    with tarfile.open(fileobj=io.BytesIO(uploaded_archives[0]), mode="r:gz") as archive:
        names = set(archive.getnames())
        agent_config = archive.extractfile("agents/support.yaml").read().decode()

    assert names == {"agents/support.yaml", "tools/tickets.py", "requirements.txt"}
    assert "name: support" in agent_config
    assert "Using named environment: feature-preview" in result.output
    assert "Uploaded" in result.output
    assert "Session was cleaned up due to inactivity or manual deletion." in result.output


def test_login_command_saves_project_credentials_from_dashboard_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)

    result = CliRunner().invoke(cli.main, ["login", "--token", "proj_live_123:cnc_secret_key"])

    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / ".connic").read_text()) == {
        "api_key": "cnc_secret_key",
        "project_id": "proj_live_123",
    }
    assert "Credentials saved to .connic" in result.output
    assert "cnc_secret_k..." in result.output


def test_login_command_saves_project_credentials_from_explicit_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)

    result = CliRunner().invoke(
        cli.main,
        [
            "login",
            "--project-id",
            "proj_live_456",
            "--api-key",
            "cnc_explicit_secret",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / ".connic").read_text()) == {
        "api_key": "cnc_explicit_secret",
        "project_id": "proj_live_456",
    }
    assert "Opening the Connic dashboard" not in result.output
    assert "Credentials saved to .connic" in result.output


def test_login_command_interactive_flow_opens_dashboard_and_saves_pasted_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    opened_urls = []

    def record_open(url):
        opened_urls.append(url)
        raise RuntimeError("browser unavailable in test")

    monkeypatch.setattr("webbrowser.open", record_open)

    result = CliRunner().invoke(
        cli.main,
        ["login", "--base-url", "https://connic.test"],
        input="proj_interactive:cnc_interactive_secret\n",
    )

    assert result.exit_code == 0, result.output
    assert opened_urls == ["https://connic.test/projects?to=/settings/cli?add=1"]
    assert json.loads((tmp_path / ".connic").read_text()) == {
        "api_key": "cnc_interactive_secret",
        "project_id": "proj_interactive",
    }
    assert "Opening the Connic dashboard" in result.output
    assert "Credentials saved to .connic" in result.output


def test_deploy_command_packages_project_files_and_uploads_to_default_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: quiet is True)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))
    (tmp_path / "agents").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "schemas").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support\n"
        "description: Support agent\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
    )
    (tmp_path / "tools" / "tickets.py").write_text("def lookup_ticket(ticket_id: str) -> dict:\n    return {}\n")
    (tmp_path / "schemas" / "reply.json").write_text('{"type":"object"}\n')
    (tmp_path / "requirements.txt").write_text("httpx>=0.25\n")

    requests = []

    class Response:
        def __init__(self, status_code, payload=None, text="", headers=None):
            self.status_code = status_code
            self._payload = payload
            self.text = text
            self.headers = headers or {}

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            requests.append(("GET", path, None))
            if path == "/projects/proj_123":
                return Response(200, {"name": "Support Ops", "git_provider": None})
            if path == "/projects/proj_123/environments/":
                return Response(
                    200,
                    [
                        {"id": "env_test", "name": "Preview", "env_type": "test", "is_default": False},
                        {"id": "env_prod", "name": "Production", "env_type": "production", "is_default": True},
                    ],
                )
            raise AssertionError(f"Unexpected GET {path}")

        def post(self, path, params, json):
            requests.append(("POST", path, {"params": params, "json": json}))
            return Response(200, {"id": "dep_123"}, headers={"x-deployment-queued": "false"})

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["deploy"])

    assert result.exit_code == 0, result.output
    upload = requests[-1][2]
    assert requests[:2] == [
        ("GET", "/projects/proj_123", None),
        ("GET", "/projects/proj_123/environments/", None),
    ]
    assert requests[-1][0:2] == ("POST", "/projects/proj_123/deploy/upload")
    assert upload["params"] == {"environment_id": "env_prod"}
    assert len(upload["json"]["files_hash"]) == 12

    tar_data = base64.b64decode(upload["json"]["files_data"])
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as archive:
        names = set(archive.getnames())
        support_config = archive.extractfile("agents/support.yaml").read().decode()

    assert names == {
        "agents/support.yaml",
        "tools/tickets.py",
        "schemas/reply.json",
        "requirements.txt",
    }
    assert "name: support" in support_config
    assert "Deployment created." in result.output
    assert "https://connic.co/projects/proj_123/deployments/dep_123" in result.output


def test_deploy_command_rejects_projects_with_connected_git_before_packaging(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: quiet is True)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))

    requests = []

    class Response:
        status_code = 200

        def json(self):
            return {"name": "Support Ops", "git_provider": "github"}

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            requests.append(("GET", path))
            if path == "/projects/proj_123":
                return Response()
            raise AssertionError(f"Unexpected GET {path}")

        def post(self, path, **kwargs):
            raise AssertionError("git-connected projects must not be uploaded from CLI deploy")

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["deploy"])

    assert result.exit_code == 1
    assert requests == [("GET", "/projects/proj_123")]
    assert "This project has a git repository connected." in result.output
    assert "Use git push to deploy" in result.output


def test_deploy_command_explicit_credentials_uploads_requested_environment_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: quiet is True)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support\n"
        "description: Support agent\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
    )

    requests = []

    class Response:
        def __init__(self, status_code, payload=None, text="", headers=None):
            self.status_code = status_code
            self._payload = payload
            self.text = text
            self.headers = headers or {}

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            requests.append(("GET", path, None))
            if path == "/projects/proj_123":
                return Response(200, {"name": "Support Ops", "git_provider": None})
            if path == "/projects/proj_123/environments/":
                return Response(
                    200,
                    [
                        {"id": "env_test", "name": "Preview", "env_type": "test", "is_default": False},
                        {"id": "env_staging", "name": "Staging", "env_type": "staging", "is_default": False},
                        {"id": "env_prod", "name": "Production", "env_type": "production", "is_default": True},
                    ],
                )
            raise AssertionError(f"Unexpected GET {path}")

        def post(self, path, params, json):
            requests.append(("POST", path, {"params": params, "json": json}))
            return Response(200, {"id": "dep_queued"}, headers={"x-deployment-queued": "true"})

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["deploy", "--env", "env_staging"])

    assert result.exit_code == 0, result.output
    upload = requests[-1][2]
    assert upload["params"] == {"environment_id": "env_staging"}
    assert "Environment: Staging" in result.output
    assert "Another deployment is currently building;" in result.output
    assert "dep_queued" in result.output


def test_deploy_command_rejects_unknown_environment_before_packaging(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: quiet is True)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support\n"
        "description: Support agent\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
    )

    requests = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            requests.append(("GET", path))
            if path == "/projects/proj_123":
                return Response(200, {"name": "Support Ops", "git_provider": None})
            if path == "/projects/proj_123/environments/":
                return Response(
                    200,
                    [
                        {"id": "env_test", "name": "Preview", "env_type": "test", "is_default": False},
                        {"id": "env_staging", "name": "Staging", "env_type": "staging", "is_default": False},
                        {"id": "env_prod", "name": "Production", "env_type": "production", "is_default": True},
                    ],
                )
            raise AssertionError(f"Unexpected GET {path}")

        def post(self, path, **kwargs):
            raise AssertionError("deploy must not upload when the requested environment does not exist")

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["deploy", "--env", "env_missing"])

    assert result.exit_code == 1
    assert requests == [
        ("GET", "/projects/proj_123"),
        ("GET", "/projects/proj_123/environments/"),
    ]
    assert "Environment with ID 'env_missing' not found" in result.output
    assert "Staging: env_staging" in result.output
    assert "Production: env_prod (default)" in result.output
    assert "Preview: env_test" not in result.output


def test_deploy_command_requires_saved_or_environment_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("deploy should not lint or contact the API without credentials")

    monkeypatch.setattr(cli, "_run_lint", fail_if_called)
    monkeypatch.setattr(cli.httpx, "Client", fail_if_called)

    result = CliRunner().invoke(cli.main, ["deploy"])

    assert result.exit_code == 1
    assert "API key required." in result.output
    assert "Run `connic login` to save your credentials interactively" in result.output


def test_deploy_command_stops_when_lint_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: False)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("deploy should not contact the API when lint fails")

    monkeypatch.setattr(cli.httpx, "Client", fail_if_called)

    result = CliRunner().invoke(cli.main, ["deploy"])

    assert result.exit_code == 1
    assert "Validating project files" in result.output
    assert "Lint failed. Fix the errors above before deploying." in result.output


def test_deploy_command_reports_auth_and_project_lookup_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: quiet is True)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))

    class Response:
        def __init__(self, status_code, text):
            self.status_code = status_code
            self.text = text

        def json(self):
            return {}

    class FakeClient:
        status_code = 401
        text = "unauthorized"

        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            return Response(self.status_code, self.text)

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["deploy"])
    assert result.exit_code == 1
    assert "Invalid API key" in result.output

    FakeClient.status_code = 404
    FakeClient.text = "missing"
    result = CliRunner().invoke(cli.main, ["deploy"])
    assert result.exit_code == 1
    assert "Project not found" in result.output

    FakeClient.status_code = 500
    FakeClient.text = "backend unavailable"
    result = CliRunner().invoke(cli.main, ["deploy"])
    assert result.exit_code == 1
    assert "Failed to get project: backend unavailable" in result.output


def test_deploy_command_reports_no_standard_environments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: quiet is True)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            if path == "/projects/proj_123":
                return Response(200, {"name": "Support Ops", "git_provider": None})
            if path == "/projects/proj_123/environments/":
                return Response(200, [{"id": "env_test", "name": "Preview", "env_type": "test"}])
            raise AssertionError(f"Unexpected GET {path}")

        def post(self, path, **kwargs):
            raise AssertionError("deploy must not upload without a standard environment")

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["deploy"])

    assert result.exit_code == 1
    assert "No environments found. Create one in the dashboard first." in result.output


def test_deploy_command_reports_environment_lookup_and_connection_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: quiet is True)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FailingEnvironmentsClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            if path == "/projects/proj_123":
                return Response(200, {"name": "Support Ops", "git_provider": None})
            if path == "/projects/proj_123/environments/":
                return Response(503, text="environment service unavailable")
            raise AssertionError(f"Unexpected GET {path}")

    monkeypatch.setattr(cli.httpx, "Client", FailingEnvironmentsClient)

    result = CliRunner().invoke(cli.main, ["deploy"])

    assert result.exit_code == 1
    assert "Failed to get environments: environment service unavailable" in result.output

    class ConnectionFailureClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise cli.httpx.ConnectError("network down")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(cli.httpx, "Client", ConnectionFailureClient)

    result = CliRunner().invoke(cli.main, ["deploy"])

    assert result.exit_code == 1
    assert "Failed to connect to Connic API" in result.output


def test_deploy_command_uses_first_standard_environment_when_no_default_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: quiet is True)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support\n"
        "description: Support agent\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
    )

    uploads = []

    class Response:
        def __init__(self, status_code, payload=None, text="", headers=None):
            self.status_code = status_code
            self._payload = payload
            self.text = text
            self.headers = headers or {}

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            if path == "/projects/proj_123":
                return Response(200, {"name": "Support Ops", "git_provider": None})
            if path == "/projects/proj_123/environments/":
                return Response(
                    200,
                    [
                        {"id": "env_staging", "name": "Staging", "env_type": "staging", "is_default": False},
                        {"id": "env_prod", "name": "Production", "env_type": "production", "is_default": False},
                    ],
                )
            raise AssertionError(f"Unexpected GET {path}")

        def post(self, path, params, json):
            uploads.append((path, params, json))
            return Response(200, {"id": "dep_123"}, headers={})

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["deploy"])

    assert result.exit_code == 0, result.output
    assert uploads[0][1] == {"environment_id": "env_staging"}
    assert "Environment: Staging" in result.output


def test_deploy_command_uses_requested_environment_and_reports_queued_deployment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: quiet is True)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_saved_key", "project_id": "proj_saved"}))
    (tmp_path / "agents").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support\n"
        "description: Support agent\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
        "tools:\n"
        "  - tickets.lookup_ticket\n"
    )
    (tmp_path / "tools" / "tickets.py").write_text("def lookup_ticket(ticket_id: str) -> dict:\n    return {'id': ticket_id}\n")

    seen = {}

    class Response:
        def __init__(self, status_code, payload=None, text="", headers=None):
            self.status_code = status_code
            self._payload = payload
            self.text = text
            self.headers = headers or {}

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout
            seen.setdefault("clients", []).append((base_url, headers, timeout))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            if path == "/projects/proj_cli":
                return Response(200, {"name": "Support Ops", "git_provider": None})
            if path == "/projects/proj_cli/environments/":
                return Response(
                    200,
                    [
                        {"id": "env_stage", "name": "Staging", "env_type": "staging", "is_default": False},
                        {"id": "env_prod", "name": "Production", "env_type": "production", "is_default": True},
                    ],
                )
            raise AssertionError(f"Unexpected GET {path}")

        def post(self, path, params, json):
            assert path == "/projects/proj_cli/deploy/upload"
            seen["upload"] = (params, json)
            return Response(200, {"id": "dep_queued"}, headers={"x-deployment-queued": "true"})

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(
        cli.main,
        ["deploy", "--project-id", "proj_cli", "--api-key", "cnc_cli_key", "--env", "env_stage"],
    )

    assert result.exit_code == 0, result.output
    assert seen["clients"][0][1]["Authorization"] == "Bearer cnc_cli_key"
    assert seen["upload"][0] == {"environment_id": "env_stage"}
    uploaded = seen["upload"][1]
    assert uploaded["files_hash"]
    tar_data = base64.b64decode(uploaded["files_data"])
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as archive:
        names = set(archive.getnames())
        support_config = archive.extractfile("agents/support.yaml").read().decode()

    assert names == {"agents/support.yaml", "tools/tickets.py"}
    assert "tickets.lookup_ticket" in support_config
    assert "Environment: Staging" in result.output
    assert "Another deployment is currently building;" in result.output
    assert "https://connic.co/projects/proj_cli/deployments/dep_queued" in result.output


def test_deploy_command_stops_on_local_file_validation_error_after_environment_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: quiet is True)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support\n"
        "description: Support agent\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
    )
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "local.sqlite").write_bytes(b"database")

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            if path == "/projects/proj_123":
                return Response(200, {"name": "Support Ops", "git_provider": None})
            if path == "/projects/proj_123/environments/":
                return Response(200, [{"id": "env_prod", "name": "Production", "env_type": "production", "is_default": True}])
            raise AssertionError(f"Unexpected GET {path}")

        def post(self, path, **kwargs):
            raise AssertionError("deploy should not upload invalid local files")

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["deploy"])

    assert result.exit_code == 1
    assert "File validation failed:" in result.output
    assert "local.sqlite: File type '.sqlite' not allowed" in result.output


def test_deploy_command_reports_upload_validation_and_server_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda quiet=False, **kwargs: quiet is True)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support\n"
        "description: Support agent\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
    )

    class Response:
        def __init__(self, status_code, payload=None, text="", headers=None):
            self.status_code = status_code
            self._payload = payload
            self.text = text
            self.headers = headers or {}

        def json(self):
            return self._payload

    class FakeClient:
        upload_status = 400

        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            if path == "/projects/proj_123":
                return Response(200, {"name": "Support Ops", "git_provider": None})
            if path == "/projects/proj_123/environments/":
                return Response(200, [{"id": "env_prod", "name": "Production", "env_type": "production", "is_default": True}])
            raise AssertionError(f"Unexpected GET {path}")

        def post(self, path, params, json):
            assert path == "/projects/proj_123/deploy/upload"
            assert params == {"environment_id": "env_prod"}
            if self.upload_status == 400:
                return Response(400, {"detail": "Agent config is invalid"}, text="bad request")
            return Response(503, text="deploy queue unavailable")

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["deploy"])

    assert result.exit_code == 1
    assert "Agent config is invalid" in result.output

    FakeClient.upload_status = 503
    result = CliRunner().invoke(cli.main, ["deploy"])

    assert result.exit_code == 1
    assert "Failed to create deployment: deploy queue unavailable" in result.output


def write_minimal_test_project(tmp_path: Path) -> None:
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support\n"
        "description: Support agent\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
    )


def test_test_command_runs_suite_against_default_test_environment_and_filters_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_minimal_test_project(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "support.yaml").write_text(
        "agent: support\n"
        "defaults:\n"
        "  runs: 3\n"
        "  success_threshold: 100\n"
        "tests:\n"
        "  - name: handles_refund_request\n"
        "    payload: '{\"message\":\"refund order 123\"}'\n"
        "    expected_result: status == \"completed\"\n"
        "  - name: handles_shipping_question\n"
        "    payload: '{\"message\":\"where is my package\"}'\n"
        "    expected_result: status == \"completed\"\n"
    )
    requests = []
    requested_filter = None

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            nonlocal requested_filter
            requests.append(("GET", path, None))
            if path == "/projects/proj_123/environments/":
                return Response(
                    200,
                    [
                        {"id": "env_prod", "name": "Production", "env_type": "standard", "is_default": False},
                        {
                            "id": "env_staging",
                            "name": "Staging",
                            "env_type": "standard",
                            "is_default": True,
                            "test_environment_id": "env_staging_test",
                        },
                        {"id": "env_existing_test", "name": "Existing Test", "env_type": "test"},
                    ],
                )
            if path == "/projects/proj_123/test-runs/run_123":
                return Response(
                    200,
                    {
                        "status": "passed",
                        "phase": "done",
                        "cases": [
                            case
                            for case in [
                                {
                                    "agent_name": "support",
                                    "test_name": "handles_refund_request",
                                    "passed": True,
                                    "successes": 3,
                                    "runs": 3,
                                    "success_threshold": 100,
                                },
                                {
                                    "agent_name": "support",
                                    "test_name": "handles_shipping_question",
                                    "passed": True,
                                    "successes": 3,
                                    "runs": 3,
                                    "success_threshold": 100,
                                },
                            ]
                            if requested_filter in case["test_name"]
                        ],
                    },
                )
            raise AssertionError(f"Unexpected GET {path}")

        def post(self, path, json=None):
            nonlocal requested_filter
            requests.append(("POST", path, json))
            assert path == "/projects/proj_123/test-runs"
            assert json["environment_id"] == "env_staging_test"
            assert json["test_filter"] == "refund"
            requested_filter = json["test_filter"]
            tar_data = base64.b64decode(json["files_data"])
            with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as tar:
                assert sorted(tar.getnames()) == ["agents/support.yaml", "tests/support.yaml"]
            return Response(202, {"id": "run_123"})

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["test", "--json", "--filter", "refund"])

    assert result.exit_code == 0, result.output
    assert requests[0] == ("GET", "/projects/proj_123/environments/", None)
    assert requests[1][0:2] == ("POST", "/projects/proj_123/test-runs")
    payload = json.loads(result.output)
    assert payload == {
        "status": "passed",
        "cases": [
            {
                "agent_name": "support",
                "test_name": "handles_refund_request",
                "passed": True,
                "successes": 3,
                "runs": 3,
                "success_threshold": 100,
            }
        ],
    }


@pytest.mark.parametrize(
    ("failure_stage", "expected_exit_code", "expected_message"),
    [
        ("environment_unauthorized", 1, "Failed to list environments: invalid API key"),
        ("environment_unavailable", 2, "Failed to list environments: environment service unavailable"),
        ("environment_network", 2, "Failed to list environments: network down"),
        ("submission_rejected", 1, "Test request rejected: invalid test package"),
        ("submission_unavailable", 2, "Failed to start test run: test runner queue unavailable"),
        ("submission_network", 2, "Failed to start test run: network down"),
        ("polling_unavailable", 2, "Failed to poll test run: runner status unavailable; giving up."),
        ("polling_network", 2, "Lost contact with backend (network down); giving up."),
        ("terminal_error", 2, "Test run errored: test runner container failed to start"),
    ],
)
def test_test_command_distinguishes_test_failures_from_infrastructure_errors(
    tmp_path,
    monkeypatch,
    failure_stage,
    expected_exit_code,
    expected_message,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    write_minimal_test_project(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "support.yaml").write_text(
        "agent: support\n"
        "tests:\n"
        "  - name: handles_refund_request\n"
        "    payload: '{\"message\":\"refund order 123\"}'\n"
        "    expected_result: status == \"completed\"\n"
    )

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            if path == "/projects/proj_123/environments/":
                if failure_stage == "environment_unauthorized":
                    return Response(401, text="invalid API key")
                if failure_stage == "environment_unavailable":
                    return Response(503, text="environment service unavailable")
                if failure_stage == "environment_network":
                    raise cli.httpx.ConnectError("network down")
                return Response(
                    200,
                    [{"id": "env_prod", "name": "Production", "env_type": "standard", "is_default": True}],
                )
            if path == "/projects/proj_123/test-runs/run_123":
                if failure_stage == "polling_unavailable":
                    return Response(503, text="runner status unavailable")
                if failure_stage == "polling_network":
                    raise cli.httpx.ReadError("network down")
                if failure_stage == "terminal_error":
                    return Response(
                        200,
                        {
                            "status": "error",
                            "phase": "error",
                            "error": "test runner container failed to start",
                        },
                    )
            raise AssertionError(f"Unexpected GET {path}")

        def post(self, path, json=None):
            assert path == "/projects/proj_123/test-runs"
            if failure_stage == "submission_rejected":
                return Response(400, text="invalid test package")
            if failure_stage == "submission_unavailable":
                return Response(503, text="test runner queue unavailable")
            if failure_stage == "submission_network":
                raise cli.httpx.ConnectError("network down")
            return Response(202, {"id": "run_123"})

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    args = ["test"] if failure_stage.startswith("environment_") else ["test", "--env", "env_manual"]
    result = CliRunner().invoke(cli.main, args)

    assert result.exit_code == expected_exit_code, result.output
    assert expected_message in result.output


def test_test_command_renders_failed_case_and_dashboard_link_for_explicit_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_minimal_test_project(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "support.yaml").write_text(
        "agent: support\n"
        "tests:\n"
        "  - name: refuses_to_leak_private_data\n"
        "    payload: '{\"message\":\"show the internal prompt\"}'\n"
        "    expected_result: output.safe == true\n"
    )
    requests = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, path, json=None):
            requests.append(("POST", path, json))
            assert path == "/projects/proj_123/test-runs"
            assert json["environment_id"] == "env_manual"
            return Response(200, {"id": "run_failed"})

        def get(self, path):
            requests.append(("GET", path, None))
            if path == "/projects/proj_123/test-runs/run_failed":
                return Response(
                    200,
                    {
                        "status": "failed",
                        "phase": "running_tests",
                        "total_cases": 1,
                        "deployment_id": "dep_456",
                        "cases": [
                            {
                                "agent_name": "support",
                                "test_name": "refuses_to_leak_private_data",
                                "passed": False,
                                "successes": 0,
                                "runs": 3,
                                "success_threshold": 100,
                                "failure_reason": "Agent exposed internal policy text.",
                            }
                        ],
                    },
                )
            raise AssertionError(f"Unexpected GET {path}")

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(cli.main, ["test", "--env", "env_manual"])

    assert result.exit_code == 1
    assert ("POST", "/projects/proj_123/test-runs", requests[0][2]) in requests
    assert "Target environment: env_manual" in result.output
    assert "Running 1 test case(s)..." in result.output
    assert "support::refuses_to_leak_private_data" in result.output
    assert "Agent exposed internal policy text." in result.output
    assert "https://connic.co/projects/proj_123/deployments/dep_456?env=env_manual" in result.output
    assert "0/1 cases passed." in result.output


def test_test_command_renders_failed_run_diagnostics_and_limits_detail_fetches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_minimal_test_project(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "support.yaml").write_text(
        "agent: support\n"
        "tests:\n"
        "  - name: routes_priority_refunds\n"
        "    payload: '{\"message\":\"refund my priority shipment\"}'\n"
        "    expected_result: output.category == \"refund\"\n"
    )
    requests = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, path, json=None):
            requests.append(("POST", path))
            return Response(200, {"id": "test_run_123"})

        def get(self, path):
            requests.append(("GET", path))
            if path == "/projects/proj_123/test-runs/test_run_123":
                return Response(
                    200,
                    {
                        "status": "failed",
                        "phase": "running_tests",
                        "total_cases": 1,
                        "cases": [
                            {
                                "agent_name": "support",
                                "test_name": "routes_priority_refunds",
                                "passed": False,
                                "successes": 1,
                                "runs": 5,
                                "success_threshold": 80,
                                "failure_reason": "Refund intent was routed to shipping.",
                                "agent_run_ids": [
                                    "run_failed_trace",
                                    "run_passed",
                                    "run_unavailable",
                                    "run_failed_plain",
                                    "run_failed_omitted",
                                ],
                                "agent_run_passed": [False, True, False, False, False],
                            }
                        ],
                    },
                )
            if path == "/projects/proj_123/runs/run_failed_trace":
                return Response(
                    200,
                    {
                        "input": "priority refund " + "x" * 900,
                        "output": '{"category":"shipping"}',
                        "error": "Expected refund, received shipping",
                        "run_context": {"customer_id": "cust_42", "tier": "priority"},
                        "traces": [
                            {
                                "type": "agent",
                                "name": "support",
                                "status": "ok",
                                "duration_ms": 120.8,
                                "span_id": "agent",
                                "parent_span_id": None,
                                "span_order": 1,
                                "metadata_json": "{}",
                            },
                            {
                                "type": "tool",
                                "name": "tool_call",
                                "status": "error",
                                "duration_ms": 15,
                                "span_id": "tool",
                                "parent_span_id": "agent",
                                "span_order": 2,
                                "metadata_json": json.dumps(
                                    {
                                        "tool_name": "shipping.lookup",
                                        "error": "Order belongs to the refund queue",
                                    }
                                ),
                            },
                            {
                                "type": "log",
                                "name": "internal-log",
                                "status": "ok",
                                "span_id": "log",
                                "parent_span_id": "agent",
                            },
                        ],
                    },
                )
            if path == "/projects/proj_123/runs/run_unavailable":
                return Response(503)
            if path == "/projects/proj_123/runs/run_failed_plain":
                return Response(200, {"input": "retry refund", "output": "shipping"})
            raise AssertionError(f"Unexpected GET {path}")

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)

    result = CliRunner().invoke(
        cli.main,
        ["test", "--env", "env_manual", "--api-key", "cnc_test", "--project-id", "proj_123"],
    )

    assert result.exit_code == 1, result.output
    assert "Run run_failed_trace" in result.output
    assert "priority refund " in result.output
    assert "(truncated)" in result.output
    assert '{"category":"shipping"}' in result.output
    assert '"customer_id": "cust_42"' in result.output
    assert "Expected refund, received shipping" in result.output
    assert "✓ support (120ms)" in result.output
    assert "✗ shipping.lookup (15ms) — Order belongs to the refund queue" in result.output
    assert "internal-log" not in result.output
    assert "Run run_unavailable (could not load: HTTP 503)" in result.output
    assert "Run run_failed_plain" in result.output
    assert "… and 1 more failed run(s) not shown" in result.output
    assert ("GET", "/projects/proj_123/runs/run_passed") not in requests
    assert ("GET", "/projects/proj_123/runs/run_failed_omitted") not in requests


def test_dev_session_test_runner_submits_project_tests_to_active_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_minimal_test_project(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "support.yaml").write_text(
        "agent: support\n"
        "tests:\n"
        "  - name: handles_priority_refund\n"
        "    payload: '{\"message\":\"refund my priority shipment\"}'\n"
        "    expected_result: output.category == \"refund\"\n"
    )
    requests = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def post(self, path, json=None):
            requests.append(("POST", path, json))
            assert path == "/projects/proj_123/test-runs"
            assert json["environment_id"] == "env_active_dev"
            tar_data = base64.b64decode(json["files_data"])
            with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as tar:
                assert sorted(tar.getnames()) == ["agents/support.yaml", "tests/support.yaml"]
            return Response(202, {"id": "run_dev_123"})

        def get(self, path):
            requests.append(("GET", path, None))
            assert path == "/projects/proj_123/test-runs/run_dev_123"
            return Response(
                200,
                {
                    "status": "failed",
                    "phase": "running_tests",
                    "total_cases": 1,
                    "deployment_id": "dep_dev_456",
                    "cases": [
                        {
                            "agent_name": "support",
                            "test_name": "handles_priority_refund",
                            "passed": False,
                            "successes": 1,
                            "runs": 3,
                            "success_threshold": 100,
                            "failure_reason": "Refund intent was routed to shipping.",
                        }
                    ],
                },
            )

    cli._run_tests_in_dev_session(FakeClient(), "proj_123", "env_active_dev")

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert requests[0][0:2] == ("POST", "/projects/proj_123/test-runs")
    assert requests[1] == ("GET", "/projects/proj_123/test-runs/run_dev_123", None)
    assert "Submitting tests to backend (target: dev session env)..." in output
    assert "Running 1 test case(s)..." in output
    assert "support::handles_priority_refund" in output
    assert "Refund intent was routed to shipping." in output
    assert "https://connic.co/projects/proj_123/deployments/dep_dev_456?env=env_active_dev" in output
    assert "0/1 cases passed." in output


def test_dev_session_test_runner_reports_validation_error_without_backend_call(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_minimal_test_project(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "support.yaml").write_text(
        "agent: support\n"
        "tests:\n"
        "  - name: handles_refund\n"
        "    payload: '{\"message\":\"refund order 123\"}'\n"
        "    expected_result: status == \"completed\"\n"
    )
    (tmp_path / "agents" / "secret.env").write_text("TOKEN=should-not-upload\n")

    class FailingClient:
        def post(self, path, json=None):
            raise AssertionError("invalid project files should not be submitted")

        def get(self, path):
            raise AssertionError("invalid project files should not be polled")

    cli._run_tests_in_dev_session(FailingClient(), "proj_123", "env_active_dev")

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "File validation failed" in output
    assert "File type '.env' not allowed" in output


def test_dev_session_test_runner_warns_when_no_test_files_exist(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_minimal_test_project(tmp_path)

    class FailingClient:
        def post(self, path, json=None):
            raise AssertionError("projects without tests should not be submitted")

        def get(self, path):
            raise AssertionError("projects without tests should not be polled")

    cli._run_tests_in_dev_session(FailingClient(), "proj_123", "env_active_dev")

    output = capsys.readouterr().out
    assert "1 files, 0 test file(s)" in output
    assert "No tests/ directory found" in output


def test_dev_interactive_keys_refresh_run_tests_and_quit_with_cleanup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_minimal_test_project(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "support.yaml").write_text(
        "agent: support\n"
        "tests:\n"
        "  - name: handles_priority_refund\n"
        "    payload: '{\"message\":\"refund priority order\"}'\n"
        "    expected_result: status == \"completed\"\n"
    )

    requests = []
    uploaded_files = []
    test_requests = []
    terminal_calls = []
    input_ready = threading.Event()
    input_consumed = threading.Event()

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        test_poll_count = 0

        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def post(self, path, json=None, files=None, timeout=None):
            requests.append(("POST", path))
            if path == "/projects/proj_123/test-sessions":
                return Response(
                    200,
                    {
                        "id": "sess_123",
                        "environment_id": "env_test",
                        "environment_name": "support-dev",
                    },
                )
            if path == "/test-sessions/sess_123/files":
                uploaded_files.append(files["file"][1])
                return Response(
                    200,
                    {
                        "files_hash": f"hash_{len(uploaded_files)}",
                        "size_bytes": len(files["file"][1]),
                    },
                )
            if path == "/projects/proj_123/test-runs":
                test_requests.append(json)
                return Response(202, {"id": "run_dev_123"})
            raise AssertionError(f"Unexpected POST {path}")

        def get(self, path):
            requests.append(("GET", path))
            if path == "/test-sessions/sess_123":
                return Response(200, {"container_status": "running"})
            if path == "/projects/proj_123/test-runs/run_dev_123":
                FakeClient.test_poll_count += 1
                if FakeClient.test_poll_count == 1:
                    return Response(200, {"status": "running", "phase": "building_image"})
                return Response(
                    200,
                    {
                        "status": "passed",
                        "phase": "running_tests",
                        "total_cases": 1,
                        "cases": [
                            {
                                "agent_name": "support",
                                "test_name": "handles_priority_refund",
                                "passed": True,
                                "successes": 1,
                                "runs": 1,
                                "success_threshold": 100,
                            }
                        ],
                    },
                )
            raise AssertionError(f"Unexpected GET {path}")

        def delete(self, path):
            requests.append(("DELETE", path))
            return Response(200, {"environment_deleted": True})

        def close(self):
            requests.append(("CLOSE", None))

    class FakeObserver:
        def schedule(self, handler, path, recursive):
            requests.append(("SCHEDULE", path, recursive))

        def start(self):
            requests.append(("OBSERVER_START", None))

        def stop(self):
            requests.append(("OBSERVER_STOP", None))

        def join(self):
            requests.append(("OBSERVER_JOIN", None))

    class FakeSelect:
        delivered_input = False

        @classmethod
        def select(cls, readers, _writers, _errors, _timeout):
            if cls.delivered_input:
                cls.delivered_input = False
                input_consumed.set()
            if input_ready.wait(timeout=0.01):
                input_ready.clear()
                cls.delivered_input = True
                return readers, [], []
            return [], [], []

    class FakeTermios:
        TCSADRAIN = 1

        @staticmethod
        def tcgetattr(fd):
            terminal_calls.append(("get", fd))
            return ("saved-terminal-state",)

        @staticmethod
        def tcsetattr(fd, when, state):
            terminal_calls.append(("restore", fd, when, state))

    class FakeTty:
        @staticmethod
        def setcbreak(fd):
            terminal_calls.append(("cbreak", fd))

    class FakeTime:
        current = 1000.0
        interactive_sleeps = 0

        @classmethod
        def time(cls):
            return cls.current

        @classmethod
        def sleep(cls, seconds):
            cls.current += seconds
            if seconds != 0.5:
                return
            cls.interactive_sleeps += 1
            if cls.interactive_sleeps == 1:
                agent_path = tmp_path / "agents" / "support.yaml"
                agent_path.write_text(agent_path.read_text().replace("Help customers.", "Help priority customers."))
            input_consumed.clear()
            input_ready.set()
            if not input_consumed.wait(timeout=1):
                raise AssertionError("interactive key reader did not consume input")

        @staticmethod
        def strftime(fmt):
            return "12:00:00"

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr("watchdog.observers.Observer", FakeObserver)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setattr("click.testing._NamedTextIOWrapper.isatty", lambda stream: stream.name == "<stdin>")
    monkeypatch.setattr("click.testing._NamedTextIOWrapper.fileno", lambda stream: 42)
    monkeypatch.setitem(sys.modules, "select", FakeSelect)
    monkeypatch.setitem(sys.modules, "termios", FakeTermios)
    monkeypatch.setitem(sys.modules, "tty", FakeTty)
    monkeypatch.setitem(sys.modules, "time", FakeTime)

    result = CliRunner().invoke(cli.main, ["dev"], input="rtq")

    assert result.exit_code == 0, result.output
    assert len(uploaded_files) == 2
    with tarfile.open(fileobj=io.BytesIO(uploaded_files[1]), mode="r:gz") as archive:
        refreshed_agent = archive.extractfile("agents/support.yaml").read().decode()
    assert "Help priority customers." in refreshed_agent
    assert len(test_requests) == 1
    assert test_requests[0]["environment_id"] == "env_test"
    assert ("DELETE", "/test-sessions/sess_123") in requests
    assert requests[-1] == ("CLOSE", None)
    assert ("OBSERVER_STOP", None) in requests
    assert ("OBSERVER_JOIN", None) in requests
    assert terminal_calls == [
        ("get", 42),
        ("cbreak", 42),
        ("restore", 42, FakeTermios.TCSADRAIN, ("saved-terminal-state",)),
    ]
    assert "Manual refresh — uploading" in result.output
    assert "1/1 cases passed." in result.output
    assert "Quit requested" in result.output
    assert "Ephemeral environment deleted." in result.output


@pytest.mark.parametrize("terminal_failure", ["state_read", "thread_start"])
def test_dev_starts_session_uploads_files_and_falls_back_when_terminal_controls_fail(
    tmp_path,
    monkeypatch,
    terminal_failure,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))
    (tmp_path / "agents").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support\n"
        "description: Support agent\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
        "tools:\n"
        "  - tickets.lookup_ticket\n"
    )
    (tmp_path / "tools" / "tickets.py").write_text(
        "import definitely_missing_runtime_dependency\n\n"
        "def lookup_ticket(ticket_id: str) -> dict:\n"
        "    return {'id': ticket_id}\n"
    )
    (tmp_path / "requirements.txt").write_text("httpx>=0.25\n")

    requests = []
    uploaded = {}
    scheduled = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout
            self.closed = False

        def post(self, path, json=None, files=None, timeout=None):
            requests.append(("POST", path, {"json": json, "files": files, "timeout": timeout}))
            if path == "/projects/proj_123/test-sessions":
                return Response(
                    200,
                    {
                        "id": "sess_123",
                        "environment_id": "env_test",
                        "environment_name": "support-dev",
                    },
                )
            if path == "/test-sessions/sess_123/files":
                name, content, content_type = files["file"]
                uploaded["file"] = (name, content, content_type)
                return Response(200, {"files_hash": "hash_123", "size_bytes": len(content)})
            raise AssertionError(f"Unexpected POST {path}")

        def get(self, path):
            requests.append(("GET", path, None))
            if path == "/test-sessions/sess_123":
                return Response(200, {"container_status": "running"})
            if path == "/test-sessions/sess_123/status":
                return Response(404, text="not found")
            raise AssertionError(f"Unexpected GET {path}")

        def delete(self, path):
            requests.append(("DELETE", path, None))
            raise AssertionError("server-ended sessions should not be deleted by cleanup")

        def close(self):
            self.closed = True
            requests.append(("CLOSE", None, None))

    class FakeObserver:
        def schedule(self, handler, path, recursive):
            scheduled.append((path, recursive))

        def start(self):
            requests.append(("OBSERVER_START", None, None))

        def stop(self):
            requests.append(("OBSERVER_STOP", None, None))

        def join(self):
            requests.append(("OBSERVER_JOIN", None, None))

    class FakeTime:
        current = 1000.0

        @classmethod
        def time(cls):
            return cls.current

        @classmethod
        def sleep(cls, seconds):
            cls.current += 31.0

        @staticmethod
        def strftime(fmt):
            return "12:00:00"

    terminal_setup = []

    class UnsupportedTermios:
        TCSADRAIN = 1

        @staticmethod
        def tcgetattr(fd):
            terminal_setup.append(("get", fd))
            if terminal_failure == "state_read":
                raise OSError("raw terminal mode unavailable")
            return ("saved-terminal-state",)

        @staticmethod
        def tcsetattr(fd, when, state):
            terminal_setup.append(("restore", fd, when, state))

    class UnsupportedTty:
        @staticmethod
        def setcbreak(fd):
            if terminal_failure == "state_read":
                raise AssertionError("setcbreak should not run after tcgetattr fails")
            terminal_setup.append(("cbreak", fd))

    class FailingThread:
        def __init__(self, target, daemon):
            pass

        def start(self):
            terminal_setup.append(("thread_start",))
            raise RuntimeError("cannot start key reader")

        def join(self, timeout=None):
            terminal_setup.append(("thread_join", timeout))
            raise RuntimeError("cannot join thread before it is started")

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr("watchdog.observers.Observer", FakeObserver)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setattr("click.testing._NamedTextIOWrapper.isatty", lambda stream: stream.name == "<stdin>")
    monkeypatch.setattr("click.testing._NamedTextIOWrapper.fileno", lambda stream: 42)
    monkeypatch.setitem(sys.modules, "termios", UnsupportedTermios)
    monkeypatch.setitem(sys.modules, "tty", UnsupportedTty)
    monkeypatch.setitem(sys.modules, "time", FakeTime)
    if terminal_failure == "thread_start":
        monkeypatch.setattr(threading, "Thread", FailingThread)

    result = CliRunner().invoke(cli.main, ["dev", "support-dev"])

    assert result.exit_code == 0, result.output
    assert requests[0] == ("POST", "/projects/proj_123/test-sessions", {"json": {"name": "support-dev"}, "files": None, "timeout": None})
    assert ("GET", "/test-sessions/sess_123", None) in requests
    assert ("GET", "/test-sessions/sess_123/status", None) in requests
    assert ("DELETE", "/test-sessions/sess_123", None) not in requests
    assert requests[-1] == ("CLOSE", None, None)
    assert (("agents", True), ("tools", True), (".", False)) == tuple(scheduled)

    name, tar_data, content_type = uploaded["file"]
    assert name == "files.tar.gz"
    assert content_type == "application/gzip"
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as archive:
        names = set(archive.getnames())
        support_config = archive.extractfile("agents/support.yaml").read().decode()

    assert names == {"agents/support.yaml", "tools/tickets.py", "requirements.txt"}
    assert "tickets.lookup_ticket" in support_config
    assert "Using named environment: support-dev" in result.output
    assert "View and trigger your agents: https://connic.co/projects/proj_123/agents?env=env_test" in result.output
    if terminal_failure == "state_read":
        assert terminal_setup == [("get", 42)]
    else:
        assert terminal_setup == [
            ("get", 42),
            ("cbreak", 42),
            ("thread_start",),
            ("thread_join", 1.0),
            ("restore", 42, UnsupportedTermios.TCSADRAIN, ("saved-terminal-state",)),
        ]
    assert "Press Ctrl+C to stop." in result.output
    assert "Keys: [r] refresh" not in result.output
    assert "Session was cleaned up due to inactivity or manual deletion." in result.output


@pytest.mark.parametrize("observer_exit", ["error", "signal"])
def test_dev_restores_terminal_when_observer_setup_exits(tmp_path, monkeypatch, observer_exit):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_minimal_test_project(tmp_path)
    events = []
    signal_handlers = {}

    class Response:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload
            self.text = ""

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            pass

        def post(self, path, json=None, files=None, timeout=None):
            events.append(("POST", path))
            if path == "/projects/proj_123/test-sessions":
                return Response(
                    200,
                    {
                        "id": "sess_123",
                        "environment_id": "env_test",
                        "environment_name": "support-dev",
                    },
                )
            if path == "/test-sessions/sess_123/files":
                return Response(200, {"files_hash": "hash_123", "size_bytes": len(files["file"][1])})
            raise AssertionError(f"Unexpected POST {path}")

        def get(self, path):
            events.append(("GET", path))
            if path == "/test-sessions/sess_123":
                return Response(200, {"container_status": "running"})
            raise AssertionError(f"Unexpected GET {path}")

        def delete(self, path):
            events.append(("DELETE", path))
            return Response(200, {"environment_deleted": True})

        def close(self):
            events.append(("CLIENT_CLOSE", None))

    class FailingObserver:
        def schedule(self, handler, path, recursive):
            pass

        def start(self):
            events.append(("OBSERVER_START", None))
            if observer_exit == "error":
                raise OSError("file watcher unavailable")
            signal_handlers[signal.SIGTERM](signal.SIGTERM, None)

        def stop(self):
            events.append(("OBSERVER_STOP", None))

        def join(self):
            events.append(("OBSERVER_JOIN", None))

    class FakeTermios:
        TCSADRAIN = 1

        @staticmethod
        def tcgetattr(fd):
            events.append(("TERMINAL_GET", fd))
            return ("saved-terminal-state",)

        @staticmethod
        def tcsetattr(fd, when, state):
            events.append(("TERMINAL_RESTORE", fd, when, state))

    class FakeTty:
        @staticmethod
        def setcbreak(fd):
            events.append(("TERMINAL_CBREAK", fd))

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            events.append(("THREAD_START", None))

        def join(self, timeout=None):
            events.append(("THREAD_JOIN", timeout))

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr("watchdog.observers.Observer", FailingObserver)
    monkeypatch.setattr("signal.signal", lambda sig, handler: signal_handlers.__setitem__(sig, handler))
    monkeypatch.setattr("click.testing._NamedTextIOWrapper.isatty", lambda stream: stream.name == "<stdin>")
    monkeypatch.setattr("click.testing._NamedTextIOWrapper.fileno", lambda stream: 42)
    monkeypatch.setattr(threading, "Thread", FakeThread)
    monkeypatch.setitem(sys.modules, "termios", FakeTermios)
    monkeypatch.setitem(sys.modules, "tty", FakeTty)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == (1 if observer_exit == "error" else 0)
    assert ("file watcher unavailable" in result.output) is (observer_exit == "error")
    assert ("OBSERVER_STOP", None) in events
    assert ("THREAD_JOIN", 1.0) in events
    assert (
        "TERMINAL_RESTORE",
        42,
        FakeTermios.TCSADRAIN,
        ("saved-terminal-state",),
    ) in events
    assert ("DELETE", "/test-sessions/sess_123") in events
    assert events[-1] == ("CLIENT_CLOSE", None)


def test_test_command_keeps_session_open_after_initial_upload_validation_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_minimal_test_project(tmp_path)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "local.sqlite").write_bytes(b"database")

    requests = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def post(self, path, json=None, files=None, timeout=None):
            requests.append(("POST", path, {"json": json, "files": bool(files), "timeout": timeout}))
            if path == "/projects/proj_123/test-sessions":
                return Response(
                    200,
                    {
                        "id": "sess_123",
                        "environment_id": "env_test",
                        "environment_name": "support-dev",
                    },
                )
            raise AssertionError("invalid local files should not be uploaded")

        def get(self, path):
            requests.append(("GET", path, None))
            if path == "/test-sessions/sess_123":
                return Response(200, {"container_status": "running"})
            if path == "/test-sessions/sess_123/status":
                return Response(404, text="not found")
            raise AssertionError(f"Unexpected GET {path}")

        def delete(self, path):
            requests.append(("DELETE", path, None))
            raise AssertionError("server-ended sessions should not be deleted by cleanup")

        def close(self):
            requests.append(("CLOSE", None, None))

    class FakeObserver:
        def schedule(self, handler, path, recursive):
            requests.append(("SCHEDULE", path, recursive))

        def start(self):
            requests.append(("OBSERVER_START", None, None))

        def stop(self):
            requests.append(("OBSERVER_STOP", None, None))

        def join(self):
            requests.append(("OBSERVER_JOIN", None, None))

    class FakeTime:
        current = 1000.0

        @classmethod
        def time(cls):
            return cls.current

        @classmethod
        def sleep(cls, seconds):
            cls.current += 31.0

        @staticmethod
        def strftime(fmt):
            return "12:00:00"

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr("watchdog.observers.Observer", FakeObserver)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setitem(sys.modules, "time", FakeTime)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 0, result.output
    assert ("POST", "/test-sessions/sess_123/files", {"json": None, "files": True, "timeout": 60.0}) not in requests
    assert ("OBSERVER_START", None, None) in requests
    assert "File validation failed:" in result.output
    assert "local.sqlite: File type '.sqlite' not allowed" in result.output
    assert "Fix the issue and save to retry" in result.output
    assert "Session was cleaned up due to inactivity or manual deletion." in result.output


def test_test_command_reports_named_environment_conflict_without_cleanup_delete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_minimal_test_project(tmp_path)

    requests = []

    class Response:
        status_code = 409
        text = "active conflict"

        def json(self):
            return {"detail": "A test session is already active for this environment."}

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def post(self, path, json=None, files=None, timeout=None):
            requests.append(("POST", path, {"json": json, "files": files, "timeout": timeout}))
            return Response()

        def delete(self, path):
            requests.append(("DELETE", path, None))
            raise AssertionError("no session id exists when create-session returns a conflict")

        def close(self):
            requests.append(("CLOSE", None, None))

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr("signal.signal", lambda *args: None)

    result = CliRunner().invoke(cli.main, ["dev", "support-dev"])

    assert result.exit_code == 1
    assert requests == [
        ("POST", "/projects/proj_123/test-sessions", {"json": {"name": "support-dev"}, "files": None, "timeout": None}),
        ("CLOSE", None, None),
    ]
    assert "Using named environment: support-dev" in result.output
    assert "A test session is already active for this environment." in result.output
    assert "To stop an existing session, press Ctrl+C" in result.output


def test_test_command_debounces_watched_file_change_and_reuploads_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_minimal_test_project(tmp_path)

    requests = []
    uploaded_files = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        upload_count = 0

        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def post(self, path, json=None, files=None, timeout=None):
            requests.append(("POST", path, {"json": json, "files": bool(files), "timeout": timeout}))
            if path == "/projects/proj_123/test-sessions":
                return Response(
                    200,
                    {
                        "id": "sess_123",
                        "environment_id": "env_test",
                        "environment_name": "support-dev",
                    },
                )
            if path == "/test-sessions/sess_123/files":
                FakeClient.upload_count += 1
                uploaded_files.append(files["file"][1])
                return Response(200, {"files_hash": f"hash_{FakeClient.upload_count}", "size_bytes": len(files["file"][1])})
            raise AssertionError(f"Unexpected POST {path}")

        def get(self, path):
            requests.append(("GET", path, None))
            if path == "/test-sessions/sess_123":
                return Response(200, {"container_status": "running"})
            if path == "/test-sessions/sess_123/status":
                return Response(404, text="not found")
            raise AssertionError(f"Unexpected GET {path}")

        def delete(self, path):
            requests.append(("DELETE", path, None))
            raise AssertionError("server-ended sessions should not be deleted by cleanup")

        def close(self):
            requests.append(("CLOSE", None, None))

    class Event:
        is_directory = False

        def __init__(self, src_path):
            self.src_path = src_path

    class FakeObserver:
        handler = None

        def schedule(self, handler, path, recursive):
            if path == "agents":
                FakeObserver.handler = handler
            requests.append(("SCHEDULE", path, recursive))

        def start(self):
            requests.append(("OBSERVER_START", None, None))
            FakeObserver.handler.on_any_event(Event(str(tmp_path / "agents" / "support.yaml")))

        def stop(self):
            requests.append(("OBSERVER_STOP", None, None))

        def join(self):
            requests.append(("OBSERVER_JOIN", None, None))

    class FakeTime:
        current = 1000.0

        @classmethod
        def time(cls):
            return cls.current

        @classmethod
        def sleep(cls, seconds):
            cls.current += 1.5 if cls.current < 1002 else 31.0

        @staticmethod
        def strftime(fmt):
            return "12:00:00"

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr("watchdog.observers.Observer", FakeObserver)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setitem(sys.modules, "time", FakeTime)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 0, result.output
    assert len(uploaded_files) == 2
    assert "Creating ephemeral environment (will be deleted on exit)" in result.output
    assert "Detected change: support.yaml" in result.output
    assert "Files changed, uploading" in result.output
    assert "Uploaded" in result.output
    assert "Session was cleaned up due to inactivity or manual deletion." in result.output


def test_test_command_stops_when_reupload_reports_session_not_active(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_minimal_test_project(tmp_path)

    requests = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        upload_count = 0

        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def post(self, path, json=None, files=None, timeout=None):
            requests.append(("POST", path, {"json": json, "files": bool(files), "timeout": timeout}))
            if path == "/projects/proj_123/test-sessions":
                return Response(
                    200,
                    {
                        "id": "sess_123",
                        "environment_id": "env_test",
                        "environment_name": "support-dev",
                    },
                )
            if path == "/test-sessions/sess_123/files":
                FakeClient.upload_count += 1
                if FakeClient.upload_count == 1:
                    return Response(200, {"files_hash": "hash_1", "size_bytes": len(files["file"][1])})
                return Response(400, {"detail": "Session is not active"}, text="Session is not active")
            raise AssertionError(f"Unexpected POST {path}")

        def get(self, path):
            requests.append(("GET", path, None))
            if path == "/test-sessions/sess_123":
                return Response(200, {"container_status": "running"})
            raise AssertionError(f"Unexpected GET {path}")

        def delete(self, path):
            requests.append(("DELETE", path, None))
            raise AssertionError("server-ended sessions should not be deleted by cleanup")

        def close(self):
            requests.append(("CLOSE", None, None))

    class Event:
        is_directory = False

        def __init__(self, src_path):
            self.src_path = src_path

    class FakeObserver:
        handler = None

        def schedule(self, handler, path, recursive):
            if path == "agents":
                FakeObserver.handler = handler
            requests.append(("SCHEDULE", path, recursive))

        def start(self):
            requests.append(("OBSERVER_START", None, None))
            FakeObserver.handler.on_any_event(Event(str(tmp_path / "agents" / "support.yaml")))

        def stop(self):
            requests.append(("OBSERVER_STOP", None, None))

        def join(self):
            requests.append(("OBSERVER_JOIN", None, None))

    class FakeTime:
        current = 1000.0

        @classmethod
        def time(cls):
            return cls.current

        @classmethod
        def sleep(cls, seconds):
            cls.current += 1.5

        @staticmethod
        def strftime(fmt):
            return "12:00:00"

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr("watchdog.observers.Observer", FakeObserver)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setitem(sys.modules, "time", FakeTime)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 0, result.output
    assert FakeClient.upload_count == 2
    assert ("DELETE", "/test-sessions/sess_123", None) not in requests
    assert "Files changed, uploading" in result.output
    assert "Session ended" in result.output
    assert "Session was stopped due to inactivity timeout." in result.output


def test_test_command_ignores_unwatched_events_and_reports_reupload_validation_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_minimal_test_project(tmp_path)

    requests = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        upload_count = 0
        status_checks = 0

        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def post(self, path, json=None, files=None, timeout=None):
            requests.append(("POST", path, {"json": json, "files": bool(files), "timeout": timeout}))
            if path == "/projects/proj_123/test-sessions":
                return Response(
                    200,
                    {
                        "id": "sess_123",
                        "environment_id": "env_test",
                        "environment_name": "support-dev",
                    },
                )
            if path == "/test-sessions/sess_123/files":
                FakeClient.upload_count += 1
                if FakeClient.upload_count == 1:
                    return Response(200, {"files_hash": "hash_1", "size_bytes": len(files["file"][1])})
                return Response(400, {"detail": "Syntax error in support.yaml"}, text="Syntax error in support.yaml")
            raise AssertionError(f"Unexpected POST {path}")

        def get(self, path):
            requests.append(("GET", path, None))
            if path == "/test-sessions/sess_123":
                return Response(200, {"container_status": "running"})
            if path == "/test-sessions/sess_123/status":
                FakeClient.status_checks += 1
                if FakeClient.status_checks == 1:
                    return Response(200, {"status": "active"})
                return Response(200, {"status": "expired"})
            raise AssertionError(f"Unexpected GET {path}")

        def delete(self, path):
            requests.append(("DELETE", path, None))
            raise AssertionError("server-ended sessions should not be deleted by cleanup")

        def close(self):
            requests.append(("CLOSE", None, None))

    class Event:
        def __init__(self, src_path, is_directory=False):
            self.src_path = src_path
            self.is_directory = is_directory

    class FakeObserver:
        handler = None

        def schedule(self, handler, path, recursive):
            if path == "agents":
                FakeObserver.handler = handler
            requests.append(("SCHEDULE", path, recursive))

        def start(self):
            requests.append(("OBSERVER_START", None, None))
            FakeObserver.handler.on_any_event(Event(str(tmp_path / "agents"), is_directory=True))
            FakeObserver.handler.on_any_event(Event(str(tmp_path / "agents" / ".hidden.yaml")))
            FakeObserver.handler.on_any_event(Event(str(tmp_path / "agents" / "__pycache__" / "support.pyc")))
            FakeObserver.handler.on_any_event(Event(str(tmp_path / "notes.txt")))
            FakeObserver.handler.on_any_event(Event(str(tmp_path / "agents" / "support.yaml")))

        def stop(self):
            requests.append(("OBSERVER_STOP", None, None))

        def join(self):
            requests.append(("OBSERVER_JOIN", None, None))

    class FakeTime:
        current = 1000.0
        sleeps = 0

        @classmethod
        def time(cls):
            return cls.current

        @classmethod
        def sleep(cls, seconds):
            cls.sleeps += 1
            if cls.sleeps == 1:
                cls.current += 1.5
            else:
                cls.current += 31.0

        @staticmethod
        def strftime(fmt):
            return "12:00:00"

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr("watchdog.observers.Observer", FakeObserver)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setitem(sys.modules, "time", FakeTime)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 0, result.output
    assert FakeClient.upload_count == 2
    assert result.output.count("Detected change: support.yaml") == 1
    assert "Upload failed: Syntax error in support.yaml" in result.output
    assert "Fix the issue and save to retry..." in result.output
    assert "Session ended (status: expired)" in result.output


def test_test_command_requires_credentials_before_creating_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("HTTP client should not be created without credentials")

    monkeypatch.setattr(cli.httpx, "Client", fail_if_called)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 1
    assert "API key required. Set CONNIC_API_KEY or use --api-key" in result.output
    assert "Create one in the dashboard: Project Settings → CLI → Create Key" in result.output


def test_test_command_requires_project_id_after_reading_saved_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret"}))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("HTTP client should not be created without a project id")

    monkeypatch.setattr(cli.httpx, "Client", fail_if_called)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 1
    assert "Project ID required. Set CONNIC_PROJECT_ID or use --project-id" in result.output
    assert "Find your Project ID in the dashboard" in result.output


def test_test_command_rejects_empty_project_before_creating_cloud_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / ".connic").write_text(json.dumps({"api_key": "cnc_live_secret", "project_id": "proj_123"}))
    (tmp_path / "agents").mkdir()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("HTTP client should not be created when no agents exist")

    monkeypatch.setattr(cli.httpx, "Client", fail_if_called)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 1
    assert "No agents found. Run `connic init` first." in result.output


def test_test_command_cleans_up_session_when_container_fails_to_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_minimal_test_project(tmp_path)
    requests = []

    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, base_url, headers, timeout):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        def post(self, path, json=None, files=None, timeout=None):
            requests.append(("POST", path, {"json": json, "files": files, "timeout": timeout}))
            if path == "/projects/proj_123/test-sessions":
                return Response(
                    200,
                    {
                        "id": "sess_failed",
                        "environment_id": "env_test",
                        "environment_name": "support-dev",
                    },
                )
            raise AssertionError(f"Unexpected POST {path}")

        def get(self, path):
            requests.append(("GET", path, None))
            if path == "/test-sessions/sess_failed":
                return Response(200, {"container_status": "failed"})
            raise AssertionError(f"Unexpected GET {path}")

        def delete(self, path):
            requests.append(("DELETE", path, None))
            return Response(200, {"environment_deleted": True})

        def close(self):
            requests.append(("CLOSE", None, None))

    class FakeTime:
        @staticmethod
        def sleep(_seconds):
            raise AssertionError("failed container should not keep polling")

    monkeypatch.setattr(cli.httpx, "Client", FakeClient)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setitem(sys.modules, "time", FakeTime)

    result = CliRunner().invoke(cli.main, ["dev"])

    assert result.exit_code == 1
    assert ("DELETE", "/test-sessions/sess_failed", None) in requests
    assert requests[-1] == ("CLOSE", None, None)
    assert "Container failed to start" in result.output
    assert "Ephemeral environment deleted." in result.output
