import json
import sys

import httpx
import pytest
from click.testing import CliRunner

from connic import cli


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    monkeypatch.setattr(cli, "_run_lint", lambda **kwargs: True)
    monkeypatch.delenv("CONNIC_API_KEY", raising=False)
    monkeypatch.delenv("CONNIC_PROJECT_ID", raising=False)
    (tmp_path / ".connic").write_text(json.dumps({"project_id": "proj_123", "api_key": "cnc_secret"}))
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents/support.yaml").write_text(
        'version: "1.0"\nname: support\ndescription: Support agent\ntype: llm\n'
        "model: openai/gpt-4o\nsystem_prompt: Help customers.\n"
    )
    return tmp_path


@pytest.fixture
def api(monkeypatch, request):
    requests = []
    environment_list_status = getattr(request, "param", 200)
    environments = [
        {"id": "env_prod", "name": "Production", "env_type": "production", "is_default": True, "can_deploy": False},
        {"id": "env_staging", "name": "Staging", "env_type": "staging", "can_deploy": True},
        {"id": "env_test", "name": "Preview", "env_type": "test", "can_deploy": True},
    ]
    session_body = {}

    def handle(request):
        requests.append(request)
        path = request.url.path
        if path.endswith("/environments/"):
            if environment_list_status != 200:
                return httpx.Response(environment_list_status, json={"detail": "Environment list unavailable"})
            return httpx.Response(200, json=environments)
        if path.endswith("/projects/proj_123"):
            return httpx.Response(200, json={"name": "Support", "git_provider": None})
        if path.endswith("/deploy/upload"):
            return httpx.Response(200, json={"id": "deploy_123"})
        if path.endswith("/test-sessions"):
            session_body.clear()
            session_body.update(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "session_123", "environment_id": "env_test",
                "environment_name": session_body.get("name", "quick-test"),
            })
        if path.endswith("/test-sessions/session_123"):
            if request.method == "DELETE":
                return httpx.Response(200, json={"environment_deleted": "name" not in session_body})
            return httpx.Response(200, json={"container_status": "running"})
        if path.endswith("/test-sessions/session_123/files"):
            return httpx.Response(200, json={"files_hash": "hash", "size_bytes": 100})
        raise AssertionError(f"Unexpected request: {request.method} {path}")

    client_class = httpx.Client
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: client_class(transport=httpx.MockTransport(handle), **kwargs))
    return environments, requests


@pytest.fixture
def dev_runtime(monkeypatch):
    class Observer:
        def schedule(self, *args, **kwargs):
            return object()

        def start(self):
            pass

        def stop(self):
            pass

        def join(self):
            pass

    class Clock:
        @staticmethod
        def time():
            return 1000.0

        @staticmethod
        def sleep(seconds):
            raise KeyboardInterrupt

    monkeypatch.setattr("watchdog.observers.Observer", Observer)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setitem(sys.modules, "time", Clock)


def uploaded(requests):
    return [request for request in requests if request.url.path.endswith("/deploy/upload")]


def test_deploy_lists_only_permitted_standard_environments_without_local_validation(project, api, monkeypatch):
    def no_lint(**kwargs):
        raise AssertionError("Listing environments must not inspect local source files")

    monkeypatch.setattr(cli, "_run_lint", no_lint)
    _, requests = api

    result = CliRunner().invoke(cli.main, ["deploy", "--list"])

    assert result.exit_code == 0, result.output
    assert "Staging" in result.output and "env_staging" in result.output
    assert "Production" not in result.output and "Preview" not in result.output
    assert all(request.method == "GET" for request in requests)


def test_deploy_shows_names_and_confirms_selected_target(project, api):
    _, requests = api

    result = CliRunner().invoke(cli.main, ["deploy"], input="1\ny\n")

    assert result.exit_code == 0, result.output
    assert "1. Staging" in result.output
    assert "Confirm deployment:" in result.output
    assert "Environment: Staging (env_staging)" in result.output
    assert "Deploy to this environment? [y/N]" in result.output
    assert uploaded(requests)[0].url.params["environment_id"] == "env_staging"


@pytest.mark.parametrize("confirmation", ["n\n", "\n", ""])
def test_deploy_never_uploads_without_confirmation(project, api, confirmation):
    _, requests = api

    result = CliRunner().invoke(cli.main, ["deploy", "--env", "Staging"], input=confirmation)

    assert result.exit_code == (1 if not confirmation else 0), result.output
    assert not uploaded(requests)


@pytest.mark.parametrize("target", ["Staging", "staging", "env_staging"])
def test_deploy_accepts_names_and_ids_for_explicit_automation(project, api, target):
    _, requests = api

    result = CliRunner().invoke(cli.main, ["deploy", "--env", target, "--yes"])

    assert result.exit_code == 0, result.output
    assert len(uploaded(requests)) == 1
    assert "Deploy to this environment?" not in result.output


@pytest.mark.parametrize("target", ["Production", "env_prod"])
def test_deploy_cannot_select_disallowed_environment_by_name_or_id(project, api, target):
    _, requests = api

    result = CliRunner().invoke(cli.main, ["deploy", "--env", target, "--yes"])

    assert result.exit_code == 1
    assert "not available for deployment" in result.output
    assert not uploaded(requests)


def test_deploy_rejects_ambiguous_names_and_accepts_explicit_id(project, api):
    environments, requests = api
    environments.append({"id": "env_staging_2", "name": "Staging", "env_type": "staging", "can_deploy": True})

    ambiguous = CliRunner().invoke(cli.main, ["deploy", "--env", "Staging", "--yes"])
    resolved = CliRunner().invoke(cli.main, ["deploy", "--env", "env_staging_2", "--yes"])

    assert ambiguous.exit_code == 1
    assert "Select an environment ID" in ambiguous.output
    assert resolved.exit_code == 0, resolved.output
    assert uploaded(requests)[0].url.params["environment_id"] == "env_staging_2"


def test_deploy_non_terminal_invalid_selection_reprompts(project, api):
    _, requests = api

    result = CliRunner().invoke(cli.main, ["deploy"], input="missing\n2\n1\ny\n")

    assert result.exit_code == 0, result.output
    assert "not a valid integer" in result.output
    assert "not in the range" in result.output
    assert len(uploaded(requests)) == 1


def test_deploy_menu_distinguishes_duplicate_names_and_keeps_project_default(project, api):
    environments, requests = api
    environments.append({
        "id": "env_staging_2", "name": "Staging", "env_type": "staging",
        "can_deploy": True, "is_default": True,
    })

    result = CliRunner().invoke(cli.main, ["deploy"], input="\ny\n")

    assert result.exit_code == 0, result.output
    assert "Staging (env_staging)" in result.output
    assert "Staging (env_staging_2)" in result.output
    assert "default" in result.output.lower()
    assert uploaded(requests)[0].url.params["environment_id"] == "env_staging_2"


def test_deploy_cancelled_selection_never_uploads(project, api):
    _, requests = api

    result = CliRunner().invoke(cli.main, ["deploy"], input="")

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert not uploaded(requests)


def test_deploy_yes_requires_explicit_target(project, api):
    _, requests = api

    result = CliRunner().invoke(cli.main, ["deploy", "--yes"])

    assert result.exit_code == 2
    assert "--yes requires --env" in result.output
    assert requests == []


def test_deploy_empty_permission_list_is_successful_read_only_result(project, api):
    environments, requests = api
    environments.clear()

    result = CliRunner().invoke(cli.main, ["deploy", "--list"])

    assert result.exit_code == 0, result.output
    assert "No environments are available" in result.output
    assert not uploaded(requests)


@pytest.mark.parametrize(("gate_enabled", "flags", "summary"), [
    (True, [], "Tests: run when present"),
    (False, [], "Tests: disabled for this environment"),
    (True, ["--skip-tests"], "Tests: skipped (--skip-tests)"),
])
def test_deploy_confirmation_reports_environment_test_policy(project, api, gate_enabled, flags, summary):
    environments, _ = api
    environments[1]["deploy_gate_tests_enabled"] = gate_enabled

    result = CliRunner().invoke(cli.main, ["deploy", "--env", "Staging", *flags], input="n\n")

    assert result.exit_code == 0, result.output
    assert summary in result.output


def test_dev_interactive_quick_test_does_not_save_preference_or_credentials(project, api, dev_runtime):
    _, requests = api
    (project / ".connic").unlink()

    result = CliRunner().invoke(
        cli.main, ["dev", "--project-id", "proj_123", "--api-key", "cnc_flag_secret"], input="2\n",
    )

    assert result.exit_code == 0, result.output
    assert "Quick test" in result.output and "Reusable test environment" in result.output
    assert json.loads(requests[0].content) == {}
    assert not (project / ".connic").exists()


def test_dev_defaults_to_reusable_and_prompts_for_name(project, api, dev_runtime):
    _, requests = api

    result = CliRunner().invoke(cli.main, ["dev"], input="\nmy-feature\n")

    assert result.exit_code == 0, result.output
    assert "1. Reusable test environment" in result.output
    assert "2. Quick test" in result.output
    assert "Dev environment [1]" in result.output
    assert "Environment name:" in result.output
    create = next(request for request in requests if request.url.path.endswith("/test-sessions"))
    assert json.loads(create.content) == {"name": "my-feature"}
    assert json.loads((project / ".connic").read_text())["preferred_dev_environment"]["name"] == "my-feature"


@pytest.mark.parametrize(("quick_args", "quick_input"), [([], "3\n"), (["--quick"], "")])
def test_dev_named_choice_remains_preferred_after_quick_test(project, api, dev_runtime, quick_args, quick_input):
    environments, requests = api
    first = CliRunner().invoke(cli.main, ["dev"], input="\nmy-feature\n")
    assert first.exit_code == 0, first.output
    config = json.loads((project / ".connic").read_text())
    assert config["api_key"] == "cnc_secret"
    assert config["preferred_dev_environment"] == {"project_id": "proj_123", "name": "my-feature"}
    environments.append({"id": "env_saved", "name": "my-feature", "env_type": "test", "can_use_dev_session": True})
    requests.clear()

    quick = CliRunner().invoke(cli.main, ["dev", *quick_args], input=quick_input)

    assert quick.exit_code == 0, quick.output
    quick_create = next(request for request in requests if request.url.path.endswith("/test-sessions"))
    assert json.loads(quick_create.content) == {}
    assert json.loads((project / ".connic").read_text()) == config
    requests.clear()

    second = CliRunner().invoke(cli.main, ["dev"], input="\n")

    assert second.exit_code == 0, second.output
    assert "1. Reuse my-feature (preferred)" in second.output
    assert "2. Reusable test environment" in second.output
    assert "3. Quick test" in second.output
    assert "Dev environment [1]" in second.output
    create = next(request for request in requests if request.url.path.endswith("/test-sessions"))
    assert json.loads(create.content) == {"name": "my-feature"}


@pytest.mark.parametrize("api", [403], indirect=True)
@pytest.mark.parametrize(("selection", "expected_body"), [
    ("\n", {"name": "my-feature"}),
    ("3\n", {}),
    ("2\nnew-feature\n", {"name": "new-feature"}),
])
def test_dev_keeps_preferred_and_other_options_when_key_cannot_list_environments(
    project, api, dev_runtime, selection, expected_body,
):
    _, requests = api
    cli._save_dev_preference("proj_123", "my-feature")

    result = CliRunner().invoke(cli.main, ["dev"], input=selection)

    assert result.exit_code == 0, result.output
    assert "1. Reuse my-feature (preferred)" in result.output
    assert "Dev environment [1]" in result.output
    create = next(request for request in requests if request.url.path.endswith("/test-sessions"))
    assert json.loads(create.content) == expected_body


@pytest.mark.parametrize("api", [401, 503], indirect=True)
def test_dev_reports_preference_lookup_failures_other_than_permission_denied(project, api):
    _, requests = api
    cli._save_dev_preference("proj_123", "my-feature")

    result = CliRunner().invoke(cli.main, ["dev"], input="\n")

    assert result.exit_code == 1
    assert "Failed to get environments: Environment list unavailable" in result.output
    assert not any(request.method == "POST" for request in requests)


@pytest.mark.parametrize("invalid_environment", [
    None,
    {"id": "env_saved", "name": "my-feature", "env_type": "production"},
    {"id": "env_saved", "name": "my-feature", "env_type": "test", "can_use_dev_session": False},
    {"id": "env_saved", "name": "my-feature", "env_type": "test", "is_ephemeral": True},
])
def test_dev_does_not_default_to_missing_or_unusable_preference(project, api, dev_runtime, invalid_environment):
    environments, requests = api
    cli._save_dev_preference("proj_123", "my-feature")
    if invalid_environment:
        environments.append(invalid_environment)

    result = CliRunner().invoke(cli.main, ["dev"], input="\nnew-feature\n")

    assert result.exit_code == 0, result.output
    assert "no longer available" in result.output
    assert "Reuse my-feature" not in result.output
    create = next(request for request in requests if request.url.path.endswith("/test-sessions"))
    assert json.loads(create.content) == {"name": "new-feature"}


def test_dev_ignores_preference_for_another_project(project, api, dev_runtime):
    _, requests = api
    cli._save_dev_preference("another_project", "my-feature")

    result = CliRunner().invoke(cli.main, ["dev"], input="\nnew-feature\n")

    assert result.exit_code == 0, result.output
    assert "Reuse my-feature" not in result.output
    assert not any(request.url.path.endswith("/environments/") for request in requests)


def test_dev_cancel_creates_no_session_and_does_not_overwrite_preference(project, api):
    _, requests = api
    previous = (project / ".connic").read_text()

    result = CliRunner().invoke(cli.main, ["dev"], input="")

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert not requests
    assert (project / ".connic").read_text() == previous


def test_dev_quick_flag_and_name_are_mutually_exclusive(project, api):
    _, requests = api

    result = CliRunner().invoke(cli.main, ["dev", "named", "--quick"])

    assert result.exit_code == 2
    assert "Use either --quick or an environment name" in result.output
    assert not requests


def test_dev_preference_preserves_existing_configuration(project):
    config = {"api_key": "cnc_secret", "project_id": "proj_123", "custom_option": True}
    (project / ".connic").write_text(json.dumps(config))

    cli._save_dev_preference("proj_123", "my-feature")

    assert json.loads((project / ".connic").read_text()) == {
        **config, "preferred_dev_environment": {"project_id": "proj_123", "name": "my-feature"},
    }


def test_dev_does_not_overwrite_invalid_configuration(project, capsys):
    (project / ".connic").write_text("{invalid")

    cli._save_dev_preference("proj_123", "my-feature")

    assert (project / ".connic").read_text() == "{invalid"
    assert "Could not save the preferred dev environment" in capsys.readouterr().out


@pytest.mark.parametrize("new_project", ["proj_123", "another_project"])
def test_login_preserves_preference_only_for_same_project(project, new_project):
    cli._save_dev_preference("proj_123", "my-feature")

    result = CliRunner().invoke(cli.main, ["login", "--token", f"{new_project}:new_key"])

    assert result.exit_code == 0, result.output
    config = json.loads((project / ".connic").read_text())
    assert config["api_key"] == "new_key"
    assert ("preferred_dev_environment" in config) == (new_project == "proj_123")
