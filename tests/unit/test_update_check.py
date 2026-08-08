import json
import subprocess
import sys

import pytest

from connic import update_check


@pytest.fixture(autouse=True)
def isolated_update_state(monkeypatch, tmp_path):
    monkeypatch.delenv("CONNIC_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(update_check, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.chdir(tmp_path)


def _skill(version: str | None) -> str:
    metadata = f'\nmetadata:\n  version: "{version}"' if version else ""
    return f"---\nname: connic{metadata}\n---\n\n# Connic\n"


def _status(*, sdk=False, skill=False):
    return update_check.UpdateStatus(
        current_sdk_version="1.0.0",
        latest_sdk_version="2.0.0" if sdk else "1.0.0",
        current_skill_version="1.0.0" if skill else None,
        latest_skill_version="2.0.0" if skill else "1.0.0",
        sdk_update_available=sdk,
        skill_update_available=skill,
        installed_skill_paths=(),
        missing_skill_paths=(),
    )


def test_skill_version_reads_standard_metadata_version():
    assert update_check._skill_version(_skill("1.2.3")) == "1.2.3"
    assert update_check._skill_version(_skill(None)) is None


def test_skill_version_ignores_non_mapping_frontmatter():
    assert update_check._skill_version("---\n- invalid\n---\n") is None


def test_get_update_status_honors_environment_disable(monkeypatch):
    monkeypatch.setenv("CONNIC_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(
        update_check,
        "_fetch_remote_versions",
        lambda **kwargs: pytest.fail("disabled checks must not fetch"),
    )

    assert update_check.get_update_status(force=True) is None
    assert update_check.print_update_hint() == update_check.UpdateAction.DISABLED


def test_reminder_preference_blocks_automatic_but_not_manual_check(monkeypatch):
    update_check.set_reminders_enabled(False)
    calls = []
    monkeypatch.setattr(
        update_check,
        "_fetch_remote_versions",
        lambda **kwargs: calls.append(kwargs) or ("2.0.0", "1.0.0"),
    )

    assert update_check.get_update_status() is None
    assert update_check.get_manual_update_status() is not None
    assert calls == [{"force": True}]


def test_enable_update_reminders_clears_persisted_disable():
    update_check.set_reminders_enabled(False)
    assert update_check.reminders_enabled() is False

    update_check.enable_update_reminders()

    assert update_check.reminders_enabled() is True
    assert json.loads(update_check.CACHE_FILE.read_text())["reminders_disabled"] is False


def test_fresh_legacy_cache_is_migrated_without_network(monkeypatch):
    update_check.CACHE_FILE.write_text(
        json.dumps(
            {
                "last_check": 9999,
                "latest_version": "99.0.0",
            }
        )
    )
    monkeypatch.setattr(update_check.time, "time", lambda: 10000)
    monkeypatch.setattr(
        update_check.httpx,
        "get",
        lambda *args, **kwargs: pytest.fail("fresh cache must not fetch"),
    )

    status = update_check.get_update_status()

    assert status.sdk_update_available is True
    cache = json.loads(update_check.CACHE_FILE.read_text())
    assert cache["latest_sdk_version"] == "99.0.0"
    assert "latest_version" not in cache


def test_remote_sdk_and_skill_versions_are_cached(monkeypatch):
    monkeypatch.setattr(update_check.time, "time", lambda: 1234)
    requested = []

    class Response:
        def __init__(self, *, data=None, text=""):
            self._data = data
            self.text = text

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    def get(url, timeout):
        requested.append((url, timeout))
        if url == update_check.PYPI_URL:
            return Response(data={"info": {"version": "2.0.0"}})
        return Response(text=_skill("3.0.0"))

    monkeypatch.setattr(update_check.httpx, "get", get)

    assert update_check._fetch_remote_versions() == ("2.0.0", "3.0.0")
    assert requested == [
        (update_check.PYPI_URL, 3),
        (update_check.SKILL_URL, 3),
    ]
    assert json.loads(update_check.CACHE_FILE.read_text()) == {
        "last_check": 1234,
        "latest_sdk_version": "2.0.0",
        "latest_skill_version": "3.0.0",
    }


def test_fresh_cache_is_used_for_both_remote_versions(monkeypatch):
    update_check.CACHE_FILE.write_text(
        json.dumps(
            {
                "last_check": 9999,
                "latest_sdk_version": "2.0.0",
                "latest_skill_version": "3.0.0",
            }
        )
    )
    monkeypatch.setattr(update_check.time, "time", lambda: 10000)
    monkeypatch.setattr(
        update_check.httpx,
        "get",
        lambda *args, **kwargs: pytest.fail("fresh cache must not fetch"),
    )

    assert update_check._fetch_remote_versions() == ("2.0.0", "3.0.0")


def test_network_failures_are_cached_to_avoid_delaying_every_command(monkeypatch):
    def offline(*args, **kwargs):
        raise TimeoutError("offline")

    monkeypatch.setattr(update_check.time, "time", lambda: 1234)
    monkeypatch.setattr(update_check.httpx, "get", offline)

    assert update_check.check_for_updates() is None
    assert json.loads(update_check.CACHE_FILE.read_text())["last_check"] == 1234


def test_absent_skill_does_not_offer_skill_install(monkeypatch):
    monkeypatch.setattr(
        update_check,
        "_fetch_remote_versions",
        lambda **kwargs: (update_check.__version__, "2.0.0"),
    )

    status = update_check.get_update_status()

    assert status.skill_update_available is False
    assert status.installed_skill_paths == ()
    assert status.missing_skill_paths == ()


def test_missing_claude_destination_requires_skill_sync(monkeypatch, tmp_path):
    agents_skill = tmp_path / update_check.SKILL_PATHS[0]
    agents_skill.parent.mkdir(parents=True)
    agents_skill.write_text(_skill("2.0.0"))
    monkeypatch.setattr(
        update_check,
        "_fetch_remote_versions",
        lambda **kwargs: (update_check.__version__, "2.0.0"),
    )

    status = update_check.get_update_status(project_root=tmp_path)

    assert status.skill_update_available is True
    assert status.current_skill_version == "2.0.0"
    assert status.installed_skill_paths == (agents_skill,)
    assert status.missing_skill_paths == (tmp_path / update_check.SKILL_PATHS[1],)


def test_legacy_installed_skill_is_outdated_when_remote_is_versioned(monkeypatch, tmp_path):
    for relative_path in update_check.SKILL_PATHS:
        skill_path = tmp_path / relative_path
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(_skill(None))
    monkeypatch.setattr(
        update_check,
        "_fetch_remote_versions",
        lambda **kwargs: (update_check.__version__, "1.0.0"),
    )

    status = update_check.get_update_status(project_root=tmp_path)

    assert status.current_skill_version == "legacy"
    assert status.skill_update_available is True


def test_older_or_mismatched_installed_skills_are_outdated(monkeypatch, tmp_path):
    versions = ("1.0.0", "2.0.0")
    for relative_path, version in zip(update_check.SKILL_PATHS, versions):
        skill_path = tmp_path / relative_path
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(_skill(version))
    monkeypatch.setattr(
        update_check,
        "_fetch_remote_versions",
        lambda **kwargs: (update_check.__version__, "2.0.0"),
    )

    status = update_check.get_update_status(project_root=tmp_path)

    assert status.current_skill_version == "mixed"
    assert status.skill_update_available is True


def test_current_skill_in_both_destinations_needs_no_update(monkeypatch, tmp_path):
    for relative_path in update_check.SKILL_PATHS:
        skill_path = tmp_path / relative_path
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(_skill("2.0.0"))
    monkeypatch.setattr(
        update_check,
        "_fetch_remote_versions",
        lambda **kwargs: (update_check.__version__, "2.0.0"),
    )

    status = update_check.get_update_status(project_root=tmp_path)

    assert status.skill_update_available is False
    assert status.missing_skill_paths == ()


def test_check_for_updates_keeps_formatted_message_api(monkeypatch):
    monkeypatch.setattr(update_check, "get_update_status", lambda **kwargs: _status(sdk=True, skill=True))

    message = update_check.check_for_updates()

    assert "Updates available:" in message
    assert "SDK    1.0.0 → 2.0.0" in message
    assert "Skill  1.0.0 → 2.0.0" in message


def test_noninteractive_check_warns_without_prompting(monkeypatch, capsys):
    monkeypatch.setattr(update_check, "get_update_status", lambda **kwargs: _status(sdk=True, skill=True))
    monkeypatch.setattr(update_check, "_is_interactive", lambda: False)
    monkeypatch.setattr(
        update_check.click,
        "prompt",
        lambda *args, **kwargs: pytest.fail("noninteractive check must not prompt"),
    )

    action = update_check.print_update_hint()

    assert action == update_check.UpdateAction.NONE
    assert capsys.readouterr().err == ("Connic SDK and skill update available; run `connic update`.\n")


def test_noninteractive_skill_update_uses_update_command(monkeypatch, capsys):
    monkeypatch.setattr(update_check, "get_update_status", lambda **kwargs: _status(skill=True))
    monkeypatch.setattr(update_check, "_is_interactive", lambda: False)

    assert update_check.print_update_hint() == update_check.UpdateAction.NONE
    assert capsys.readouterr().err == (
        "Connic skill update available; run `connic update --skill`.\n"
    )


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("1", update_check.UpdateAction.BOTH),
        ("2", update_check.UpdateAction.SKIP),
        ("3", update_check.UpdateAction.DISABLED),
        ("4", update_check.UpdateAction.SDK),
        ("5", update_check.UpdateAction.SKILL),
    ],
)
def test_both_update_prompt_has_exact_options(monkeypatch, capsys, choice, expected):
    monkeypatch.setattr(update_check, "get_update_status", lambda **kwargs: _status(sdk=True, skill=True))
    monkeypatch.setattr(update_check, "_is_interactive", lambda: True)
    monkeypatch.setattr(update_check.click, "prompt", lambda *args, **kwargs: choice)

    action = update_check.print_update_hint()

    assert action == expected
    output = capsys.readouterr().err
    assert "1 Update\n" in output
    assert "2 Skip for now\n" in output
    assert "3 Skip and don't remind me again\n" in output
    assert "4 Only update SDK\n" in output
    assert "5 Only update skill\n" in output
    assert update_check.reminders_enabled() is (choice != "3")


@pytest.mark.parametrize(
    ("sdk", "skill", "expected"),
    [
        (True, False, update_check.UpdateAction.SDK),
        (False, True, update_check.UpdateAction.SKILL),
    ],
)
def test_single_update_prompt_returns_component_action(
    monkeypatch,
    capsys,
    sdk,
    skill,
    expected,
):
    monkeypatch.setattr(
        update_check,
        "get_update_status",
        lambda **kwargs: _status(sdk=sdk, skill=skill),
    )
    monkeypatch.setattr(update_check, "_is_interactive", lambda: True)
    monkeypatch.setattr(update_check.click, "prompt", lambda *args, **kwargs: "1")

    assert update_check.print_update_hint() == expected
    output = capsys.readouterr().err
    assert "1 Update\n" in output
    assert "2 Skip for now\n" in output
    assert "3 Skip and don't remind me again\n" in output
    assert "4 Only update SDK" not in output


def test_skip_only_applies_to_current_run(monkeypatch):
    monkeypatch.setattr(update_check, "get_update_status", lambda **kwargs: _status(sdk=True))
    monkeypatch.setattr(update_check, "_is_interactive", lambda: True)
    monkeypatch.setattr(update_check.click, "prompt", lambda *args, **kwargs: "2")

    assert update_check.print_update_hint() == update_check.UpdateAction.SKIP
    assert update_check.reminders_enabled() is True


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        (
            "/Users/test/.local/share/pipx/venvs/connic-composer-sdk",
            ("pipx", "upgrade", "connic-composer-sdk"),
        ),
        (
            "/Users/test/.local/share/uv/tools/connic-composer-sdk",
            ("uv", "tool", "upgrade", "connic-composer-sdk"),
        ),
        (
            "/Users/test/project/.venv",
            (
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "connic-composer-sdk",
            ),
        ),
    ],
)
def test_sdk_update_command_detects_installer(prefix, expected):
    assert update_check.get_sdk_update_command(prefix) == expected


def test_update_sdk_runs_detected_command(monkeypatch):
    command = ("uv", "tool", "upgrade", "connic-composer-sdk")
    monkeypatch.setattr(update_check, "get_sdk_update_command", lambda: command)
    calls = []
    monkeypatch.setattr(
        update_check.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)) or subprocess.CompletedProcess(args[0], 0),
    )

    assert update_check.update_sdk() is True
    assert calls == [((command,), {"check": False})]


def test_update_sdk_returns_false_when_installer_is_missing(monkeypatch):
    monkeypatch.setattr(
        update_check.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    assert update_check.update_sdk() is False


def test_cache_write_errors_do_not_break_update_check(monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    monkeypatch.setattr(update_check, "CACHE_FILE", blocker / "cache.json")
    monkeypatch.setattr(
        update_check,
        "_fetch_remote_versions",
        lambda **kwargs: (update_check.__version__, None),
    )

    assert update_check.check_for_updates() is None
