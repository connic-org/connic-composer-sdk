import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import click
import httpx
import yaml
from packaging.version import InvalidVersion, Version

from . import __version__

CACHE_FILE = Path.home() / ".cache" / "connic" / "update_check.json"
CHECK_INTERVAL = 4 * 3600
PACKAGE_NAME = "connic-composer-sdk"
PYPI_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
SKILL_URL = "https://raw.githubusercontent.com/connic-org/connic-skill/main/plugins/connic/skills/connic/SKILL.md"
PLUGIN_URL = "https://raw.githubusercontent.com/connic-org/connic-skill/main/plugins/connic/.codex-plugin/plugin.json"
SKILL_PATHS = (
    Path(".agents/skills/connic/SKILL.md"),
    Path(".claude/skills/connic/SKILL.md"),
)
CONNIC_PLUGIN_ID = "connic@connic"
PLUGIN_CLIENTS = (("Codex", "codex"), ("Claude Code", "claude"))


class UpdateAction(str, Enum):
    BOTH = "both"
    SDK = "sdk"
    SKILL = "skill"
    SKIP = "skip"
    DISABLED = "disabled"
    NONE = "none"


@dataclass(frozen=True)
class PluginInstallation:
    client: str
    executable: str
    version: str | None
    scope: str | None = None
    project_path: Path | None = None

    @property
    def label(self) -> str:
        if self.client == "Codex":
            return "Codex plugin"
        details = self.scope or "unknown scope"
        if self.project_path is not None:
            details = f"{details}: {self.project_path}"
        return f"Claude plugin ({details})"


@dataclass(frozen=True)
class UpdateStatus:
    current_sdk_version: str
    latest_sdk_version: str | None
    current_skill_version: str | None
    latest_skill_version: str | None
    sdk_update_available: bool
    skill_update_available: bool
    installed_skill_paths: tuple[Path, ...]
    missing_skill_paths: tuple[Path, ...]
    sdk_check_succeeded: bool = True
    skill_check_succeeded: bool = True
    local_skill_update_available: bool = False
    latest_plugin_version: str | None = None
    installed_plugins: tuple[PluginInstallation, ...] = ()
    plugin_check_succeeded: bool = True
    client_check_failures: tuple[str, ...] = ()

    @property
    def has_updates(self) -> bool:
        return self.check_complete and (
            self.sdk_update_available or self.skill_update_available
        )

    @property
    def plugin_update_available(self) -> bool:
        return any(
            _plugin_needs_update(installation, self.latest_plugin_version)
            for installation in self.installed_plugins
        )

    @property
    def check_complete(self) -> bool:
        return (
            self.sdk_check_succeeded
            and (not self.installed_skill_paths or self.skill_check_succeeded)
            and (not self.installed_plugins or self.plugin_check_succeeded)
            and not self.client_check_failures
        )

    @property
    def action(self) -> UpdateAction:
        if not self.check_complete:
            return UpdateAction.NONE
        if self.sdk_update_available and self.skill_update_available:
            return UpdateAction.BOTH
        if self.sdk_update_available:
            return UpdateAction.SDK
        if self.skill_update_available:
            return UpdateAction.SKILL
        return UpdateAction.NONE


def _parse_version(version_str: str) -> Version:
    return Version(version_str)


def _is_newer(candidate: str | None, current: str | None) -> bool:
    if not candidate or not current:
        return False
    try:
        return _parse_version(candidate) > _parse_version(current)
    except InvalidVersion:
        return False


def _load_cache() -> dict:
    try:
        data = json.loads(CACHE_FILE.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}

    if data.get("latest_version") and not data.get("latest_sdk_version"):
        data["latest_sdk_version"] = data["latest_version"]
        data.pop("latest_version", None)
        _save_cache(data)
    return data


def _save_cache(data: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(data))
    except OSError:
        pass


def _read_cache() -> dict | None:
    data = _load_cache()
    last_check = data.get("last_check")
    has_plugin_check = "plugin_check_succeeded" in data
    if (
        has_plugin_check
        and isinstance(last_check, (int, float))
        and time.time() - last_check < CHECK_INTERVAL
    ):
        return data
    return None


def _write_cache(latest_version: str, latest_skill_version: str | None = None) -> None:
    data = _load_cache()
    data.update(
        {
            "last_check": time.time(),
            "latest_sdk_version": latest_version,
            "latest_skill_version": latest_skill_version,
        }
    )
    _save_cache(data)


def reminders_enabled() -> bool:
    return not _load_cache().get("reminders_disabled", False)


def set_reminders_enabled(enabled: bool) -> None:
    data = _load_cache()
    data["reminders_disabled"] = not enabled
    _save_cache(data)


def enable_update_reminders() -> None:
    set_reminders_enabled(True)


def _skill_version(contents: str) -> str | None:
    lines = contents.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    try:
        closing_index = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
        frontmatter = yaml.safe_load("\n".join(lines[1:closing_index])) or {}
    except (StopIteration, yaml.YAMLError):
        return None

    if not isinstance(frontmatter, dict):
        return None
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return None
    version = metadata.get("version")
    return str(version) if version is not None else None


def _fetch_remote_versions(
    force: bool = False,
) -> tuple[str | None, str | None, str | None, bool, bool, bool]:
    cached = _read_cache()
    if cached is not None and not force:
        latest_sdk = cached.get("latest_sdk_version")
        latest_skill = cached.get("latest_skill_version")
        latest_plugin = cached.get("latest_plugin_version")
        sdk_check_succeeded = cached.get("sdk_check_succeeded")
        if not isinstance(sdk_check_succeeded, bool):
            sdk_check_succeeded = latest_sdk is not None
        skill_check_succeeded = cached.get("skill_check_succeeded")
        if not isinstance(skill_check_succeeded, bool):
            skill_check_succeeded = latest_skill is not None
        plugin_check_succeeded = cached.get("plugin_check_succeeded")
        if not isinstance(plugin_check_succeeded, bool):
            plugin_check_succeeded = latest_plugin is not None
        return (
            latest_sdk,
            latest_skill,
            latest_plugin,
            sdk_check_succeeded,
            skill_check_succeeded,
            plugin_check_succeeded,
        )

    previous = _load_cache()
    latest_sdk = previous.get("latest_sdk_version")
    latest_skill = previous.get("latest_skill_version")
    latest_plugin = previous.get("latest_plugin_version")
    sdk_check_succeeded = False
    skill_check_succeeded = False
    plugin_check_succeeded = False
    try:
        response = httpx.get(PYPI_URL, timeout=3)
        response.raise_for_status()
        fetched_sdk = response.json()["info"]["version"]
        _parse_version(fetched_sdk)
        latest_sdk = fetched_sdk
        sdk_check_succeeded = True
    except (httpx.HTTPError, OSError, KeyError, TypeError, ValueError, InvalidVersion):
        pass

    try:
        response = httpx.get(SKILL_URL, timeout=3)
        response.raise_for_status()
        fetched_skill = _skill_version(response.text)
        if fetched_skill is not None:
            _parse_version(fetched_skill)
            latest_skill = fetched_skill
            skill_check_succeeded = True
    except (httpx.HTTPError, OSError, InvalidVersion):
        pass

    try:
        response = httpx.get(PLUGIN_URL, timeout=3)
        response.raise_for_status()
        fetched_plugin = response.json()["version"]
        _parse_version(fetched_plugin)
        latest_plugin = fetched_plugin
        plugin_check_succeeded = True
    except (httpx.HTTPError, OSError, KeyError, TypeError, ValueError, InvalidVersion):
        pass

    data = previous
    data.update(
        {
            "last_check": time.time(),
            "latest_sdk_version": latest_sdk,
            "latest_skill_version": latest_skill,
            "latest_plugin_version": latest_plugin,
            "sdk_check_succeeded": sdk_check_succeeded,
            "skill_check_succeeded": skill_check_succeeded,
            "plugin_check_succeeded": plugin_check_succeeded,
        }
    )
    data.pop("latest_version", None)
    _save_cache(data)
    return (
        latest_sdk,
        latest_skill,
        latest_plugin,
        sdk_check_succeeded,
        skill_check_succeeded,
        plugin_check_succeeded,
    )


def _installed_plugins_for_client(
    label: str,
    executable: str,
) -> tuple[tuple[PluginInstallation, ...], bool]:
    if shutil.which(executable) is None:
        return (), True

    try:
        result = subprocess.run(
            [executable, "plugin", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            return (), False
        data = json.loads(result.stdout)
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        return (), False

    plugins = data.get("installed") if executable == "codex" and isinstance(data, dict) else data
    if not isinstance(plugins, list):
        return (), False

    installations = []
    seen = set()
    for plugin in plugins:
        if not isinstance(plugin, dict):
            continue
        if (plugin.get("pluginId") or plugin.get("id")) != CONNIC_PLUGIN_ID:
            continue
        if plugin.get("installed", True) is False:
            continue

        scope = plugin.get("scope") if isinstance(plugin.get("scope"), str) else None
        raw_project_path = plugin.get("projectPath")
        project_path = Path(raw_project_path) if isinstance(raw_project_path, str) else None
        key = scope, project_path
        if key in seen:
            continue
        seen.add(key)
        version = plugin.get("version")
        installations.append(
            PluginInstallation(
                client=label,
                executable=executable,
                version=version if isinstance(version, str) else None,
                scope=scope,
                project_path=project_path,
            )
        )
    return tuple(installations), True


def get_installed_plugins() -> tuple[tuple[PluginInstallation, ...], tuple[str, ...]]:
    installations = []
    failures = []
    for label, executable in PLUGIN_CLIENTS:
        client_installations, succeeded = _installed_plugins_for_client(label, executable)
        installations.extend(client_installations)
        if not succeeded:
            failures.append(label)
    return tuple(installations), tuple(failures)


def _plugin_needs_update(
    installation: PluginInstallation,
    latest_plugin_version: str | None,
) -> bool:
    if latest_plugin_version is None:
        return False
    if installation.version is None:
        return True
    try:
        return _parse_version(latest_plugin_version) > _parse_version(installation.version)
    except InvalidVersion:
        return True


def _installed_skill_status(
    project_root: Path,
    latest_skill_version: str | None,
) -> tuple[str | None, bool, tuple[Path, ...], tuple[Path, ...]]:
    paths = tuple(project_root / relative_path for relative_path in SKILL_PATHS)
    installed = tuple(path for path in paths if path.is_file())
    if not installed:
        return None, False, (), ()

    missing = tuple(path for path in paths if path not in installed)
    versions: list[str | None] = []
    for path in installed:
        try:
            versions.append(_skill_version(path.read_text()))
        except OSError:
            versions.append(None)

    unique_versions = set(versions)
    if len(unique_versions) == 1:
        installed_version = versions[0] or "legacy"
    else:
        installed_version = "mixed"

    legacy = any(version is None for version in versions)
    outdated = bool(latest_skill_version) and any(version is None or _is_newer(latest_skill_version, version) for version in versions)
    return installed_version, outdated or legacy and bool(latest_skill_version), installed, missing


def get_update_status(
    *,
    force: bool = False,
    project_root: str | Path | None = None,
) -> UpdateStatus | None:
    """Return SDK and installed-skill update state.

    ``force`` bypasses both the reminder preference and the four-hour remote cache.
    ``CONNIC_NO_UPDATE_CHECK`` always disables the check.
    """
    if os.environ.get("CONNIC_NO_UPDATE_CHECK"):
        return None
    if not force and not reminders_enabled():
        return None

    (
        latest_sdk,
        latest_skill,
        latest_plugin,
        sdk_check_succeeded,
        skill_check_succeeded,
        plugin_check_succeeded,
    ) = _fetch_remote_versions(force=force)
    root = Path.cwd() if project_root is None else Path(project_root)
    current_skill, local_skill_available, installed, missing = _installed_skill_status(
        root,
        latest_skill,
    )
    installed_plugins, client_check_failures = get_installed_plugins()
    plugin_available = any(
        _plugin_needs_update(installation, latest_plugin)
        for installation in installed_plugins
    )
    return UpdateStatus(
        current_sdk_version=__version__,
        latest_sdk_version=latest_sdk,
        current_skill_version=current_skill,
        latest_skill_version=latest_skill,
        sdk_update_available=_is_newer(latest_sdk, __version__),
        skill_update_available=local_skill_available or plugin_available,
        installed_skill_paths=installed,
        missing_skill_paths=missing,
        sdk_check_succeeded=sdk_check_succeeded,
        skill_check_succeeded=skill_check_succeeded,
        local_skill_update_available=local_skill_available,
        latest_plugin_version=latest_plugin,
        installed_plugins=installed_plugins,
        plugin_check_succeeded=plugin_check_succeeded,
        client_check_failures=client_check_failures,
    )


def get_manual_update_status(
    project_root: str | Path | None = None,
) -> UpdateStatus | None:
    return get_update_status(force=True, project_root=project_root)


def _format_update_message(status: UpdateStatus) -> str:
    lines = ["Updates available:"]
    if status.sdk_update_available:
        lines.append(f"  SDK    {status.current_sdk_version} → {status.latest_sdk_version}")
    if status.local_skill_update_available:
        if status.latest_skill_version:
            lines.append(
                f"  Project skill  {status.current_skill_version or 'unknown'} → "
                f"{status.latest_skill_version}"
            )
        else:
            lines.append("  Project skill  installed copies need syncing")
    for installation in status.installed_plugins:
        if _plugin_needs_update(installation, status.latest_plugin_version):
            lines.append(
                f"  {installation.label}  {installation.version or 'unknown'} → "
                f"{status.latest_plugin_version}"
            )
    if (
        status.skill_update_available
        and not status.local_skill_update_available
        and not status.installed_plugins
    ):
        lines.append(
            f"  Skill  {status.current_skill_version or 'unknown'} → "
            f"{status.latest_skill_version or 'unknown'}"
        )
    return "\n".join(lines)


def check_for_updates(
    *,
    force: bool = False,
    project_root: str | Path | None = None,
) -> str | None:
    """Return a formatted update status for compatibility with existing callers."""
    status = get_update_status(force=force, project_root=project_root)
    if status and status.check_complete and status.has_updates:
        return _format_update_message(status)
    return None


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def _noninteractive_warning(status: UpdateStatus) -> str:
    components = []
    if status.sdk_update_available:
        components.append("SDK")
    if status.local_skill_update_available:
        components.append("skill")
    if status.plugin_update_available:
        components.append("plugin")
    if (
        status.skill_update_available
        and not status.local_skill_update_available
        and not status.plugin_update_available
    ):
        components.append("skill")
    command = "connic update --skill" if status.action == UpdateAction.SKILL else "connic update"
    if len(components) > 2:
        component_names = f"{', '.join(components[:-1])}, and {components[-1]}"
    else:
        component_names = " and ".join(components)
    return f"Connic {component_names} update available; run `{command}`."


def print_update_hint() -> UpdateAction:
    """Offer available updates and return the selected action without installing."""
    if os.environ.get("CONNIC_NO_UPDATE_CHECK") or not reminders_enabled():
        return UpdateAction.DISABLED

    status = get_update_status()
    if not status or not status.check_complete or not status.has_updates:
        return UpdateAction.NONE

    if not _is_interactive():
        click.secho(_noninteractive_warning(status), fg="yellow", err=True)
        return UpdateAction.NONE

    click.echo(_format_update_message(status), err=True)
    click.echo(err=True)
    click.echo("1 Update", err=True)
    click.echo("2 Skip for now", err=True)
    click.echo("3 Skip and don't remind me again", err=True)
    choices = ["1", "2", "3"]
    if status.sdk_update_available and status.skill_update_available:
        click.echo("4 Only update SDK", err=True)
        click.echo("5 Only update skill/plugins", err=True)
        choices.extend(["4", "5"])

    choice = click.prompt(
        "Select an option",
        type=click.Choice(choices),
        default="2",
        show_default=False,
        err=True,
    )
    if choice == "1":
        return status.action
    if choice == "2":
        return UpdateAction.SKIP
    if choice == "3":
        set_reminders_enabled(False)
        return UpdateAction.DISABLED
    if choice == "4":
        return UpdateAction.SDK
    return UpdateAction.SKILL


def get_sdk_update_command(prefix: str | Path | None = None) -> tuple[str, ...]:
    environment = str(prefix or sys.prefix).replace("\\", "/").lower()
    if "/pipx/venvs/" in environment:
        return "pipx", "upgrade", PACKAGE_NAME
    if "/uv/tools/" in environment:
        return "uv", "tool", "upgrade", PACKAGE_NAME
    return sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE_NAME


def update_sdk() -> bool:
    try:
        completed = subprocess.run(get_sdk_update_command(), check=False)
    except OSError:
        return False
    return completed.returncode == 0
