import json
import os
import time
from pathlib import Path

import click
import httpx

from . import __version__

CACHE_FILE = Path.home() / ".cache" / "connic" / "update_check.json"
CHECK_INTERVAL = 4 * 3600  # 4 hours
PYPI_URL = "https://pypi.org/pypi/connic-composer-sdk/json"


def _parse_version(version_str: str) -> tuple:
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(part)
    return tuple(parts)


def _read_cache() -> dict | None:
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text())
            if time.time() - data.get("last_check", 0) < CHECK_INTERVAL:
                return data
    except Exception:
        pass
    return None


def _write_cache(latest_version: str) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps({
            "last_check": time.time(),
            "latest_version": latest_version,
        }))
    except Exception:
        pass


def _format_update_message(latest: str) -> str:
    return (
        f"\n  Update available: {__version__} -> {latest}\n"
        f"  Run: pip install --upgrade connic-composer-sdk\n"
    )


def check_for_updates() -> str | None:
    """Check PyPI for a newer version. Returns an update message or None."""
    if os.environ.get("CONNIC_NO_UPDATE_CHECK"):
        return None

    try:
        cache = _read_cache()
        if cache:
            latest = cache.get("latest_version", "")
            if latest and _parse_version(latest) > _parse_version(__version__):
                return _format_update_message(latest)
            return None

        resp = httpx.get(PYPI_URL, timeout=3)
        resp.raise_for_status()
        latest = resp.json()["info"]["version"]
        _write_cache(latest)

        if _parse_version(latest) > _parse_version(__version__):
            return _format_update_message(latest)
    except Exception:
        pass

    return None


def print_update_hint() -> None:
    """Check for updates and schedule the hint to print after the command."""
    msg = check_for_updates()
    if msg:
        import atexit
        atexit.register(lambda: click.secho(msg, fg="yellow", err=True))
