import atexit
import json

from connic import update_check


def test_check_for_updates_returns_none_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("CONNIC_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(update_check, "CACHE_FILE", tmp_path / "cache.json")

    assert update_check.check_for_updates() is None


def test_check_for_updates_uses_fresh_cache_without_network(monkeypatch, tmp_path):
    cache = tmp_path / "update_check.json"
    cache.write_text(json.dumps({"last_check": 9999, "latest_version": "99.0.0"}))
    monkeypatch.setattr(update_check, "CACHE_FILE", cache)
    monkeypatch.setattr(update_check.time, "time", lambda: 10000)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fresh cache should avoid PyPI request")

    monkeypatch.setattr(update_check.httpx, "get", fail_if_called)

    assert "Update available" in update_check.check_for_updates()


def test_check_for_updates_fetches_pypi_and_writes_cache(monkeypatch, tmp_path):
    cache = tmp_path / "connic" / "update_check.json"
    monkeypatch.setattr(update_check, "CACHE_FILE", cache)
    monkeypatch.setattr(update_check.time, "time", lambda: 1234)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"info": {"version": "99.1.0"}}

    monkeypatch.setattr(update_check.httpx, "get", lambda url, timeout: Response())

    message = update_check.check_for_updates()

    assert "99.1.0" in message
    assert json.loads(cache.read_text()) == {"last_check": 1234, "latest_version": "99.1.0"}


def test_check_for_updates_suppresses_network_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(update_check, "CACHE_FILE", tmp_path / "missing.json")

    def raise_timeout(*args, **kwargs):
        raise TimeoutError("offline")

    monkeypatch.setattr(update_check.httpx, "get", raise_timeout)

    assert update_check.check_for_updates() is None


def test_check_for_updates_stale_cache_refreshes_from_pypi(monkeypatch, tmp_path):
    """When the JSON cache is older than CHECK_INTERVAL, PyPI is consulted again."""
    cache = tmp_path / "stale.json"
    monkeypatch.setattr(update_check, "CACHE_FILE", cache)
    old_ts = 0
    cache.write_text(json.dumps({"last_check": old_ts, "latest_version": "0.0.1"}))
    monkeypatch.setattr(update_check.time, "time", lambda: 10**9)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"info": {"version": update_check.__version__}}

    monkeypatch.setattr(update_check.httpx, "get", lambda url, timeout: Response())

    assert update_check.check_for_updates() is None
    data = json.loads(cache.read_text())
    assert data["latest_version"] == update_check.__version__


def test_check_for_updates_fresh_cache_same_version_returns_none(monkeypatch, tmp_path):
    cache = tmp_path / "fresh.json"
    monkeypatch.setattr(update_check, "CACHE_FILE", cache)
    cache.write_text(
        json.dumps({"last_check": 9999, "latest_version": update_check.__version__}),
    )
    monkeypatch.setattr(update_check.time, "time", lambda: 10000)

    def must_not_call_pypi(*args, **kwargs):
        raise AssertionError("must not call PyPI")

    monkeypatch.setattr(update_check.httpx, "get", must_not_call_pypi)

    assert update_check.check_for_updates() is None


def test_check_for_updates_malformed_cache_falls_back_to_pypi(monkeypatch, tmp_path):
    cache = tmp_path / "bad.json"
    monkeypatch.setattr(update_check, "CACHE_FILE", cache)
    cache.write_text("not-json")
    monkeypatch.setattr(update_check.time, "time", lambda: 1)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"info": {"version": update_check.__version__}}

    monkeypatch.setattr(update_check.httpx, "get", lambda url, timeout: Response())

    assert update_check.check_for_updates() is None


def test_parse_version_handles_prerelease_numeric_segments():
    assert update_check._parse_version("1.2.3a1") == (1, 2, "3a1")
    assert update_check._parse_version("2.0") > update_check._parse_version("1.99")


def test_print_update_hint_registers_atexit_when_update_available(monkeypatch):
    monkeypatch.setattr(update_check, "check_for_updates", lambda: "please upgrade")
    registered = []
    monkeypatch.setattr(atexit, "register", lambda fn: registered.append(fn))

    update_check.print_update_hint()

    assert len(registered) == 1
    echoed = []
    monkeypatch.setattr(update_check.click, "secho", lambda msg, **kw: echoed.append((msg, kw)))
    registered[0]()
    assert echoed[0][0] == "please upgrade"
    assert echoed[0][1].get("fg") == "yellow"
    assert echoed[0][1].get("err") is True


def test_print_update_hint_no_atexit_when_no_update(monkeypatch):
    monkeypatch.setattr(update_check, "check_for_updates", lambda: None)
    registered = []
    monkeypatch.setattr(atexit, "register", lambda fn: registered.append(fn))

    update_check.print_update_hint()

    assert registered == []


def test_write_cache_swallows_errors_when_cache_path_is_invalid(monkeypatch, tmp_path):
    """If the cache file cannot be written (e.g. parent is a file), PyPI check still completes."""
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    cache_path = blocker / "update_check.json"
    monkeypatch.setattr(update_check, "CACHE_FILE", cache_path)
    monkeypatch.delenv("CONNIC_NO_UPDATE_CHECK", raising=False)

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"info": {"version": update_check.__version__}}

    monkeypatch.setattr(update_check.httpx, "get", lambda url, timeout: Response())

    assert update_check.check_for_updates() is None
