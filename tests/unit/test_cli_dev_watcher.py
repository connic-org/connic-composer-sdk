import io
import shutil
import sys
import tarfile
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner
from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileClosedEvent,
    FileCreatedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileOpenedEvent,
)
from watchdog.observers.api import ObservedWatch

from connic import cli


class ScheduledObserver:
    """Deliver filesystem events only to handlers whose watches cover the path."""

    def __init__(self):
        self.watches = {}
        self.started = False
        self.stopped = False
        self.joined = False

    def schedule(self, handler, path, recursive=False):
        watch = ObservedWatch(path, recursive=recursive)
        self.watches[watch] = handler
        return watch

    def unschedule(self, watch):
        self.watches.pop(watch, None)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self):
        self.joined = True

    def emit(self, event):
        assert self.started
        paths = [Path(event.src_path).absolute()]
        if getattr(event, "dest_path", ""):
            paths.append(Path(event.dest_path).absolute())
        registrations = list(self.watches.items())
        if isinstance(event, (DirDeletedEvent, DirMovedEvent)):
            removed = Path(event.src_path).absolute()
            for watch in list(self.watches):
                root = Path(watch.path).absolute()
                if root == removed or removed in root.parents:
                    self.unschedule(watch)
        for watch, handler in registrations:
            root = Path(watch.path).absolute()
            if any(
                path == root or path.parent == root or (watch.is_recursive and root in path.parents)
                for path in paths
            ):
                handler.dispatch(event)


@pytest.fixture
def dev_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "support.yaml").write_text(
        'version: "1.0"\n'
        "name: support\n"
        "description: Support agent\n"
        "type: llm\n"
        "model: openai/gpt-4o\n"
        "system_prompt: Help customers.\n"
    )
    return tmp_path


def run_dev(monkeypatch, changes=(), observer=None, app_uploads=None, output=None, app_failures=None):
    uploads = []
    observer = observer or ScheduledObserver()

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Client:
        closed = False
        deleted = False

        def __init__(self, **kwargs):
            pass

        def post(self, path, json=None, files=None, timeout=None):
            if path == "/projects/proj_123/test-sessions":
                return Response({"id": "sess_123", "environment_id": "env_test", "environment_name": "support-dev"})
            content = files["file"][1]
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
                members = {member.name: archive.extractfile(member).read() for member in archive.getmembers()}
            if app_uploads is not None and path.startswith("/test-sessions/sess_123/apps/"):
                key = path.rsplit("/", 1)[1]
                failure = app_failures.pop(0) if app_failures else None
                if isinstance(failure, Exception):
                    raise failure
                if failure is not None:
                    response = Response({"detail": "Service unavailable"})
                    response.status_code = failure
                    return response
                app_uploads.append((key, members))
                return Response({"url": f"https://{key}-1a2b3c4d.connic.app"})
            assert path == "/test-sessions/sess_123/files"
            uploads.append(members)
            return Response({"files_hash": f"hash_{len(uploads)}", "size_bytes": len(content)})

        def get(self, path):
            assert path in {"/test-sessions/sess_123", "/test-sessions/sess_123/status"}
            return Response({"container_status": "running", "status": "active"})

        def delete(self, path):
            assert path == "/test-sessions/sess_123"
            Client.deleted = True
            return Response({"environment_deleted": True})

        def close(self):
            Client.closed = True

    class Clock:
        current = 1000.0
        sleeps = 0

        @classmethod
        def time(cls):
            return cls.current

        @classmethod
        def sleep(cls, seconds):
            cls.current += 2.0
            step, settling = divmod(cls.sleeps, 2)
            cls.sleeps += 1
            if step >= len(changes):
                raise KeyboardInterrupt
            if not settling:
                changes[step](observer)

        @staticmethod
        def strftime(fmt):
            return "12:00:00"

    monkeypatch.setattr(cli.httpx, "Client", Client)
    monkeypatch.setattr("watchdog.observers.Observer", lambda: observer)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    monkeypatch.setitem(sys.modules, "time", Clock)

    result = CliRunner().invoke(cli.main, ["dev", "--quick", "--api-key", "cnc_test_secret", "--project-id", "proj_123"])

    assert result.exit_code == 0, result.output
    assert observer.started and observer.stopped and observer.joined
    assert Client.deleted and Client.closed
    if output is not None:
        output.append(result.output)
    return uploads


def test_dev_starts_with_one_agent_and_no_optional_directories(dev_project, monkeypatch):
    uploads = run_dev(monkeypatch)

    assert len(uploads) == 1
    assert set(uploads[0]) == {"agents/support.yaml"}


@pytest.mark.parametrize(
    "relative_path",
    [
        "tools/nested/helper.py",
        "middleware/nested/helper.py",
        "schemas/nested/reply.json",
        "guardrails/nested/helper.py",
        "hooks/nested/helper.py",
        "tests/nested/support.yaml",
    ],
)
def test_dev_syncs_optional_directory_created_after_start_and_later_edits(dev_project, monkeypatch, relative_path):
    path = dev_project / relative_path

    def create_files(observer):
        path.parent.parent.mkdir()
        observer.emit(DirCreatedEvent(str(path.parent.parent)))
        path.parent.mkdir()
        observer.emit(DirCreatedEvent(str(path.parent)))
        path.write_text("{}\n")
        observer.emit(FileCreatedEvent(str(path)))

    def edit_file(observer):
        path.write_text('{"updated": true}\n')
        observer.emit(FileModifiedEvent(str(path)))

    uploads = run_dev(monkeypatch, [create_files, edit_file])

    assert len(uploads) == 3
    assert set(uploads[0]) == {"agents/support.yaml"}
    assert uploads[1][relative_path] == b"{}\n"
    assert uploads[2][relative_path] == b'{"updated": true}\n'


def test_dev_syncs_prepopulated_directory_moved_into_supported_path(dev_project, monkeypatch):
    incoming = dev_project / "incoming"
    (incoming / "nested").mkdir(parents=True)
    (incoming / "nested" / "helper.py").write_text("VERSION = 1\n")
    hooks = dev_project / "hooks"

    def move_directory(observer):
        incoming.rename(hooks)
        observer.emit(DirMovedEvent(str(incoming), str(hooks)))

    def edit_file(observer):
        path = hooks / "nested" / "helper.py"
        path.write_text("VERSION = 2\n")
        observer.emit(FileModifiedEvent(str(path)))

    uploads = run_dev(monkeypatch, [move_directory, edit_file])

    assert len(uploads) == 3
    assert set(uploads[0]) == {"agents/support.yaml"}
    assert uploads[1]["hooks/nested/helper.py"] == b"VERSION = 1\n"
    assert uploads[2]["hooks/nested/helper.py"] == b"VERSION = 2\n"


def test_dev_syncs_directory_deletion_recreation_and_later_edits(dev_project, monkeypatch):
    tools = dev_project / "tools"
    tools.mkdir()
    path = tools / "helper.py"
    path.write_text("VERSION = 1\n")

    def delete_directory(observer):
        shutil.rmtree(tools)
        observer.emit(DirDeletedEvent(str(tools)))

    def recreate_directory(observer):
        tools.mkdir()
        path.write_text("VERSION = 2\n")
        observer.emit(DirCreatedEvent(str(tools)))

    def edit_file(observer):
        path.write_text("VERSION = 3\n")
        observer.emit(FileModifiedEvent(str(path)))

    uploads = run_dev(monkeypatch, [delete_directory, recreate_directory, edit_file])

    assert len(uploads) == 4
    assert uploads[0]["tools/helper.py"] == b"VERSION = 1\n"
    assert "tools/helper.py" not in uploads[1]
    assert uploads[2]["tools/helper.py"] == b"VERSION = 2\n"
    assert uploads[3]["tools/helper.py"] == b"VERSION = 3\n"


@pytest.mark.parametrize("schedule_error", [FileNotFoundError, NotADirectoryError])
def test_dev_recovers_when_directory_disappears_while_registering_watch(dev_project, monkeypatch, schedule_error):
    tools = dev_project / "tools"
    path = tools / "helper.py"

    class RacingObserver(ScheduledObserver):
        failed_once = False

        def schedule(self, handler, path, recursive=False):
            if Path(path).name == "tools" and not self.failed_once:
                self.failed_once = True
                tools.rmdir()
                raise schedule_error("Directory changed while registering watch")
            return super().schedule(handler, path, recursive=recursive)

    def create_directory(observer):
        tools.mkdir()
        observer.emit(DirCreatedEvent(str(tools)))

    def recreate_directory(observer):
        tools.mkdir()
        path.write_text("VERSION = 1\n")
        observer.emit(DirCreatedEvent(str(tools)))

    def edit_file(observer):
        path.write_text("VERSION = 2\n")
        observer.emit(FileModifiedEvent(str(path)))

    uploads = run_dev(monkeypatch, [create_directory, recreate_directory, edit_file], observer=RacingObserver())

    assert len(uploads) == 4
    assert "tools/helper.py" not in uploads[1]
    assert uploads[2]["tools/helper.py"] == b"VERSION = 1\n"
    assert uploads[3]["tools/helper.py"] == b"VERSION = 2\n"


@pytest.mark.parametrize(
    ("event_class", "relative_path"),
    [
        (FileOpenedEvent, "agents/support.yaml"),
        (FileClosedEvent, "agents/support.yaml"),
        (DirModifiedEvent, "agents"),
        (FileModifiedEvent, "agents/.draft.yaml"),
        (FileModifiedEvent, "agents/__pycache__/helper.py"),
        (FileModifiedEvent, "agents/helper.pyc"),
        (FileModifiedEvent, "notes.txt"),
        (DirCreatedEvent, "unrelated"),
    ],
)
def test_dev_ignores_events_without_supported_content_changes(dev_project, monkeypatch, event_class, relative_path):
    def emit_event(observer):
        observer.emit(event_class(str(dev_project / relative_path)))

    uploads = run_dev(monkeypatch, [emit_event])

    assert len(uploads) == 1


def test_dev_syncs_atomic_file_move_from_hidden_source(dev_project, monkeypatch):
    tools = dev_project / "tools"
    tools.mkdir()

    def replace_file(observer):
        source = tools / ".helper.tmp.py"
        destination = tools / "helper.py"
        source.write_text("VERSION = 1\n")
        source.rename(destination)
        observer.emit(FileMovedEvent(str(source), str(destination)))

    uploads = run_dev(monkeypatch, [replace_file])

    assert len(uploads) == 2
    assert uploads[1]["tools/helper.py"] == b"VERSION = 1\n"


def test_dev_keeps_replacement_watch_when_directory_is_recreated_before_delete_event(dev_project, monkeypatch):
    tools = dev_project / "tools"
    tools.mkdir()
    path = tools / "helper.py"
    path.write_text("VERSION = 1\n")

    def replace_directory(observer):
        shutil.rmtree(tools)
        tools.mkdir()
        path.write_text("VERSION = 2\n")
        observer.emit(DirDeletedEvent(str(tools)))
        observer.emit(DirCreatedEvent(str(tools)))

    def edit_file(observer):
        path.write_text("VERSION = 3\n")
        observer.emit(FileModifiedEvent(str(path)))

    uploads = run_dev(monkeypatch, [replace_directory, edit_file])

    assert len(uploads) == 3
    assert uploads[1]["tools/helper.py"] == b"VERSION = 2\n"
    assert uploads[2]["tools/helper.py"] == b"VERSION = 3\n"


BUILD_SCRIPT = (
    'if grep -q broken src/main.js; then echo "Build error in src/main.js"; exit 1; fi\n'
    "mkdir -p dist && cp src/main.js dist/index.html\n"
)


def write_app(project, key="portal", source="v1\n"):
    app_dir = project / "apps" / key
    (app_dir / "src").mkdir(parents=True)
    (app_dir / "package.json").write_text('{"name": "portal"}\n')
    (app_dir / "package-lock.json").write_text("{}\n")
    (app_dir / "build.sh").write_text(BUILD_SCRIPT)
    (app_dir / "src" / "main.js").write_text(source)
    (project / "apps" / f"{key}.yaml").write_text("build:\n  command: sh build.sh\n  output: dist\n")
    return app_dir


@pytest.fixture
def app_project(dev_project, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    write_app(dev_project)
    return dev_project


def test_dev_uploads_app_files_without_sources_and_builds_apps_locally(app_project, monkeypatch):
    app_uploads = []
    output = []

    uploads = run_dev(monkeypatch, app_uploads=app_uploads, output=output)

    assert set(uploads[0]) == {"agents/support.yaml", "apps/portal.yaml"}
    assert app_uploads == [("portal", {"index.html": b"v1\n"})]
    assert "portal → https://portal-1a2b3c4d.connic.app" in output[0]


def test_dev_rebuilds_apps_only_when_their_sources_change(app_project, monkeypatch):
    app_dir = app_project / "apps" / "portal"
    source = app_dir / "src" / "main.js"

    def edit_source(observer):
        source.write_text("v2\n")
        observer.emit(FileModifiedEvent(str(source)))

    def touch_dependencies_and_output(observer):
        for path in (app_dir / "node_modules" / "react" / "index.js", app_dir / "dist" / "index.html"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("generated\n")
            observer.emit(FileModifiedEvent(str(path)))

    def save_without_changes(observer):
        source.write_text("v2\n")
        observer.emit(FileModifiedEvent(str(source)))

    app_uploads = []
    uploads = run_dev(
        monkeypatch, [edit_source, touch_dependencies_and_output, save_without_changes], app_uploads=app_uploads,
    )

    assert len(uploads) == 1
    assert app_uploads == [("portal", {"index.html": b"v1\n"}), ("portal", {"index.html": b"v2\n"})]


def test_dev_keeps_the_last_good_build_when_an_app_build_fails(app_project, monkeypatch):
    source = app_project / "apps" / "portal" / "src" / "main.js"

    def break_build(observer):
        source.write_text("broken\n")
        observer.emit(FileModifiedEvent(str(source)))

    def fix_build(observer):
        source.write_text("v3\n")
        observer.emit(FileModifiedEvent(str(source)))

    app_uploads = []
    output = []
    run_dev(monkeypatch, [break_build, fix_build], app_uploads=app_uploads, output=output)

    assert app_uploads == [("portal", {"index.html": b"v1\n"}), ("portal", {"index.html": b"v3\n"})]
    assert "portal: `sh build.sh` exited with code 1" in output[0]
    assert "Build error in src/main.js" in output[0]
    assert "The last good build stays live at https://portal-1a2b3c4d.connic.app" in output[0]


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (httpx.ReadTimeout("timed out"), "portal: Upload failed: timed out"),
        (httpx.ConnectError(""), "portal: Upload failed: ConnectError"),
        (503, "portal: Upload failed: Service unavailable"),
    ],
)
def test_dev_retries_a_failed_app_upload_on_the_next_save(app_project, monkeypatch, failure, message):
    source = app_project / "apps" / "portal" / "src" / "main.js"

    def save_without_changes(observer):
        source.write_text("v1\n")
        observer.emit(FileModifiedEvent(str(source)))

    app_uploads = []
    output = []
    uploads = run_dev(
        monkeypatch, [save_without_changes], app_uploads=app_uploads, output=output, app_failures=[failure],
    )

    assert len(uploads) == 1
    assert app_uploads == [("portal", {"index.html": b"v1\n"})]
    assert message in output[0]


def test_dev_keeps_the_session_when_packing_an_app_build_fails(app_project, monkeypatch):
    build_app = cli._build_app
    calls = []

    def flaky_build(key, build):
        calls.append(key)
        if len(calls) == 2:
            raise OSError("No space left on device")
        return build_app(key, build)

    monkeypatch.setattr(cli, "_build_app", flaky_build)
    source = app_project / "apps" / "portal" / "src" / "main.js"

    def edit_source(observer):
        source.write_text("v2\n")
        observer.emit(FileModifiedEvent(str(source)))

    def save_without_changes(observer):
        observer.emit(FileModifiedEvent(str(source)))

    app_uploads = []
    output = []
    run_dev(monkeypatch, [edit_source, save_without_changes], app_uploads=app_uploads, output=output)

    assert app_uploads == [("portal", {"index.html": b"v1\n"}), ("portal", {"index.html": b"v2\n"})]
    assert "portal: Could not package the build: No space left on device" in output[0]
    assert "The last good build stays live at https://portal-1a2b3c4d.connic.app" in output[0]


def test_dev_waits_for_a_source_change_after_a_failed_build(app_project, monkeypatch):
    source = app_project / "apps" / "portal" / "src" / "main.js"

    def break_build(observer):
        source.write_text("broken\n")
        observer.emit(FileModifiedEvent(str(source)))

    def save_without_changes(observer):
        observer.emit(FileModifiedEvent(str(source)))

    output = []
    run_dev(monkeypatch, [break_build, save_without_changes], app_uploads=[], output=output)

    assert output[0].count("`sh build.sh` exited with code 1") == 1


def test_dev_builds_apps_added_after_start_once_their_file_is_uploaded(app_project, monkeypatch):
    def add_app(observer):
        write_app(app_project, key="admin", source="admin\n")
        observer.emit(FileCreatedEvent(str(app_project / "apps" / "admin.yaml")))

    app_uploads = []
    uploads = run_dev(monkeypatch, [add_app], app_uploads=app_uploads)

    assert set(uploads[1]) == {"agents/support.yaml", "apps/portal.yaml", "apps/admin.yaml"}
    assert app_uploads == [("portal", {"index.html": b"v1\n"}), ("admin", {"index.html": b"admin\n"})]


def test_dev_reports_missing_node_once_and_keeps_syncing_agents(app_project, monkeypatch):
    write_app(app_project, key="admin")
    monkeypatch.setattr(cli.shutil, "which", lambda tool: None if tool == "node" else f"/usr/bin/{tool}")
    source = app_project / "apps" / "portal" / "src" / "main.js"

    def edit_source(observer):
        source.write_text("v2\n")
        observer.emit(FileModifiedEvent(str(source)))

    app_uploads = []
    output = []
    uploads = run_dev(monkeypatch, [edit_source], app_uploads=app_uploads, output=output)

    assert len(uploads) == 1
    assert app_uploads == []
    assert output[0].count("Node.js isn't installed or isn't on PATH; install Node.js 24 to build apps locally") == 1


def test_dev_stops_before_creating_a_session_when_an_app_file_is_invalid(app_project, monkeypatch):
    (app_project / "apps" / "portal.yaml").write_text(
        "build: {command: sh build.sh, output: dist}\naccess: {runs: {read: doc.id == user.id}}\n"
    )

    result = CliRunner().invoke(cli.main, ["dev", "--quick", "--api-key", "cnc_test_secret", "--project-id", "proj_123"])

    assert result.exit_code == 1
    assert "access.runs.read: 'doc' isn't available here; use run, user" in result.output


@pytest.mark.parametrize(
    ("command", "available", "missing"),
    [
        ("npm run build", {"node", "npm"}, None),
        ("pnpm build", {"node", "npm"}, "pnpm"),
        ("NODE_ENV=production npm run build", {"node", "npm"}, None),
        ("npm run build", {"npm"}, "node"),
    ],
)
def test_missing_build_tool_checks_node_and_the_command(monkeypatch, command, available, missing):
    monkeypatch.setattr(cli.shutil, "which", lambda tool: f"/usr/bin/{tool}" if tool in available else None)

    assert cli._missing_build_tool(command) == missing


@pytest.mark.parametrize(
    ("relative", "output", "watched"),
    [
        (("portal", "src", "App.tsx"), "dist", True),
        (("portal", "package.json"), "dist", True),
        (("portal", "dist"), "dist", False),
        (("portal", "dist", "index.html"), "dist", False),
        (("portal", "build"), "dist", False),
        (("portal", "public", "build", "app.js"), "public/build", False),
        (("portal", "public", "logo.png"), "public/build", True),
        (("portal", "node_modules", "react", "index.js"), "dist", False),
        (("portal", ".vite", "deps.js"), "dist", False),
    ],
)
def test_app_watch_path_ignores_dependencies_and_build_output(relative, output, watched):
    assert cli._app_watch_path(relative, output) is watched


def test_upload_app_build_posts_the_archive_and_reports_rejections():
    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.payload = payload
            self.text = str(payload)

        def json(self):
            return self.payload

    class Client:
        def __init__(self, response):
            self.response = response
            self.requests = []

        def post(self, path, files=None, timeout=None):
            self.requests.append((path, files))
            return self.response

    client = Client(Response(200, {"url": "https://portal-1a2b3c4d.connic.app"}))
    assert cli._upload_app_build(client, "sess_123", "portal", b"archive") == "https://portal-1a2b3c4d.connic.app"
    assert client.requests == [
        ("/test-sessions/sess_123/apps/portal", {"file": ("portal.tar.gz", b"archive", "application/gzip")}),
    ]

    client = Client(Response(413, {"detail": "App output exceeds 25 MB"}))
    with pytest.raises(ValueError, match="Upload failed: App output exceeds 25 MB"):
        cli._upload_app_build(client, "sess_123", "portal", b"archive")


def test_build_app_rejects_missing_output_and_symlinks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_app(tmp_path)
    build = cli.AppBuild(command="true", output="dist")

    with pytest.raises(ValueError, match="The build didn't create apps/portal/dist/"):
        cli._build_app("portal", build)

    (tmp_path / "apps" / "portal" / "dist").mkdir()
    (tmp_path / "apps" / "portal" / "dist" / "secret").symlink_to(tmp_path / "agents" / "support.yaml")
    with pytest.raises(ValueError, match="Symbolic links are not allowed: apps/portal/dist/secret"):
        cli._build_app("portal", build)
