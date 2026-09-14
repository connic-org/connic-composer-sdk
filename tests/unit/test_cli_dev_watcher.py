import io
import shutil
import sys
import tarfile
from pathlib import Path

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


def run_dev(monkeypatch, changes=(), observer=None):
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
            assert path == "/test-sessions/sess_123/files"
            content = files["file"][1]
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
                uploads.append({member.name: archive.extractfile(member).read() for member in archive.getmembers()})
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

    result = CliRunner().invoke(cli.main, ["dev", "--api-key", "cnc_test_secret", "--project-id", "proj_123"])

    assert result.exit_code == 0, result.output
    assert observer.started and observer.stopped and observer.joined
    assert Client.deleted and Client.closed
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
