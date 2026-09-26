import click
import pytest
import questionary
from click.testing import CliRunner
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from connic import cli


@pytest.fixture
def terminal_select(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    select = questionary.select

    def run(keys, *, choices=None, default=0):
        with create_pipe_input() as pipe:
            if keys is None:
                pipe.close()
            else:
                pipe.send_text(keys)

            def select_with_input(*args, **kwargs):
                return select(*args, **kwargs, input=pipe, output=DummyOutput())

            monkeypatch.setattr(questionary, "select", select_with_input)
            return cli._select_option("Environment", choices or ["Reusable", "Quick", "Another"], default=default)

    return run


@pytest.mark.parametrize(("keys", "default", "expected"), [
    ("\r", 0, 0),
    ("\r", 2, 2),
    ("\x1b[B\r", 0, 1),
    ("\x1b[A\r", 2, 1),
    ("\x1b[B\x1b[B\x1b[A\r", 0, 1),
])
def test_terminal_selection_uses_arrows_and_enter(terminal_select, keys, default, expected):
    assert terminal_select(keys, default=default) == expected


def test_terminal_selection_returns_index_for_duplicate_labels(terminal_select):
    assert terminal_select("\x1b[B\r", choices=["Staging", "Staging"]) == 1


@pytest.mark.parametrize("key", ["\x03", None], ids=["interrupt", "end-of-input"])
def test_terminal_selection_cancels_cleanly(terminal_select, key):
    with pytest.raises(click.Abort):
        terminal_select(key)


@pytest.mark.parametrize(("stdin_tty", "stdout_tty"), [(False, True), (True, False), (False, False)])
def test_selection_uses_numbered_fallback_when_a_stream_is_not_a_terminal(monkeypatch, capsys, stdin_tty, stdout_tty):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: stdin_tty)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: stdout_tty)
    monkeypatch.setattr(questionary, "select", lambda *args, **kwargs: pytest.fail("Must not open a terminal prompt"))
    prompts = []

    def prompt(message, **kwargs):
        prompts.append((message, kwargs))
        return 2

    monkeypatch.setattr(cli.click, "prompt", prompt)

    assert cli._select_option("Environment", ["Reusable", "Quick"], default=1) == 1
    assert "1. Reusable" in capsys.readouterr().out
    assert prompts[0][1]["default"] == 2


def test_dumb_terminal_uses_numbered_fallback(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setattr(questionary, "select", lambda *args, **kwargs: pytest.fail("Must not open a terminal prompt"))
    monkeypatch.setattr(cli.click, "prompt", lambda *args, **kwargs: 1)

    assert cli._select_option("Environment", ["Reusable", "Quick"]) == 0


@pytest.mark.parametrize(("answer", "expected"), [("\n", 1), ("1\n", 0), ("invalid\n3\n2\n", 1)])
def test_numbered_fallback_accepts_default_and_validates_input(monkeypatch, answer, expected):
    monkeypatch.setattr(questionary, "select", lambda *args, **kwargs: pytest.fail("Must not open a terminal prompt"))

    @click.command()
    def command():
        click.echo(f"Selected: {cli._select_option('Environment', ['Reusable', 'Quick'], default=1)}")

    result = CliRunner().invoke(command, input=answer)

    assert result.exit_code == 0, result.output
    assert f"Selected: {expected}" in result.output


def test_numbered_fallback_aborts_on_end_of_input():
    @click.command()
    def command():
        cli._select_option("Environment", ["Reusable", "Quick"])

    result = CliRunner().invoke(command, input="")

    assert result.exit_code == 1
    assert "Aborted!" in result.output
    assert "Traceback" not in result.output
