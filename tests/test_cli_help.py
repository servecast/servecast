"""The command line's help is the user-facing contract: snapshot it, hold it to the style, and scan it.

Every help page is compared with tests/snapshots/cli/<command>.txt. After a deliberate change, regenerate them with
`SERVECAST_UPDATE_SNAPSHOTS=1 pytest tests/test_cli_help.py` and review the diff like code.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest
import typer
import typer.rich_utils
from typer.testing import CliRunner

from servecast import __version__
from servecast.cli.app import app

# Typer reads the help width and the terminal mode from the environment once, at import (GITHUB_ACTIONS
# forces a terminal, and a forced terminal with TERM=dumb renders 80 columns). Pin both here, before any
# page is rendered, so the pages are the same on every machine.
typer.rich_utils.MAX_WIDTH = 100
typer.rich_utils.FORCE_TERMINAL = False

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "tests" / "snapshots" / "cli"
ENV = {"NO_COLOR": "1"}
UPDATE = os.environ.get("SERVECAST_UPDATE_SNAPSHOTS") == "1"

ROOT_COMMAND = typer.main.get_command(app)


def _is_group(command: object) -> bool:
    """Typer bundles its own Click, so a group is recognised by what it has, not by its class."""
    return bool(getattr(command, "commands", None))


def _paths(command, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    paths = [prefix]
    if _is_group(command):
        for name, sub in sorted(command.commands.items()):
            paths += _paths(sub, (*prefix, name))
    return paths


def _command(path: tuple[str, ...]):
    command = ROOT_COMMAND
    for name in path:
        assert _is_group(command)
        command = command.commands[name]
    return command


PATHS = _paths(ROOT_COMMAND)
LEAVES = [p for p in PATHS if not _is_group(_command(p))]


def _help(path: tuple[str, ...]) -> str:
    result = CliRunner().invoke(app, [*path, "--help"], env=ENV, prog_name="servecast")
    assert result.exit_code == 0, result.output
    return result.output


def _name(path: tuple[str, ...]) -> str:
    return "_".join(path) or "servecast"


@pytest.mark.parametrize("path", PATHS, ids=_name)
def test_help_pages_match_their_snapshots(path: tuple[str, ...]) -> None:
    snapshot = SNAPSHOTS / f"{_name(path)}.txt"
    text = _help(path)
    if UPDATE:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(text, encoding="utf-8")
    assert snapshot.is_file(), f"no snapshot for {path}; run with SERVECAST_UPDATE_SNAPSHOTS=1 and review it"
    assert text == snapshot.read_text(encoding="utf-8")


def test_no_snapshot_is_left_without_a_command() -> None:
    expected = {f"{_name(p)}.txt" for p in PATHS}
    assert {p.name for p in SNAPSHOTS.glob("*.txt")} == expected


@pytest.mark.parametrize("path", LEAVES, ids=_name)
def test_every_command_follows_the_help_style(path: tuple[str, ...]) -> None:
    command = _command(path)
    summary = (command.help or "").strip().splitlines()[0]
    assert summary.endswith(".") and len(summary) <= 72, summary
    epilog = command.epilog or ""
    assert "**Examples**" in epilog and "See also:" in epilog
    for parameter in command.params:
        if parameter.param_type_name == "option" and parameter.name != "help":
            assert parameter.help, f"{path}: option {parameter.opts} has no help"


@pytest.mark.parametrize("path", LEAVES, ids=_name)
def test_every_example_names_a_command_that_exists(path: tuple[str, ...]) -> None:
    for line in (_command(path).epilog or "").splitlines():
        if not line.startswith("servecast "):
            continue
        words = re.findall(r"[a-z][a-z-]*", line.split(" -", 1)[0].removeprefix("servecast "))
        command = ROOT_COMMAND
        for word in words:
            if not _is_group(command) or word not in command.commands:
                break
            command = command.commands[word]
        assert not _is_group(command), f"{path}: example does not reach a command: {line}"


def test_the_version_is_printed() -> None:
    result = CliRunner().invoke(app, ["--version"], env=ENV)
    assert result.exit_code == 0 and result.output.strip() == f"servecast {__version__}"


def test_no_help_page_carries_anything_the_privacy_scanner_refuses() -> None:
    spec = importlib.util.spec_from_file_location("privacy_scan", ROOT / "tools" / "privacy_scan.py")
    assert spec is not None and spec.loader is not None
    scanner = importlib.util.module_from_spec(spec)
    sys.modules["privacy_scan"] = scanner
    spec.loader.exec_module(scanner)
    findings = [f for path in PATHS for f in scanner.scan_text(_help(path), _name(path))]
    assert findings == []


def test_the_cli_reference_is_generated_from_the_help() -> None:
    import subprocess

    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_cli_reference.py"), "--check"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def _sources(command: object) -> list[str]:
    """Every text a help page is rendered from: the command's help and epilog, and each parameter's help."""
    texts = [getattr(command, "help", None) or "", getattr(command, "epilog", None) or ""]
    texts += [getattr(p, "help", None) or "" for p in getattr(command, "params", [])]
    return texts


@pytest.mark.parametrize("path", PATHS, ids=_name)
def test_no_help_text_holds_a_placeholder_the_renderer_drops(path: tuple[str, ...]) -> None:
    """The help's markdown renderer reads <word> as an HTML tag and drops it: placeholders are written in capitals."""
    for text in _sources(_command(path)):
        assert not re.search(r"<[A-Za-z_][\w-]*>", text), text


def test_an_unexpected_failure_is_one_line_and_debug_shows_the_traceback(monkeypatch: pytest.MonkeyPatch,
                                                                         capsys: pytest.CaptureFixture[str]) -> None:
    from servecast.cli import app as cli
    from servecast.cli import doctor_cmd

    def broken(*_args: object) -> None:
        raise RuntimeError("an internal fault")

    monkeypatch.setattr(doctor_cmd, "main", broken)
    monkeypatch.setattr(sys, "argv", ["servecast", "doctor"])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    err = capsys.readouterr().err
    assert exited.value.code == 1 and "an internal fault" in err and "--debug" in err and "Traceback" not in err
    monkeypatch.setattr(sys, "argv", ["servecast", "--debug", "doctor"])
    with pytest.raises(RuntimeError, match="an internal fault"):
        cli.main()
