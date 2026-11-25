"""AGENTS.md shows commands that an agent will run as written, so each one must exist; and Claude Code, which reads
CLAUDE.md first, must be sent on to AGENTS.md."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servecast.cli.app import app

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "AGENTS.md").read_text(encoding="utf-8")


def words(command: str) -> str:
    """The command words of a command AGENTS.md shows, up to its first option, placeholder or redirection."""
    out = []
    for token in command.split():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", token):
            break
        out.append(token)
    return " ".join(out)


COMMANDS = sorted({words(c) for c in re.findall(r"`servecast ([^`]*)`", TEXT)})


def test_agents_md_shows_the_commands_it_is_for() -> None:
    assert {"predict", "manifest show example-a100", "manifest validate", "replicate verify"} <= set(COMMANDS)


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_agents_md_shows_exists(command: str) -> None:
    r = CliRunner().invoke(app, [*command.split(), "--help"])
    assert r.exit_code == 0, (command, r.output)


def test_claude_md_imports_agents_md() -> None:
    assert (ROOT / "CLAUDE.md").read_text(encoding="utf-8") == "@AGENTS.md\n"
