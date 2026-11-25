"""`servecast completion install`: the completion script where the shell looks for it, and the one line its start-up
file needs. Running it again changes nothing; every change it makes is printed.

bash (4.4 or newer): ~/.bash_completions/servecast.sh, sourced from ~/.bashrc (a macOS login shell reads
~/.bash_profile, which should source ~/.bashrc). zsh: ~/.zfunc/_servecast, with ~/.zfunc on fpath and compinit in
$ZDOTDIR/.zshrc (~/.zshrc without ZDOTDIR). fish: $XDG_CONFIG_HOME/fish/completions/servecast.fish (~/.config
without XDG_CONFIG_HOME), which fish loads.

Exit statuses: 0 installed or already there; 2 the shell cannot be told; 3 a file cannot be written.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NoReturn

import typer

SHELLS = ("bash", "zsh", "fish")
PROG, VAR = "servecast", "_SERVECAST_COMPLETE"


def fail(status: int, message: str) -> NoReturn:
    typer.echo(f"servecast completion: {message}", err=True)
    raise typer.Exit(status)


def script(shell: str) -> str:
    # Typer bundles its own Click; its completion helper is the one that matches it (pinned in constraints/ci.txt).
    from typer._completion_shared import get_completion_script

    return get_completion_script(prog_name=PROG, complete_var=VAR, shell=shell)


def _running_shell() -> str | None:
    """The shell this process runs under, when the process tree tells."""
    try:
        import shellingham

        return str(shellingham.detect_shell()[0])
    except (ImportError, OSError, RuntimeError):
        return None


def detect() -> str | None:
    """The running shell, else the login shell ($SHELL); None when neither is bash, zsh or fish."""
    for name in (_running_shell(), Path(os.environ.get("SHELL", "")).name):
        if name in SHELLS:
            return name
    return None


def _write(path: Path, text: str) -> str:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return f"unchanged {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return f"wrote {path}"


def _append_once(path: Path, line: str) -> str:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if line in text.splitlines():
        return f"unchanged {path} (it already has: {line})"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(("" if not text or text.endswith("\n") else "\n") + f"# servecast completion\n{line}\n")
    return f"added to {path}: {line}"


def install(shell: str, home: Path) -> list[str]:
    """The changes made, one line each."""
    if shell == "bash":
        target = home / ".bash_completions" / f"{PROG}.sh"
        return [_write(target, script(shell)), _append_once(home / ".bashrc", f"source '{target}'")]
    if shell == "zsh":
        rc = Path(os.environ.get("ZDOTDIR") or home) / ".zshrc"
        return [_write(home / ".zfunc" / f"_{PROG}", script(shell)),
                _append_once(rc, "fpath+=~/.zfunc; autoload -Uz compinit; compinit")]
    config = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    return [_write(config / "fish" / "completions" / f"{PROG}.fish", script(shell) + "\n")]


def main(shell: str | None) -> None:
    name = shell or detect()
    if name is None:
        fail(2, "cannot tell which shell this is; name it: servecast completion install bash, zsh or fish")
    try:
        changes = install(name, Path.home())
    except OSError as error:
        fail(3, f"cannot write {error.filename}: {error.strerror}")
    for change in changes:
        typer.echo(change)
    typer.echo(f"open a new {name} session to use it")
