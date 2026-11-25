"""Shell completion: the scripts print, and a simulated TAB press completes commands, values and options."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servecast.cli.app import app

SERVECAST = Path(sys.executable).with_name("servecast")


def complete(env: dict[str, str]) -> str:
    """What the shell receives when it asks `servecast` to complete, as the printed script does."""
    return subprocess.run([str(SERVECAST)], env={**os.environ, **env}, capture_output=True, text=True).stdout


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_the_script_prints_for_each_shell(shell: str) -> None:
    result = CliRunner().invoke(app, ["completion", "show", shell])
    assert result.exit_code == 0 and "_SERVECAST_COMPLETE" in result.output


def test_a_command_completes() -> None:
    out = complete({"_SERVECAST_COMPLETE": "complete_zsh", "_TYPER_COMPLETE_ARGS": "servecast li"})
    assert '"list"' in out


def test_an_argument_completes_to_its_values() -> None:
    out = complete({"_SERVECAST_COMPLETE": "complete_zsh", "_TYPER_COMPLETE_ARGS": "servecast replicate run "})
    assert all(f'"{name}"' in out for name in ("rq1", "rq2", "rq3", "ablation", "appendix"))


def test_an_option_completes() -> None:
    out = complete({"_SERVECAST_COMPLETE": "complete_bash", "COMP_WORDS": "servecast predict --f", "COMP_CWORD": "2"})
    assert "--format" in out.split()


#: the environment of an install: a scratch home, and none of the caller's own locations for start-up files
def sandbox(home: Path, **extra: str | None) -> dict[str, str | None]:
    return {"HOME": str(home), "ZDOTDIR": None, "XDG_CONFIG_HOME": None, **extra}


INSTALLED = {"bash": ".bash_completions/servecast.sh", "zsh": ".zfunc/_servecast",
             "fish": ".config/fish/completions/servecast.fish"}
RC = {"bash": ".bashrc", "zsh": ".zshrc"}


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_install_writes_the_script_and_one_line_and_says_so(tmp_path: Path, shell: str) -> None:
    rc = tmp_path / RC[shell] if shell in RC else None
    if rc is not None:
        rc.write_text("export KEEP=1\n", encoding="utf-8")
    r = CliRunner().invoke(app, ["completion", "install", shell], env=sandbox(tmp_path))
    assert r.exit_code == 0, r.output
    script = tmp_path / INSTALLED[shell]
    assert "_SERVECAST_COMPLETE" in script.read_text(encoding="utf-8") and str(script) in r.stdout
    if rc is not None:
        text = rc.read_text(encoding="utf-8")
        assert text.startswith("export KEEP=1\n") and str(rc) in r.stdout
        before = text
        again = CliRunner().invoke(app, ["completion", "install", shell], env=sandbox(tmp_path))
        assert again.exit_code == 0 and rc.read_text(encoding="utf-8") == before
        assert "unchanged" in again.stdout


def test_install_refuses_a_shell_it_cannot_tell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from servecast.cli import completion_cmd

    monkeypatch.setattr(completion_cmd, "_running_shell", lambda: None)
    r = CliRunner().invoke(app, ["completion", "install"], env=sandbox(tmp_path, SHELL="/bin/tcsh"))
    assert r.exit_code == 2 and "bash, zsh or fish" in r.stderr
    assert not any(tmp_path.iterdir())


def test_install_reads_the_login_shell_when_detection_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from servecast.cli import completion_cmd

    monkeypatch.setattr(completion_cmd, "_running_shell", lambda: None)
    r = CliRunner().invoke(app, ["completion", "install"], env=sandbox(tmp_path, SHELL="/usr/bin/fish"))
    assert r.exit_code == 0 and (tmp_path / INSTALLED["fish"]).is_file()


def test_install_honours_zdotdir_and_names_an_unwritable_file(tmp_path: Path) -> None:
    zdot = tmp_path / "zdot"
    zdot.mkdir()
    r = CliRunner().invoke(app, ["completion", "install", "zsh"], env=sandbox(tmp_path, ZDOTDIR=str(zdot)))
    assert r.exit_code == 0 and "compinit" in (zdot / ".zshrc").read_text(encoding="utf-8")
    assert not (tmp_path / ".zshrc").exists()
    other = tmp_path / "other"
    (other / ".bashrc").mkdir(parents=True)
    r = CliRunner().invoke(app, ["completion", "install", "bash"], env=sandbox(other))
    assert r.exit_code == 3 and ".bashrc" in r.stderr and "Traceback" not in r.output


def test_fish_honours_xdg_config_home(tmp_path: Path) -> None:
    xdg = tmp_path / "xdg"
    r = CliRunner().invoke(app, ["completion", "install", "fish"], env=sandbox(tmp_path, XDG_CONFIG_HOME=str(xdg)))
    assert r.exit_code == 0 and (xdg / "fish" / "completions" / "servecast.fish").is_file()
    assert not (tmp_path / ".config").exists()
