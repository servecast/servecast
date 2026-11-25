"""Generate docs/reference/cli.md from the command line's own help pages, so the two cannot drift.

    python tools/gen_cli_reference.py            # write the file
    python tools/gen_cli_reference.py --check    # exit 1 if the file is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import typer
import typer.rich_utils
from typer.testing import CliRunner

from servecast.cli.app import app

# Typer reads the help width and the terminal mode from the environment once, at import (GITHUB_ACTIONS
# forces a terminal, and a forced terminal with TERM=dumb renders 80 columns). Pin both here, before any
# page is rendered, so the pages are the same on every machine.
typer.rich_utils.MAX_WIDTH = 100
typer.rich_utils.FORCE_TERMINAL = False

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "reference" / "cli.md"
ENV = {"NO_COLOR": "1"}


def _paths(command, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    paths = [prefix]
    for name, sub in sorted((getattr(command, "commands", None) or {}).items()):
        paths += _paths(sub, (*prefix, name))
    return paths


def render() -> str:
    parts = [
        "# Command-line reference",
        "",
        "Generated from `servecast --help` by `python tools/gen_cli_reference.py`; do not edit by hand.",
    ]
    for path in _paths(typer.main.get_command(app)):
        result = CliRunner().invoke(app, [*path, "--help"], env=ENV, prog_name="servecast")
        if result.exit_code != 0:
            raise RuntimeError(f"help failed for {path}: {result.output}")
        page = "\n".join(line.rstrip() for line in result.output.splitlines()).strip("\n")
        parts += ["", f"## `{' '.join(('servecast', *path))}`", "", "```text", page, "```"]
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the CLI reference from the help pages.")
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed file is out of date")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.is_file() else ""
        if current != text:
            print(f"{TARGET.relative_to(ROOT)} is out of date: run python tools/gen_cli_reference.py", file=sys.stderr)
            return 1
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
