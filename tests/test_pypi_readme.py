"""The PyPI page: built from README.md and CHANGELOG.md by hatch-fancy-pypi-readme (pyproject.toml), and read here
from the installed package's metadata, which is what a wheel uploads. An editable install keeps the page it was
installed with, so after editing README.md, CHANGELOG.md or that table, run `pip install -e .` again."""

from __future__ import annotations

import importlib.metadata as md
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STALE = "the installed page differs from README.md: run `pip install -e .` again"


def page() -> str:
    return str(md.metadata("servecast").json["description"])


def rewritten(text: str) -> str:
    """README text as the PyPI page shows it: with pyproject.toml's substitutions applied."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for sub in config["tool"]["hatch"]["metadata"]["hooks"]["fancy-pypi-readme"]["substitutions"]:
        text = re.sub(sub["pattern"], sub["replacement"], text)
    return text.replace("$HFPR_VERSION", md.version("servecast"))


def section(text: str, heading: str) -> str:
    """A `## ` section of a Markdown file, from its heading to the next one."""
    start = text.index(f"\n{heading}\n") + 1
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else len(text)]


def test_every_link_and_image_on_the_page_is_absolute() -> None:
    """A relative link resolves against pypi.org there: the README's logo and its 11 links into the docs broke."""
    targets = re.findall(r"\]\(([^)\s]+)", page()) + re.findall(r'(?:href|src)="([^"]+)"', page())
    assert targets
    assert [t for t in targets if not t.startswith(("https://", "#"))] == []


@pytest.mark.parametrize("heading", ["## Quick start", "## What it does", "## Documentation", "## Citation",
                                     "## License"])
def test_the_page_shows_the_readmes_sections_word_for_word(heading: str) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert rewritten(section(readme, heading)).strip() in page(), STALE


def test_the_page_links_the_repository_and_the_guide_instead_of_the_long_gpu_flow() -> None:
    assert '<a href="https://github.com/servecast/servecast">GitHub</a>' in page()
    assert "https://servecast.github.io/servecast/guides/calibrate/" in page()
    assert "servecast profile calibration -c example-a100.yaml" not in page()


def test_the_page_ends_with_the_newest_changelog_section() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    newest = changelog.split("\n## ")[1]
    heading, body = newest.split("\n", 1)
    assert re.fullmatch(r"\d+\.\d+\.\d+ \((\d{4}-\d{2}-\d{2}|unreleased)\)", heading), heading
    assert page().rstrip().endswith(f"## What's new in {md.version('servecast')}\n\n{body.strip()}"), STALE
