"""The package's seven layers exist, import, and say what they are."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

import servecast

ROOT = Path(__file__).resolve().parents[1]
LAYERS = ("core", "model", "calibrate", "profile", "compare", "paper", "cli")


@pytest.mark.parametrize("layer", LAYERS)
def test_every_layer_imports_and_documents_itself(layer: str) -> None:
    module = importlib.import_module(f"servecast.{layer}")
    assert module.__doc__ and "May import" in module.__doc__


def test_the_version_matches_the_project_metadata() -> None:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert servecast.__version__ == meta["project"]["version"]
