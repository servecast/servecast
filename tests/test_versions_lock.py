"""The released component modules match versions.lock (the review's DC-01 and Q1: comments promised a lock that did
not exist, so an edit to a released module that left the reference outputs unchanged passed CI)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tool():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("versions_lock", ROOT / "tools" / "versions_lock.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_released_module_matches_its_lock() -> None:
    lock = tool()
    assert lock.released(), "no released module found"
    assert lock.problems() == []


def test_an_edit_to_a_released_module_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tool()
    now = lock.released()
    path = next(iter(now))
    monkeypatch.setattr(lock, "released", lambda: {**now, path: "0" * 64})
    assert lock.problems() == [f"{path}: changed (sha256 000000000000, locked {lock.locked()[path][:12]})"]


def test_a_released_module_left_out_of_the_lock_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tool()
    now = lock.released()
    monkeypatch.setattr(lock, "released", lambda: {**now, "src/servecast/model/law/v2099_01.py": "f" * 64})
    assert lock.problems() == ["src/servecast/model/law/v2099_01.py: released but not in versions.lock"]
