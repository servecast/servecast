"""The workload-law cache: a law read back is the one that was written, and what the fitter fell back to is said."""

from __future__ import annotations

import json
from typing import Any

import pytest

from servecast.core import pool as pools
from servecast.core.paths import cache_dir
from servecast.model import run as R
from servecast.model import workload as workloads

REPORT = {"prompt_classes": 1, "coupling_groups": 1, "branches": 1, "stages": [1], "phase_cap": 12,
          "footprint_tol": 0.01, "reanchored": False}


class _Law:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    def to_dict(self) -> dict[str, Any]:
        return self.body


@pytest.fixture
def fits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The fitter replaced by a stub that records each call: the cache's behaviour, not the fit's, is under test."""
    calls: list[int] = []

    def fit_law(p: list[int], g: list[int], *, block_size: int, phase_cap: int) -> tuple[_Law, dict[str, Any]]:
        calls.append(len(p))
        return _Law({"prompt_lengths": list(p)}), dict(REPORT)

    monkeypatch.setattr(workloads.get("2026-10"), "fit_law", fit_law)
    return calls


def _cached(p: list[int], g: list[int]) -> Any:
    return cache_dir() / "workload_laws" / "cd88bc61" / "cap12-block16" / f"{pools.digest(p, g)}.json"


def law(p: list[int], g: list[int]) -> dict[str, Any]:
    return R.workload_law(p, g, "2026-10", 12, block_size=16)


def test_a_cached_law_is_read_back_without_fitting_again(fits: list[int]) -> None:
    p, g = [101, 203, 305], [11, 21, 31]
    first = law(p, g)
    assert law(p, g) == first and fits == [3]


def test_an_altered_cached_law_is_fitted_again(fits: list[int]) -> None:
    """The audit multiplied the prompt lengths of a cached law by 3: the same manifest then predicted a TTFT of 681.2 ms
    instead of 174.9 ms, with no warning."""
    p, g = [102, 204, 306], [12, 22, 32]
    good = law(p, g)
    path = _cached(p, g)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["law"]["prompt_lengths"] = [3 * x for x in doc["law"]["prompt_lengths"]]
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert law(p, g) == good and fits == [3, 3]
    assert json.loads(path.read_text(encoding="utf-8")) == good


@pytest.mark.parametrize("body", ["{not json", json.dumps({"law": {"prompt_lengths": [1]}})])
def test_a_cache_file_without_a_valid_checksum_is_fitted_again(fits: list[int], body: str) -> None:
    p, g = [103, 205, 307], [13, 23, 33]
    path = _cached(p, g)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    assert law(p, g)["law"] == {"prompt_lengths": p} and fits == [3]


@pytest.mark.parametrize(("fit", "says"), [
    ({"footprint_tol": 0.01, "reanchored": False}, None),
    ({"footprint_tol": 0.025, "reanchored": False}, "only within 2.5 %"),
    ({"footprint_tol": 0.025, "reanchored": True}, "re-anchored"),
    ({}, None),  # a shipped trace's law records no fallback
])
def test_the_fitters_fallbacks_are_said(fit: dict[str, Any], says: str | None) -> None:
    warnings = R.fit_warnings({"fit": fit})
    assert (warnings == []) if says is None else (len(warnings) == 1 and says in warnings[0])


# the fit is bounded by --timeout ---------------------------------------------------------------------------------

def _pool(n: int) -> tuple[list[int], list[int]]:
    return [97 + (37 * i) % 1900 for i in range(n)], [3 + (13 * i) % 400 for i in range(n)]


def test_a_fit_that_runs_out_of_time_is_stopped_and_nothing_is_cached() -> None:
    """--timeout bounded the load points but not the fit before them: the guide's own synthetic example fitted for
    about 13 minutes under --timeout 60."""
    import time

    p, g = _pool(3000)
    started = time.perf_counter()
    with pytest.raises(R.FitTimeoutError, match="did not finish in 2 s"):
        R.workload_law(p, g, "2026-10", 12, block_size=16, seconds=2)
    assert time.perf_counter() - started < 20
    assert not _cached(p, g).exists()


def test_a_fit_in_time_gives_the_law_the_unbounded_fit_gives() -> None:
    """The same law from the child process as in this one (a small pool and phase cap keep the two fits short)."""
    p, g = _pool(24)
    bounded = R.workload_law(p, g, "2026-10", 2, block_size=16, seconds=600)
    (cache_dir() / "workload_laws" / "cd88bc61" / "cap2-block16" / f"{pools.digest(p, g)}.json").unlink()
    assert R.workload_law(p, g, "2026-10", 2, block_size=16) == bounded
