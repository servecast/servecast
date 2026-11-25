"""Every shipped calibration keeps its numbers: its beat durations for each of the six known LLMs, on the 120
compositions of law.json, equal the reference taken from ServeCast before the change (tools/shipped_beats.py). It
replaces the byte lock of the shipped data: a renamed or reformatted calibration passes, a changed number fails."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from servecast.model import geometry
from servecast.model import predict as P
from servecast.model.population.v2026_10 import Beat
from tests.components.inputs import load

pytestmark = pytest.mark.exact

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((Path(__file__).parent / "goldens" / "shipped-beats.json").read_text(encoding="utf-8"))
BEATS = [Beat(prefill=tuple(map(tuple, c["prefill"])), decode=tuple(map(tuple, c["decode"])))
         for c in load("law")["compositions"]]
_spec = importlib.util.spec_from_file_location("shipped_beats", ROOT / "tools" / "shipped_beats.py")
assert _spec is not None and _spec.loader is not None
TOOL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TOOL)


def _key(name: str) -> str:
    return str(TOOL.key(json.loads(P.shipped_calibrations()[name].read_text(encoding="utf-8"))))


def test_the_reference_covers_every_shipped_calibration_and_was_made_from_a_clean_tree() -> None:
    assert {_key(name) for name in P.shipped_calibrations()} == set(REFERENCE["cases"])
    assert REFERENCE["made_by"]["dirty"] is False


@pytest.mark.parametrize("name", sorted(P.shipped_calibrations()))
def test_every_shipped_calibration_keeps_its_beats(name: str) -> None:
    calibration, want = P.load_calibration(name), REFERENCE["cases"][_key(name)]
    for llm, g in geometry.KNOWN.items():
        clock = P.clock(calibration, g)
        assert [clock(b) for b in BEATS] == want[llm], f"{name} on {llm}"
