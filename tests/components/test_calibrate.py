"""The calibration procedures 2026-10 and 2026-11 against the shipped calibrations: every shipped calibration
refitted from the profiling cells it was fitted on.

These are reference-environment checks: there every refit is identical, and these tests also hold a relative 1e-6,
declared before the port. On other machines a least-squares solver's linear algebra rounds differently, and a
parameter at its bound moves far in relative terms: on the reference machine, a one-ulp change of every measured beat
time moves A100 Qwen2.5-3B's beta_req (7.9e-9, at its lower bound 0) by 4.9e-4 relative while the fitted beat times
move by 6e-10. The refits therefore run where the `exact` marker runs, and print each parameter's deviation when
they fail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from servecast.calibrate.procedure import v2026_10 as C10
from servecast.calibrate.procedure import v2026_11 as C11
from servecast.core import calibration as calibrations
from servecast.core import gpus
from servecast.model import geometry
from servecast.model import predict as P

PROFILING = P.shipped_profiling()["a100_sxm_40gb/c62ebe75"].parents[1]  # the package's data/profiling
#: the relative tolerance, declared before the port
TOLERANCE = 1e-6


PAIRS = sorted(PROFILING.glob("*/d9cbf1d3/*.json"))


def relative(got: list[float], want: list[float]) -> list[float]:
    assert len(got) == len(want)
    return [abs(a - b) / max(abs(b), 1e-12) for a, b in zip(got, want, strict=True)]


def check(got: dict, want: dict) -> None:
    parameters = want["law"]["parameters"]
    dev = relative(got["values"], list(parameters.values()))
    shown = ", ".join(f"{n} {d:.1e}" for n, d in zip(parameters, dev, strict=True))
    assert got["success"] and max(dev) <= TOLERANCE, f"relative deviation per parameter: {shown}"


@pytest.mark.exact
@pytest.mark.parametrize("path", PAIRS, ids=lambda p: f"{p.parts[-3]}-{p.stem}")
def test_the_papers_calibration_is_refitted(path: Path) -> None:
    prof = json.loads(path.read_text(encoding="utf-8"))
    got = C10.fit(prof, geometry.resolve(prof["llm"]), gpus.get(prof["gpu"]))
    want = P.load_calibration(f"{prof['gpu']}/d9cbf1d3/{path.stem}")
    check(got, want)
    assert got["peak_fraction"] == calibrations.peak_fraction(want)


@pytest.mark.exact
def test_the_newest_calibration_is_refitted_grid_included() -> None:
    prof = json.loads((PROFILING / "a100_sxm_40gb" / "c62ebe75.json").read_text(encoding="utf-8"))
    got = C11.fit(prof, gpus.get(prof["gpu"]))
    want = P.load_calibration("a100_sxm_40gb/c62ebe75")
    check(got, want)
    grid = want["law"]["eta_c_grid"]
    assert got["eta_c_grid"]["points"] == grid["points"] and got["eta_c_grid"]["rule"] == "interpolation"
    assert sorted(got["pins"]) == ["beta_req", "eta_c", "eta_w"]


def test_every_shipped_calibration_has_its_profiling_data() -> None:
    assert len(PAIRS) == 18 and (PROFILING / "a100_sxm_40gb" / "c62ebe75.json").is_file()


def test_a_probe_class_without_a_share_is_refused() -> None:
    with pytest.raises(ValueError, match="no weight share"):
        C10.reweight([{"cls": "warmup", "w": 1.0}], C10.shares())
