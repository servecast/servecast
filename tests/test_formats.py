"""The output files: what a reader accepts, refuses and ignores."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from servecast.core import formats as f

SUMMARY = {
    "mean_ttft_ms": 182.0, "p95_ttft_ms": 410.0, "mean_tpot_ms": 21.5, "p95_tpot_ms": 27.0,
    "output_throughput_tok_s": 2900.0, "request_throughput_rps": 11.2,
    "num_ttft_samples": 4000, "num_tpot_samples": 3990, "duration_ms": 360000.0,
    "ci95": {"mean_ttft_ms": {"low": 175.0, "high": 189.0}},
}
PREDICTION = {
    "schema": "servecast-prediction/1",
    "points": [{"load": {"concurrency": 64, "think_time_ms": 1000}, "summary": SUMMARY}],
    "estimand": {"rule": "steady_state", "ttft": 0.10, "tpot": 0.05, "throughput": 0.05},
    "validity": {"status": "inside"},
    "provenance": {"components": {"cost law": "2026-11"}, "servecast_version": "0.0.1", "seed": 0},
    "resolved_manifest": {},
}


def test_a_prediction_file_is_read() -> None:
    prediction = f.Prediction.model_validate(PREDICTION)
    assert prediction.points[0].summary.p95_ttft_ms == 410.0


def test_an_unknown_major_is_refused() -> None:
    with pytest.raises(ValidationError, match="reads servecast-prediction/1, not servecast-prediction/2"):
        f.Prediction.model_validate({**PREDICTION, "schema": "servecast-prediction/2"})


def test_a_file_of_another_kind_is_refused() -> None:
    with pytest.raises(ValidationError, match="servecast-prediction/1"):
        f.Prediction.model_validate({**PREDICTION, "schema": "servecast-measurement/1"})


def test_unknown_fields_inside_a_known_major_are_ignored() -> None:
    newer = {**PREDICTION, "added_later": {"x": 1}, "points": [{**PREDICTION["points"][0], "extra": 3}]}
    assert f.Prediction.model_validate(newer).points[0].load.concurrency == 64


def test_a_missing_field_is_refused() -> None:
    with pytest.raises(ValidationError, match="validity"):
        f.Prediction.model_validate({k: v for k, v in PREDICTION.items() if k != "validity"})


@pytest.mark.parametrize("kind", sorted(f.SCHEMAS))
def test_every_format_publishes_a_json_schema(kind: str) -> None:
    schema = f.SCHEMAS[kind].model_json_schema(by_alias=True)
    assert schema["type"] == "object" and schema.get("properties")
    if kind != "manifest":
        assert "schema" in schema["properties"]


def test_the_formats_reference_is_generated_from_the_models() -> None:
    import subprocess
    import sys
    from pathlib import Path

    tool = Path(__file__).resolve().parents[1] / "tools" / "gen_formats_reference.py"
    proc = subprocess.run([sys.executable, str(tool), "--check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("interval", [{"low": float("nan"), "high": 2.0}, {"low": 3.0, "high": 2.0},
                                      {"low": 1.0, "high": float("inf")}])
def test_a_broken_interval_is_refused(interval: dict) -> None:
    point = {"load": {"concurrency": 64, "think_time_ms": 1000},
             "summary": {**SUMMARY, "ci95": {"mean_ttft_ms": interval}}}
    with pytest.raises(ValidationError, match="interval"):
        f.Prediction.model_validate({**PREDICTION, "points": [point]})


@pytest.mark.parametrize(("part", "field"), [("load", "think_time_ms"), ("summary", "duration_ms"),
                                             ("summary", "wall_time_ms")])
def test_a_load_or_a_duration_that_is_not_finite_is_refused(part: str, field: str) -> None:
    point = {"load": {"concurrency": 64, "think_time_ms": 1000}, "summary": dict(SUMMARY)}
    point[part] = {**point[part], field: float("inf")}
    with pytest.raises(ValidationError, match="finite"):
        f.Prediction.model_validate({**PREDICTION, "points": [point]})
