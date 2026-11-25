"""Every output directory explains itself: run.json beside the outputs, valid against its documented format."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from servecast.cli.app import app
from servecast.core import formats, runrecord

RUN = CliRunner()


def test_predict_writes_its_manifest_and_a_run_record(tmp_path: Path) -> None:
    cfg = tmp_path / "m.yaml"
    assert RUN.invoke(app, ["manifest", "init", "--out", str(cfg)]).exit_code == 0
    out = tmp_path / "out"
    r = RUN.invoke(app, ["predict", "-c", str(cfg), "--output-dir", str(out)])
    assert r.exit_code == 0, r.output
    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    formats.RunRecord.model_validate(record)
    assert record["inputs"]["manifest"]["sha256"] == runrecord.sha256_of(cfg)
    assert record["inputs"]["calibration"]["sha256"] == json.loads(
        (out / "prediction.json").read_text(encoding="utf-8"))["provenance"]["calibration_sha256"]
    assert set(record["outputs"]) == {"prediction.json", "pool.jsonl", "manifest.yaml"}
    assert all((out / name).is_file() for name in record["outputs"])
    # the manifest as it ran is a manifest: it predicts the same numbers again
    again = tmp_path / "again"
    r = RUN.invoke(app, ["predict", "-c", str(out / "manifest.yaml"), "--output-dir", str(again)])
    assert r.exit_code == 0, r.output
    points = [json.loads((d / "prediction.json").read_text(encoding="utf-8"))["points"] for d in (out, again)]
    for run in points:  # the one field that is not a prediction: how long the simulation took on this machine
        for point in run:
            point["summary"].pop("wall_time_ms")
    assert points[0] == points[1]
    assert yaml.safe_load((out / "manifest.yaml").read_text(encoding="utf-8"))["engine"]["model"]


def test_calibrate_writes_the_calibration_and_a_run_record(tmp_path: Path) -> None:
    out = tmp_path / "cal"
    r = RUN.invoke(app, ["calibrate", "--data", "a100_sxm_40gb/d9cbf1d3/llama-3-8b", "--versions", "2026-10",
                         "--name", "mine/llama", "--output-dir", str(out)])
    assert r.exit_code == 0, r.output
    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    formats.RunRecord.model_validate(record)
    calibration = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    assert record["outputs"] == ["calibration.json"] and calibration["name"] == "mine/llama"
    assert record["inputs"]["profiling data"]["sha256"] == calibration["fit"]["profiling_sha256"]
