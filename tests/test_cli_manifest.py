"""`servecast manifest validate | schema | explain`, end to end through the command line."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servecast.cli.app import app
from tests.test_manifest import ANOTHER_TOOLS_QUICKSTART, PREDICT_WITH_A_TRACE

RUN = CliRunner()


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "m.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def template(tmp_path: Path) -> Path:
    """The manifest `manifest init` writes: its calibration and its lengths ship, so predict can run it."""
    path = tmp_path / "m.yaml"
    assert RUN.invoke(app, ["manifest", "init", "--out", str(path)]).exit_code == 0
    return path


def test_a_valid_manifest_says_so(tmp_path: Path) -> None:
    result = RUN.invoke(app, ["manifest", "validate", "-c", str(template(tmp_path))])
    assert result.exit_code == 0 and "valid" in result.output, result.output


@pytest.mark.parametrize(("setting", "message"), [
    # the review's F10, MF-04 and DC-04: validate called each of these "valid", and predict refused it
    ("calibration.use=nosuch", "calibration.use: no calibration 'nosuch'"),
    ("traffic.source={type: trace, paths: [no-such-trace.jsonl], format: mooncake}", "no-such-trace.jsonl"),
    ("engine.hardware=h100_sxm", "engine.hardware: h100_sxm, but the calibration"),
    ("versions={as_of: 2026-11, calibration_procedure: d9cbf1d3}", "calibration procedure"),
    ("engine.workers.aggregated.kv_cache.capacity.blocks=2", "fewer tokens than one request"),
])
def test_validate_refuses_what_predict_refuses(tmp_path: Path, setting: str, message: str) -> None:
    path = template(tmp_path)
    validated = RUN.invoke(app, ["manifest", "validate", "-c", str(path), "--set", setting])
    predicted = RUN.invoke(app, ["predict", "-c", str(path), "--set", setting, "--output-dir", str(tmp_path / "o")])
    assert validated.exit_code == predicted.exit_code == 2, (validated.output, predicted.output)
    assert message in validated.output and message in predicted.output, (validated.output, predicted.output)


@pytest.mark.parametrize(("setting", "lines"), [
    # the review's MF-13: a field that takes a value or a mapping printed one line per alternative, each with
    # pydantic's tag for it in the path (versions.constrained-str, versions.VersionsSpec, ...)
    ("versions=2026-13",
     ["versions: Input should be a year-month such as 2026-11, or a mapping (servecast manifest explain versions)"]),
    ("versions={as_of: 2026-10, estimater: 2026-11}", ["versions.estimater: unknown field; did you mean estimator?"]),
    ("engine.context_length=maximum",
     ["engine.context_length: Input should be a valid integer, unable to parse string as an integer, or 'max'"]),
    ("profile.clock=locked", ["profile.clock: Input should be 'auto' or 'measured', or a mapping (servecast manifest "
                              "explain profile.clock)"]),
    # the review's MF-19: "mode" for a transposed "model"; no hint in a synthetic source, whose fields were looked up
    # in the trace source's; and three lines about a type that does not apply, after it was refused
    ("engine.modle=x", ["engine.modle: unknown field; did you mean model?"]),
    ("traffic.source={type: synthetic, input_token: {mean: 5}, output_tokens: {mean: 4}}",
     ["traffic.source.input_tokens: Field required",
      "traffic.source.input_token: unknown field; did you mean input_tokens?"]),
    ("traffic.load={type: poisson, rate: 3}",
     ["traffic.load.type: closed workloads only: use type: concurrency with concurrency and think_time_ms"]),
])
def test_each_problem_is_one_line_on_the_path_the_user_wrote(tmp_path: Path, setting: str, lines: list[str]) -> None:
    path = template(tmp_path)
    result = RUN.invoke(app, ["manifest", "validate", "-c", str(path), "--set", setting])
    got = [line.removeprefix(f"{path}: ") for line in result.output.splitlines() if line.strip()]
    assert result.exit_code == 2 and got == lines, got


def test_the_shipped_example_names_the_command_that_saves_its_calibration(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """The review's F14, N9 and MF-14: on a fresh install the shipped example was "valid", and predict refused it
    without saying that Get started saves my-a100 first."""
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path / "fresh"))
    result = RUN.invoke(app, ["manifest", "validate", "-c", "example-a100"])
    assert result.exit_code == 2, result.output
    assert "servecast calibrate --data <profiling data> --name my-a100" in result.output, result.output


def test_json_output_reports_what_predict_refuses(tmp_path: Path) -> None:
    result = RUN.invoke(app, ["manifest", "validate", "-c", str(template(tmp_path)), "--set", "calibration.use=nosuch",
                              "--format", "json"])
    report = json.loads(result.stdout)
    assert result.exit_code == 2 and report["valid"] is False
    assert [p["path"] for p in report["problems"]] == ["calibration.use"], report


def test_every_problem_is_reported_with_its_file_and_path(tmp_path: Path) -> None:
    path = write(tmp_path, ANOTHER_TOOLS_QUICKSTART)
    result = RUN.invoke(app, ["manifest", "validate", "-c", str(path)])
    assert result.exit_code == 2
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) >= 4 and all(line.startswith(f"{path}: ") for line in lines)


def test_set_takes_part_in_the_validation(tmp_path: Path) -> None:
    path = write(tmp_path, PREDICT_WITH_A_TRACE)
    result = RUN.invoke(app, ["manifest", "validate", "-c", str(path), "--set", "traffic.load.concurrency=0"])
    assert result.exit_code == 2 and "traffic.load.concurrency" in result.output


@pytest.mark.parametrize(("setting", "field"), [
    # the review's MF-07: an infinite think time was valid, then predict crashed with an OverflowError
    ("traffic.load.think_time_ms=.inf", "traffic.load.think_time_ms"),
    ("traffic.load.think_time_ms=.nan", "traffic.load.think_time_ms"),
    # the review's MF-17: a mean below one token was valid, explain said "> 0", and predict refused it
    ("traffic.source={type: synthetic, input_tokens: {mean: 0.5}, output_tokens: {mean: 100}}",
     "traffic.source.input_tokens.mean"),
])
def test_a_value_predict_cannot_take_is_not_valid(tmp_path: Path, setting: str, field: str) -> None:
    path = write(tmp_path, PREDICT_WITH_A_TRACE)
    result = RUN.invoke(app, ["manifest", "validate", "-c", str(path), "--set", setting])
    assert result.exit_code == 2 and field in result.output, result.output


@pytest.mark.parametrize("text", ["", "- a list\n", "just a string\n"])
def test_a_file_that_is_not_a_mapping_says_so(tmp_path: Path, text: str) -> None:
    """The review's MF-18: validate reported "(manifest): unknown field" for an empty file."""
    result = RUN.invoke(app, ["manifest", "validate", "-c", str(write(tmp_path, text))])
    assert result.exit_code == 2 and "a manifest is a YAML mapping" in result.output, result.output
    assert "unknown field" not in result.output


def test_explain_gives_the_mean_lengths_real_limit() -> None:
    result = RUN.invoke(app, ["manifest", "explain", "traffic.source.input_tokens.mean"])
    assert result.exit_code == 0 and ">= 1" in result.output, result.output


def test_json_output_is_machine_readable(tmp_path: Path) -> None:
    path = write(tmp_path, ANOTHER_TOOLS_QUICKSTART)
    result = RUN.invoke(app, ["manifest", "validate", "-c", str(path), "--format", "json"])
    report = json.loads(result.output)
    assert result.exit_code == 2 and report["valid"] is False and report["problems"]


def test_a_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    result = RUN.invoke(app, ["manifest", "validate", "-c", str(tmp_path / "absent.yaml")])
    assert result.exit_code == 2 and "no such file" in result.output


def test_the_schemas_print_as_json() -> None:
    manifest = json.loads(RUN.invoke(app, ["manifest", "schema"]).output)
    assert manifest["additionalProperties"] is False
    prediction = json.loads(RUN.invoke(app, ["manifest", "schema", "--kind", "prediction"]).output)
    assert "schema" in prediction["properties"]


def test_explain_prints_meaning_type_and_limits() -> None:
    result = RUN.invoke(app, ["manifest", "explain", "traffic.load.think_time_ms"])
    assert result.exit_code == 0 and "Mean think time" in result.output and "> 0" in result.output


def test_explain_suggests_a_close_field() -> None:
    result = RUN.invoke(app, ["manifest", "explain", "traffic.load.concurency"])
    assert result.exit_code == 2 and "did you mean concurrency?" in result.output
