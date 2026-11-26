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


def test_a_valid_manifest_says_so(tmp_path: Path) -> None:
    result = RUN.invoke(app, ["manifest", "validate", "-c", str(write(tmp_path, PREDICT_WITH_A_TRACE))])
    assert result.exit_code == 0 and "valid" in result.output


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
