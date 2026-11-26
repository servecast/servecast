"""A command never replaces a file it did not write without being told to, and a refused command leaves nothing
behind (findings F02, F04, F05 and DC-21 of the outside review)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servecast.cli.app import app
from tests.test_compare import pair

RUN = CliRunner()


def run(*args: str) -> tuple[int, str]:
    result = RUN.invoke(app, list(args))
    return result.exit_code, result.output


def test_compare_never_writes_over_the_prediction_it_scores(tmp_path: Path) -> None:
    """The review's F02: --output pred/prediction.json replaced the prediction, which could not be scored again."""
    pred, meas = pair(tmp_path)
    before = (pred / "prediction.json").read_bytes()
    code, output = run("compare", str(pred), str(meas), "--output", str(pred / "prediction.json"))
    assert code == 3 and "is not a comparison" in output, output
    assert (pred / "prediction.json").read_bytes() == before
    assert run("compare", str(pred), str(meas))[0] == 0  # an earlier comparison is still replaced, as before
    assert run("compare", str(pred), str(meas))[0] == 0


@pytest.mark.parametrize("command", [("manifest", "init", "--out"), ("workload", "sample", "-c", "{m}", "--out")])
def test_an_existing_out_file_is_kept_unless_overwrite_is_given(tmp_path: Path, command: tuple[str, ...]) -> None:
    """The review's F04: --out replaced any file there without a word."""
    m = tmp_path / "m.yaml"
    assert run("manifest", "init", "--out", str(m))[0] == 0
    keep = tmp_path / "keep.txt"
    keep.write_text("mine", encoding="utf-8")
    args = [a.format(m=m) for a in command] + [str(keep)]
    code, output = run(*args)
    assert code == 3 and "--overwrite" in output and keep.read_text(encoding="utf-8") == "mine", output
    assert run(*args, "--overwrite")[0] == 0 and keep.read_text(encoding="utf-8") != "mine"


def test_out_dash_is_standard_output_for_init_and_refused_for_sample(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """The review's F04: --out - wrote a file named "-"."""
    monkeypatch.chdir(tmp_path)
    code, output = run("manifest", "init", "--out", "-")
    assert code == 0 and "engine:" in output and not (tmp_path / "-").exists(), output
    assert run("manifest", "init", "--out", "m.yaml")[0] == 0
    code, output = run("workload", "sample", "-c", "m.yaml", "--out", "-")
    assert code == 2 and "give its path" in output and not (tmp_path / "-").exists(), output


def test_a_taken_calibration_name_is_refused_before_the_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The review's F05: the name was checked after the whole fit, and refused with exit 2."""
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path / "data"))
    first = ("calibrate", "--data", "a100_sxm_40gb/c62ebe75", "--name", "mine", "--output-dir", str(tmp_path / "a"))
    assert run(*first)[0] == 0
    code, output = run("calibrate", "--data", "a100_sxm_40gb/c62ebe75", "--name", "mine", "--output-dir",
                       str(tmp_path / "b"))
    assert code == 3 and "a calibration named mine exists" in output and "fitting" not in output, output


def test_an_output_dir_holding_other_files_is_refused_and_an_earlier_calibration_is_replaced(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The review's F05: calibration.json and run.json went into any folder, over what was there."""
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path / "data"))
    out = tmp_path / "out"
    out.mkdir()
    (out / "profiling.json").write_text("{}", encoding="utf-8")
    code, output = run("calibrate", "--data", "a100_sxm_40gb/c62ebe75", "--name", "a", "--output-dir", str(out))
    assert code == 3 and "profiling.json" in output and not (out / "calibration.json").exists(), output
    again = tmp_path / "again"
    assert run("calibrate", "--data", "a100_sxm_40gb/c62ebe75", "--name", "b", "--output-dir", str(again))[0] == 0
    assert run("calibrate", "--data", "a100_sxm_40gb/c62ebe75", "--name", "c", "--output-dir", str(again))[0] == 0
    assert json.loads((again / "calibration.json").read_text(encoding="utf-8"))["name"] == "c"


def test_a_refused_prediction_leaves_no_empty_output_folder(tmp_path: Path) -> None:
    """The review's DC-21: a predict refused for its calibration left an empty folder behind."""
    m = tmp_path / "m.yaml"
    assert run("manifest", "init", "--out", str(m))[0] == 0
    out = tmp_path / "out"
    code, output = run("predict", "-c", str(m), "--with", "nosuch", "--output-dir", str(out))
    assert code == 2 and not out.exists(), output
    out.mkdir()  # a folder the user made stays, empty or not
    assert run("predict", "-c", str(m), "--with", "nosuch", "--output-dir", str(out))[0] == 2 and out.is_dir()
