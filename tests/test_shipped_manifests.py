"""The manifests that ship with ServeCast: found by name, listed, printed by manifest show, valid, and shown on the
get-started page as they are; and a calibration fitted on a shipped calibration's profiling data counts as validated
where that one is."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servecast.cli.app import app
from servecast.core import manifest
from servecast.model import predict as P
from servecast.model import run as R
from tests.test_docs_examples import fences_in

ROOT = Path(__file__).resolve().parents[1]
RUN = CliRunner()


def test_every_shipped_manifest_is_valid_and_found_by_name() -> None:
    assert "example-a100" in manifest.shipped()
    for name, path in manifest.shipped().items():
        assert manifest.find(Path(name)) == path
        _, problems = manifest.validate(manifest.load(path))
        assert problems == [], name


def test_a_file_wins_over_a_shipped_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "example-a100").write_text("engine: {}\n", encoding="utf-8")
    assert manifest.find(Path("example-a100")) == Path("example-a100")


def test_list_manifests_shows_each_one() -> None:
    rows = json.loads(RUN.invoke(app, ["list", "manifests", "--format", "json"]).output)
    row = next(r for r in rows if r["manifest"] == "example-a100")
    assert row["llm"] == "Qwen/Qwen2.5-7B" and row["calibration"] == "my-a100"


def test_manifest_show_prints_a_shipped_manifest_as_it_is() -> None:
    for name, path in manifest.shipped().items():
        r = RUN.invoke(app, ["manifest", "show", name])
        assert r.exit_code == 0 and r.stdout == path.read_text(encoding="utf-8"), name


def test_manifest_show_refuses_a_name_that_does_not_ship_and_names_those_that_do() -> None:
    r = RUN.invoke(app, ["manifest", "show", "my-a100"])
    assert r.exit_code == 2 and r.stdout == ""
    assert "no manifest 'my-a100' ships with ServeCast; shipped: example-a100" in r.stderr


def test_the_get_started_page_shows_the_shipped_manifest_as_it_is() -> None:
    page = (ROOT / "docs" / "get-started.md").read_text(encoding="utf-8")
    shown = next(body for language, body in fences_in(page) if language.split(" ")[0] == "yaml")
    without_markers = re.sub(r" # \(\d+\)!", "", shown)
    assert without_markers == manifest.shipped()["example-a100"].read_text(encoding="utf-8")


def test_a_calibration_fitted_on_shipped_data_is_validated_where_the_shipped_one_is() -> None:
    shipped = P.load_calibration("a100_sxm_40gb/c62ebe75")
    sha = hashlib.sha256(P.shipped_profiling()["a100_sxm_40gb/c62ebe75"].read_bytes()).hexdigest()
    mine = {**shipped, "name": "my-a100", "fit": {"profiling_sha256": sha}}
    assert R.validated_on(mine, "my-a100") == R.VALIDATED_ON["a100_sxm_40gb/c62ebe75"]
    other = {**mine, "fit": {"profiling_sha256": "0" * 64}}
    assert R.validated_on(other, "my-a100") == ()
