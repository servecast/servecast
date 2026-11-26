"""`servecast list components | laws` and `servecast manifest init`."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from servecast.cli.app import app
from servecast.core import manifest, versions

RUN = CliRunner()


def test_list_components_shows_every_entry_by_name_with_its_summary() -> None:
    rows = json.loads(RUN.invoke(app, ["list", "components", "--format", "json"]).output)
    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == {name for key in versions.KIND_OF if key != "calibration_procedure"
                            for name in versions.entries(key)}
    assert by_name["d9cbf1d3"]["in_paper"] is True and by_name["d9cbf1d3"]["default"] is False
    assert by_name["fe3db391"]["in_paper"] is False and by_name["fe3db391"]["released"] == "2026-11"
    assert all(row["summary"] == versions.entry(row["versions_key"], row["name"]).summary for row in rows)


def test_list_components_prints_aligned_columns() -> None:
    lines = RUN.invoke(app, ["list", "components"]).output.splitlines()
    assert lines[0].startswith("component") and len(lines) == 1 + 7


def test_list_laws_names_each_law_and_states_no_number_beyond_the_paper() -> None:
    output = RUN.invoke(app, ["list", "laws"]).output
    newest = next(line for line in output.splitlines() if line.startswith("c62ebe75"))
    assert "per GPU" in newest
    assert not any(ch.isdigit() for ch in newest.removeprefix("c62ebe75").replace("2026-11", "", 1))


@pytest.mark.parametrize("kind", ["predict", "profile"])
def test_manifest_init_writes_a_valid_manifest_pinned_to_the_newest_release(kind: str) -> None:
    text = RUN.invoke(app, ["manifest", "init", "--template", kind]).output
    _, problems = manifest.validate(yaml.safe_load(text))
    assert problems == []
    assert f"versions: {versions.newest_release()}" in text


def test_the_starter_calibration_ships() -> None:
    """A new cost law without a per-GPU A100 calibration would make the first example fail: this test says so."""
    from servecast.model import predict as P

    assert yaml.safe_load(versions.template("predict"))["calibration"]["use"] in P.shipped_calibrations()


class _December(date):
    @classmethod
    def today(cls) -> date:
        return date(2026, 12, 1)


def test_the_first_example_still_works_after_this_month(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The starter manifest named the calibration after the current month: on 1 December 2026 it named
    a100_sxm_40gb/2026-12, which does not ship, and the first example of the documentation failed. Since components
    have names nothing reads today's date: unset `versions` means the defaults, and the template pins the newest
    release."""
    import datetime

    monkeypatch.setattr(datetime, "date", _December)
    cfg = tmp_path / "m.yaml"
    cfg.write_text(RUN.invoke(app, ["manifest", "init", "--template", "predict"]).output, encoding="utf-8")
    r = RUN.invoke(app, ["predict", "-c", str(cfg), "--output-dir", str(tmp_path / "out")])
    assert r.exit_code == 0, r.output
