"""Smaller findings of the outside review about what the command line accepts and says: --sweep values (F17), the
folder calibrate writes given to --with (F19), compare's flag for a measurement without a vLLM version (F21), the
hub's 401 and the id said once (F22), and a local copy with another context length (MF-12)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from servecast.cli.app import app
from tests.test_compare import measurement, pair

RUN = CliRunner()


def run(*args: str) -> tuple[int, str]:
    result = RUN.invoke(app, list(args))
    return result.exit_code, result.output


def template(tmp_path: Path) -> Path:
    path = tmp_path / "m.yaml"
    assert run("manifest", "init", "--out", str(path))[0] == 0
    return path


@pytest.mark.parametrize(("sweep", "message"), [
    ("traffic.load.concurrency=4,,8", "has an empty value"),        # ran 4 and 8 without a word
    ("traffic.load.concurrency=4,4", "4 given more than once"),     # computed the same point twice
])
def test_a_sweep_with_an_empty_or_a_repeated_value_is_refused(tmp_path: Path, sweep: str, message: str) -> None:
    code, output = run("predict", "-c", str(template(tmp_path)), "--sweep", sweep, "--output-dir", str(tmp_path / "o"))
    assert code == 2 and message in output, output


def test_with_takes_the_folder_calibrate_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from servecast.cli import predict_cmd

    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path / "data"))
    out = tmp_path / "cal"
    assert run("calibrate", "--data", "a100_sxm_40gb/c62ebe75", "--name", "mine", "--output-dir", str(out))[0] == 0
    doc, _, _ = predict_cmd.find_calibration(str(out))
    assert doc["name"] == "mine"


def test_compare_says_when_the_measurement_does_not_record_its_vllm_version(tmp_path: Path) -> None:
    meas = measurement()
    meas["provenance"].pop("vllm_version", None)
    _, output = run("compare", *map(str, pair(tmp_path, meas=meas)))
    assert "does not record its vLLM version" in output and "vLLM None" not in output, output


def test_an_unknown_id_names_it_once_and_does_not_only_suspect_a_gated_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from servecast.core import hub
    from servecast.model import run as R

    def refused(repo_id: str, revision: str = "main", *, offline: bool = False) -> dict:
        raise ValueError(f"{repo_id}: the hub refused config.json (401): no such repository, or a gated one")

    monkeypatch.setattr(hub, "fetch_config", refused)
    with pytest.raises(R.ManifestError) as stopped:
        R.resolve_llm("nosuch/model")
    assert str(stopped.value).count("nosuch/model") == 1 and "no such repository" in str(stopped.value)


def test_a_local_copy_with_another_context_length_is_the_same_llm() -> None:
    """The review's MF-12: a copy that differed only in max_position_embeddings was a transfer."""
    from servecast.model import geometry
    from servecast.model import run as R

    llm = geometry.resolve("Qwen/Qwen2.5-7B")
    shorter = {**R.architecture(llm), "max_context": 32768}
    assert R._same_llm(R.architecture(llm), shorter)
    assert not R._same_llm(R.architecture(llm), {**shorter, "layers": 27})
