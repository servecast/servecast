"""`servecast calibrate` and `predict` with profiling data: the fitted calibration is the shipped one, and predicts the
same numbers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from servecast.cli.app import app
from servecast.core import calibration as calibrations
from servecast.model.predict import shipped_profiling

RUN = CliRunner()
PROFILING = shipped_profiling()["a100_sxm_40gb/c62ebe75"].parents[1]  # the package's data/profiling
H100_LLAMA = PROFILING / "h100_sxm" / "d9cbf1d3" / "llama-3-8b.json"


@pytest.fixture(autouse=True)
def private_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SERVECAST_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)  # calibrate writes ./servecast-calibration by default: never into the checkout


def manifest(calibration: dict) -> dict:
    return {"engine": {"model": "meta-llama/Meta-Llama-3-8B", "hardware": "h100_sxm",
                       "workers": {"aggregated": {"scheduler": {"max_batched_tokens": 2048, "max_sequences": 512},
                                                  "kv_cache": {"capacity": {"blocks": 9965}}}}},
            "calibration": calibration, "versions": "2026-10",
            "traffic": {"source": {"type": "trace", "paths": ["azure-conversation"], "format": "lengths"},
                        "load": {"type": "concurrency", "concurrency": 8, "think_time_ms": 1000},
                        "stop": {"steady_state": {}}}}


def predict(tmp_path: Path, doc: dict, out: str) -> dict:
    cfg = tmp_path / f"{out}.yaml"
    cfg.write_text(yaml.safe_dump(doc), encoding="utf-8")
    r = RUN.invoke(app, ["predict", "-c", str(cfg), "--output-dir", str(tmp_path / out)])
    assert r.exit_code == 0, r.output
    return json.loads((tmp_path / out / "prediction.json").read_text())["points"][0]["summary"]


@pytest.mark.exact
def test_a_fitted_calibration_predicts_what_the_shipped_one_does(tmp_path: Path) -> None:
    r = RUN.invoke(app, ["calibrate", "--data", str(H100_LLAMA), "--versions", "2026-10", "--name", "mine/llama",
                         "--format", "json"])
    assert r.exit_code == 0, r.output
    fitted = json.loads(r.stdout)["calibration"]
    shipped = json.loads((Path(__file__).resolve().parents[1] / "src/servecast/data/calibrations/h100_sxm/d9cbf1d3/"
                          "llama-3-8b.json").read_text())
    assert fitted["law"] == shipped["law"] and fitted["conditions"] == shipped["conditions"]
    want = predict(tmp_path, manifest({"use": "h100_sxm/d9cbf1d3/llama-3-8b"}), "shipped")
    for calibration, out in (({"use": "mine/llama"}, "mine"), ({"data": str(H100_LLAMA)}, "fly")):
        got = predict(tmp_path, manifest(calibration), out)
        assert {k: v for k, v in got.items() if k != "wall_time_ms"} == {
            k: v for k, v in want.items() if k != "wall_time_ms"}, out


@pytest.mark.parametrize("args, message", [
    (["--data", str(H100_LLAMA), "--versions", "2026-11"], "measured for d9cbf1d3"),
    (["--data", str(H100_LLAMA), "--versions", "2026-10", "--name", "h100_sxm/d9cbf1d3/llama-3-8b"],
     "name of a shipped calibration"),
    (["--versions", "2026-10"], "give either -c MANIFEST"),
    (["--data", "nowhere.json", "--versions", "2026-10"], "no profiling data"),
])
def test_what_calibrate_refuses(args: list[str], message: str) -> None:
    r = RUN.invoke(app, ["calibrate", *args])
    assert r.exit_code == 2 and message in r.output, r.output


def test_the_default_name_is_local_and_never_a_shipped_one(tmp_path: Path) -> None:
    r = RUN.invoke(app, ["calibrate", "--data", str(H100_LLAMA), "--versions", "2026-10"],
                   env={"SERVECAST_DATA_DIR": str(tmp_path)})
    assert r.exit_code == 0, r.output
    assert (tmp_path / "calibrations" / "local" / "h100_sxm" / "d9cbf1d3" / "llama-3-8b.json").is_file()


# what a probe must hold before it is fitted, and the fit's error -------------------------------------------------

A100_LLM = PROFILING / "a100_sxm_40gb" / "d9cbf1d3" / "deepseek-coder-1.3b-base.json"
A100_GPU = PROFILING / "a100_sxm_40gb" / "c62ebe75.json"
SHIPPED_A100 = Path(__file__).resolve().parents[1] / "src/servecast/data/calibrations/a100_sxm_40gb/c62ebe75.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("cls", ["decode", "prefill", "mixed"])
def test_a_probe_without_one_class_is_refused(cls: str) -> None:
    from servecast.calibrate import fit as F

    doc = _load(A100_LLM)
    doc["cells"] = [c for c in doc["cells"] if c["cls"] != cls]
    with pytest.raises(ValueError, match=f"no {cls} cell"):
        F.calibrate(doc, "sha")


def test_a_probe_with_no_cell_is_refused_before_the_optimizer_returns_its_seed() -> None:
    from servecast.calibrate import fit as F

    doc = _load(A100_LLM)
    doc["cells"] = []
    with pytest.raises(ValueError, match="no decode, prefill or mixed cell"):
        F.calibrate(doc, "sha")


def test_a_synthetic_model_without_a_cell_is_refused() -> None:
    from servecast.calibrate import fit as F

    doc = _load(A100_GPU)
    doc["models"][1]["cells"] = []
    with pytest.raises(ValueError, match=doc["models"][1]["llm"]):
        F.calibrate(doc, "sha")


def test_calibrate_prints_and_saves_the_fit_error(tmp_path: Path) -> None:
    r = RUN.invoke(app, ["calibrate", "--data", str(A100_LLM), "--versions", "2026-10", "--format", "json"])
    assert r.exit_code == 0, r.output
    err = json.loads(r.stdout)["calibration"]["fit"]["error_pct"]
    assert set(err) == {"mape", "max"} and 0 < err["mape"] <= err["max"]
    r = RUN.invoke(app, ["calibrate", "--data", str(A100_LLM), "--versions", "2026-10", "--name", "again"])
    assert r.exit_code == 0 and "training MAPE 10.47 % over the 976 cells" in r.output, r.output


def test_the_training_mape_is_the_papers_table_5_row() -> None:
    """The six A100 fits, priced with their shipped coefficients: the median and range the paper prints."""
    import statistics as st

    from servecast.calibrate import fit as F
    from servecast.core import gpus

    shipped = SHIPPED_A100.parent / "d9cbf1d3"
    mapes = sorted(F.training_error(_load(f), list(_load(shipped / f.name)["law"]["parameters"].values()),
                                    calibrations.peak_fraction(_load(shipped / f.name)),
                                    gpus.get("a100_sxm_40gb"))["mape"]
                   for f in sorted((PROFILING / "a100_sxm_40gb" / "d9cbf1d3").glob("*.json")))
    row = next(r for r in _load(SHIPPED_A100.parents[2] / "paper" / "tables.json")["tables"]["rq3-cost"]["rows"]
               if r[0] == "A100")
    assert (f"{st.median(mapes):.1f}", f"{mapes[0]:.1f}--{mapes[-1]:.1f}") == (row[1], row[2])


def test_the_error_prices_each_cell_as_the_fit_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """The newest law's shipped calibration, priced with its own form, reproduces its probe better than with the
    paper's form: the switches the error passes to the cell price are the fit's."""
    import statistics as st

    from servecast.calibrate import fit as F
    from servecast.core import gpus

    prof, cal = _load(A100_GPU), _load(SHIPPED_A100)
    args = (prof, list(cal["law"]["parameters"].values()), calibrations.peak_fraction(cal), gpus.get("a100_sxm_40gb"))
    right = st.median(abs(e) for e in F.fit_errors(*args))
    monkeypatch.setitem(F.FORM_OF_LAW, "c62ebe75", {})
    assert right < st.median(abs(e) for e in F.fit_errors(*args))


@pytest.mark.parametrize("name", ["../../escape", "/abs/path", "", "a//b", "a/./b", "a b"])
def test_a_name_that_is_not_a_calibration_name_is_refused(tmp_path: Path, name: str) -> None:
    """`--name ../../escape` wrote outside the data directory, an absolute name was an internal error, and an empty
    one fell back to the default name without a word."""
    r = RUN.invoke(app, ["calibrate", "--data", str(A100_LLM), "--versions", "2026-10", "--name", name])
    assert r.exit_code == 2 and "--name" in r.output, r.output
    assert not list(tmp_path.glob("**/escape.json"))


@pytest.mark.parametrize("name", ["my-a100", "team/a100.v2/llama_3"])
def test_a_calibration_name_may_have_folders(name: str) -> None:
    r = RUN.invoke(app, ["calibrate", "--data", str(A100_LLM), "--versions", "2026-10", "--name", name])
    assert r.exit_code == 0 and f"use it with --with {name}" in r.output, r.output


def test_calibrate_refuses_a_local_llm_the_laws_do_not_model(tmp_path: Path) -> None:
    """calibrate read a local LLM's config.json through the model's geometry, which applied fewer refusals than
    predict's reader: a plain-MLP model was calibrated as a gated one."""
    from servecast.calibrate import fit as F

    llm = tmp_path / "neox"
    llm.mkdir()
    (llm / "config.json").write_text(json.dumps({
        "model_type": "gpt_neox", "num_hidden_layers": 24, "hidden_size": 2048, "intermediate_size": 8192,
        "num_attention_heads": 16, "vocab_size": 50304, "max_position_embeddings": 2048, "torch_dtype": "float16"}),
        encoding="utf-8")
    doc = _load(A100_LLM)
    doc["llm"] = str(llm)
    with pytest.raises(ValueError, match="feed-forward"):
        F.calibrate(doc, "sha")


def test_the_profiling_data_name_their_law_so_versions_is_optional(tmp_path: Path) -> None:
    r = RUN.invoke(app, ["calibrate", "--data", str(H100_LLAMA), "--name", "inferred", "--format", "json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["calibration"]["law"]["cost_law"] == "d9cbf1d3"


@pytest.mark.parametrize(("hardware", "status"), [("h100_sxm", 0), ("a100_sxm_40gb", 2)])
def test_calibrate_fits_profiling_data_of_the_manifests_gpu_only(tmp_path: Path, hardware: str, status: int) -> None:
    """engine.hardware is the GPU calibrated: profiling data measured on another one are refused."""
    doc = manifest({"data": str(H100_LLAMA)})
    doc["engine"]["hardware"] = hardware
    cfg = tmp_path / "m.yaml"
    cfg.write_text(yaml.safe_dump(doc), encoding="utf-8")
    r = RUN.invoke(app, ["calibrate", "-c", str(cfg), "--name", "mine/llama"])
    assert r.exit_code == status, r.output
    if status:
        assert "engine.hardware: a100_sxm_40gb, but the profiling data" in r.output and "h100_sxm" in r.output
