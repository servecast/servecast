"""GPU-free, end to end: a manifest is measured by `servecast profile serving` on a fake GPU host (the mock engine,
a fake NVML: tests/test_cli_profile.py), predicted by `servecast predict`, and the two are scored by `servecast
compare`, through the same commands and files a GPU host would use. Plumbing only: the mock's numbers are evidence of
nothing.

The requests come from a shipped trace, whose workload law ships too, so `predict` fits nothing and the test takes
seconds. The same run with lengths drawn from a mean (`synthetic`) fits a workload law to the drawn requests, which
takes minutes on a cold cache, and runs with SERVECAST_SLOW=1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from servecast.cli.app import app
from servecast.core import formats
from servecast.core.manifest import Manifest
from servecast.model import geometry
from servecast.model import predict as P
from servecast.profile import serving
from tests.mock_engine import FixedClock, MockEngine, VirtualTimeLoop
from tests.test_cli_examples import SLOW_ON
from tests.test_cli_profile import mock_host
from tests.test_compare import LLAMA3_8B

SOURCES: dict[str, dict[str, Any]] = {
    "trace": {"type": "trace", "paths": ["burstgpt"]},
    "synthetic": {"type": "synthetic", "input_tokens": {"mean": 200}, "output_tokens": {"mean": 12}},
}


def manifest(source: str) -> dict[str, Any]:
    return {
        "engine": {"model": "meta-llama/Meta-Llama-3-8B", "hardware": "a100_sxm_40gb",
                   "workers": {"aggregated": {"kv_cache": {"capacity": {"blocks": 9965}}}}},
        "calibration": {"use": "a100_sxm_40gb/c62ebe75", "transfer": True}, "versions": "2026-11",
        "traffic": {"source": SOURCES[source], "load": {"type": "concurrency", "concurrency": 8, "think_time_ms": 1000},
                    "stop": {"steady_state": {}}, "pool_size": 2000},
    }


@pytest.fixture(scope="module", params=sorted(SOURCES))
def measured(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    if request.param == "synthetic" and not SLOW_ON:
        pytest.skip("fits a workload law to the drawn requests, minutes on a cold cache; SERVECAST_SLOW=1 runs it")
    out = tmp_path_factory.mktemp(request.param)
    (out / "m.yaml").write_text(yaml.safe_dump(manifest(request.param)), encoding="utf-8")
    calibration = P.load_calibration("a100_sxm_40gb/c62ebe75")
    engines: list[MockEngine] = []

    def engine(model: str, d: Any, backend: Any) -> MockEngine:
        engines.append(MockEngine(P.clock(calibration, geometry.resolve(model)), d))
        return engines[-1]

    with pytest.MonkeyPatch.context() as mp:
        mock_host(mp, engine=engine, clock_mhz=float(calibration["conditions"]["clock_mhz"]))
        r = CliRunner().invoke(app, ["profile", "serving", "-c", str(out / "m.yaml"), "--output-dir",
                                     str(out / "measurement"), "--timeout", "7200", "--format", "json"])
    assert r.exit_code == 0, r.output
    return out, {"printed": json.loads(r.stdout), "engines": engines}


def test_profile_serving_writes_a_valid_measurement(measured: tuple[Path, dict]) -> None:
    out, run = measured
    doc = json.loads((out / "measurement" / "measurement.json").read_text(encoding="utf-8"))
    formats.Measurement.model_validate(doc)
    assert run["printed"] == doc and len(run["engines"]) == 1
    assert run["engines"][0].beats > 0 and run["engines"][0].shut_down and len(doc["points"]) == 1
    assert doc["points"][0]["kv_blocks"] == 9965 and doc["provenance"]["components"]["length_sampler"] == "fe3db391"
    assert doc["provenance"]["vllm_version"] == "0.25.1"
    assert sorted(p.name for p in (out / "measurement").iterdir()) == [
        "SHA256SUMS", "engine.log", "environment.json", "measurement.json", "pool.jsonl", "requests.jsonl"]
    env = json.loads((out / "measurement" / "environment.json").read_text(encoding="utf-8"))
    assert env["gpu"]["name"] == "NVIDIA A100-SXM4-40GB" and env["gpu"]["compute_capability"] == "8.0"
    assert (env["clock"]["policy"], env["clock"]["lock_mhz"], env["clock"]["chosen_by"]) == (
        "locked", 1200, "calibration")                                   # the calibration's level, held under load
    assert env["attention"]["flash_attn_version"] == 2
    assert not {"uuid", "serial", "bus_id", "hostname", "processes"} & set(env["gpu"])


def test_predict_and_compare_pair_the_mock_measurement(measured: tuple[Path, dict]) -> None:
    out, _ = measured
    r = CliRunner().invoke(app, ["predict", "-c", str(out / "m.yaml"), "--output-dir", str(out / "prediction")])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(app, ["compare", str(out / "prediction"), str(out / "measurement"), "--format", "json"])
    assert r.exit_code == 0, r.output
    doc = json.loads(r.stdout)
    formats.Comparison.model_validate(doc)
    assert [p["load"]["concurrency"] for p in doc["points"]] == [8]
    assert not [f for f in doc["flags"] if "differ" in f or "named" in f or "clock" in f]
    # with the clock simulated, the mock and the model price every engine step alike; what is left is the mock's
    # simpler scheduler (a request's whole KV reserved at admission, no preemption), declared before the run as 10 %
    for name in ("mean_ttft_ms", "mean_tpot_ms", "request_throughput_rps"):
        error = doc["points"][0]["metrics"][name]["signed_error_pct"]
        assert error is not None and abs(error) < 10, (name, error)


def test_a_list_of_load_levels_is_measured_predicted_and_paired_in_the_order_listed(tmp_path: Path) -> None:
    """The workload's shape as a user writes it, load levels listed in the manifest: profile serving measures them in
    the order listed on one engine, predict predicts the same manifest, and compare pairs them level by level."""
    doc = manifest("trace")
    doc["traffic"]["load"] = [{"type": "concurrency", "concurrency": 8, "think_time_ms": 1000},
                              {"type": "concurrency", "concurrency": 4, "think_time_ms": 500}]
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    calibration = P.load_calibration("a100_sxm_40gb/c62ebe75")
    engines: list[MockEngine] = []

    def engine(model: str, d: Any, backend: Any) -> MockEngine:
        engines.append(MockEngine(P.clock(calibration, geometry.resolve(model)), d))
        return engines[-1]

    with pytest.MonkeyPatch.context() as mp:
        mock_host(mp, engine=engine, clock_mhz=float(calibration["conditions"]["clock_mhz"]))
        r = CliRunner().invoke(app, ["profile", "serving", "-c", str(tmp_path / "m.yaml"), "--output-dir",
                                     str(tmp_path / "measurement"), "--timeout", "7200"])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(app, ["predict", "-c", str(tmp_path / "m.yaml"), "--output-dir",
                                 str(tmp_path / "prediction")])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(app, ["compare", str(tmp_path / "prediction"), str(tmp_path / "measurement"), "--format",
                                 "json"])
    assert r.exit_code == 0, r.output
    levels = [(8, 1000.0), (4, 500.0)]
    measured = json.loads((tmp_path / "measurement" / "measurement.json").read_text(encoding="utf-8"))
    assert [(p["load"]["concurrency"], p["load"]["think_time_ms"]) for p in measured["points"]] == levels
    assert len(engines) == 1 and engines[0].shut_down
    compared = json.loads(r.stdout)
    assert [(p["load"]["concurrency"], p["load"]["think_time_ms"]) for p in compared["points"]] == levels
    assert not [f for f in compared["flags"] if f.endswith((" only", "differ"))], compared["flags"]


def test_the_readmes_gpu_flow_on_the_shipped_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The README's flow on a GPU host, on the fake one, with the shipped manifest as it ships: profile calibration
    measures the per-GPU law's profiling data (the synthetic models, then the GEMM grid), calibrate fits it under the
    name the manifest uses, then predict, profile serving and compare all read example-a100."""
    from importlib import resources

    from servecast.cli import profile_cmd
    from tests.test_cli_profile import never, per_gpu_host

    qwen = json.loads(resources.files("servecast").joinpath("data", "registry", "models.json")
                      .read_text(encoding="utf-8"))["models"]["Qwen/Qwen2.5-7B"]
    calibration = P.load_calibration("a100_sxm_40gb/c62ebe75")
    engines: list[MockEngine] = []

    def engine(model: str, d: Any, backend: Any) -> MockEngine:
        engines.append(MockEngine(P.clock(calibration, geometry.resolve(model)), d))
        return engines[-1]

    mock_host(monkeypatch, engine=engine, probe=never, clock_mhz=float(calibration["conditions"]["clock_mhz"]))
    per_gpu_host(monkeypatch, tmp_path)
    monkeypatch.setattr(profile_cmd, "read_config", lambda model: dict(qwen))
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path / "data"))
    runner = CliRunner()
    steps = [["profile", "calibration", "-c", "example-a100", "--output-dir", str(tmp_path / "profiling")],
             ["calibrate", "--data", str(tmp_path / "profiling"), "--name", "my-a100", "--overwrite",
              "--output-dir", str(tmp_path / "calibration")],
             ["predict", "-c", "example-a100", "--output-dir", str(tmp_path / "prediction")],
             ["profile", "serving", "-c", "example-a100", "--output-dir", str(tmp_path / "measurement"),
              "--timeout", "7200"],
             ["compare", str(tmp_path / "prediction"), str(tmp_path / "measurement"), "--format", "json"]]
    for step in steps:
        r = runner.invoke(app, step)
        assert r.exit_code == 0, (step, r.output)
    compared = json.loads(r.stdout)
    assert [(p["load"]["concurrency"], p["load"]["think_time_ms"]) for p in compared["points"]] == [(64, 1000.0)]
    assert not [f for f in compared["flags"] if "differ" in f or "named" in f or "clock" in f], compared["flags"]
    profiled = json.loads((tmp_path / "profiling" / "environment.json").read_text(encoding="utf-8"))
    served = json.loads((tmp_path / "measurement" / "environment.json").read_text(encoding="utf-8"))
    assert profiled["engine"]["kv_blocks"] == 9965 and profiled["clock"]["chosen_by"] == "sweep"
    assert (served["clock"]["lock_mhz"], served["clock"]["chosen_by"]) == (profiled["clock"]["lock_mhz"],
                                                                         "calibration")
    assert len(engines) == 1 and engines[0].shut_down


def test_a_mock_measurement_does_not_depend_on_the_computers_speed(tmp_path: Path) -> None:
    """On the simulated clock the same manifest measures to the same numbers, run after run."""
    setup = serving.prepare(Manifest.model_validate(manifest("trace")), "a100_sxm_40gb", LLAMA3_8B)
    calibration = P.load_calibration("a100_sxm_40gb/c62ebe75")
    clock = P.clock(calibration, geometry.resolve("meta-llama/Meta-Llama-3-8B"))
    runs = [serving.serve(setup, tmp_path / str(i), engine=MockEngine(clock, setup.deployment),
                          sampler=FixedClock(1200.0), prompt=list(range(max(setup.pool.input_tokens))),
                          environment={"vllm": "0.25.1"}, timeout_s=300.0, loop_factory=VirtualTimeLoop)
            for i in range(2)]
    assert runs[0][0]["summary"] == runs[1][0]["summary"]
