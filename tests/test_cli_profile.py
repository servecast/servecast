"""`servecast doctor`, `servecast profile serving` and `servecast profile calibration` on a fake GPU host: a fake
NVML (tests/test_profile_gpu.py), the GPU-free mock engine on a simulated clock (tests/mock_engine.py) and the
simulated engine of the calibration probe (tests/mock_probe_engine.py). What the commands decide is tested here: which
computers they refuse and why, before the GPU is touched; the clock lock, released even on a failure; the files and
their checksums; powering off only once they verify; and profiling data that `servecast calibrate` fits. The serving
measurement itself is tested in tests/test_profile_end_to_end.py, through the command, and the probe in
tests/test_profile_probe.py. Plumbing only: the mocks' numbers are evidence of nothing about vLLM.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from servecast.cli import doctor_cmd, profile_cmd
from servecast.cli.app import app
from servecast.core import formats
from servecast.core.manifest import ClockLock
from servecast.model import geometry
from servecast.model import predict as P
from servecast.profile import gpu as G
from servecast.profile import probe as PR
from servecast.profile import probe_gpu as PG
from servecast.profile import serving, vllm_engine
from tests.mock_engine import FixedClock, MockEngine, VirtualTimeLoop
from tests.mock_probe_engine import SimEngine
from tests.test_compare import LLAMA3_8B
from tests.test_profile_gpu import T4, fake_nvml, loaded, smi_refused

L40S = ("NVIDIA L40S", (8, 9))       # compute capability 8.9, not in the shipped registry
CALIBRATION = "a100_sxm_40gb/c62ebe75"


def mock_engine(model: str = "meta-llama/Meta-Llama-3-8B") -> Callable[..., MockEngine]:
    """The mock engine, each beat priced by the shipped A100 calibration."""
    beat = P.clock(P.load_calibration(CALIBRATION), geometry.resolve(model))
    return lambda _model, d, _backend: MockEngine(beat, d)


def probe_engine(calibration: str = "a100_sxm_40gb/d9cbf1d3/llama-3-8b", drift: float = 0.0) -> Callable[..., Any]:
    """The probe's simulated engine, each step priced by a shipped calibration of Llama-3-8B; `drift` slows the
    engine down by that fraction over a probe of 4,000 steps."""
    beat = P.clock(P.load_calibration(calibration), geometry.resolve("meta-llama/Meta-Llama-3-8B"))
    steps = [0]

    def price(comp: Any) -> float:
        steps[0] += 1
        return beat(comp) * (1.0 + drift * steps[0] / 4000)
    return lambda _model, d, _backend, *, kv_tokens, sample_clock: SimEngine(
        price, np.random.default_rng(0), d.max_batched_tokens, d.max_sequences, 1200.0)


def mock_host(mp: pytest.MonkeyPatch, *, nvml: Any = None, vllm: str | None = "0.25.1", clock_mhz: float = 1200.0,
              engine: Callable[..., Any] | None = None,
              probe: Callable[..., Any] | None = None) -> types.SimpleNamespace:
    """A GPU host as `doctor` and the profile commands see it. The real `detect` runs, on Linux, on the fake NVML."""
    nv = nvml if nvml is not None else fake_nvml()
    detect = G.detect
    mp.setattr(G, "detect", lambda env=None: detect({}, system="Linux", import_nvml=lambda: nv, amd=lambda: False))
    mp.setattr(G, "sampler", lambda backend: FixedClock(clock_mhz))
    version = doctor_cmd._version
    mp.setattr(doctor_cmd, "_version", lambda dist: vllm if dist == "vllm" else version(dist))
    mp.setattr(profile_cmd, "read_config", lambda model: dict(LLAMA3_8B))
    mp.setattr(profile_cmd, "prompt", lambda model, n: list(range(n)))
    mp.setattr(profile_cmd, "engine", engine if engine is not None else mock_engine())
    mp.setattr(profile_cmd, "loop_factory", VirtualTimeLoop)
    mp.setattr(profile_cmd, "probe_engine", probe if probe is not None else probe_engine())
    mp.setattr(profile_cmd, "measure_under_load", loaded)    # the dense GEMM, on the fake NVML's clock
    mp.setattr(G, "nvidia_smi", smi_refused())                # no passwordless sudo
    mp.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return nv


def manifest(**profile: Any) -> dict[str, Any]:
    return {
        "engine": {"model": "meta-llama/Meta-Llama-3-8B", "hardware": "a100_sxm_40gb",
                   "workers": {"aggregated": {"kv_cache": {"capacity": {"blocks": 9965}}}}},
        "calibration": {"use": CALIBRATION, "transfer": True}, "versions": "2026-11",
        "traffic": {"source": {"type": "trace", "paths": ["burstgpt"]},
                    "load": {"type": "concurrency", "concurrency": 8, "think_time_ms": 1000},
                    "stop": {"steady_state": {}}, "pool_size": 2000},
        "profile": profile}


def run(tmp_path: Path, *args: str, doc: dict[str, Any] | None = None) -> Any:
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(doc or manifest()), encoding="utf-8")
    return CliRunner().invoke(app, ["profile", "serving", "-c", str(tmp_path / "m.yaml"),
                                    "--output-dir", str(tmp_path / "measurement"), *args])


def never(*_: Any) -> Any:
    raise AssertionError("the engine was built although the command should have stopped before")


# doctor ----------------------------------------------------------------------------------------------------

def test_doctor_on_this_computer_says_what_runs() -> None:
    r = CliRunner().invoke(app, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "profile calibration: " in r.stdout and "every other command runs" in r.stdout
    doc = json.loads(CliRunner().invoke(app, ["doctor", "--format", "json"]).stdout)
    assert set(doc["profile"]) == {"serving", "calibration"}


def test_doctor_on_a_gpu_host_that_can_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_host(monkeypatch)
    r = CliRunner().invoke(app, ["doctor"])
    assert r.exit_code == 0, r.output
    assert ("GPU: NVIDIA A100-SXM4-40GB (a100_sxm_40gb), 40 GiB, compute capability 8.0, driver 580.82"
            in r.stdout)
    assert "profile serving: ready" in r.stdout
    assert "profile calibration: ready" in r.stdout
    doc = json.loads(CliRunner().invoke(app, ["doctor", "--format", "json"]).stdout)
    assert doc["gpu"]["id"] == "a100_sxm_40gb" and doc["profile"]["serving"] == {"ready": True, "why_not": None}


@pytest.mark.parametrize(("host", "why"), [
    ({"vllm": None}, "vLLM 0.25.1 is not installed; install the profile extra: pip install 'servecast[profile]'"),
    ({"vllm": "0.26.0"}, "vLLM 0.26.0 is installed; the measurement runs 0.25.1"),
    ({"nvml": fake_nvml(devices=(T4,))}, "Tesla T4 has compute capability 7.5"),
    ({"nvml": fake_nvml(devices=(L40S,))}, "NVIDIA L40S is not in ServeCast's GPU registry"),
])
def test_doctor_names_what_profile_is_missing(monkeypatch: pytest.MonkeyPatch, host: dict[str, Any],
                                              why: str) -> None:
    mock_host(monkeypatch, **host)
    r = CliRunner().invoke(app, ["doctor"])
    assert r.exit_code == 0, r.output
    assert f"profile serving: cannot run here: {why}" in r.stdout


def test_a_driver_too_old_for_the_installed_pytorch_is_named_by_doctor_and_profile(monkeypatch: pytest.MonkeyPatch,
                                                                                  tmp_path: Path) -> None:
    """Before this check, doctor said ready and vLLM failed only when it started, its reason in engine.log."""
    mock_host(monkeypatch, nvml=fake_nvml(driver_version="570.86", cuda=12080), engine=never)
    monkeypatch.setattr(G, "torch_cuda", lambda: "13.0")
    r = CliRunner().invoke(app, ["doctor"])
    assert r.exit_code == 0, r.output
    assert ("profile serving: cannot run here: the NVIDIA driver 570.86 supports CUDA up to 12.8, and the installed "
            "PyTorch is built for CUDA 13.0") in r.stdout
    r = run(tmp_path)
    assert r.exit_code == 3, r.output
    assert "this computer cannot profile: the NVIDIA driver 570.86" in r.stderr
    assert "install a driver that supports CUDA 13 (580 or newer)" in r.stderr


def test_doctor_does_not_print_the_bit_exact_reference_environment_but_its_json_has_it() -> None:
    r = CliRunner().invoke(app, ["doctor"])
    assert r.exit_code == 0 and "reference environment" not in r.stdout and "bit-exact" not in r.stdout
    assert "reference_environment" in json.loads(CliRunner().invoke(app, ["doctor", "--format", "json"]).stdout)


# profile serving: what it refuses, before the GPU is touched -----------------------------------------------

@pytest.mark.parametrize(("host", "doc", "status", "phrase"), [
    ({"nvml": fake_nvml(processes=(4242,))}, None, 3, "the GPU is in use by process(es) 4242"),
    ({}, {"engine": {"hardware": "h100_sxm"}}, 2, "engine.hardware: h100_sxm, but this machine's GPU is a100_sxm_40gb"),
    ({"vllm": None}, None, 3, "pip install 'servecast[profile]'"),
    ({"vllm": "0.26.0"}, None, 3, "vLLM 0.26.0 is installed"),
    ({"nvml": fake_nvml(devices=(T4,))}, None, 3, "this computer cannot profile: Tesla T4 has compute capability 7.5"),
    ({"nvml": fake_nvml(devices=(L40S,))}, None, 2, "add it to"),
    ({}, {"engine": {"gpu_clock_mhz": 1200}}, 2, "engine.gpu_clock_mhz: a measurement runs at the GPU's own clock"),
])
def test_profile_serving_refuses_before_the_engine_starts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                          host: dict[str, Any], doc: dict[str, Any] | None,
                                                          status: int, phrase: str) -> None:
    mock_host(monkeypatch, engine=never, **host)
    m = manifest()
    for key, value in (doc or {}).items():
        m[key] = {**m[key], **value}
    r = run(tmp_path, doc=m)
    assert r.exit_code == status, r.output
    assert phrase in r.stderr and "Traceback" not in r.output
    assert not (tmp_path / "measurement").exists()      # a refused run leaves nothing behind


def test_a_sweep_of_a_manifest_that_lists_levels_is_refused_before_the_engine_starts(monkeypatch: pytest.MonkeyPatch,
                                                                                      tmp_path: Path) -> None:
    mock_host(monkeypatch, engine=never)
    m = manifest()
    m["traffic"]["load"] = [{"type": "concurrency", "concurrency": 8, "think_time_ms": 1000},
                            {"type": "concurrency", "concurrency": 4, "think_time_ms": 1000}]
    r = run(tmp_path, "--sweep", "traffic.load.concurrency=2,16", doc=m)
    assert r.exit_code == 2 and "--sweep: traffic.load lists 2 levels" in r.stderr, r.output
    assert not (tmp_path / "measurement").exists()


def test_a_computer_without_an_nvidia_gpu_is_refused(tmp_path: Path) -> None:
    """On this computer, unless it is a GPU host with the profile extra."""
    try:
        G.detect()
    except G.CannotProfileError:
        r = run(tmp_path)
        assert r.exit_code == 3, r.output
        assert "this computer cannot profile: " in r.stderr and "Every other command runs on it." in r.stderr
        assert not (tmp_path / "measurement").exists()
    else:  # pragma: no cover - a GPU host
        pytest.skip("this computer has an NVIDIA GPU that NVML sees")


def test_a_non_empty_output_directory_is_refused_without_overwrite(monkeypatch: pytest.MonkeyPatch,
                                                                   tmp_path: Path) -> None:
    mock_host(monkeypatch, engine=never)
    (tmp_path / "measurement").mkdir()
    (tmp_path / "measurement" / "measurement.json").write_text("{}", encoding="utf-8")
    r = run(tmp_path)
    assert r.exit_code == 3 and "is not empty; give --overwrite" in r.stderr, r.output


# profile serving: a run on the mock ------------------------------------------------------------------------

def test_a_failing_engine_ends_the_run_and_the_machine_stays_on(monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path) -> None:
    def out_of_memory(*_: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    nv = mock_host(monkeypatch, engine=out_of_memory)
    marker = tmp_path / "powered-off"
    monkeypatch.setattr(profile_cmd, "POWEROFF", (sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"))
    r = run(tmp_path, "--poweroff-when-safe", doc=manifest(clock={"lock": 1200}))
    assert r.exit_code == 1, r.output
    assert "vLLM could not start (RuntimeError: CUDA out of memory)" in r.stderr
    assert not marker.exists()
    assert nv.calls == [("lock", 1200), ("unlock", 0)]   # the lock is released after a failure too


@pytest.mark.parametrize("how", ["flag", "manifest"])
def test_the_machine_powers_off_once_the_files_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                      how: str) -> None:
    mock_host(monkeypatch)
    marker = tmp_path / "powered-off"
    monkeypatch.setattr(profile_cmd, "POWEROFF", (sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"))
    args = ("--poweroff-when-safe",) if how == "flag" else ()
    r = run(tmp_path, *args, doc=manifest(poweroff_when_safe=how == "manifest"))
    assert r.exit_code == 0, r.output
    assert "every file matches SHA256SUMS; powering off" in r.stderr and marker.exists()


def test_files_that_do_not_match_their_checksums_keep_the_machine_on(monkeypatch: pytest.MonkeyPatch,
                                                                     tmp_path: Path) -> None:
    mock_host(monkeypatch)
    marker = tmp_path / "powered-off"
    monkeypatch.setattr(profile_cmd, "POWEROFF", (sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"))
    monkeypatch.setattr(profile_cmd, "verify", lambda out_dir: ["requests.jsonl"])
    r = run(tmp_path, "--poweroff-when-safe")
    assert r.exit_code == 0, r.output
    assert "not powered off: requests.jsonl do not match SHA256SUMS" in r.stderr and not marker.exists()


def test_a_refused_power_off_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_host(monkeypatch)
    monkeypatch.setattr(profile_cmd, "POWEROFF", (sys.executable, "-c", "raise SystemExit(1)"))
    r = run(tmp_path, "--poweroff-when-safe")
    assert r.exit_code == 0, r.output
    assert "failed" in r.stderr and "power the machine off yourself" in r.stderr


def test_engine_log_holds_what_a_child_process_prints_on_either_stream(tmp_path: Path,
                                                                       capfd: pytest.CaptureFixture[str]) -> None:
    """vLLM's engine core is a child process; it logs on standard output, and its block count is read from the log."""
    child = ("import sys; print('(EngineCore pid=1) INFO [kv_cache_utils.py:2084] Overriding num_gpu_blocks=11167 "
             "with num_gpu_blocks_override=9965', flush=True); sys.stderr.write('a warning\\n')")
    with serving.output_to(tmp_path / "engine.log") as log:
        subprocess.run([sys.executable, "-c", child], check=True)
    text = log.read_text(encoding="utf-8")
    assert "a warning" in text and vllm_engine.computed_blocks(text) == 11167
    shown = capfd.readouterr()
    assert "Overriding num_gpu_blocks=11167" in shown.out and "a warning" in shown.err


def test_checksums_catch_a_changed_or_missing_file(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "SHA256SUMS").write_text(f"{hashlib.sha256(b'{}').hexdigest()}  a.json\n{'0' * 64}  b.json\n",
                                         encoding="utf-8")
    assert profile_cmd.verify(tmp_path) == ["b.json"]
    (tmp_path / "a.json").write_text("{ }", encoding="utf-8")
    assert profile_cmd.verify(tmp_path) == ["a.json", "b.json"]
    assert profile_cmd.verify(tmp_path / "nowhere") == ["SHA256SUMS"]


def test_the_clock_is_locked_for_the_run_and_released_after_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profile_cmd, "measure_under_load", loaded)
    monkeypatch.setattr(G, "nvidia_smi", smi_refused())
    nv = fake_nvml(hold_mhz=1275)
    backend = G.Nvidia(nv, 0)
    with profile_cmd.clock_policy(backend, "measured") as policy:
        assert policy == {"policy": "measured"}
    assert nv.calls == []
    with pytest.raises(KeyError), profile_cmd.clock_policy(backend, ClockLock(lock=1200)) as policy:
        assert (policy["policy"], policy["lock_mhz"], policy["chosen_by"]) == ("locked", 1200, "manifest")
        raise KeyError("a failure inside the run")
    assert nv.calls == [("lock", 1200), ("unlock", 0)]          # released after a failure too
    with profile_cmd.clock_policy(backend, "auto", target_mhz=1200.0) as policy:
        assert (policy["lock_mhz"], policy["chosen_by"]) == (1200, "calibration")
    with profile_cmd.clock_policy(backend, "auto") as policy:     # the levels on the probe's 25 MHz grid
        assert (policy["lock_mhz"], policy["chosen_by"]) == (1275, "sweep")
        assert [v["mhz"] for v in policy["levels"]] == [1350, 1275]
    with profile_cmd.clock_policy(backend, "auto", target_mhz=1350.0) as policy:   # the calibration's level is
        assert policy["policy"] == "measured"                                    # above what this GPU holds
        why = policy["lock_refused"]
        assert why.startswith("the clock did not hold 1350 MHz under load (the clock averaged 1275 MHz") and \
            "the power cap held the clock down in 100 % of the samples" in why
    assert nv.state["locked"] is None
    refused = G.Nvidia(fake_nvml(lockable=False), 0)
    with profile_cmd.clock_policy(refused, "auto") as policy:
        assert policy["policy"] == "measured" and "the clock cannot be locked here" in policy["lock_refused"]


def test_profile_serving_locks_at_the_level_its_calibration_was_made_at(monkeypatch: pytest.MonkeyPatch,
                                                                       tmp_path: Path) -> None:
    nv = mock_host(monkeypatch)
    r = run(tmp_path)
    assert r.exit_code == 0, r.output
    clock = json.loads((tmp_path / "measurement" / "environment.json").read_text(encoding="utf-8"))["clock"]
    assert (clock["policy"], clock["lock_mhz"], clock["chosen_by"]) == ("locked", 1200, "calibration")
    assert clock["levels"][0]["held"] and nv.calls == [("lock", 1200), ("unlock", 0)]
    assert "the clock is locked at 1200 MHz, the level its calibration was made at" in r.stderr


def test_where_the_clock_cannot_be_locked_the_run_measures_it_and_says_why(monkeypatch: pytest.MonkeyPatch,
                                                                         tmp_path: Path) -> None:
    mock_host(monkeypatch, nvml=fake_nvml(lockable=False))
    r = run(tmp_path)
    assert r.exit_code == 0, r.output
    clock = json.loads((tmp_path / "measurement" / "environment.json").read_text(encoding="utf-8"))["clock"]
    assert clock["policy"] == "measured" and "locking needs root, or sudo without a password" in clock["lock_refused"]
    assert "the clock is measured instead" in r.stderr


@pytest.mark.parametrize(("use", "phrase"), [
    ("my-a100", "calibration.use: no calibration 'my-a100'; shipped: a100_sxm_40gb/c62ebe75"),
    ("broken.json", "is not a calibration ServeCast can read"),
])
def test_profile_serving_refuses_a_calibration_it_cannot_use_before_the_gpu(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use: str, phrase: str) -> None:
    """Without the calibration's clock, the serving locked at a level of its own and was compared with a prediction
    made at another. No host is mocked here, so a command that reached the GPU would end with status 3."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "broken.json").write_text("not json", encoding="utf-8")
    doc = manifest()
    doc["calibration"] = {"use": use, "transfer": True}
    r = run(tmp_path, doc=doc)
    assert r.exit_code == 2, r.output
    assert phrase in r.stderr
    assert use != "my-a100" or "profile serving locks the clock at the level the calibration was made at" in r.stderr


def test_profile_serving_reports_each_checkpoint_of_its_stop_rule(monkeypatch: pytest.MonkeyPatch,
                                                                   tmp_path: Path) -> None:
    mock_host(monkeypatch)
    r = run(tmp_path)
    assert r.exit_code == 0, r.output
    lines = [line for line in r.stderr.splitlines() if " requests after " in line]
    assert lines, r.stderr
    shape = (r"servecast profile serving: 8 clients: [\d,]+ requests after [\d.]+ min; (the widest 95 % interval is "
             r"that of .+, ±[\d.]+ % \(target ±[\d.]+ %\)|too few requests for a 95 % interval yet)")
    assert all(re.fullmatch(shape, line) for line in lines), lines


def test_a_checkpoint_line_names_the_interval_the_run_waits_for() -> None:
    checkpoint = {"n_at": 2431, "elapsed_s": 366.0, "worst": "TTFT", "worst_rel_hw": 0.142, "target": 0.1,
                  "stop": False}
    assert serving.checkpoint_line(64, checkpoint) == (
        "64 clients: 2,431 requests after 6.1 min; the widest 95 % interval is that of mean TTFT, ±14.2 % "
        "(target ±10 %)")
    early = {**checkpoint, "worst": None, "worst_rel_hw": float("inf")}
    assert serving.checkpoint_line(64, early).endswith("; too few requests for a 95 % interval yet")


# profile calibration ---------------------------------------------------------------------------------------

def probe_manifest(versions: str = "2026-10", **engine: Any) -> dict[str, Any]:
    """The engine of the paper's A100 probes: context 4,096, 512 sequences, 9,965 KV blocks."""
    worker = {"kv_cache": {"capacity": {"blocks": 9965}}, "scheduler": {"max_sequences": 512}}
    return {"engine": {"model": "meta-llama/Meta-Llama-3-8B", "hardware": "a100_sxm_40gb", "context_length": 4096,
                       "workers": {"aggregated": worker}, **engine},
            "calibration": {"use": "a100_sxm_40gb/d9cbf1d3/llama-3-8b"}, "versions": versions,
            "traffic": manifest()["traffic"]}


def run_probe(tmp_path: Path, *args: str, doc: dict[str, Any] | None = None) -> Any:
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(doc or probe_manifest()), encoding="utf-8")
    return CliRunner().invoke(app, ["profile", "calibration", "-c", str(tmp_path / "m.yaml"),
                                    "--output-dir", str(tmp_path / "profiled"), *args])


def serving_manifest(versions: str, blocks: int = 23488) -> dict[str, Any]:
    """A manifest written for serving, as the README's flow has it: vLLM's KV block count for the LLM it serves
    (23,488 is Qwen2.5-7B's on the A100, the shipped manifest's), and no probe settings."""
    return {"engine": {"model": "meta-llama/Meta-Llama-3-8B", "hardware": "a100_sxm_40gb",
                       "workers": {"aggregated": {"kv_cache": {"capacity": {"blocks": blocks}}}}},
            "calibration": {"use": "a100_sxm_40gb/d9cbf1d3/llama-3-8b"}, "versions": versions,
            "traffic": manifest()["traffic"]}


PAPER_ENGINE = {"context_length": 4096, "max_sequences": 512, "max_batched_tokens": 2048, "block_size": 16,
                "memory_fraction": 0.9}


@pytest.mark.parametrize(("blocks", "probed"), [(23488, 9965), (2870, 2870)])
def test_profile_calibration_probes_on_the_papers_engine_capped_by_the_manifests_blocks(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, blocks: int, probed: int) -> None:
    mock_host(monkeypatch)
    r = run_probe(tmp_path, doc=serving_manifest("2026-10", blocks))
    assert r.exit_code == 0, r.output
    engine = json.loads((tmp_path / "profiled" / "environment.json").read_text(encoding="utf-8"))["engine"]
    assert {k: engine[k] for k in PAPER_ENGINE} == PAPER_ENGINE and engine["kv_blocks"] == probed


def test_a_gpu_too_small_for_the_per_gpu_probe_is_refused_before_anything_runs(monkeypatch: pytest.MonkeyPatch,
                                                                               tmp_path: Path) -> None:
    mock_host(monkeypatch, nvml=fake_nvml(devices=(("NVIDIA GeForce RTX 3090 Ti", (8, 6)),)), probe=never)
    per_gpu_host(monkeypatch, tmp_path)
    doc = serving_manifest("2026-11")
    doc["engine"]["hardware"] = "rtx_3090_ti"
    r = run_probe(tmp_path, doc=doc)
    assert r.exit_code == 3, r.output
    assert "this GPU cannot hold the per-GPU probe, which runs every synthetic model with 9965 KV blocks" in r.stderr
    assert "at most" in r.stderr and not (tmp_path / "profiled").exists()


def test_profile_calibration_writes_profiling_data_that_calibrate_fits(monkeypatch: pytest.MonkeyPatch,
                                                                       tmp_path: Path) -> None:
    mock_host(monkeypatch)
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path / "data"))
    r = run_probe(tmp_path)
    assert r.exit_code == 0, r.output
    out = tmp_path / "profiled"
    assert sorted(p.name for p in out.iterdir()) == ["SHA256SUMS", "engine.log", "environment.json", "probe.json",
                                                     "profiling.json"]
    assert profile_cmd.verify(out) == []
    probe = formats.Probe.model_validate(json.loads((out / "probe.json").read_text(encoding="utf-8")))
    assert probe.audit.clean and probe.audit.compositions == 178
    data = formats.ProfilingData.model_validate(json.loads((out / "profiling.json").read_text(encoding="utf-8")))
    assert (data.cost_law, data.llm, data.reference_mhz, len(data.cells or [])) == (
        "d9cbf1d3", "meta-llama/Meta-Llama-3-8B", 1200.0, 976)
    assert "178 of 178 designed" in r.stdout and "976 cells at 1200 MHz" in r.stdout
    clock = json.loads((out / "environment.json").read_text(encoding="utf-8"))["clock"]   # the fake holds 1200 MHz:
    assert (clock["policy"], clock["lock_mhz"], clock["chosen_by"]) == ("locked", 1200, "sweep")   # swept to it
    assert [v["mhz"] for v in clock["levels"]] == [1350, 1275, 1200] and clock["levels"][-1]["held"]
    r = CliRunner().invoke(app, ["calibrate", "--data", str(out), "--name", "mine/llama-3-8b", "--output-dir",
                                 str(tmp_path / "calibration")])
    assert r.exit_code == 0, r.output
    m = probe_manifest()
    m["calibration"] = {"use": "mine/llama-3-8b"}
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(m), encoding="utf-8")
    r = CliRunner().invoke(app, ["predict", "-c", str(tmp_path / "p.yaml"), "--output-dir", str(tmp_path / "pred")])
    assert r.exit_code == 0, r.output


@pytest.mark.parametrize(("doc", "status", "phrase"), [
    (probe_manifest(gpu_clock_mhz=1200), 2, "engine.gpu_clock_mhz: the probe runs at the GPU's own clock"),
    # the manifest's context is for serving; its block count caps the probe engine, here below one probe request
    (serving_manifest("2026-10", blocks=200), 2, "200 blocks hold fewer tokens than one request of the context "
                                                 "length (4096)"),
])
def test_profile_calibration_refuses_before_the_engine_starts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                              doc: dict[str, Any], status: int, phrase: str) -> None:
    mock_host(monkeypatch, probe=never)
    r = run_probe(tmp_path, doc=doc)
    assert r.exit_code == status, r.output
    assert phrase in r.stderr and not (tmp_path / "profiled").exists()


def test_a_probe_whose_audit_has_flags_writes_no_profiling_data(monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path) -> None:
    mock_host(monkeypatch, probe=probe_engine(drift=0.10))
    r = run_probe(tmp_path)
    assert r.exit_code == 1, r.output
    assert "no profiling data was written" in r.stderr and "reference drift" in r.stdout
    out = tmp_path / "profiled"
    assert (out / "probe.json").is_file() and not (out / "profiling.json").exists()
    assert profile_cmd.verify(out) == []


def per_gpu_host(mp: pytest.MonkeyPatch, tmp_path: Path, *, grid_speedup: float = 1.0) -> None:
    """The per-GPU probe on the fake host: synthetic directories made from the base model's config, each synthetic
    model probed on the simulated engine priced by the shipped per-GPU calibration, and the shipped GEMM grid as the
    grid timed (`grid_speedup` makes every point that many times faster).

    Every synthetic model is priced at the calibration's eta_c, the one value the procedure fits them with. Priced
    each at its own eta_c read off the grid (0.57 to 0.85 around 0.70), as the law prices an LLM it predicts, the
    fitted KV efficiency, per-layer latency and per-token cost move by +1.2, +1.7 and -5 %, with a 3.5 % training
    error: a property of the per-GPU procedure, not of the probe."""
    import copy

    from tests.test_profile_probe import BASE_CONFIG

    calibration = copy.deepcopy(P.load_calibration("a100_sxm_40gb/c62ebe75"))
    grid = calibration["law"]["eta_c_grid"]
    grid.update(rule="curve", curve={"a": 1.0 / calibration["law"]["parameters"]["eta_c"], "b": 0.0})

    def synthetic_model(name: str) -> Path:
        out = tmp_path / "synthetic" / name.removeprefix("synthetic/")
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(PG.synthetic_config(name, BASE_CONFIG)), encoding="utf-8")
        return out

    def probe_synthetic(job: dict[str, Any]) -> dict[str, Any]:
        d = job["design"]
        eng = SimEngine(P.clock(calibration, geometry.resolve(job["model"])), np.random.default_rng(0), d["budget"],
                        d["max_num_seqs"], 1200.0)
        spans: list[dict[str, Any]] = []
        PR.drive(eng, d, lambda: len(eng.beats), spans)
        return {"spans": spans, "beats": [dataclasses.asdict(b) for b in eng.beats], "clock": eng.clock,
                "audit": PR.audit(eng.beats, spans, d, eng.clock, lock_mhz=job.get("lock_mhz"))}

    shipped = json.loads(P.shipped_profiling()["a100_sxm_40gb/c62ebe75"].read_text(encoding="utf-8"))

    def gemm_grid(shapes: Any, sample_clock: Any) -> list[dict[str, Any]]:
        return [{"model": r["model"], "reps": [[{**p, "s_per_iter": p["s_per_iter"] / grid_speedup, "iters": 100,
                                                 "sm_mhz": 1200.0} for p in rep] for rep in r["reps"]]}
                for r in shipped["gemm_grid"]["runs"]]
    mp.setattr(profile_cmd, "synthetic_model", synthetic_model)
    mp.setattr(profile_cmd, "probe_synthetic", probe_synthetic)
    mp.setattr(profile_cmd, "gemm_grid", gemm_grid)


def test_the_per_gpu_probe_fits_back_to_the_law_that_priced_it(monkeypatch: pytest.MonkeyPatch,
                                                               tmp_path: Path) -> None:
    mock_host(monkeypatch, probe=never)
    per_gpu_host(monkeypatch, tmp_path)
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path / "data"))
    r = run_probe(tmp_path, doc=serving_manifest("2026-11"))   # 3 of the 8 models could not hold its 23,488 blocks
    assert r.exit_code == 0, r.output
    out = tmp_path / "profiled"
    engine = json.loads((out / "environment.json").read_text(encoding="utf-8"))["engine"]
    assert {k: engine[k] for k in PAPER_ENGINE} == PAPER_ENGINE and engine["kv_blocks"] == 9965
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(["SHA256SUMS", "engine.log", "environment.json", "gemm.json", "profiling.json",
                            *(f"probe-{n.removeprefix('synthetic/')}.json" for n in PG.POOL)])
    assert profile_cmd.verify(out) == []
    data = formats.ProfilingData.model_validate(json.loads((out / "profiling.json").read_text(encoding="utf-8")))
    assert data.cost_law == "c62ebe75" and [m.llm for m in data.models or []] == list(PG.POOL)
    assert data.reference_mhz == 1200.0 and data.gemm_grid is not None and len(data.gemm_grid.runs) == 24
    r = CliRunner().invoke(app, ["calibrate", "--data", str(out), "--name", "mine/a100", "--output-dir",
                                 str(tmp_path / "calibration")])
    assert r.exit_code == 0, r.output
    got = json.loads((tmp_path / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    want = P.load_calibration("a100_sxm_40gb/c62ebe75")["law"]["parameters"]
    for name, value in want.items():
        fitted = got["law"]["parameters"][name]
        assert (abs(fitted / value - 1) < 0.01) if value else abs(fitted) < 1e-9, (name, fitted, value)


def test_a_gemm_grid_faster_than_the_gpu_writes_no_profiling_data(monkeypatch: pytest.MonkeyPatch,
                                                                  tmp_path: Path) -> None:
    mock_host(monkeypatch, probe=never)
    per_gpu_host(monkeypatch, tmp_path, grid_speedup=3.0)
    r = run_probe(tmp_path, doc=probe_manifest("2026-11"))
    assert r.exit_code == 1, r.output
    assert "the GEMM grid's audit has flags" in r.stderr and "of the GPU's peak" in r.stdout
    out = tmp_path / "profiled"
    assert (out / "gemm.json").is_file() and not (out / "profiling.json").exists()
