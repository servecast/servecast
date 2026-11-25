"""A first session without a GPU, as a user would run it: every command that needs no GPU, through the installed
`servecast` program, in one scratch home, each step reading what the steps before it wrote.

Only two things come from outside the session: the profiling data a GPU host would have produced (the shipped A100
calibration's own), and, for `compare`, a measurement made by the GPU-free mock, since no command makes one without a
GPU. Everything else is the program's own output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from servecast.core import formats
from servecast.core.manifest import Manifest
from servecast.model import geometry
from servecast.model import predict as P
from servecast.model.predict import shipped_profiling
from servecast.profile import serving
from tests.mock_engine import FixedClock, MockEngine, VirtualTimeLoop

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.executable).parent
#: Qwen2.5-7B's config.json, the fields the cost laws read (the template's LLM)
QWEN25_7B = {"num_hidden_layers": 28, "hidden_size": 3584, "intermediate_size": 18944, "num_attention_heads": 28,
             "num_key_value_heads": 4, "vocab_size": 152064, "max_position_embeddings": 131072,
             "sliding_window": 131072, "use_sliding_window": False, "torch_dtype": "bfloat16"}


class Session:
    """A user's shell: one home, the installed program first on PATH, nothing of the caller's start-up files."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.env = {**os.environ, "HOME": str(home), "SERVECAST_CACHE_DIR": str(home / ".cache" / "servecast"),
                    "SERVECAST_DATA_DIR": str(home / ".local" / "share" / "servecast"), "NO_COLOR": "1",
                    "PATH": f"{BIN}{os.pathsep}{os.environ.get('PATH', '')}", "ZDOTDIR": str(home),
                    "XDG_CONFIG_HOME": str(home / ".config")}

    def run(self, line: str, status: int = 0) -> subprocess.CompletedProcess[str]:
        r = subprocess.run(["bash", "-o", "pipefail", "-c", line], cwd=self.home, env=self.env, capture_output=True,
                           text=True, timeout=1800)
        assert r.returncode == status, f"{line}: status {r.returncode}, expected {status}\n{r.stdout[-3000:]}\n" \
                                       f"{r.stderr[-3000:]}"
        assert "Traceback" not in r.stderr and "internal error" not in r.stderr, r.stderr[-3000:]
        return r

    def json(self, relative: str) -> dict:
        return json.loads((self.home / relative).read_text(encoding="utf-8"))


def _summary(doc: dict) -> list[dict]:
    return [{k: v for k, v in p["summary"].items() if k != "wall_time_ms"} for p in doc["points"]]


def test_a_first_session_without_a_gpu(tmp_path: Path) -> None:
    s = Session(tmp_path)

    # the program, and what it can do on this computer
    assert s.run("servecast --version").stdout.startswith("servecast ")
    s.run("servecast doctor")
    assert "a100_sxm_40gb/c62ebe75" in s.run("servecast list calibrations").stdout

    # a manifest, checked and explained
    s.run("servecast manifest init --template predict > m.yaml")
    s.run("servecast manifest validate -c m.yaml")
    assert "think" in s.run("servecast manifest explain traffic.load.think_time_ms").stdout

    # a prediction, then a load curve
    out = s.run("servecast predict -c m.yaml").stdout
    assert "converged" in out and " yes" in out
    first = s.json("servecast-output/prediction.json")
    formats.Prediction.model_validate(first, strict=True)
    assert first["points"][0]["converged"] and (tmp_path / "servecast-output" / "pool.jsonl").is_file()
    s.run("servecast predict -c m.yaml --sweep traffic.load.concurrency=16,32 --jobs 2 --output-dir curve")
    curve = s.json("curve/prediction.json")
    rps = [p["summary"]["request_throughput_rps"] for p in curve["points"]]
    assert [p["load"]["concurrency"] for p in curve["points"]] == [16, 32] and rps[0] < rps[1]

    # the workload
    assert "input_tokens" in s.run("servecast workload describe azure-conversation").stdout
    s.run("servecast workload sample -c m.yaml --out pool-sample.jsonl")
    assert (tmp_path / "pool-sample.jsonl").is_file()

    # a calibration of one's own, from profiling data, then a prediction with it
    (tmp_path / "profiling").mkdir()
    shutil.copyfile(shipped_profiling()["a100_sxm_40gb/c62ebe75"],
                    tmp_path / "profiling" / "profiling.json")
    s.run("servecast calibrate --data ./profiling --versions 2026-11 --name my-a100")
    assert "my-a100" in s.run("servecast list calibrations").stdout
    s.run("servecast predict -c m.yaml --with my-a100 --output-dir mine")
    mine = s.json("mine/prediction.json")
    # refitted from the shipped calibration's own data: the same calibration, so the same prediction
    for a, b in zip(_summary(first), _summary(mine), strict=True):
        assert a["mean_ttft_ms"] == pytest.approx(b["mean_ttft_ms"], rel=1e-3)
        assert a["request_throughput_rps"] == pytest.approx(b["request_throughput_rps"], rel=1e-3)

    # the paper
    assert "rq1" in s.run("servecast replicate list").stdout
    assert ", 0 differ" in s.run("servecast replicate verify").stdout
    s.run("servecast replicate predict --with a100_sxm_40gb/d9cbf1d3/llama-3-8b --to qwen2.5-7b --trace burstgpt "
          "--clients 66")

    # a measurement of the same manifest (made by the GPU-free mock), and the comparison
    manifest = Manifest.model_validate(yaml.safe_load((tmp_path / "m.yaml").read_text(encoding="utf-8")))
    setup = serving.prepare(manifest, "a100_sxm_40gb", QWEN25_7B)
    calibration = P.load_calibration("a100_sxm_40gb/c62ebe75")
    engine = MockEngine(P.clock(calibration, geometry.resolve("Qwen/Qwen2.5-7B")), setup.deployment)
    serving.serve(setup, tmp_path / "measurement", engine=engine,
                  sampler=FixedClock(float(calibration["conditions"]["clock_mhz"])),
                  prompt=list(range(max(setup.pool.input_tokens))), environment={"vllm": "0.25.1"},
                  timeout_s=7200.0, loop_factory=VirtualTimeLoop)
    table = s.run("servecast compare servecast-output measurement").stdout
    assert "mean TTFT ms" in table and "flag:" not in table, table
    comparison = s.json("servecast-output/comparison.json")
    formats.Comparison.model_validate(comparison)
    for name in ("mean_ttft_ms", "mean_tpot_ms", "request_throughput_rps"):
        error = comparison["points"][0]["metrics"][name]["signed_error_pct"]
        assert error is not None and abs(error) < 10, (name, error)

    # a mistake is named, with its fix, and nothing is written over
    refused = s.run("servecast predict -c m.yaml --set engine.backend_version=0.26.0 --overwrite", status=2)
    assert "engine.backend_version" in refused.stderr
    assert s.json("servecast-output/prediction.json") == first

    # shell completion
    assert "_SERVECAST_COMPLETE" in s.run("servecast completion show bash").stdout
