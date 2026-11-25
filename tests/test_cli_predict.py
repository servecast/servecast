"""`servecast predict`: the paper's and the newest law's numbers through the command line, and every refusal with its
exit status."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from servecast.cli.app import app
from servecast.core import versions
from servecast.core.formats import Prediction

RUN = CliRunner()
GOLDENS = Path(__file__).parent / "goldens"
PAPER_DATA = Path(__file__).resolve().parents[1] / "src" / "servecast" / "data" / "paper"
HOMES = {"paper.json": PAPER_DATA / "campaigns.json", "tables.json": PAPER_DATA / "tables.json",
         "newest-a100.json": GOLDENS / "newest-a100.json"}


def campaign(name: str, gpu: str, llm: str, trace: str) -> dict:
    doc = json.loads(HOMES[name].read_text(encoding="utf-8"))
    return next(c for c in doc["campaigns"] if c.get("gpu", doc.get("gpu")) == gpu and c["llm"] == llm
                and c["trace"] == trace)


def manifest_for(camp: dict, gpu: str, calibration: str, versions: str, concurrency: int,
                 transfer: bool = False) -> dict:
    s = camp["prediction"]
    return {"engine": {"model": camp["llm"], "hardware": gpu,
                       "workers": {"aggregated": {"scheduler": {"max_batched_tokens": s["max_batched_tokens"],
                                                                "max_sequences": s["max_sequences"]},
                                                  "kv_cache": {"block_size": s["block_size"], "prefix_caching": False,
                                                               "capacity": {"blocks": s["kv_blocks"]}}}}},
            "calibration": {"use": calibration, "transfer": transfer}, "versions": versions,
            "traffic": {"source": {"type": "trace", "paths": [camp["trace"]], "format": "lengths"},
                        "load": {"type": "concurrency", "concurrency": concurrency, "think_time_ms": 1000},
                        "stop": {"steady_state": {}}, "seed": s["seed"]}}


def predict(tmp_path: Path, manifest: dict, *args: str) -> tuple[int, str, Path]:
    cfg = tmp_path / "m.yaml"
    cfg.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    out = tmp_path / "out"
    r = RUN.invoke(app, ["predict", "-c", str(cfg), "--output-dir", str(out), *args])
    return r.exit_code, r.output, out


@pytest.mark.exact
def test_the_papers_prediction_through_the_command_line(tmp_path: Path) -> None:
    camp = campaign("paper.json", "a100_sxm_40gb", "deepseek-ai/deepseek-llm-7b-base", "azure-conversation")
    code, output, out = predict(tmp_path, manifest_for(camp, "a100_sxm_40gb",
                                                       "a100_sxm_40gb/d9cbf1d3/deepseek-llm-7b-base", "2026-10", 12))
    assert code == 0, output
    doc = json.loads((out / "prediction.json").read_text())
    Prediction.model_validate(doc)
    want = next(c for c in camp["cells"] if c["concurrency"] == 12)["predicted"][camp["llm"]]
    s = doc["points"][0]["summary"]
    assert s["mean_ttft_ms"] == want["mean_ttft_s"] * 1e3 and s["mean_tpot_ms"] == want["mean_tpot_s"] * 1e3
    assert s["request_throughput_rps"] == want["throughput_rps"]
    assert doc["provenance"]["components"] == {**versions.PAPER}  # the paper's components, by name


@pytest.mark.exact
def test_the_newest_laws_prediction_through_the_command_line(tmp_path: Path) -> None:
    camp = campaign("newest-a100.json", "a100_sxm_40gb", "meta-llama/Llama-3.2-3B", "burstgpt")
    cell = camp["cells"][0]
    code, output, out = predict(tmp_path, manifest_for(camp, "a100_sxm_40gb", "a100_sxm_40gb/c62ebe75", "2026-11",
                                                       cell["concurrency"], transfer=True))
    assert code == 0, output
    s = json.loads((out / "prediction.json").read_text())["points"][0]["summary"]
    want = cell["predicted"]
    assert s["mean_ttft_ms"] == want["mean_ttft_s"] * 1e3 and s["p95_ttft_ms"] == want["p95_ttft_s"] * 1e3
    assert s["p95_tpot_ms"] == want["p95_tpot_s"] * 1e3 and s["num_ttft_samples"] == want["ttft_samples"]
    assert s["request_throughput_rps"] == want["throughput_rps"]


BASE = {"engine": {"model": "meta-llama/Meta-Llama-3-8B", "hardware": "a100_sxm_40gb",
                   "workers": {"aggregated": {"kv_cache": {"capacity": {"blocks": 9965}}}}},
        "calibration": {"use": "a100_sxm_40gb/d9cbf1d3/llama-3-8b"}, "versions": "2026-10",
        "traffic": {"source": {"type": "trace", "paths": ["azure-conversation"], "format": "lengths"},
                    "load": {"type": "concurrency", "concurrency": 4, "think_time_ms": 1000},
                    "stop": {"steady_state": {}}}}


def _with(path: str, value: object) -> dict:
    doc = json.loads(json.dumps(BASE))
    node = doc
    keys = path.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    if value is None:
        node.pop(keys[-1], None)
    else:
        node[keys[-1]] = value
    return doc


@pytest.mark.parametrize("manifest, args, status, message", [
    (_with("engine.workers.aggregated.kv_cache.capacity", None), (), 2,
     "kv_cache.capacity: required: write {blocks: N}"),
    (_with("engine.hardware", "h100_sxm"), (), 2, "engine.hardware: h100_sxm, but the calibration"),
    (_with("calibration.use", "a100_sxm_40gb/c62ebe75"), (), 2, "predates cost law c62ebe75, released 2026-11"),
    (_with("traffic.stop.steady_state", {"ttft_p95": 0.1}), (), 2, "does not judge percentiles"),
    (_with("calibration.use", "nowhere/2026-10/x"), (), 2, "no calibration"),
    (BASE, ("--sweep", "engine.model=a,b"), 2, "cannot be swept"),
    (_with("traffic.source.paths", ["no-such-trace.jsonl"]), (), 2, "no such file"),
    (_with("engine.workers.aggregated.kv_cache.capacity", {"blocks": 2}), (), 2, "fewer tokens than one request"),
    (_with("engine.backend_version", "0.26.0"), (), 2, "engine.backend_version: 0.26.0: ServeCast models vLLM 0.25.1"),
    # the audit's case: --to keeps a block count that, for DeepSeek-LLM-7B, needs 172 GiB of KV on a 40 GiB A100
    (_with("engine.workers.aggregated.kv_cache.capacity", {"blocks": 23488}),
     ("--to", "deepseek-ai/deepseek-llm-7b-base"), 2, "at most 3264 blocks fit"),
    (_with("engine.context_length", 0), (), 2, "engine.context_length"),
    # the manifest audit's case: one request of the trace fits, and the workload law's fit ended in an internal error
    (_with("engine.context_length", 64), (), 2, "a pool needs at least 100 requests"),
    # Typer kept only the last --sweep: the first was dropped without a word
    (BASE, ("--sweep", "traffic.load.think_time_ms=500,1000", "--sweep", "traffic.load.concurrency=2,4"), 2,
     "--sweep: one field per run"),
    (_with("traffic.load", [{"type": "concurrency", "concurrency": 2, "think_time_ms": 1000},
                            {"type": "concurrency", "concurrency": 4, "think_time_ms": 1000}]),
     ("--sweep", "traffic.load.concurrency=8,16"), 2, "--sweep: traffic.load lists 2 levels"),
])
def test_a_manifest_that_cannot_run_is_refused_with_its_fix(tmp_path: Path, manifest: dict, args: tuple, status: int,
                                                            message: str) -> None:
    code, output, _ = predict(tmp_path, manifest, *args)
    assert code == status and message in output, output


def test_a_used_output_directory_is_refused_without_overwrite(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "prediction.json").write_text("{}")
    code, output, _ = predict(tmp_path, BASE)
    assert code == 3 and "--overwrite" in output


def test_a_sweep_predicts_each_load_point(tmp_path: Path) -> None:
    code, output, out = predict(tmp_path, BASE, "--sweep", "traffic.load.concurrency=2,4", "--format", "json")
    assert code == 0, output
    doc = json.loads((out / "prediction.json").read_text())
    assert [p["load"]["concurrency"] for p in doc["points"]] == [2, 4]
    assert all(p["converged"] for p in doc["points"])
    assert (out / "pool.jsonl").is_file()


def test_a_list_of_levels_predicts_what_a_sweep_over_the_same_levels_predicts(tmp_path: Path) -> None:
    """Each level is predicted on its own, with the run's seed: the list and the sweep are the same experiment."""
    levels = [{"type": "concurrency", "concurrency": 4, "think_time_ms": 1000},
              {"type": "concurrency", "concurrency": 2, "think_time_ms": 1000}]
    (tmp_path / "list").mkdir()
    (tmp_path / "sweep").mkdir()
    code, output, listed = predict(tmp_path / "list", _with("traffic.load", levels))
    assert code == 0, output
    code, output, swept = predict(tmp_path / "sweep", BASE, "--sweep", "traffic.load.concurrency=4,2")
    assert code == 0, output
    a, b = (json.loads((d / "prediction.json").read_text())["points"] for d in (listed, swept))
    assert [p["load"] for p in a] == [p["load"] for p in b] == [{"concurrency": 4, "think_time_ms": 1000.0},
                                                                {"concurrency": 2, "think_time_ms": 1000.0}]

    def numbers(point: dict) -> dict:
        return {k: v for k, v in point["summary"].items() if k != "wall_time_ms"}  # the computer's time, not a result
    assert [numbers(p) for p in a] == [numbers(p) for p in b]


def test_capture_per_request_writes_the_window_requests(tmp_path: Path) -> None:
    code, output, out = predict(tmp_path, BASE, "--capture-per-request")
    assert code == 0, output
    doc = json.loads((out / "prediction.json").read_text())
    lines = (out / "requests.jsonl").read_text().splitlines()
    assert len(lines) == doc["points"][0]["summary"]["num_ttft_samples"]
    first = json.loads(lines[0])
    assert {"input_tokens", "output_tokens", "ttft_ms", "tpot_ms", "preemptions"} <= set(first)
    s = doc["points"][0]["summary"]
    assert s["output_throughput_tok_s"] > s["request_throughput_rps"] > 0


LLAMA3_8B_CONFIG = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 32, "hidden_size": 4096,
                    "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8,
                    "vocab_size": 128256, "max_position_embeddings": 8192, "torch_dtype": "bfloat16"}


def _local_llm(tmp_path: Path, name: str, config: dict) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def test_an_llm_in_a_local_directory_keeps_its_name_and_its_calibration(tmp_path: Path) -> None:
    """The calibration's own LLM, read from a local copy of its config.json: not a transfer, and recorded under the
    name the manifest gives (which a measurement of the same manifest records too)."""
    llm = _local_llm(tmp_path, "Meta-Llama-3-8B", LLAMA3_8B_CONFIG)
    code, output, out = predict(tmp_path, _with("engine.model", str(llm)))
    assert code == 0, output
    doc = json.loads((out / "prediction.json").read_text())
    assert doc["resolved_manifest"]["engine"]["model"] == str(llm)
    assert doc["validity"] == {"status": "inside", "reasons": []}


def test_a_local_llm_the_cost_laws_do_not_model_is_refused_with_its_reason(tmp_path: Path) -> None:
    llm = _local_llm(tmp_path, "moe", {**LLAMA3_8B_CONFIG, "num_local_experts": 8, "num_experts_per_tok": 2})
    code, output, _ = predict(tmp_path, _with("engine.model", str(llm)))
    assert code == 2 and "mixture-of-experts" in output, output


def test_overwrite_removes_what_described_the_previous_prediction(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    for name in ("prediction.json", "comparison.json", "requests.jsonl", "notes.txt"):
        (out / name).write_text("old", encoding="utf-8")
    code, output, out = predict(tmp_path, BASE, "--overwrite")
    assert code == 0, output
    assert not (out / "comparison.json").exists() and not (out / "requests.jsonl").exists()
    assert (out / "notes.txt").read_text(encoding="utf-8") == "old"
    assert "removed" in output and "comparison.json" in output


def test_the_resolved_manifest_runs_again_as_a_manifest(tmp_path: Path) -> None:
    code, output, out = predict(tmp_path, BASE)
    assert code == 0, output
    first = json.loads((out / "prediction.json").read_text())
    (tmp_path / "again").mkdir()
    code, output, out = predict(tmp_path / "again", first["resolved_manifest"])
    assert code == 0, output
    again = json.loads((out / "prediction.json").read_text())
    for doc in (first, again):
        for point in doc["points"]:
            del point["summary"]["wall_time_ms"]          # the time this computer took, not a result
    assert again["points"] == first["points"] and again["resolved_manifest"] == first["resolved_manifest"]


def test_an_architecture_that_is_not_the_llms_is_refused(tmp_path: Path) -> None:
    arch = {"layers": 28, "hidden": 4096, "ffn": 14336, "heads": 32, "kv_heads": 8, "kv_width": 1024,
            "vocab": 128256, "max_context": 8192}
    code, output, _ = predict(tmp_path, _with("engine.architecture", arch))
    assert code == 2 and "engine.architecture" in output and "layers" in output, output


def test_an_output_directory_that_cannot_be_written_is_a_refused_resource(tmp_path: Path) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("", encoding="utf-8")
    cfg = tmp_path / "m.yaml"
    cfg.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    r = RUN.invoke(app, ["predict", "-c", str(cfg), "--output-dir", str(blocker / "out")])
    assert r.exit_code == 3 and "cannot write" in r.output and "Traceback" not in r.output, r.output


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="root writes anywhere")
def test_a_read_only_output_directory_is_refused_before_the_prediction_runs(tmp_path: Path) -> None:
    """mkdir(exist_ok=True) succeeds on an existing read-only directory, so it was found only when the outputs were
    written, after the load points had run."""
    out = tmp_path / "ro"
    out.mkdir()
    out.chmod(0o555)
    cfg = tmp_path / "m.yaml"
    cfg.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    try:
        r = RUN.invoke(app, ["predict", "-c", str(cfg), "--output-dir", str(out)])
    finally:
        out.chmod(0o755)
    assert r.exit_code == 3 and "cannot write" in r.output and "load point" not in r.output, r.output


@pytest.mark.parametrize(("to", "sets", "original", "warns"), [
    ("Qwen/Qwen2.5-3B", [], "meta-llama/Meta-Llama-3-8B", True),
    ("Qwen/Qwen2.5-3B", ["engine.workers.aggregated.kv_cache.capacity.blocks=9965"], "meta-llama/Meta-Llama-3-8B",
     False),
    ("meta-llama/Meta-Llama-3-8B", [], "meta-llama/Meta-Llama-3-8B", False),  # --to the manifest's own LLM
    (None, [], "meta-llama/Meta-Llama-3-8B", False),
])
def test_to_warns_when_it_keeps_a_block_count_set_for_another_llm(to: str | None, sets: list[str], original: str,
                                                                   warns: bool) -> None:
    from servecast.cli import predict_cmd

    warning = predict_cmd.carried_capacity(to, sets, original)
    assert (warning is not None) == warns
    if warning:
        assert original in warning and "capacity.blocks=" in warning


def test_the_carried_block_count_warning_reaches_the_user_and_the_prediction(tmp_path: Path) -> None:
    code, output, out = predict(tmp_path, BASE, "--to", "Qwen/Qwen2.5-3B")
    assert code == 0, output
    assert "warning: --to Qwen/Qwen2.5-3B: the KV-cache block count" in output
    warnings = json.loads((out / "prediction.json").read_text(encoding="utf-8"))["warnings"]
    assert any("block count" in w for w in warnings), warnings


# an input error is the user's (exit 2), never an internal error -------------------------------------------------

@pytest.mark.parametrize("body", ["{not json", "[1, 2]"])
def test_a_calibration_file_that_is_not_one_is_an_input_error(tmp_path: Path, body: str) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(body, encoding="utf-8")
    code, output, _ = predict(tmp_path, BASE, "--with", str(bad))
    assert code == 2 and "calibration.use" in output and "internal error" not in output, output


def test_workload_sample_reports_an_unknown_version_as_an_input_error(tmp_path: Path) -> None:
    cfg = tmp_path / "m.yaml"
    cfg.write_text(yaml.safe_dump({k: v for k, v in BASE.items() if k != "versions"}), encoding="utf-8")
    r = RUN.invoke(app, ["workload", "sample", "-c", str(cfg), "--set", "versions.cost_law=2099-01",
                         "--out", str(tmp_path / "pool.jsonl")])
    assert r.exit_code == 2 and "versions" in r.output and "internal error" not in r.output, r.output


def test_timeout_also_bounds_the_workload_law_fit(tmp_path: Path) -> None:
    """The fit before the load points ran unbounded: the guide's synthetic example fitted for minutes under
    --timeout 60. The default 10,000-request synthetic pool cannot be fitted in 1 s."""
    doc = _with("traffic.source", {"type": "synthetic", "input_tokens": {"mean": 1200},
                                   "output_tokens": {"mean": 250, "variance": 40000}})
    doc = {**doc, "versions": "2026-11", "calibration": {"use": "a100_sxm_40gb/c62ebe75", "transfer": True}}
    code, output, out = predict(tmp_path, doc, "--timeout", "1")
    assert code == 124 and "did not finish in 1 s" in output and not (out / "prediction.json").exists(), output
