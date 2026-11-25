"""`servecast compare`: what it refuses, what it flags, and the numbers it writes.

Every resolved manifest here is built the way the two writers build theirs, from a YAML manifest through
`deployment.resolved_sections`; nothing on the measurement side is copied from the prediction.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from servecast.cli.app import app
from servecast.compare import score as S
from servecast.core import deployment, formats, gpus
from servecast.core import pool as pools
from servecast.core.manifest import Manifest

P_LENGTHS, G_LENGTHS = [120, 900, 33], [40, 7, 512]
DIGEST = pools.digest(P_LENGTHS, G_LENGTHS)
#: Meta-Llama-3-8B's config.json, the fields that define its architecture
LLAMA3_8B = {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
             "num_key_value_heads": 8, "vocab_size": 128256, "max_position_embeddings": 8192}
YAML: dict[str, Any] = {
    "engine": {"model": "meta-llama/Meta-Llama-3-8B", "hardware": "a100_sxm_40gb",
               "workers": {"aggregated": {"kv_cache": {"capacity": {"blocks": 9965}}}}},
    "traffic": {"source": {"type": "trace", "paths": ["traces/trace.jsonl"], "format": "mooncake"},
                "load": {"type": "concurrency", "concurrency": 40, "think_time_ms": 1000.0},
                "stop": {"steady_state": {}}},
}
SYNTHETIC_YAML: dict[str, Any] = {**YAML, "traffic": {
    **YAML["traffic"], "source": {"type": "synthetic", "input_tokens": {"mean": 200}, "output_tokens": {"mean": 50}}}}
COMPONENTS = {"cost_law": "2026-11", "calibration_procedure": "2026-11", "population_model": "2026-10",
              "workload_fitter": "2026-10", "length_sampler": "2026-11", "estimator": "2026-10"}


def sections(yaml_doc: dict[str, Any], *, config: dict[str, Any] = LLAMA3_8B, model: str | None = None,
             clock: float | None = None, components: dict[str, str] = COMPONENTS) -> dict[str, Any]:
    """What a writer records for `yaml_doc`: the shared resolver's engine and traffic sections."""
    m = Manifest.model_validate(yaml_doc)
    arch = deployment.architecture_from_config(config)
    d = deployment.resolve(m, gpus.get("a100_sxm_40gb"), arch["max_context"], "llm", architecture=arch)
    return deployment.resolved_sections(m, model=model or str(m.engine.model), architecture=arch,
                                        hardware="a100_sxm_40gb", deployment=d, components=components,
                                        gpu_clock_mhz=clock)


def estimand_of(man: dict[str, Any]) -> dict[str, Any]:
    """The stop rule a writer applied: the one its resolved manifest records."""
    return {"rule": "steady_state", **{k: v for k, v in man["traffic"]["stop"]["steady_state"].items()
                                       if v is not None}}


def summary(ttft: float, tpot: float, rps: float, *, ci: bool = True) -> dict[str, Any]:
    doc: dict[str, Any] = {"mean_ttft_ms": ttft, "p95_ttft_ms": 2 * ttft, "mean_tpot_ms": tpot,
                           "p95_tpot_ms": 1.5 * tpot, "output_throughput_tok_s": 100 * rps,
                           "request_throughput_rps": rps,
                           "num_ttft_samples": 4000, "num_tpot_samples": 4000, "duration_ms": 1e6}
    if ci:
        doc["ci95"] = {name: {"low": 0.95 * value, "high": 1.05 * value}
                       for name, value in (("mean_ttft_ms", ttft), ("mean_tpot_ms", tpot),
                                           ("request_throughput_rps", rps), ("p95_ttft_ms", 2 * ttft),
                                           ("p95_tpot_ms", 1.5 * tpot))}
    return doc


def prediction(yaml_doc: dict[str, Any] | None = None, *, components: dict[str, str] = COMPONENTS,
               **edit: Any) -> dict[str, Any]:
    man = sections(yaml_doc or YAML, clock=1200.0, components=components)
    man["calibration"] = {"use": "a100_sxm_40gb/c62ebe75"}
    man["versions"] = {"as_of": "2026-11", **components}
    doc = {"schema": "servecast-prediction/1",
           "points": [{"load": {"concurrency": n, "think_time_ms": 1000.0}, "summary": summary(t, 20.0, r)}
                      for n, t, r in ((40, 110.0, 30.0), (80, 220.0, 50.0))],
           "estimand": estimand_of(man),
           "validity": {"status": "inside"},
           "provenance": {"components": dict(components), "servecast_version": "0.1.0", "seed": 0,
                          "pool_sha256": DIGEST, "calibration": "a100_sxm_40gb/c62ebe75"},
           "resolved_manifest": man}
    doc.update(edit)
    return doc


def measurement(yaml_doc: dict[str, Any] | None = None, *, config: dict[str, Any] = LLAMA3_8B,
                model: str | None = None, components: dict[str, str] = COMPONENTS, **edit: Any) -> dict[str, Any]:
    if yaml_doc is None:
        yaml_doc = copy.deepcopy(YAML)
        yaml_doc["traffic"]["source"]["paths"] = ["/mnt/elsewhere/trace.jsonl"]
    man = sections(yaml_doc, config=config, model=model, components=components)
    man["profile"] = {"clock": "measured"}
    doc = {"schema": "servecast-measurement/1",
           "points": [{"load": {"concurrency": n, "think_time_ms": 1000.0}, "summary": summary(t, 25.0, r),
                       "converged": True, "stop_reason": "converged", "measured_clock_mhz": 1205.0,
                       "kv_blocks": 9965, "kv_blocks_computed": 10400}
                      for n, t, r in ((40, 100.0, 32.0), (80, 200.0, 40.0))],
           "estimand": estimand_of(man),
           "provenance": {"components": dict(components), "servecast_version": "0.1.0", "seed": 0,
                          "pool_sha256": DIGEST, "vllm_version": "0.25.1"},
           "resolved_manifest": man}
    doc.update(edit)
    return doc


def write(directory: Path, name: str, doc: dict[str, Any], *, pool: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(doc), encoding="utf-8")
    if pool:
        pools.write(directory / "pool.jsonl", P_LENGTHS, G_LENGTHS, {"source": "trace", "files": ["test"]})
    return directory


def pair(tmp_path: Path, pred: dict[str, Any] | None = None, meas: dict[str, Any] | None = None, *,
         pool: bool = True) -> tuple[Path, Path]:
    return (write(tmp_path / "pred", "prediction.json", pred or prediction(), pool=pool),
            write(tmp_path / "meas", "measurement.json", meas or measurement(), pool=pool))


def yaml_with(path: str, value: Any) -> dict[str, Any]:
    doc = copy.deepcopy(YAML)
    node = doc
    keys = path.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
    return doc


# scoring ------------------------------------------------------------------------------------------------------


def test_a_matching_pair_is_scored_point_by_point(tmp_path: Path) -> None:
    doc = S.score(*pair(tmp_path))
    formats.Comparison.model_validate(doc)
    first = doc["points"][0]
    assert first["load"] == {"concurrency": 40, "think_time_ms": 1000.0}
    ttft = first["metrics"]["mean_ttft_ms"]
    assert ttft["predicted"] == 110.0 and ttft["measured"] == 100.0
    assert ttft["signed_error_pct"] == pytest.approx(10.0, rel=1e-12)
    assert ttft["within_ci"] is False and ttft["measured_ci95"] == {"low": 95.0, "high": 105.0}
    assert first["metrics"]["request_throughput_rps"]["signed_error_pct"] == pytest.approx(-6.25, rel=1e-12)
    assert first["metrics"]["mean_tpot_ms"]["within_ci"] is False
    assert first["measured_certified"] is True and first["predicted_converged"] is True
    assert doc["flags"] == []
    assert doc["checks"] == {"manifest": True, "estimand": True, "pool": True, "sampler": None, "estimator": True}


def test_a_prediction_on_the_interval_bound_is_inside(tmp_path: Path) -> None:
    meas = measurement()
    meas["points"][0]["summary"]["ci95"]["mean_ttft_ms"] = {"low": 100.0, "high": 110.0}
    assert S.score(*pair(tmp_path, meas=meas))["points"][0]["metrics"]["mean_ttft_ms"]["within_ci"] is True


def test_trace_paths_and_what_only_one_side_reads_do_not_count(tmp_path: Path) -> None:
    meas = measurement(yaml_with("engine.workers.aggregated.kv_cache.capacity.memory_fraction", 0.95))
    assert S.score(*pair(tmp_path, meas=meas))["flags"] == []


@pytest.mark.parametrize(("path", "value", "named"), [
    ("engine.workers.aggregated.scheduler.max_sequences", 128, "scheduler.max_sequences"),
    ("engine.workers.aggregated.kv_cache.capacity.blocks", 9000, "capacity.blocks"),
    # blocks of 32 tokens, half as many (the same tokens): 9,965 blocks of 32 need 52.9 GiB with the weights, which the
    # resolver refuses on 40 GiB, and a block size that is not a multiple of 16 is refused as FlashAttention refuses it
    ("engine.workers.aggregated.kv_cache", {"block_size": 32, "capacity": {"blocks": 4982}}, "kv_cache.block_size"),
    ("engine.context_length", 4096, "engine.context_length"),
    ("traffic.seed", 1, "traffic.seed"),
    ("traffic.stop.steady_state", {"tpot": 0.1}, "steady_state.tpot"),
])
def test_a_different_manifest_is_refused_naming_the_field(tmp_path: Path, path: str, value: Any,
                                                          named: str) -> None:
    with pytest.raises(S.NotComparableError, match=re.escape(named)):
        S.score(*pair(tmp_path, meas=measurement(yaml_with(path, value))))


def test_a_different_architecture_is_refused_and_a_different_name_is_flagged(tmp_path: Path) -> None:
    other = {**LLAMA3_8B, "num_hidden_layers": 28}
    with pytest.raises(S.NotComparableError, match=r"engine\.architecture\.layers"):
        S.score(*pair(tmp_path, meas=measurement(config=other)))
    doc = S.score(*pair(tmp_path / "b", meas=measurement(model="/models/llama-3-8b")))
    assert any("named" in f and "/models/llama-3-8b" in f for f in doc["flags"])


def test_a_different_pool_is_refused(tmp_path: Path) -> None:
    meas = measurement()
    meas["provenance"]["pool_sha256"] = "0" * 64
    with pytest.raises(S.NotComparableError, match="request pools differ"):
        S.score(*pair(tmp_path, meas=meas))


def test_a_pair_without_pool_digests_is_refused(tmp_path: Path) -> None:
    pred, meas = prediction(), measurement()
    pred["provenance"]["pool_sha256"] = meas["provenance"]["pool_sha256"] = None
    with pytest.raises(S.NotComparableError, match="no request-pool digest"):
        S.score(*pair(tmp_path, pred, meas, pool=False))


def test_a_pool_file_that_is_not_the_recorded_pool_is_refused(tmp_path: Path) -> None:
    pred_dir, meas_dir = pair(tmp_path)
    pools.write(meas_dir / "pool.jsonl", [1, 2], [3, 4], {"source": "trace", "files": ["other"]})
    with pytest.raises(S.NotComparableError, match="not the pool its provenance records"):
        S.score(pred_dir, meas_dir)


def test_a_malformed_pool_file_is_refused_by_name(tmp_path: Path) -> None:
    pred_dir, meas_dir = pair(tmp_path)
    lines = (meas_dir / "pool.jsonl").read_text(encoding="utf-8").splitlines()
    lines[1] = '{"input_tokens": null, "output_tokens": 4}'
    (meas_dir / "pool.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(S.NotComparableError, match="not a request pool"):
        S.score(pred_dir, meas_dir)


@pytest.mark.parametrize(("component", "source"), [("length_sampler", "synthetic"), ("estimator", "trace"),
                                                   ("estimator", "synthetic")])
def test_a_different_sampler_or_estimator_is_refused(tmp_path: Path, component: str, source: str) -> None:
    yaml_doc = SYNTHETIC_YAML if source == "synthetic" else None
    meas = measurement(yaml_doc)
    meas["provenance"]["components"][component] = "2026-12"
    with pytest.raises(S.NotComparableError, match=component.replace("_", " ")):
        S.score(*pair(tmp_path, prediction(yaml_doc), meas))


def test_the_length_sampler_does_not_count_for_lengths_read_from_a_trace(tmp_path: Path) -> None:
    meas = measurement()
    meas["provenance"]["components"]["length_sampler"] = "2026-12"
    doc = S.score(*pair(tmp_path, meas=meas))
    assert doc["flags"] == [] and doc["checks"]["sampler"] is None


def test_two_predictions_are_refused(tmp_path: Path) -> None:
    pred_dir = write(tmp_path / "pred", "prediction.json", prediction())
    other = write(tmp_path / "other", "measurement.json", prediction())
    with pytest.raises(S.NotComparableError, match="not a measurement"):
        S.score(pred_dir, other)
    with pytest.raises(S.NotComparableError, match=r"no measurement\.json"):
        S.score(pred_dir, pred_dir)


@pytest.mark.parametrize(("where", "value"), [("request_throughput_rps", -50.0), ("mean_ttft_ms", math.inf)])
def test_a_negative_or_infinite_measurement_is_refused(tmp_path: Path, where: str, value: float) -> None:
    meas = measurement()
    meas["points"][0]["summary"][where] = value
    with pytest.raises(S.NotComparableError, match="finite non-negative"):
        S.score(*pair(tmp_path, meas=meas))


def test_an_interval_for_an_unknown_metric_is_refused(tmp_path: Path) -> None:
    meas = measurement()
    meas["points"][0]["summary"]["ci95"] = {"ttft": {"low": 1.0, "high": 2.0}}
    with pytest.raises(S.NotComparableError, match="not metrics"):
        S.score(*pair(tmp_path, meas=meas))


def test_a_measured_point_must_say_whether_it_converged(tmp_path: Path) -> None:
    meas = measurement()
    del meas["points"][0]["converged"]
    with pytest.raises(S.NotComparableError, match="converged"):
        S.score(*pair(tmp_path, meas=meas))


# pairing ------------------------------------------------------------------------------------------------------


def test_load_points_pair_by_clients_and_think_time(tmp_path: Path) -> None:
    meas = measurement()
    meas["points"][1]["load"]["concurrency"] = 120
    doc = S.score(*pair(tmp_path, meas=meas))
    assert [p["load"]["concurrency"] for p in doc["points"]] == [40]
    assert any("clients 80" in f and "predicted only" in f for f in doc["flags"])
    assert any("clients 120" in f and "measured only" in f for f in doc["flags"])
    meas["points"][0]["load"]["think_time_ms"] = 500.0
    with pytest.raises(S.NotComparableError, match="no load point in common"):
        S.score(*pair(tmp_path / "b", meas=meas))


def test_a_think_time_off_by_its_last_bit_still_pairs(tmp_path: Path) -> None:
    meas = measurement()
    for point in meas["points"]:
        point["load"]["think_time_ms"] = 1000.0 * (1 + 2.0 ** -52)
    assert len(S.score(*pair(tmp_path, meas=meas))["points"]) == 2


def test_a_load_point_listed_twice_is_refused(tmp_path: Path) -> None:
    meas = measurement()
    meas["points"][1]["load"] = dict(meas["points"][0]["load"])
    with pytest.raises(S.NotComparableError, match="twice"):
        S.score(*pair(tmp_path, meas=meas))


# flags --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("clock", "flagged"), [(1205.0, False), (1217.0, False), (1219.0, True), (1150.0, True),
                                                 (None, True), (math.nan, True)])
def test_a_clock_missing_or_beyond_the_tolerance_is_flagged(tmp_path: Path, clock: float | None,
                                                            flagged: bool) -> None:
    meas = measurement()
    meas["points"][0]["measured_clock_mhz"] = clock
    doc = S.score(*pair(tmp_path, meas=meas))
    assert any("clients 40" in f and "clock" in f for f in doc["flags"]) is flagged


def test_a_prediction_without_a_clock_is_flagged(tmp_path: Path) -> None:
    pred = prediction()
    del pred["resolved_manifest"]["engine"]["gpu_clock_mhz"]
    flags = S.score(*pair(tmp_path, pred=pred))["flags"]
    assert any("prediction records no GPU clock" in f for f in flags)


def test_another_vllm_version_is_flagged(tmp_path: Path) -> None:
    meas = measurement()
    meas["provenance"]["vllm_version"] = "0.26.0"
    assert any("vLLM 0.26.0" in f for f in S.score(*pair(tmp_path, meas=meas))["flags"])


def test_a_kv_capacity_unrecorded_or_not_served_is_flagged(tmp_path: Path) -> None:
    meas = measurement()
    meas["points"][0]["kv_blocks"] = 10400
    meas["points"][1]["kv_blocks_computed"] = 9000
    flags = S.score(*pair(tmp_path, meas=meas))["flags"]
    assert any("clients 40" in f and "ran with 10400" in f for f in flags)
    assert any("clients 80" in f and "exceed the 9000" in f for f in flags)
    meas["points"][0]["kv_blocks"] = None
    flags = S.score(*pair(tmp_path / "b", meas=meas))["flags"]
    assert any("clients 40" in f and "does not record the KV-cache blocks" in f for f in flags)


def test_an_unconverged_run_is_flagged_and_marked_on_either_side(tmp_path: Path) -> None:
    pred, meas = prediction(), measurement()
    meas["points"][1].update(converged=False, stop_reason="seconds_budget")
    pred["points"][0].update(converged=False, stop_reason="seconds_budget")
    doc = S.score(*pair(tmp_path, pred, meas))
    assert doc["points"][1]["measured_certified"] is False and doc["points"][0]["predicted_converged"] is False
    assert any("clients 80" in f and "measurement did not converge" in f for f in doc["flags"])
    assert any("clients 40" in f and "prediction did not converge" in f for f in doc["flags"])


def test_a_metric_without_an_error_is_flagged(tmp_path: Path) -> None:
    meas = measurement()
    meas["points"][0]["summary"]["request_throughput_rps"] = 0.0
    meas["points"][0]["summary"]["p95_tpot_ms"] = None
    doc = S.score(*pair(tmp_path, meas=meas))
    metrics = doc["points"][0]["metrics"]
    assert metrics["request_throughput_rps"]["signed_error_pct"] is None
    assert metrics["p95_tpot_ms"]["signed_error_pct"] is None
    assert any("clients 40" in f and "p95_tpot_ms" in f and "request_throughput_rps" in f for f in doc["flags"])


# the command --------------------------------------------------------------------------------------------------


def test_the_command_writes_the_comparison_and_exits_0(tmp_path: Path) -> None:
    pred_dir, meas_dir = pair(tmp_path)
    r = CliRunner().invoke(app, ["compare", str(pred_dir), str(meas_dir)])
    assert r.exit_code == 0, r.output
    assert "mean TTFT ms" in r.stdout and "+10.0" in r.stdout
    written = json.loads((pred_dir / "comparison.json").read_text(encoding="utf-8"))
    assert written["schema"] == "servecast-comparison/1"
    out = tmp_path / "c.json"
    r = CliRunner().invoke(app, ["compare", str(pred_dir), str(meas_dir), "--output", str(out), "--format", "json"])
    assert r.exit_code == 0 and json.loads(r.stdout) == json.loads(out.read_text(encoding="utf-8"))


def test_the_command_refuses_with_status_2_and_names_the_difference(tmp_path: Path) -> None:
    meas = measurement()
    meas["provenance"]["components"]["estimator"] = "2026-11"
    r = CliRunner().invoke(app, ["compare", *map(str, pair(tmp_path, meas=meas))])
    assert r.exit_code == 2 and "estimator differs" in r.stderr
    assert not (tmp_path / "pred" / "comparison.json").exists()


def test_a_real_prediction_pairs_with_a_measurement_of_the_same_yaml(tmp_path: Path) -> None:
    """The prediction side is real: `predict` on a manifest. The measurement side is what the profile writer records
    for the SAME manifest, through the shared resolver, with the LLM's real config.json; nothing is copied."""
    from tests.test_cli_predict import BASE, predict

    code, output, out = predict(tmp_path, BASE)
    assert code == 0, output
    pred = json.loads((out / "prediction.json").read_text(encoding="utf-8"))
    assert pred["resolved_manifest"]["engine"]["gpu_clock_mhz"] == 1200.0
    point = copy.deepcopy(pred["points"][0])
    point.update(converged=True, measured_clock_mhz=1210.0, kv_blocks=9965, kv_blocks_computed=10400)
    meas = measurement(copy.deepcopy(BASE), model="meta-llama/Meta-Llama-3-8B")
    meas.update(points=[point], estimand=pred["estimand"],
                provenance={**pred["provenance"], "vllm_version": "0.25.1"})
    meas_dir = write(tmp_path / "meas", "measurement.json", meas, pool=False)
    (meas_dir / "pool.jsonl").write_bytes((out / "pool.jsonl").read_bytes())
    doc = S.score(out, meas_dir)
    assert doc["flags"] == [] and doc["points"][0]["metrics"]["mean_ttft_ms"]["signed_error_pct"] == 0.0


# the stop rule ------------------------------------------------------------------------------------------------

NEWEST = {**COMPONENTS, "estimator": "2026-11"}


def test_a_measurement_stopped_by_a_looser_rule_is_refused(tmp_path: Path) -> None:
    """Both judge the 95th percentiles; the measurement stopped at 90 % half-widths, the prediction at 10 %."""
    loose = yaml_with("traffic.stop.steady_state", {"ttft_p95": 0.9, "tpot_p95": 0.9})
    loose["traffic"]["source"]["paths"] = ["/mnt/elsewhere/trace.jsonl"]
    with pytest.raises(S.NotComparableError, match=r"steady_state\.ttft_p95"):
        S.score(*pair(tmp_path, prediction(components=NEWEST), measurement(loose, components=NEWEST)))


def test_an_estimand_that_is_not_the_recorded_stop_rule_is_refused(tmp_path: Path) -> None:
    meas = measurement()
    meas["estimand"]["ttft"] = 0.5
    with pytest.raises(S.NotComparableError, match=r"stop rules differ at ttft.*0\.1.*0\.5"):
        S.score(*pair(tmp_path, meas=meas))
    pred = prediction()
    pred["estimand"]["tpot"] = 0.5
    with pytest.raises(S.NotComparableError, match="the prediction's estimand is not the stop rule"):
        S.score(*pair(tmp_path / "b", pred=pred))


# every field of the resolved manifest is compared, or left out on purpose ----------------------------------------

def _leaves(doc: dict[str, Any], prefix: str = "") -> list[str]:
    out = []
    for key, value in doc.items():
        if isinstance(value, dict) and value:
            out += _leaves(value, f"{prefix}{key}.")
        else:
            out.append(prefix + key)
    return out


def test_every_field_a_writer_records_is_compared_or_left_out_on_purpose() -> None:
    from servecast.core import manifest

    compared, left_out = manifest.SEMANTIC_FIELDS, manifest.NOT_SEMANTIC
    assert not set(compared) & set(left_out)
    for doc in (prediction(SYNTHETIC_YAML, components=NEWEST), measurement(SYNTHETIC_YAML, components=NEWEST),
                prediction(), measurement()):
        for leaf in _leaves(doc["resolved_manifest"]):
            owners = [p for p in (*compared, *left_out) if leaf == p or leaf.startswith(p + ".")]
            assert len(owners) == 1, f"{leaf}: compared or left out by {owners or 'nothing'}"


#: one change per compared field, each making a different experiment. Written out here, not read from SEMANTIC_FIELDS,
#: so a field dropped from that tuple fails this test.
DIFFERENT_EXPERIMENT = [
    ("engine.mode", "disaggregated"), ("engine.backend", "sglang"), ("engine.hardware", "h100_sxm"),
    ("engine.context_length", 4096), ("engine.architecture.layers", 28), ("engine.architecture.vocab", 32000),
    ("engine.workers.aggregated.parallelism.tensor", 2),
    ("engine.workers.aggregated.scheduler.max_batched_tokens", 4096),
    ("engine.workers.aggregated.scheduler.max_sequences", 128), ("engine.workers.aggregated.kv_cache.block_size", 32),
    ("engine.workers.aggregated.kv_cache.prefix_caching", True),
    ("engine.workers.aggregated.kv_cache.capacity.blocks", 9000),
    ("engine.workers.aggregated.kv_cache.capacity.type", "fraction"), ("traffic.seed", 1),
    ("traffic.stop.steady_state.ttft", 0.2), ("traffic.stop.steady_state.tpot_p95", 0.9),
]


@pytest.mark.parametrize(("path", "value"), DIFFERENT_EXPERIMENT)
def test_a_measurement_of_another_experiment_is_refused_naming_the_field(tmp_path: Path, path: str,
                                                                         value: Any) -> None:
    meas = measurement()
    node = meas["resolved_manifest"]
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    with pytest.raises(S.NotComparableError, match=re.escape(path)):
        S.score(*pair(tmp_path, meas=meas))


def test_a_field_this_version_does_not_compare_is_flagged_when_it_differs(tmp_path: Path) -> None:
    meas = measurement()
    meas["resolved_manifest"]["engine"]["workers"]["aggregated"]["scheduler"]["policy"] = "priority"
    meas["resolved_manifest"]["engine"]["workers"]["aggregated"]["chunked_prefill"] = False
    flags = S.score(*pair(tmp_path, meas=meas))["flags"]
    assert any("scheduler.policy" in f and "not compared" in f for f in flags)
    assert any("aggregated.chunked_prefill" in f for f in flags)


# what a certified number needs ----------------------------------------------------------------------------------

def test_a_certified_measurement_without_its_intervals_is_flagged(tmp_path: Path) -> None:
    meas = measurement()
    meas["points"][0]["summary"]["ci95"] = {}
    flags = S.score(*pair(tmp_path, meas=meas))["flags"]
    assert any("clients 40" in f and "no 95 % interval" in f and "mean_ttft_ms" in f and "request_throughput_rps" in f
               for f in flags)


def test_a_pool_matched_on_its_digest_alone_is_flagged(tmp_path: Path) -> None:
    flags = S.score(*pair(tmp_path, pool=False))["flags"]
    assert any("no pool.jsonl" in f and "recorded digest" in f for f in flags)


def test_a_capacity_that_is_not_a_whole_number_is_refused_not_a_crash(tmp_path: Path) -> None:
    pred, meas = prediction(), measurement()
    for doc in (pred, meas):
        doc["resolved_manifest"]["engine"]["workers"]["aggregated"]["kv_cache"]["capacity"]["blocks"] = "9965"
    with pytest.raises(S.NotComparableError, match=r"capacity\.blocks.*whole number"):
        S.score(*pair(tmp_path, pred, meas))


@pytest.mark.parametrize(("predicted", "measured", "paired"), [
    (1000.0000004999999, 1000.0000005, True),
    (1000.0, 1000.0 * (1 + 2.0 ** -52), True),
    (1000.0, 1000.4, False),
])
def test_think_times_pair_when_equal_to_float_precision(tmp_path: Path, predicted: float, measured: float,
                                                        paired: bool) -> None:
    pred, meas = prediction(), measurement()
    for doc, think in ((pred, predicted), (meas, measured)):
        for point in doc["points"]:
            point["load"]["think_time_ms"] = think
    if paired:
        assert len(S.score(*pair(tmp_path, pred, meas))["points"]) == 2
    else:
        with pytest.raises(S.NotComparableError, match="no load point in common"):
            S.score(*pair(tmp_path, pred, meas))


def test_a_vllm_version_servecast_does_not_model_is_flagged_even_when_both_sides_agree(tmp_path: Path) -> None:
    pred, meas = prediction(), measurement()
    pred["resolved_manifest"]["engine"]["backend_version"] = "0.26.0"
    meas["resolved_manifest"]["engine"]["backend_version"] = "0.26.0"
    meas["provenance"]["vllm_version"] = "0.26.0"
    flags = S.score(*pair(tmp_path, pred, meas))["flags"]
    assert any("vLLM 0.26.0" in f and "ServeCast models vLLM 0.25.1" in f for f in flags)
    meas["provenance"]["vllm_version"] = "0.25.1"
    flags = S.score(*pair(tmp_path / "b", pred, meas))["flags"]
    assert any("measured on vLLM 0.25.1" in f and "0.26.0" in f for f in flags)


@pytest.mark.parametrize(("where", "value"), [("converged", "false"), ("kv_blocks_computed", "10400"),
                                              ("summary.mean_ttft_ms", "100.0"),
                                              ("summary.ci95.mean_ttft_ms.low", "95.0")])
def test_a_value_of_the_wrong_type_is_refused_not_coerced(tmp_path: Path, where: str, value: str) -> None:
    meas = measurement()
    node = meas["points"][0]
    keys = where.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    with pytest.raises(S.NotComparableError, match="not a measurement"):
        S.score(*pair(tmp_path, meas=meas))


def test_a_resolved_manifest_without_what_every_writer_records_is_refused(tmp_path: Path) -> None:
    pred, meas = prediction(), measurement()
    pred["resolved_manifest"], meas["resolved_manifest"] = {}, {}
    with pytest.raises(S.NotComparableError, match=r"lacks .*engine\.architecture"):
        S.score(*pair(tmp_path, pred, meas))
    pred, meas = prediction(), measurement()
    del pred["resolved_manifest"]["engine"]["architecture"], meas["resolved_manifest"]["engine"]["architecture"]
    with pytest.raises(S.NotComparableError, match=r"lacks engine\.architecture"):
        S.score(*pair(tmp_path / "b", pred, meas))


def test_json_output_says_it_replaced_a_comparison(tmp_path: Path) -> None:
    pred_dir, meas_dir = pair(tmp_path)
    (pred_dir / "comparison.json").write_text("{}", encoding="utf-8")
    r = CliRunner().invoke(app, ["compare", str(pred_dir), str(meas_dir), "--format", "json"])
    assert r.exit_code == 0 and json.loads(r.stdout)["schema"] == "servecast-comparison/1"
    assert "replaces the earlier comparison" in r.stderr


def test_an_output_that_cannot_be_written_is_a_refused_resource(tmp_path: Path) -> None:
    pred_dir, meas_dir = pair(tmp_path)
    r = CliRunner().invoke(app, ["compare", str(pred_dir), str(meas_dir), "--output", str(tmp_path)])
    assert r.exit_code == 3 and "cannot write" in r.stderr and "Traceback" not in r.output


@pytest.mark.parametrize("architecture", [[], "", 0, False, {"x": 1}, {**{k: 1 for k in (
    "layers", "hidden", "ffn", "heads", "kv_heads", "kv_width", "vocab")}, "max_context": "8192"}])
def test_an_architecture_that_is_not_an_llms_is_refused_even_when_both_sides_agree(tmp_path: Path,
                                                                                   architecture: Any) -> None:
    pred, meas = prediction(), measurement()
    pred["resolved_manifest"]["engine"]["architecture"] = architecture
    meas["resolved_manifest"]["engine"]["architecture"] = architecture
    with pytest.raises(S.NotComparableError, match=r"engine\.architecture as .*not an LLM's architecture"):
        S.score(*pair(tmp_path, pred, meas))


def test_an_empty_compared_section_counts_as_missing(tmp_path: Path) -> None:
    pred, meas = prediction(), measurement()
    pred["resolved_manifest"]["traffic"]["stop"]["steady_state"] = {}
    meas["resolved_manifest"]["traffic"]["stop"]["steady_state"] = {}
    with pytest.raises(S.NotComparableError, match=r"lacks traffic\.stop\.steady_state"):
        S.score(*pair(tmp_path, pred, meas))
