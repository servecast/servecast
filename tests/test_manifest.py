"""The manifest: what it accepts, what it refuses, and how it says so."""

from __future__ import annotations

import json
import textwrap

import pytest
import yaml

from servecast.core import manifest as m

PREDICT_WITH_A_TRACE = """
engine:
  model: meta-llama/Meta-Llama-3-8B
  hardware: a100_sxm_40gb
  workers: {aggregated: {parallelism: {tensor: 1}, kv_cache: {capacity: {blocks: 9965}}}}
calibration: {use: a100_sxm_40gb/c62ebe75}
traffic:
  source: {type: trace, paths: [azure_conv.jsonl], format: mooncake}
  load: {type: concurrency, concurrency: 64, think_time_ms: 1000}
  stop: {steady_state: {}}
"""

FIT_WITH_MEAN_AND_VARIANCE = """
engine:
  model: Qwen/Qwen2.5-7B
  hardware: h100_sxm
  workers: {aggregated: {parallelism: {tensor: 1}, kv_cache: {capacity: {blocks: 65806}}}}
versions: 2026-11
calibration: {data: ./h100-profiling-data/}
traffic:
  source:
    type: synthetic
    input_tokens: {mean: 1200}
    output_tokens: {mean: 250, variance: 40000}
  load: {type: concurrency, concurrency: 80, think_time_ms: 1000}
  stop: {steady_state: {}}
"""

ANOTHER_TOOLS_QUICKSTART = """
engine: {mode: aggregated, model: Qwen/Qwen3-32B-FP8, hardware: h200_sxm, backend: vllm, workers: {aggregated: {}}}
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 10}
  stop: {requests: 100}
"""


def problems(text: str) -> list[str]:
    _, found = m.validate(yaml.safe_load(textwrap.dedent(text)))
    return [str(p) for p in found]


def paths(text: str) -> set[str]:
    _, found = m.validate(yaml.safe_load(textwrap.dedent(text)))
    return {p.path for p in found}


@pytest.mark.parametrize("text", [PREDICT_WITH_A_TRACE, FIT_WITH_MEAN_AND_VARIANCE])
def test_the_plans_examples_are_valid(text: str) -> None:
    assert problems(text) == []


def test_a_file_written_for_another_tool_is_refused_with_every_reason_at_once() -> None:
    found = problems(ANOTHER_TOOLS_QUICKSTART)
    joined = "\n".join(found)
    assert "traffic.source.input_tokens: fixed lengths are not supported" in joined
    assert "traffic.source.output_tokens: fixed lengths are not supported" in joined
    assert "traffic.stop.requests: finite runs are not supported" in joined
    assert "traffic.load.think_time_ms" in joined
    assert len(found) >= 4


def test_a_refused_field_is_not_reported_twice_and_paths_are_what_the_user_wrote() -> None:
    found = paths(ANOTHER_TOOLS_QUICKSTART)
    assert not any(".synthetic." in p or ".trace." in p for p in found)  # no union tag in a path
    reported = problems(ANOTHER_TOOLS_QUICKSTART)
    assert sum("input_tokens" in p for p in reported) == 1


def test_a_typo_gets_a_suggestion() -> None:
    text = PREDICT_WITH_A_TRACE.replace("concurrency: 64", "concurency: 64")
    assert any("traffic.load.concurency: unknown field; did you mean concurrency?" in p for p in problems(text))


@pytest.mark.parametrize(
    ("extra", "path"),
    [
        ("router: {policy: kv_router}", "router"),
        ("evaluation: {sla: {ttft_ms: 500}}", "evaluation"),
        ("execution: {resources: {}}", "execution"),
    ],
)
def test_sections_another_tool_has_are_refused_by_name(extra: str, path: str) -> None:
    found = problems(PREDICT_WITH_A_TRACE + extra + "\n")
    assert f"{path}: not supported by ServeCast; remove it" in found


def test_engine_fields_not_modelled_are_refused_by_name() -> None:
    text = PREDICT_WITH_A_TRACE.replace(
        "workers: {aggregated: {parallelism: {tensor: 1}, kv_cache: {capacity: {blocks: 9965}}}}",
        "workers: {aggregated: {parallelism: {tensor: 1, replicas: 2}, timing: {type: fixed},"
        " kv_cache: {host_offload: {num_host_blocks: 8}, capacity: {blocks: 9965}}}, decode: {}}\n"
        "  speculation: {draft: 2}",
    )
    found = paths(text)
    assert {"engine.workers.aggregated.parallelism.replicas", "engine.workers.aggregated.timing",
            "engine.workers.aggregated.kv_cache.host_offload", "engine.workers.decode", "engine.speculation"} <= found


def test_tensor_parallelism_above_one_is_refused() -> None:
    text = PREDICT_WITH_A_TRACE.replace("tensor: 1", "tensor: 2")
    assert any("tensor parallelism above 1 is not supported yet" in p for p in problems(text))


@pytest.mark.parametrize(
    ("load", "expected"),
    [
        ("{type: poisson, requests_per_second: 8}", "closed workloads only"),
        ("{type: concurrency, concurrency: 64, think_time_ms: 0}", "greater than 0"),
    ],
)
def test_open_loop_and_zero_think_time_are_refused(load: str, expected: str) -> None:
    text = PREDICT_WITH_A_TRACE.replace("{type: concurrency, concurrency: 64, think_time_ms: 1000}", load)
    assert any(expected in p for p in problems(text))


ONE_LEVEL = "{type: concurrency, concurrency: 64, think_time_ms: 1000}"
LEVEL_16 = "{type: concurrency, concurrency: 16, think_time_ms: 1000}"


def test_the_load_is_one_level_or_a_list_of_levels_in_the_order_listed() -> None:
    one, _ = m.validate(yaml.safe_load(textwrap.dedent(PREDICT_WITH_A_TRACE)))
    listed = PREDICT_WITH_A_TRACE.replace(ONE_LEVEL, f"[{LEVEL_16}, {{type: concurrency, concurrency: 64, "
                                                     "think_time_ms: 500}]")
    many, found = m.validate(yaml.safe_load(textwrap.dedent(listed)))
    assert one is not None and one.traffic is not None
    assert many is not None and many.traffic is not None, found
    assert [(lv.concurrency, lv.think_time_ms) for lv in one.traffic.levels()] == [(64, 1000.0)]
    assert [(lv.concurrency, lv.think_time_ms) for lv in many.traffic.levels()] == [(16, 1000.0), (64, 500.0)]


@pytest.mark.parametrize(("load", "expected"), [
    ("[]", "traffic.load: List should have at least 1 item"),
    (f"[{LEVEL_16}, {LEVEL_16}]", "traffic.load: levels 0 and 1 are the same (16 clients, think time 1000 ms)"),
    (f"[{LEVEL_16}, {{type: concurrency, concurrency: 0, think_time_ms: 1000}}]",
     "traffic.load.1.concurrency: Input should be greater than or equal to 1"),
    (f"[{LEVEL_16}, {{type: concurrency, concurency: 4, think_time_ms: 1000}}]",
     "traffic.load.1.concurency: unknown field; did you mean concurrency?"),
    (f"[{LEVEL_16}, {{type: poisson, requests_per_second: 8}}]", "traffic.load.1.type: closed workloads only"),
])
def test_a_list_of_levels_is_refused_entry_by_entry_at_the_path_written(load: str, expected: str) -> None:
    found = problems(PREDICT_WITH_A_TRACE.replace(ONE_LEVEL, load))
    assert any(p.startswith(expected) for p in found), found


def test_explain_reads_a_levels_field_through_the_list() -> None:
    assert "type: Load | list of Load" in m.explain("traffic.load")
    assert m.explain("traffic.load.1.concurrency").splitlines()[1:] == \
        m.explain("traffic.load.concurrency").splitlines()[1:]


def test_session_workloads_are_refused() -> None:
    text = FIT_WITH_MEAN_AND_VARIANCE.replace("type: synthetic", "type: synthetic-session")
    assert any("session workloads are not supported" in p for p in problems(text))


def test_a_dotted_version_is_refused_with_the_reason() -> None:
    text = FIT_WITH_MEAN_AND_VARIANCE.replace("versions: 2026-11", "versions: 2026.10")
    assert any("hyphen" in p and "2026.1" in p for p in problems(text))


@pytest.mark.parametrize(
    ("versions", "ok"),
    [
        ("2026-10", True),
        ("2026-13", False),
        ("{as_of: 2026-11, estimator: 2026-10}", True),
        ("{cost_law: 2026-11b}", True),
        ("{as_of: 2026-11b}", False),
        ("{estimator: 2026.11}", False),
    ],
)
def test_version_forms(versions: str, ok: bool) -> None:
    text = FIT_WITH_MEAN_AND_VARIANCE.replace("versions: 2026-11", f"versions: {versions}")
    assert (problems(text) == []) is ok


def test_a_calibration_is_either_used_or_fitted() -> None:
    text = PREDICT_WITH_A_TRACE.replace("calibration: {use: a100_sxm_40gb/c62ebe75}",
                                        "calibration: {use: a100_sxm_40gb/c62ebe75, data: ./x}")
    assert any("either `use` or `data`" in p for p in problems(text))


def test_the_kv_capacity_is_an_input_given_in_blocks() -> None:
    assert problems(PREDICT_WITH_A_TRACE) == []
    without_blocks = PREDICT_WITH_A_TRACE.replace("capacity: {blocks: 9965}", "capacity: {type: fixed}")
    assert any(p.startswith("engine.workers.aggregated.kv_cache.capacity.blocks:") for p in problems(without_blocks))
    zero = PREDICT_WITH_A_TRACE.replace("blocks: 9965", "blocks: 0")
    assert any(p.startswith("engine.workers.aggregated.kv_cache.capacity.blocks:") for p in problems(zero))


def test_the_engines_own_kv_sizing_is_refused_with_how_to_get_the_blocks() -> None:
    text = PREDICT_WITH_A_TRACE.replace("capacity: {blocks: 9965}", "capacity: {type: default, memory_fraction: 0.9}")
    refusal = next(p for p in problems(text) if p.startswith("engine.workers.aggregated.kv_cache.capacity.type:"))
    assert "GPU KV cache size" in refusal and "--num-gpu-blocks-override" in refusal


@pytest.mark.parametrize("text", [
    PREDICT_WITH_A_TRACE.replace(", kv_cache: {capacity: {blocks: 9965}}", ""),
    PREDICT_WITH_A_TRACE.replace("workers: {aggregated: {parallelism: {tensor: 1}, kv_cache: {capacity: "
                                 "{blocks: 9965}}}}", "context_length: 4096"),
    "calibration: {data: ./profiling}\n",
])
def test_every_manifest_gives_its_kv_capacity_and_one_without_it_is_refused_with_the_fix(text: str) -> None:
    refusal = [p for p in problems(text) if p.startswith("engine.workers.aggregated.kv_cache.capacity:")]
    assert len(refusal) == 1 and "required" in refusal[0], problems(text)
    assert "GPU KV cache size" in refusal[0] and "--num-gpu-blocks-override" in refusal[0]


def test_a_missing_kv_capacity_is_reported_with_the_other_problems() -> None:
    text = PREDICT_WITH_A_TRACE.replace(", kv_cache: {capacity: {blocks: 9965}}", "")
    text = text.replace("think_time_ms", "thinktime_ms")
    assert {"engine.workers.aggregated.kv_cache.capacity", "traffic.load.thinktime_ms"} <= paths(text)


def test_explain_says_the_kv_capacity_is_required() -> None:
    assert "\n  required" in m.explain("engine.workers.aggregated.kv_cache.capacity")


def test_set_overrides_apply_left_to_right_and_replace_whole_values() -> None:
    raw = yaml.safe_load(PREDICT_WITH_A_TRACE)
    out, issues = m.apply_sets(raw, ["traffic.load.concurrency=80", "traffic.load.concurrency=96"])
    assert issues == [] and out["traffic"]["load"]["concurrency"] == 96
    out, _ = m.apply_sets(raw, ["traffic.load={type: concurrency, concurrency: 8, think_time_ms: 100}"])
    assert out["traffic"]["load"] == {"type": "concurrency", "concurrency": 8, "think_time_ms": 100}
    assert raw["traffic"]["load"]["concurrency"] == 64  # the input is not modified


def test_a_set_to_an_unknown_path_is_refused_by_validation() -> None:
    raw, _ = m.apply_sets(yaml.safe_load(PREDICT_WITH_A_TRACE), ["traffic.load.burst=3"])
    _, found = m.validate(raw)
    assert any(p.path == "traffic.load.burst" for p in found)


def test_a_malformed_set_is_reported() -> None:
    _, issues = m.apply_sets({}, ["no-equals-sign"])
    assert issues and "PATH=VALUE" in issues[0].message


def test_the_schema_refuses_unknown_fields_everywhere() -> None:
    schema = m.Manifest.model_json_schema()
    assert schema.get("additionalProperties") is False
    assert all(d.get("additionalProperties") is False for d in schema["$defs"].values() if d.get("type") == "object")


def test_a_set_inside_a_versions_date_keeps_the_date() -> None:
    """`--set versions.workload_fitter=...` on `versions: 2026-10` replaced the date with {} and so selected the
    newest components: the date is now kept as `as_of`."""
    out, issues = m.apply_sets({"versions": "2026-10"}, ["versions.workload_fitter=2026-10"])
    assert issues == [] and out["versions"] == {"as_of": "2026-10", "workload_fitter": "2026-10"}


def test_a_set_inside_any_other_value_is_refused_instead_of_replacing_it() -> None:
    raw = {"engine": {"model": "meta-llama/Meta-Llama-3-8B"}}
    out, issues = m.apply_sets(raw, ["engine.model.name=x"])
    assert out == raw and len(issues) == 1 and "engine.model" in str(issues[0]) and "not a mapping" in str(issues[0])


def test_a_key_written_twice_is_refused_naming_it(tmp_path) -> None:
    """yaml.safe_load keeps the last of two equal keys: a second `versions:` line used to win without a word."""
    path = tmp_path / "m.yaml"
    path.write_text("versions: 2026-11\ntraffic: {}\nversions: 2026-10\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError, match="duplicate key 'versions'"):
        m.load(path)
    nested = tmp_path / "n.yaml"
    nested.write_text("engine:\n  model: a\n  model: b\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError, match="duplicate key 'model'"):
        m.load(nested)


@pytest.mark.parametrize(("block_size", "accepted"), [(16, True), (32, True), (64, True), (7, False), (8, False),
                                                      (24, False)])
def test_a_block_size_flashattention_cannot_use_is_refused(block_size: int, accepted: bool) -> None:
    """vLLM 0.25.1's FlashAttention backend, which ServeCast pins on NVIDIA GPUs, takes multiples of 16 only."""
    text = PREDICT_WITH_A_TRACE.replace("kv_cache: {capacity:", f"kv_cache: {{block_size: {block_size}, capacity:")
    refusal = [p for p in problems(text) if p.startswith("engine.workers.aggregated.kv_cache.block_size:")]
    assert (refusal == []) == accepted, refusal
    if not accepted:
        assert "multiple of 16" in refusal[0] and "FlashAttention" in refusal[0]


def test_profile_gpu_is_refused_with_where_a_gpu_of_your_own_goes() -> None:
    """It was accepted and never read: a datasheet override nothing used."""
    refusal = [p for p in problems(PREDICT_WITH_A_TRACE + "profile: {gpu: {hbm_bandwidth_gb_s: 9999}}\n")
               if p.startswith("profile.gpu:")]
    assert len(refusal) == 1 and "gpus.json" in refusal[0]


@pytest.mark.parametrize(("value", "refused"), [("true", True), ("false", False)])
def test_prefix_caching_on_is_refused_by_validation(value: str, refused: bool) -> None:
    """Refused by predict only, and turned off by profile serving: now refused by every command alike."""
    text = PREDICT_WITH_A_TRACE.replace("kv_cache: {capacity:", f"kv_cache: {{prefix_caching: {value}, capacity:")
    refusal = [p for p in problems(text) if p.startswith("engine.workers.aggregated.kv_cache.prefix_caching:")]
    assert bool(refusal) == refused and (not refused or "not modelled" in refusal[0])


def test_an_unknown_gpu_is_refused_naming_the_known_ones_and_where_yours_goes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path))
    text = PREDICT_WITH_A_TRACE.replace("hardware: a100_sxm_40gb", "hardware: h100")
    refusal = [p for p in problems(text) if p.startswith("engine.hardware:")]
    assert len(refusal) == 1 and "a100_sxm_40gb" in refusal[0] and "gpus.json" in refusal[0]


def test_a_gpu_of_your_own_is_known_once_your_gpus_json_lists_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path))
    l40s = {"device_name": "NVIDIA L40S", "memory_gib": 48, "sm_count": 142, "peak_flop_per_s": 362e12,
            "memory_bandwidth_bytes_per_s": 0.864e12, "boost_clock_mhz": 2520}
    (tmp_path / "gpus.json").write_text(json.dumps({"schema": "servecast-gpus/1", "gpus": {"l40s": l40s}}),
                                        encoding="utf-8")
    text = PREDICT_WITH_A_TRACE.replace("hardware: a100_sxm_40gb", "hardware: l40s")
    assert not [p for p in problems(text) if p.startswith("engine.hardware:")]


@pytest.mark.parametrize("text", [
    PREDICT_WITH_A_TRACE.replace("  hardware: a100_sxm_40gb\n", ""),
    "calibration: {data: ./profiling}\n",
])
def test_every_manifest_names_its_gpu_and_one_without_it_is_refused_with_what_it_is_for(text: str) -> None:
    """The GPU predict simulates, and the GPU the profile commands measure and calibrate fits: never guessed."""
    refusal = [p for p in problems(text) if p.startswith("engine.hardware:")]
    assert len(refusal) == 1 and "required" in refusal[0] and "servecast list gpus" in refusal[0], problems(text)
    assert "simulates" in refusal[0] and "gpus.json" in refusal[0]


def test_explain_says_the_gpu_is_required() -> None:
    assert "\n  required" in m.explain("engine.hardware")
