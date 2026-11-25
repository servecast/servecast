"""The resolver both writers share: what an unset field becomes, and the architecture identity."""

from __future__ import annotations

import copy
import json
import re
import statistics as st
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from servecast.core import deployment, gpus
from servecast.core.manifest import Capacity, KvCache, Manifest, ManifestError
from servecast.model import geometry
from servecast.model import run as R

MINIMAL = {"engine": {"model": "meta-llama/Meta-Llama-3-8B", "workers": {"aggregated": {"kv_cache": {
    "capacity": {"blocks": 9965}}}}},
    "traffic": {"source": {"type": "trace", "paths": ["burstgpt"]},
                "load": {"type": "concurrency", "concurrency": 4, "think_time_ms": 1000}, "stop": {"steady_state": {}}}}
#: the architecture MINIMAL predicts, as both writers read it
LLAMA8B = R.architecture(geometry.KNOWN["meta-llama/Meta-Llama-3-8B"])


def resolved() -> dict:
    m = Manifest.model_validate(MINIMAL)
    d = deployment.resolve(m, gpus.get("a100_sxm_40gb"), 8192, "llm", architecture=LLAMA8B)
    return deployment.resolved_sections(m, model="m", architecture={}, hardware="a100_sxm_40gb", deployment=d,
                                        components={"estimator": "2026-10"})


@pytest.mark.parametrize(("field", "doc_says", "path", "value"), [
    (KvCache.model_fields["prefix_caching"], "Unset: off", ("kv_cache", "prefix_caching"), False),
    (KvCache.model_fields["block_size"], "Unset: 16", ("kv_cache", "block_size"), 16),
    (Capacity.model_fields["memory_fraction"], "Unset: 0.9,", ("kv_cache", "capacity", "memory_fraction"), 0.9),
])
def test_an_unset_field_becomes_what_its_doc_says(field, doc_says: str, path: tuple[str, ...], value) -> None:
    assert doc_says in (field.description or "")
    node = resolved()["engine"]["workers"]["aggregated"]
    for key in path:
        node = node[key]
    assert node == value


def test_the_vllm_defaults_and_the_version_are_filled_in() -> None:
    engine = resolved()["engine"]
    assert engine["workers"]["aggregated"]["scheduler"] == {"max_batched_tokens": 2048, "max_sequences": 256}
    assert engine["context_length"] == 8192 and engine["backend_version"] == deployment.VLLM_VERSION


@pytest.mark.parametrize("config", [
    {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
     "num_key_value_heads": 8, "vocab_size": 128256, "max_position_embeddings": 8192},
    {"num_hidden_layers": 28, "hidden_size": 3584, "intermediate_size": 18944, "num_attention_heads": 28,
     "num_key_value_heads": 4, "vocab_size": 152064, "max_position_embeddings": 131072},
    {"num_hidden_layers": 24, "hidden_size": 2048, "intermediate_size": 5504, "num_attention_heads": 16,
     "vocab_size": 32256, "max_position_embeddings": 16384, "head_dim": 128},
])
def test_both_writers_read_the_same_architecture(config: dict) -> None:
    assert R.architecture(geometry.from_config("x", config)) == deployment.architecture_from_config(config)


def test_the_paper_llms_known_offline_match_their_config() -> None:
    llama3 = {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
              "num_key_value_heads": 8, "vocab_size": 128256, "max_position_embeddings": 8192}
    assert R.architecture(geometry.resolve("meta-llama/Meta-Llama-3-8B")) == \
        deployment.architecture_from_config(llama3)


# the shared config.json reader: what the cost laws do not model is refused -------------------------------------

DENSE = {"num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336, "num_attention_heads": 32,
         "num_key_value_heads": 8, "vocab_size": 128256, "max_position_embeddings": 8192, "torch_dtype": "bfloat16"}


@pytest.mark.parametrize(("extra", "refusal"), [
    ({"num_local_experts": 8, "num_experts_per_tok": 2}, "mixture-of-experts"),
    ({"num_experts": 60, "num_experts_per_tok": 4}, "mixture-of-experts"),
    ({"n_routed_experts": 256}, "mixture-of-experts"),
    ({"ffn_config": {"moe_num_experts": 16}}, "mixture-of-experts"),
    ({"quantization_config": {"quant_method": "awq", "bits": 4}}, "quantized"),
    ({"quantization_config": {"quant_method": "gptq", "bits": 4}}, "quantized"),
    ({"quantization_config": {"quant_method": "fp8"}}, "quantized"),
    ({"torch_dtype": "float32"}, "16-bit"),
    ({"torch_dtype": None, "dtype": "float32"}, "16-bit"),
    ({"sliding_window": 4096, "max_position_embeddings": 32768}, "sliding window"),
    ({"sliding_window": 4096, "layer_types": ["sliding_attention", "full_attention"] * 16}, "sliding window"),
    # a mixture of experts named only by its class, without the expert keys
    ({"architectures": ["MixtralForCausalLM"]}, "mixture-of-experts"),
    ({"is_encoder_decoder": True}, "encoder-decoder"),
    ({"model_type": "t5", "architectures": ["T5ForConditionalGeneration"]}, "encoder-decoder"),
    # a two-matrix feed-forward block: the laws count the gated three, a third too many parameters
    ({"model_type": "gpt_neox", "architectures": ["GPTNeoXForCausalLM"]}, "feed-forward"),
    ({"model_type": "falcon"}, "feed-forward"),
    # counts that are not positive integers, and head counts that do not fit
    ({"num_attention_heads": -4}, "num_attention_heads"),
    ({"num_hidden_layers": "32"}, "num_hidden_layers"),
    ({"num_key_value_heads": 6}, "does not divide"),
    ({"hidden_size": 4097}, "not a multiple"),
    ({"head_dim": 160}, "query width"),
])
def test_what_the_cost_laws_do_not_model_is_refused_by_the_one_reader(tmp_path: Path, extra: dict,
                                                                     refusal: str) -> None:
    """The measurement reads config.json with `architecture_from_config`; predict refuses through it too."""
    config = {**DENSE, **extra}
    with pytest.raises(ValueError, match=refusal):
        deployment.architecture_from_config(config)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ManifestError, match=refusal):
        R.resolve_llm(str(tmp_path))


@pytest.mark.parametrize("extra", [
    {"sliding_window": 131072, "use_sliding_window": False, "max_window_layers": 28},
    {"sliding_window": None},
    {"sliding_window": 8192},
    {"layer_types": ["full_attention"] * 32},
    {"num_experts": 0},
    {"quantization_config": None},
    {"torch_dtype": "float16"},
    {"torch_dtype": None, "dtype": "bfloat16"},
    {"head_dim": 128},
    {"head_dim": None},
    {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
    {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]},
    {"model_type": "mistral", "is_encoder_decoder": False},
])
def test_a_dense_16_bit_full_attention_config_is_read(extra: dict) -> None:
    arch = deployment.architecture_from_config({**DENSE, **extra})
    assert (arch["layers"], arch["kv_width"], arch["max_context"]) == (32, 1024, 8192)


def test_a_config_without_a_number_the_laws_need_is_refused_naming_it() -> None:
    config = {k: v for k, v in DENSE.items() if k != "intermediate_size"}
    with pytest.raises(ValueError, match="intermediate_size"):
        deployment.architecture_from_config(config)


# every documented default is what the resolved manifest records -------------------------------------------------

def _models(annotation: object) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [m for arg in typing.get_args(annotation) for m in _models(arg)]


def unset_fields() -> dict[str, str]:
    """Every manifest field whose description says what it becomes when left unset, by dotted path."""
    out: dict[str, str] = {}

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            if "Unset:" in (field.description or ""):
                out[prefix + name] = field.description or ""
            for sub in _models(field.annotation):
                walk(sub, f"{prefix}{name}.")

    walk(Manifest, "")
    return out


SYNTHETIC = {"engine": MINIMAL["engine"],
             "traffic": {"source": {"type": "synthetic", "input_tokens": {"mean": 200}, "output_tokens": {"mean": 50}},
                         "load": {"type": "concurrency", "concurrency": 4, "think_time_ms": 1000},
                         "stop": {"steady_state": {}}}}
NEWEST = {"length_sampler": "2026-11", "estimator": "2026-11"}
#: what each documented default resolves to on an A100 with the 2026-11 sampler and estimator: the phrase its
#: description says it with, and the value both writers record
WRITTEN_BY_BOTH = {
    "engine.backend_version": ("Unset: 0.25.1", "0.25.1"),
    "engine.workers.aggregated.scheduler.max_batched_tokens": ("else 2048", 2048),
    "engine.workers.aggregated.scheduler.max_sequences": ("else 256", 256),
    "engine.workers.aggregated.kv_cache.block_size": ("Unset: 16", 16),
    "engine.workers.aggregated.kv_cache.prefix_caching": ("Unset: off", False),
    "engine.workers.aggregated.kv_cache.capacity.memory_fraction": ("Unset: 0.9", 0.9),
    "traffic.source.input_tokens.variance": ("mean (mean - 1)", 200.0 * 199.0),
    "traffic.source.output_tokens.variance": ("mean (mean - 1)", 50.0 * 49.0),
    "traffic.stop.steady_state.ttft_p95": ("Unset: 0.10", 0.10),
    "traffic.stop.steady_state.tpot_p95": ("Unset: 0.10", 0.10),
}
#: what only a prediction records (a measurement runs at the GPU's own clock, with the versions it was given)
WRITTEN_BY_PREDICT = {"engine.gpu_clock_mhz": "the clock the calibration was made at",
                      "versions": "Unset: each component's default"}


def _at(doc: dict, dotted: str) -> object:
    for key in dotted.split("."):
        doc = doc[key]
    return doc


def test_every_documented_default_has_a_test() -> None:
    assert set(unset_fields()) == set(WRITTEN_BY_BOTH) | set(WRITTEN_BY_PREDICT)


@pytest.mark.parametrize("path", sorted(WRITTEN_BY_BOTH))
def test_an_unset_field_is_recorded_as_its_documented_default(path: str) -> None:
    phrase, value = WRITTEN_BY_BOTH[path]
    assert phrase in unset_fields()[path]
    m = Manifest.model_validate(SYNTHETIC)
    d = deployment.resolve(m, gpus.get("a100_sxm_40gb"), 8192, "llm", architecture=LLAMA8B)
    doc = deployment.resolved_sections(m, model="m", architecture={}, hardware="a100_sxm_40gb", deployment=d,
                                       components=NEWEST)
    assert _at(doc, path) == pytest.approx(value) and type(_at(doc, path)) is type(value)


def test_the_percentile_targets_are_recorded_as_not_judged_by_an_estimator_that_does_not_judge_them() -> None:
    m = Manifest.model_validate(MINIMAL)
    d = deployment.resolve(m, gpus.get("a100_sxm_40gb"), 8192, "llm", architecture=LLAMA8B)
    doc = deployment.resolved_sections(m, model="m", architecture={}, hardware="a100_sxm_40gb", deployment=d,
                                       components={"estimator": "2026-10"})
    stop = doc["traffic"]["stop"]["steady_state"]
    assert stop["ttft_p95"] is None and stop["tpot_p95"] is None and stop["ttft"] == 0.10


def test_a_prediction_records_the_calibrations_clock_and_its_components_by_name() -> None:
    from servecast.core import versions
    from servecast.model import predict as P

    for path, phrase in WRITTEN_BY_PREDICT.items():
        assert phrase in unset_fields()[path]
    m = Manifest.model_validate({**MINIMAL, "calibration": {"use": "a100_sxm_40gb/c62ebe75", "transfer": True}})
    calibration = P.load_calibration("a100_sxm_40gb/c62ebe75")
    pl = R.plan(m, calibration, "a100_sxm_40gb/c62ebe75", b"{}")
    doc = R.resolved_manifest(m, pl)
    assert doc["engine"]["gpu_clock_mhz"] == float(calibration["conditions"]["clock_mhz"])
    assert doc["versions"] == {k: v for k, v in versions.resolve(None).items() if v is not None}  # no as_of


# the refusals and defaults the two writers share ---------------------------------------------------------------

def _manifest(**kv_cache: object) -> Manifest:
    doc = copy.deepcopy(MINIMAL)
    doc["engine"]["workers"]["aggregated"]["kv_cache"].update(kv_cache)
    return Manifest.model_validate(doc)


def test_prefix_caching_is_refused() -> None:
    with pytest.raises(ManifestError, match="prefix_caching"):
        deployment.resolve(_manifest(prefix_caching=True), gpus.get("a100_sxm_40gb"), 8192, "llm", architecture=LLAMA8B)


def test_a_context_beyond_the_llms_limit_is_refused() -> None:
    m = Manifest.model_validate({**MINIMAL, "engine": {**MINIMAL["engine"], "context_length": 8193}})
    with pytest.raises(ManifestError, match="exceeds"):
        deployment.resolve(m, gpus.get("a100_sxm_40gb"), 8192, "llm", architecture=LLAMA8B)
    assert deployment.resolve(m, gpus.get("a100_sxm_40gb"), 8193, "llm", architecture=LLAMA8B).context_length == 8193


@pytest.mark.parametrize(("blocks", "fits"), [(513, True), (512, False)])
def test_the_kv_cache_must_hold_one_request_of_the_context_length(blocks: int, fits: bool) -> None:
    m = _manifest(capacity={"blocks": blocks})
    if fits:
        assert deployment.resolve(m, gpus.get("a100_sxm_40gb"), 8192, "llm", architecture=LLAMA8B).kv_blocks == blocks
    else:
        with pytest.raises(ManifestError, match="fewer tokens than one request"):
            deployment.resolve(m, gpus.get("a100_sxm_40gb"), 8192, "llm", architecture=LLAMA8B)


@pytest.mark.parametrize(("gpu", "defaults"), [("h100_sxm", (8192, 1024)), ("a100_sxm_40gb", (2048, 256)),
                                               ("rtx_3090_ti", (2048, 256))])
def test_vllms_scheduler_defaults_follow_the_gpu(gpu: str, defaults: tuple[int, int]) -> None:
    assert deployment.engine_defaults(gpus.get(gpu)) == defaults


@pytest.mark.parametrize(("date", "components"), [
    ("2026-10", {"estimator": "cabeb968"}), ("2026-11", {"length_sampler": "fe3db391", "estimator": "c3ad4006"})])
def test_a_measurement_runs_the_estimator_and_the_sampler_its_date_selects(date: str, components: dict) -> None:
    assert deployment.measured_components(Manifest.model_validate({**MINIMAL, "versions": date})) == components


@pytest.mark.parametrize(("version", "accepted"), [(None, True), ("0.25.1", True), ("0.26.0", False), ("0.25", False)])
def test_a_vllm_version_servecast_does_not_model_is_refused(version: str | None, accepted: bool) -> None:
    doc = copy.deepcopy(MINIMAL)
    if version is not None:
        doc["engine"]["backend_version"] = version
    m = Manifest.model_validate(doc)
    if accepted:
        assert deployment.resolve(m, gpus.get("a100_sxm_40gb"), 8192, "llm", architecture=LLAMA8B).kv_blocks == 9965
    else:
        refusal = rf"engine\.backend_version: {re.escape(version or '')}: ServeCast models"
        with pytest.raises(ManifestError, match=refusal):
            deployment.resolve(m, gpus.get("a100_sxm_40gb"), 8192, "llm", architecture=LLAMA8B)


# the deployment must fit in the GPU's memory ----------------------------------------------------------------------

DS7B = R.architecture(geometry.KNOWN["deepseek-ai/deepseek-llm-7b-base"])


def test_the_memory_count_is_the_geometrys() -> None:
    """core cannot import the model's geometry (the layer contract), so it counts bytes itself: the same counts."""
    for g in geometry.KNOWN.values():
        kv, weights = deployment.memory_needed(R.architecture(g), blocks=1, block_size=1)
        assert kv == g.kv_bytes_per_token and weights == g.w_bytes + g.head_bytes, g.name


def test_a_kv_cache_that_does_not_fit_in_the_gpus_memory_is_refused_with_both_sizes() -> None:
    """The audit's case: the starter's 23,488 blocks carried to DeepSeek-LLM-7B, 172 GiB of KV on a 40 GiB A100."""
    m = _manifest(capacity={"blocks": 23488})
    with pytest.raises(ManifestError, match=r"172\.0 GiB of KV cache; with 12\.1 GiB of weights.*at most 3264 blocks"):
        deployment.resolve(m, gpus.get("a100_sxm_40gb"), 4096, "ds", architecture=DS7B)


@pytest.mark.parametrize(("blocks", "fraction", "fits"), [
    (3264, None, True), (3265, None, False),  # the boundary at the default fraction, 0.9
    (3400, None, False), (3400, 0.95, True),  # the same cache under a larger --gpu-memory-utilization
])
def test_the_bound_is_the_memory_fraction_of_the_gpus_memory(blocks: int, fraction: float | None,
                                                              fits: bool) -> None:
    m = _manifest(capacity={"blocks": blocks, **({"memory_fraction": fraction} if fraction else {})})
    if fits:
        assert deployment.resolve(m, gpus.get("a100_sxm_40gb"), 4096, "ds", architecture=DS7B).kv_blocks == blocks
    else:
        with pytest.raises(ManifestError, match=r"more than 0\.9 x 40 GiB"):
            deployment.resolve(m, gpus.get("a100_sxm_40gb"), 4096, "ds", architecture=DS7B)


def test_no_paper_campaign_is_refused() -> None:
    """Every deployment the paper measured fits the bound: the check refuses none of them."""
    path = Path(deployment.__file__).parents[1] / "data" / "paper" / "campaigns.json"
    checked = 0
    for c in json.loads(path.read_text(encoding="utf-8"))["campaigns"]:
        e = c["engine"]
        if e is None:  # three campaigns record no engine settings
            continue
        kv, weights = deployment.memory_needed(R.architecture(geometry.KNOWN[c["llm"]]),
                                               blocks=e["num_gpu_blocks_override"] or e["num_gpu_blocks"],
                                               block_size=e["block_size"])
        assert kv + weights <= e["gpu_memory_utilization"] * gpus.get(c["gpu"]).memory_gib * deployment.GIB, c
        checked += 1
    assert checked == 51


# a request must fit the context length ---------------------------------------------------------------------------

def _trace_manifest(tmp_path: Path, lengths: list[tuple[int, int]]) -> Manifest:
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps({"input_tokens": a, "output_tokens": b}) + "\n" for a, b in lengths),
                    encoding="utf-8")
    doc = copy.deepcopy(MINIMAL)
    doc["traffic"]["source"] = {"type": "trace", "paths": [str(path)], "format": "lengths"}
    return Manifest.model_validate(doc)


def test_a_trace_request_longer_than_the_context_is_left_out_as_vllm_refuses_it(tmp_path: Path) -> None:
    """The sampler draws such a request again; a trace now leaves it out, as vLLM would refuse it. Exactly the
    context length still fits."""
    m = _trace_manifest(tmp_path, [(100, 50), (4000, 96), (4000, 200), (90, 6), (4096, 1)] * 40)
    pool = deployment.build_pool(m, {}, 4096)
    assert (pool.input_tokens, pool.output_tokens) == ([100, 4000, 90] * 40, [50, 96, 6] * 40)
    assert pool.warnings == ["80 of 200 requests need more than the context length (4096 tokens); vLLM would refuse "
                             "them, so they are left out"]


@pytest.mark.parametrize(("lengths", "refusal"), [
    ([(100, 50)] * 99, r"the trace has 99 requests; a pool needs at least 100 requests"),
    ([(100, 50)] + [(5000, 10)] * 199, r"only 1 of the trace's 200 requests fit the context length \(4096 tokens\); "
                                       r"a pool needs at least 100 requests.*raise engine.context_length"),
    ([(100, 50)] * 100, None),
])
def test_a_trace_that_leaves_fewer_than_100_requests_is_refused(tmp_path: Path, lengths: list,
                                                               refusal: str | None) -> None:
    """As a synthetic pool (traffic.pool_size >= 100): with a few requests the workload law cannot be fitted, and
    predict ended in an internal error instead of naming the field."""
    m = _trace_manifest(tmp_path, lengths)
    if refusal is None:
        assert len(deployment.build_pool(m, {}, 4096).input_tokens) == 100
    else:
        with pytest.raises(ManifestError, match=refusal):
            deployment.build_pool(m, {}, 4096)


def test_a_trace_with_no_request_that_fits_the_context_is_refused(tmp_path: Path) -> None:
    m = _trace_manifest(tmp_path, [(5000, 10), (4096, 1)])
    with pytest.raises(ManifestError, match=r"none of the 2 requests fits the context length \(4096 tokens\)"):
        deployment.build_pool(m, {}, 4096)


@pytest.mark.parametrize("value", [0, -5])
def test_a_context_length_that_is_not_positive_is_refused(value: int) -> None:
    doc = copy.deepcopy(MINIMAL)
    doc["engine"]["context_length"] = value
    with pytest.raises(ValidationError, match="context_length"):
        Manifest.model_validate(doc)


def test_max_needs_the_llms_own_limit_and_an_explicit_length_does_not() -> None:
    """A config.json without max_position_embeddings gives no limit: `max` used to become 0."""
    with pytest.raises(ManifestError, match="max_position_embeddings"):
        deployment.resolve(_manifest(), gpus.get("a100_sxm_40gb"), 0, "llm", architecture=LLAMA8B)
    m = Manifest.model_validate({**MINIMAL, "engine": {**MINIMAL["engine"], "context_length": 4096}})
    assert deployment.resolve(m, gpus.get("a100_sxm_40gb"), 0, "llm", architecture=LLAMA8B).context_length == 4096


# synthetic lengths: what was drawn, against what was asked ------------------------------------------------------

def _synthetic(input_tokens: dict, output_tokens: dict) -> Manifest:
    doc = copy.deepcopy(MINIMAL)
    doc["traffic"]["source"] = {"type": "synthetic", "input_tokens": input_tokens, "output_tokens": output_tokens}
    return Manifest.model_validate(doc)


def test_synthetic_lengths_far_from_the_request_are_said() -> None:
    """The audit's case: an input mean of 1200 with a 2048-token context drew a mean of 683.8, without a word (a pair
    longer than the context is drawn again)."""
    pool = deployment.build_pool(_synthetic({"mean": 1200}, {"mean": 250, "variance": 40000}),
                                 {"length_sampler": "2026-11"}, 2048)
    said = [w for w in pool.warnings if w.startswith("the synthetic input lengths drawn have mean")]
    assert len(said) == 1 and "from the 1200 asked for" in said[0], pool.warnings
    assert abs(st.fmean(pool.input_tokens) / 1200 - 1) > 0.05


def test_synthetic_lengths_that_match_the_request_say_nothing() -> None:
    pool = deployment.build_pool(_synthetic({"mean": 100, "variance": 2500}, {"mean": 50}),
                                 {"length_sampler": "2026-11"}, 8192)
    assert pool.warnings == []


# what a per-GPU calibration covers: its synthetic models, the LLMs it was validated on, its GEMM grid ------------

SHIPPED_PER_GPU = json.loads((Path(deployment.__file__).parents[1] / "data" / "calibrations" / "a100_sxm_40gb"
                              / "c62ebe75.json").read_text(encoding="utf-8"))


def test_the_shipped_per_gpu_calibration_covers_the_six_llms_it_was_validated_on() -> None:
    """Five of the six are wider than every synthetic model (3072) or have an FFN ratio below the grid's 2.75: the
    synthetic models alone would label them outside, although the calibration's shipped predictions are theirs."""
    for g in geometry.KNOWN.values():
        assert R.coverage_reasons(SHIPPED_PER_GPU, "a100_sxm_40gb/c62ebe75", g) == [], g.name


def test_an_llm_beyond_what_the_calibration_covers_is_outside_with_the_reason() -> None:
    big = geometry.Geometry("llama-2-13b", n_layers=40, d_model=5120, d_ff=13824, d_kv=5120, h_kv=40, n_heads=40,
                            vocab=32000, max_context=4096)
    reasons = R.coverage_reasons(SHIPPED_PER_GPU, "a100_sxm_40gb/c62ebe75", big)
    assert len(reasons) == 2 and "layer count 40" in reasons[0] and "12-36" in reasons[0]
    assert "width 5120" in reasons[1] and "1024-4096" in reasons[1]


def test_a_calibration_validated_on_nothing_covers_only_its_synthetic_models() -> None:
    """A per-GPU calibration of your own has no validated LLMs: an LLM wider than its synthetic models is outside."""
    reasons = R.coverage_reasons(SHIPPED_PER_GPU, "local/a100_sxm_40gb/c62ebe75",
                                 geometry.KNOWN["meta-llama/Meta-Llama-3-8B"])
    assert reasons == ["the LLM's width 4096 lies outside 1024-3072, the range the calibration was fitted or "
                       "validated on"]


# what vLLM 0.25.1 refuses to start with ------------------------------------------------------------------------

@pytest.mark.parametrize(("scheduler", "refused"), [
    ({"max_batched_tokens": 256, "max_sequences": 256}, False),   # equal: vLLM starts
    ({"max_batched_tokens": 255, "max_sequences": 256}, True),
    ({"max_batched_tokens": 255}, True),                          # the A100's default is 256 sequences
    ({"max_sequences": 2049}, True),                              # the A100's default is 2048 tokens
    ({}, False),
])
def test_a_token_budget_below_the_sequence_cap_is_refused_as_vllm_refuses_it(scheduler: dict, refused: bool) -> None:
    """vLLM 0.25.1's SchedulerConfig: max_num_batched_tokens must be >= max_num_seqs, defaults included."""
    doc = copy.deepcopy(MINIMAL)
    doc["engine"]["workers"]["aggregated"]["scheduler"] = scheduler
    m = Manifest.model_validate(doc)
    if refused:
        with pytest.raises(ManifestError, match=r"scheduler: max_batched_tokens \(\d+\) is below max_sequences"):
            deployment.resolve(m, gpus.get("a100_sxm_40gb"), 8192, "llm", architecture=LLAMA8B)
    else:
        deployment.resolve(m, gpus.get("a100_sxm_40gb"), 8192, "llm", architecture=LLAMA8B)
