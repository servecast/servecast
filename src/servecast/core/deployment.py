"""What a manifest's deployment and workload resolve to. The prediction and the measurement both call this module, so
they describe the same experiment by construction and `compare` can pair them field by field.

It fills in vLLM 0.25.1's defaults, resolves the context length, builds the request pool, and writes the resolved
manifest's `engine` and `traffic` sections. The LLM's architecture enters as the numbers that define it
(`architecture`), so a model named by a local path on one machine and a hub id on another is still recognised as the
same model.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from servecast.core import estimator as estimators
from servecast.core import gpus, versions
from servecast.core import pool as pools
from servecast.core import sampler as samplers
from servecast.core.manifest import (
    KV_CAPACITY_REQUIRED,
    MIN_POOL,
    PREFIX_CACHING_NOT_MODELLED,
    Manifest,
    ManifestError,
    SyntheticSource,
    TraceSource,
    VersionsSpec,
)

#: tokens per KV-cache block when the manifest leaves it unset: vLLM 0.25.1's default on these GPUs
DEFAULT_BLOCK_SIZE = 16
#: the fraction of GPU memory vLLM may use when the manifest leaves it unset: the paper's measurements used 0.9
DEFAULT_MEMORY_FRACTION = 0.9
#: the serving engine and the version every calibration and measurement uses
VLLM_VERSION = "0.25.1"
#: the target relative half-width of a percentile when the manifest gives none (estimators that judge percentiles)
DEFAULT_P95_TARGET = 0.10


def as_of(m: Manifest) -> str | None:
    """The manifest's year-month, if it gives one."""
    if isinstance(m.versions, str):
        return m.versions
    if isinstance(m.versions, VersionsSpec):
        return m.versions.as_of
    return None


def stop_targets(m: Manifest, estimator_id: str) -> dict[str, float]:
    """The stop rule's targets, in the manifest's names: the means always, the 95th percentiles where the estimator
    judges them (by default at DEFAULT_P95_TARGET)."""
    assert m.traffic is not None
    s = m.traffic.stop.steady_state
    targets = {"throughput": s.throughput, "e2e": s.e2e, "ttft": s.ttft, "tpot": s.tpot}
    judges_p95 = hasattr(estimators.get(estimator_id), "quantile_ci")
    for key in ("ttft_p95", "tpot_p95"):
        value = getattr(s, key)
        if value is not None and not judges_p95:
            judging = [n for n in estimators.RELEASED if hasattr(estimators.get(n), "quantile_ci")]
            raise ManifestError(f"traffic.stop.steady_state.{key}: the steady-state estimator {estimator_id} does not "
                                f"judge percentiles; remove it or choose estimator {', '.join(judging)}")
        if judges_p95:
            targets[key] = DEFAULT_P95_TARGET if value is None else value
    return targets


def measured_components(m: Manifest) -> dict[str, str]:
    """The components a measurement uses, resolved from the manifest's versions as a prediction resolves them: the
    steady-state estimator, and the length sampler when the date has one. The cost law and the calibration procedure
    play no part in a measurement."""
    try:
        resolved = versions.resolve(m.versions)
    except ValueError as error:
        raise ManifestError(f"versions: {error}") from error
    return {k: str(resolved[k]) for k in ("length_sampler", "estimator") if resolved.get(k) is not None}


#: bytes in a GiB, the unit of `Gpu.memory_gib`
GIB = 2**30


def memory_needed(architecture: Mapping[str, int], *, blocks: int, block_size: int,
                  bytes_per_elt: int = 2) -> tuple[float, float]:
    """The bytes of the KV cache (`blocks` blocks of `block_size` tokens) and of the weights, in the 16-bit precision
    the cost laws model. The weights are the decoder layers and the vocabulary matrix, the terms the laws read; an
    untied LLM's input embeddings, the norms and the biases are left out, so the sum is a lower bound and the check in
    `resolve` never refuses a deployment that fits. The model's `Geometry` gives the same two counts (a test ties
    them): core cannot import it."""
    layers, d, ffn, kv_width, vocab = (int(architecture[k]) for k in ("layers", "hidden", "ffn", "kv_width", "vocab"))
    kv = float(blocks * block_size * 2 * bytes_per_elt * layers * kv_width)
    weights = float(bytes_per_elt * (layers * (d * (d + 2 * kv_width) + d * d + 3 * d * ffn) + vocab * d))
    return kv, weights


def engine_defaults(gpu: gpus.Gpu) -> tuple[int, int]:
    """vLLM 0.25.1's defaults for `vllm serve`: 8192 tokens and 1024 sequences on a GPU with at least 70 GiB other
    than the A100, else 2048 and 256."""
    big = gpu.memory_gib >= 70 and not gpu.id.startswith("a100")
    return (8192, 1024) if big else (2048, 256)


@dataclasses.dataclass(frozen=True)
class Deployment:
    """The engine settings a manifest resolves to on one GPU."""

    block_size: int
    max_batched_tokens: int
    max_sequences: int
    kv_blocks: int
    context_length: int
    memory_fraction: float


def resolve(m: Manifest, gpu: gpus.Gpu, max_context: int, llm_name: str, *,
            architecture: Mapping[str, int]) -> Deployment:
    """The engine settings, with every refusal the prediction and the measurement share."""
    worker = m.engine.workers.aggregated
    cap = worker.kv_cache.capacity
    if cap is None:
        raise ManifestError(f"engine.workers.aggregated.kv_cache.capacity: required: {KV_CAPACITY_REQUIRED}")
    if m.engine.backend_version is not None and m.engine.backend_version != VLLM_VERSION:
        raise ManifestError(f"engine.backend_version: {m.engine.backend_version}: ServeCast models vLLM "
                            f"{VLLM_VERSION}, the version every calibration and measurement uses; remove the field "
                            f"or write {VLLM_VERSION}")
    if worker.kv_cache.prefix_caching:
        raise ManifestError(f"engine.workers.aggregated.kv_cache.prefix_caching: {PREFIX_CACHING_NOT_MODELLED}")
    tokens, seqs = engine_defaults(gpu)
    budget = worker.scheduler.max_batched_tokens or tokens
    sequences = worker.scheduler.max_sequences or seqs
    if budget < sequences:  # vLLM 0.25.1's SchedulerConfig refuses it: max_num_batched_tokens >= max_num_seqs
        unset = "" if worker.scheduler.max_batched_tokens and worker.scheduler.max_sequences else (
            f"; unset, each is vLLM's default on this GPU ({tokens} tokens, {seqs} sequences)")
        raise ManifestError(f"engine.workers.aggregated.scheduler: max_batched_tokens ({budget}) is below "
                            f"max_sequences ({sequences}){unset}. vLLM 0.25.1 refuses to start: raise "
                            "max_batched_tokens or lower max_sequences")
    block_size = worker.kv_cache.block_size or DEFAULT_BLOCK_SIZE
    if m.engine.context_length == "max" and not max_context:
        raise ManifestError(f"engine.context_length: max: {llm_name}'s config.json gives no max_position_embeddings, "
                            "so its limit is unknown; write the context length in tokens")
    context = max_context if m.engine.context_length == "max" else int(m.engine.context_length)
    if max_context and context > max_context:
        raise ManifestError(f"engine.context_length: {context} exceeds {llm_name}'s limit of {max_context}")
    if (cap.blocks - 1) * block_size < context:
        raise ManifestError(f"engine.workers.aggregated.kv_cache.capacity.blocks: {cap.blocks} blocks hold fewer "
                            f"tokens than one request of the context length ({context}); vLLM would refuse to start")
    fraction = cap.memory_fraction or DEFAULT_MEMORY_FRACTION
    kv, weights = memory_needed(architecture, blocks=cap.blocks, block_size=block_size)
    limit = fraction * gpu.memory_gib * GIB
    if kv + weights > limit:
        room = int((limit - weights) // (kv / cap.blocks)) if limit > weights else 0
        fit = f"at most {room} blocks fit" if room > 0 else "the weights alone exceed it"
        raise ManifestError(f"engine.workers.aggregated.kv_cache.capacity.blocks: {cap.blocks} blocks of {llm_name} "
                            f"hold {kv / GIB:.1f} GiB of KV cache; with {weights / GIB:.1f} GiB of weights that is "
                            f"{(kv + weights) / GIB:.1f} GiB, more than {fraction:g} x {gpu.memory_gib} GiB "
                            f"(memory_fraction x the GPU's memory), which vLLM could not allocate: {fit}. Give the "
                            f"count vLLM logs for this LLM on this GPU ('GPU KV cache size: T tokens', blocks = T / "
                            f"block_size)")
    return Deployment(block_size=block_size, max_batched_tokens=budget, max_sequences=sequences, kv_blocks=cap.blocks,
                      context_length=context, memory_fraction=fraction)


#: the config.json numbers every cost law reads
REQUIRED_CONFIG = ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "vocab_size")
#: config.json keys that give a mixture-of-experts model its number of experts
EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
#: model families (config.json's model_type) whose feed-forward block has two matrices, not the gated three of
#: Llama-style LLMs that the cost laws count: priced as gated, P came out a third too large
UNGATED_MLP = ("gpt2", "gpt_neo", "gpt_neox", "gptj", "gpt_bigcode", "codegen", "opt", "bloom", "falcon", "mpt", "phi",
               "starcoder2")
#: model families with an encoder and a decoder
ENCODER_DECODER = ("t5", "mt5", "umt5", "bart", "mbart", "pegasus", "marian", "led", "prophetnet", "whisper")
#: the config.json counts the laws read: each must be a positive integer when present
#: how far a synthetic pool's mean or variance may fall from the request before the prediction says so
SYNTHETIC_TOLERANCE = 0.05
COUNT_KEYS = ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "vocab_size",
              "num_key_value_heads", "head_dim")


def _sliding_window(config: Mapping[str, Any]) -> int | None:
    """The attention window, in tokens, when the LLM attends over a sliding window shorter than its context."""
    if any("sliding" in str(kind) for kind in config.get("layer_types") or ()):
        return int(config.get("sliding_window") or 0)
    window = config.get("sliding_window")
    if not window or config.get("use_sliding_window") is False:
        return None
    longest = int(config.get("max_position_embeddings") or 0)
    return int(window) if not longest or int(window) < longest else None


def architecture_from_config(config: Mapping[str, Any]) -> dict[str, int]:
    """The numbers that define a dense LLM's architecture, read from its config.json the way the cost laws read
    them (a missing head_dim is hidden_size / heads; a missing KV-head count is the head count). The prediction and
    the measurement both read config.json through this function, so they refuse the same LLMs: a mixture of
    experts, quantized weights, weights other than 16-bit, and attention over a sliding window, none of which the cost
    laws model. ValueError names what was refused."""
    missing = [key for key in REQUIRED_CONFIG if key not in config]
    if missing:
        raise ValueError(f"config.json lacks {', '.join(missing)}")
    model_type = str(config.get("model_type") or "").lower()
    classes = [str(c) for c in config.get("architectures") or []]
    if config.get("is_encoder_decoder") or model_type in ENCODER_DECODER:
        named = model_type or (classes[0] if classes else "is_encoder_decoder")
        raise ValueError(f"an encoder-decoder model ({named} in config.json) is not supported: the cost laws model "
                         "decoder-only LLMs")
    ffn = config.get("ffn_config")
    nested: Mapping[str, Any] = ffn if isinstance(ffn, Mapping) else {}
    experts = [key for key in EXPERT_KEYS if config.get(key) or nested.get(key)]
    experts += [c for c in classes if "moe" in c.lower() or "mixtral" in c.lower()]
    if experts:
        raise ValueError(f"a mixture-of-experts model ({experts[0]} in config.json) is not supported: the cost laws "
                         "model dense LLMs")
    if model_type in UNGATED_MLP:
        raise ValueError(f"a {model_type} model has a two-matrix feed-forward block; the cost laws count the gated "
                         "three-matrix block of Llama-style LLMs")
    bad = [key for key in COUNT_KEYS if config.get(key) is not None
           and (isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0)]
    if bad:
        raise ValueError("config.json's " + ", ".join(f"{key} ({config[key]!r})" for key in bad)
                         + (" is" if len(bad) == 1 else " are") + " not a positive integer")
    if config.get("quantization_config"):
        method = dict(config["quantization_config"]).get("quant_method") or "unnamed"
        raise ValueError(f"quantized weights ({method} in config.json) are not supported: the cost laws model 16-bit "
                         "weights")
    dtype = str(config.get("torch_dtype") or config.get("dtype") or "bfloat16")
    if dtype not in ("bfloat16", "float16"):
        raise ValueError(f"weights in {dtype}: the cost laws model 16-bit weights")
    window = _sliding_window(config)
    if window is not None:
        raise ValueError(f"attention over a sliding window ({window or 'some'} tokens) is not supported: the cost "
                         "laws model attention over the whole context")
    heads, hidden = int(config["num_attention_heads"]), int(config["hidden_size"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    if heads % kv_heads:
        raise ValueError(f"num_key_value_heads {kv_heads} does not divide num_attention_heads {heads}")
    if config.get("head_dim") is None:
        if hidden % heads:
            raise ValueError(f"hidden_size {hidden} is not a multiple of num_attention_heads {heads}, and config.json "
                             "gives no head_dim")
        head_dim = hidden // heads
    else:
        head_dim = int(config["head_dim"])
        if heads * head_dim != hidden:
            raise ValueError(f"the query width, num_attention_heads x head_dim = {heads * head_dim}, is not "
                             f"hidden_size {hidden}: the cost laws count attention projections of hidden_size x "
                             "hidden_size")
    return {"layers": int(config["num_hidden_layers"]), "hidden": int(config["hidden_size"]),
            "ffn": int(config["intermediate_size"]), "heads": heads, "kv_heads": kv_heads,
            "kv_width": kv_heads * head_dim, "vocab": int(config["vocab_size"]),
            "max_context": int(config.get("max_position_embeddings") or 0)}


def check_architecture(m: Manifest, arch: Mapping[str, int]) -> None:
    """A manifest that gives `engine.architecture` (a resolved manifest run again) must give the LLM's own numbers."""
    if m.engine.architecture is None:
        return
    given = m.engine.architecture.model_dump()
    if given != dict(arch):
        wrong = ", ".join(f"{k} {given.get(k)} (config.json: {v})" for k, v in arch.items() if given.get(k) != v)
        raise ManifestError(f"engine.architecture: not {m.engine.model}'s: {wrong}; remove the field to read it "
                            "from config.json")


@dataclasses.dataclass(frozen=True)
class Pool:
    """The request pool and where it came from."""

    input_tokens: list[int]
    output_tokens: list[int]
    header: dict[str, Any]
    warnings: list[str]
    output_spec: Any = None          # the sampler's output-length law, for a synthetic pool


def build_pool(m: Manifest, components: Mapping[str, str | None], context: int) -> Pool:
    """Every request of the trace files in order, or `pool_size` draws of the length sampler."""
    assert m.traffic is not None
    src = m.traffic.source
    warnings: list[str] = []
    if isinstance(src, TraceSource):
        p: list[int] = []
        g: list[int] = []
        names = []
        for path in src.paths:
            if not path.exists() and str(path) in pools.SHIPPED_TRACES:
                doc = pools.shipped_trace(str(path))
                p += doc["input_tokens"]
                g += doc["output_tokens"]
            else:
                if not path.is_file():
                    raise ManifestError(f"traffic.source.paths: {path}: no such file (the shipped traces are "
                                        f"{', '.join(pools.SHIPPED_TRACES)})")
                try:
                    pp, gg, ww = pools.read_trace(path, src.format)
                except ValueError as error:
                    raise ManifestError(f"traffic.source.paths: {error}") from error
                p += pp
                g += gg
                warnings += ww
            names.append(path.name)
        fits = [a + b <= context for a, b in zip(p, g, strict=True)]
        if not any(fits):
            raise ManifestError(f"traffic.source: none of the {len(p)} requests fits the context length ({context} "
                                "tokens); vLLM would refuse every one")
        if not all(fits):
            # as vLLM refuses them; the length sampler draws such a request again
            warnings.append(f"{fits.count(False)} of {len(p)} requests need more than the context length ({context} "
                            "tokens); vLLM would refuse them, so they are left out")
            p = [a for a, keep in zip(p, fits, strict=True) if keep]
            g = [b for b, keep in zip(g, fits, strict=True) if keep]
        if len(p) < MIN_POOL:  # as a synthetic pool: with fewer, the workload law cannot be fitted
            left_out = not all(fits)
            where = (f"only {len(p)} of the trace's {len(fits)} requests fit the context length ({context} tokens)"
                     if left_out else f"the trace has {len(p)} requests")
            raise ManifestError(f"traffic.source: {where}; a pool needs at least {MIN_POOL} requests, as "
                                "traffic.pool_size does for synthetic lengths"
                                + ("; raise engine.context_length or give a trace of shorter requests" if left_out
                                   else "; give a longer trace"))
        return Pool(p, g, {"source": "trace", "files": names}, warnings)
    assert isinstance(src, SyntheticSource)
    sampler_id = components.get("length_sampler")
    if sampler_id is None:
        raise ManifestError("traffic.source: synthetic lengths need a length sampler; the selected versions have none "
                            "(versions: 2026-11 or later)")
    sampler = samplers.get(sampler_id)
    try:
        spec_in = sampler.match(src.input_tokens.mean, src.input_tokens.variance)
        spec_out = sampler.match(src.output_tokens.mean, src.output_tokens.variance)
        drawn = sampler.pool(spec_in, spec_out, size=m.traffic.pool_size, seed=m.traffic.seed, context_length=context)
    except ValueError as error:
        raise ManifestError(f"traffic.source: {error}") from error
    header = {"source": "synthetic", "sampler": sampler_id, "seed": m.traffic.seed,
              "requested": drawn["requested"], "realized": drawn["realized"], "laws": drawn["laws"]}
    # a pair longer than the context is drawn again, so the pool can be far from the request: said, never silent
    for side, spec in (("input", src.input_tokens), ("output", src.output_tokens)):
        asked = {"mean": spec.mean, "variance": spec.variance}
        for stat, want in asked.items():
            got = float(drawn["realized"][side][stat])
            if want and abs(got / want - 1) > SYNTHETIC_TOLERANCE:
                off = 100 * (got / want - 1)
                warnings.append(f"the synthetic {side} lengths drawn have {stat} {got:.1f}, {off:+.0f} % from the "
                                f"{want:g} asked for: a pair longer than the context ({context} tokens) is drawn again")
    return Pool(drawn["input_tokens"], drawn["output_tokens"], header, warnings, spec_out)


def resolved_sections(m: Manifest, *, model: str, architecture: Mapping[str, int], hardware: str,
                      deployment: Deployment, components: Mapping[str, str | None],
                      gpu_clock_mhz: float | None = None) -> dict[str, Any]:
    """The resolved manifest's `engine` and `traffic` sections, with every default filled in: vLLM's, the stop rule's
    percentile targets (for an estimator that judges them; null otherwise) and an unset length variance (the length
    sampler's). Both writers call this with the components they run, and `manifest.semantic_view` compares what it
    writes."""
    doc = m.model_dump(mode="json")
    engine = doc["engine"]
    engine.update(model=model, hardware=hardware, context_length=deployment.context_length,
                  architecture=dict(architecture), backend_version=engine.get("backend_version") or VLLM_VERSION)
    if gpu_clock_mhz is not None:
        engine["gpu_clock_mhz"] = float(gpu_clock_mhz)
    worker = engine["workers"]["aggregated"]
    worker["scheduler"] = {"max_batched_tokens": deployment.max_batched_tokens,
                           "max_sequences": deployment.max_sequences}
    worker["kv_cache"]["block_size"] = deployment.block_size
    worker["kv_cache"]["prefix_caching"] = False
    worker["kv_cache"]["capacity"]["memory_fraction"] = deployment.memory_fraction
    traffic = doc["traffic"]
    if traffic is not None and m.traffic is not None:
        estimator = components.get("estimator")
        if estimator is not None:
            targets = stop_targets(m, str(estimator))
            stop = traffic["stop"]["steady_state"]
            traffic["stop"]["steady_state"] = {key: targets.get(key) for key in stop}
        source, sampler = traffic["source"], components.get("length_sampler")
        if source["type"] == "synthetic" and sampler is not None:
            for key in ("input_tokens", "output_tokens"):
                if source[key]["variance"] is None:
                    source[key]["variance"] = float(samplers.get(sampler).match(source[key]["mean"]).variance)
    return {"engine": engine, "traffic": traffic}


def load_points(m: Manifest, sweep: tuple[str, list[str]] | None) -> list[tuple[int, float]]:
    """(clients, think time in ms) of each load point: the manifest's levels in the order listed, or one per value of
    the swept field. predict and profile serving read the same manifest and take the same --sweep, so their load
    curves pair point by point."""
    assert m.traffic is not None
    levels = m.traffic.levels()
    if sweep is None:
        return [(level.concurrency, level.think_time_ms) for level in levels]
    if len(levels) > 1:
        raise ManifestError(f"--sweep: traffic.load lists {len(levels)} levels; list the levels you want there, or "
                            "sweep a manifest with one level")
    load = levels[0]
    field, values = sweep
    try:
        if field == "traffic.load.concurrency":
            return [(int(v), load.think_time_ms) for v in values]
        if field == "traffic.load.think_time_ms":
            return [(load.concurrency, float(v)) for v in values]
    except ValueError as error:
        raise ManifestError(f"--sweep: {error}") from error
    raise ManifestError(f"--sweep: {field} cannot be swept; this version sweeps traffic.load.concurrency or "
                        "traffic.load.think_time_ms")
