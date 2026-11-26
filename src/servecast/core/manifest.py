"""The manifest: one strict YAML document that holds every input of a command.

Its sections follow a widely used layout for serving simulators (`engine`, `traffic`); ServeCast's own sections
(`calibration`, `versions`, `profile`) are additive. Nothing that decides what is estimated has a default: the KV
cache's size and the workload's source, load and stop rule must be written out. A field ServeCast does not model is
refused by name, never ignored.
"""

from __future__ import annotations

import copy
import difflib
import re
import typing
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, ValidationError, field_validator, model_validator

YEAR_MONTH = r"^\d{4}-(0[1-9]|1[0-2])$"
#: the month a component was released (2026-11; 2026-11b for a second one that month)
COMPONENT_RELEASE = r"^\d{4}-(0[1-9]|1[0-2])[a-z]?$"
#: a component's name: eight hexadecimal characters starting with a letter
COMPONENT_NAME = r"^[a-f][0-9a-f]{7}$"
#: a component in `versions`: its name, or the month it was released
COMPONENT_ID = r"^(\d{4}-(0[1-9]|1[0-2])[a-z]?|[a-f][0-9a-f]{7})$"
DOTTED_ID_HINT = ("write the version as a string with a hyphen, e.g. 2026-10: YAML reads 2026.10 as the number "
                  "2026.1, which is also what January (2026.1) gives")


class Strict(BaseModel):
    # no field of a manifest is infinite or not a number: an infinite think time used to pass and crash predict
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _no_number(value: Any) -> Any:
    if isinstance(value, int | float) and not isinstance(value, bool):
        raise ValueError(DOTTED_ID_HINT)
    return value


# engine ----------------------------------------------------------------------------------------------------

class Parallelism(Strict):
    tensor: int = Field(1, ge=1, description="Tensor-parallel degree. Only 1 is supported in this version.")

    @field_validator("tensor")
    @classmethod
    def _only_one(cls, value: int) -> int:
        if value != 1:
            raise ValueError("tensor parallelism above 1 is not supported yet: the cost laws price one GPU")
        return value


class Scheduler(Strict):
    max_batched_tokens: int | None = Field(
        None, ge=1, description="Most tokens in one engine step (vLLM --max-num-batched-tokens). Unset: vLLM "
        "0.25.1's default for `vllm serve`: 8192 on a GPU with at least 70 GiB other than the A100, else 2048.")
    max_sequences: int | None = Field(
        None, ge=1, description="Most requests running at once (vLLM --max-num-seqs). Unset: vLLM 0.25.1's default "
        "for `vllm serve`: 1024 on a GPU with at least 70 GiB other than the A100, else 256.")


class Capacity(Strict):
    type: Literal["fixed"] = Field(
        "fixed", description="fixed: the engine gets exactly `blocks` KV-cache blocks. The KV capacity is an input, "
        "as in the paper; ServeCast does not estimate it.")
    blocks: int = Field(ge=1, description="KV-cache blocks the engine gets (vLLM --num-gpu-blocks-override). vLLM "
                                          "logs its own count at start-up as 'GPU KV cache size: T tokens' "
                                          "(blocks = T / block_size).")
    memory_fraction: float | None = Field(
        None, gt=0, le=1, description="Fraction of GPU memory vLLM may use (--gpu-memory-utilization), passed to the "
        "engine by profile serving. Unset: 0.9, the value the paper's measurements used. predict and profile serving "
        "refuse a KV cache that, with the weights, needs more than this fraction of the GPU's memory.")


class KvCache(Strict):
    block_size: int | None = Field(None, ge=1, description="Tokens per KV-cache block, a multiple of 16: vLLM 0.25.1's "
                                                           "FlashAttention backend takes no other. Unset: 16, vLLM "
                                                           "0.25.1's default on these GPUs.")
    prefix_caching: bool | None = Field(
        None, description="vLLM's prefix caching. ServeCast computes every prompt in full, so true is refused, and "
        "profile serving runs vLLM with it off (its prompts all start with the same tokens). Unset: off.")
    capacity: Capacity | None = Field(
        None, description="How much KV cache the engine gets. Required: ServeCast does not estimate it, and every "
        "command refuses a manifest without it.")

    @field_validator("block_size")
    @classmethod
    def _flash_attention_blocks(cls, value: int | None) -> int | None:
        if value is not None and value % 16:
            raise ValueError(f"{value} is not a multiple of 16: vLLM 0.25.1's FlashAttention backend, which ServeCast "
                             "pins on NVIDIA GPUs, takes multiples of 16 only")
        return value


class Worker(Strict):
    parallelism: Parallelism = Field(default_factory=Parallelism, description="How the LLM is split across GPUs.")
    scheduler: Scheduler = Field(default_factory=Scheduler, description="The engine's batching limits.")
    kv_cache: KvCache = Field(default_factory=KvCache, description="The engine's KV cache.")


class Workers(Strict):
    aggregated: Worker = Field(default_factory=Worker, description="The single engine that serves every phase.")


class Architecture(Strict):
    """The numbers that define an LLM's architecture, as ServeCast reads them from its config.json."""

    layers: int = Field(ge=1, description="Transformer layers (num_hidden_layers).")
    hidden: int = Field(ge=1, description="Hidden width (hidden_size).")
    ffn: int = Field(ge=1, description="Feed-forward width (intermediate_size).")
    heads: int = Field(ge=1, description="Query heads (num_attention_heads).")
    kv_heads: int = Field(ge=1, description="Key-value heads (num_key_value_heads).")
    kv_width: int = Field(ge=1, description="Key-value heads times the head size.")
    vocab: int = Field(ge=1, description="Vocabulary size (vocab_size).")
    max_context: int = Field(ge=0, description="The LLM's own context limit (max_position_embeddings); 0 when "
                                               "config.json gives none.")


class Engine(Strict):
    mode: Literal["aggregated"] = Field("aggregated", description="One engine serving prefill and decode.")
    model: str | None = Field(None, description="The LLM to predict: a Hugging Face id or a local directory "
                                                "holding config.json. `--to` sets it.")
    hardware: str | None = Field(None, description="The GPU, as an id that servecast list gpus lists, e.g. "
                                                   "a100_sxm_40gb (shipped, or in gpus.json in your data directory). "
                                                   "Required: the GPU predict simulates, with a calibration of that "
                                                   "GPU; the GPU profile calibration and profile serving measure, "
                                                   "which refuse another; the GPU whose profiling data calibrate "
                                                   "fits.")
    backend: Literal["vllm"] = Field("vllm", description="The serving engine.")
    backend_version: str | None = Field(None, description="The vLLM version. ServeCast models vLLM 0.25.1, the "
                                                          "version every calibration and measurement uses; another "
                                                          "is refused. Unset: 0.25.1.")
    context_length: Annotated[int, Field(ge=1)] | Literal["max"] = Field(
        "max", description="Longest prompt plus output, in tokens; max: the LLM's own limit (its config.json's "
        "max_position_embeddings). A trace request longer than this is left out, as vLLM would refuse it.")
    gpu_clock_mhz: float | None = Field(
        None, gt=0, description="The GPU clock to predict at, in MHz. This version predicts at the calibration's clock "
        "only: predict refuses another value, and the profile commands refuse the field (profile.clock locks the "
        "clock). Unset: the clock the calibration was made at.")
    architecture: Architecture | None = Field(
        None, description="Written by ServeCast in a resolved manifest: the numbers that define the LLM, read from "
        "its config.json. When given, they must be the LLM's.")
    workers: Workers = Field(default_factory=Workers, description="The engine's settings.")


# calibration and versions ----------------------------------------------------------------------------------

class Calibration(Strict):
    use: str | None = Field(None, description="A calibration name or path. `--with` sets it.")
    data: Path | None = Field(None, description="A profiling-data directory to fit a calibration from.")
    transfer: bool = Field(False, description="Predict an LLM other than the one the calibration was fitted on: a "
                                              "transfer. A per-GPU calibration is fitted on synthetic models, so "
                                              "every LLM it predicts is a transfer. `--to` sets it.")

    @model_validator(mode="after")
    def _not_both(self) -> Calibration:
        if self.use is not None and self.data is not None:
            raise ValueError("give either `use` or `data`, not both")
        return self


YearMonth = Annotated[str, Field(pattern=YEAR_MONTH)]
ComponentId = Annotated[str, Field(pattern=COMPONENT_ID)]


class VersionsSpec(Strict):
    as_of: YearMonth | None = Field(None, description="Each component at its newest entry up to this month.")
    cost_law: ComponentId | None = Field(None, description="Override: the cost law, by name (or release month).")
    calibration_procedure: ComponentId | None = Field(None, description="Override: the calibration procedure, part "
                                                      "of its cost law's entry: the cost law's name, if set.")
    population_model: ComponentId | None = Field(None, description="Override: the population model, by name.")
    workload_fitter: ComponentId | None = Field(None, description="Override: the workload fitter, by name.")
    length_sampler: ComponentId | None = Field(None, description="Override: the length sampler, by name.")
    estimator: ComponentId | None = Field(None, description="Override: the steady-state estimator, by name.")

    @field_validator("*", mode="before")
    @classmethod
    def _not_a_number(cls, value: Any) -> Any:
        return _no_number(value)


# traffic ---------------------------------------------------------------------------------------------------

class LengthSpec(Strict):
    mean: float = Field(ge=1, description="Mean length, in tokens.")
    variance: float | None = Field(
        None, gt=0, description="Variance, in tokens squared. Unset: the length sampler's default; sampler 2026-11 "
        "uses mean (mean - 1), the variance of a geometric distribution.")


class TraceSource(Strict):
    type: Literal["trace"] = Field(description="Lengths taken from a trace.")
    paths: list[Path] = Field(min_length=1, description="Trace files, or the name of a trace the package ships: "
                                                        "azure-conversation, burstgpt, servegen.")
    format: Literal["mooncake", "lengths"] = Field(
        "mooncake", description="mooncake: JSONL with input_length and output_length; lengths: JSONL or CSV with "
        "input_tokens and output_tokens.")


class SyntheticSource(Strict):
    type: Literal["synthetic"] = Field(description="Lengths drawn from a mean and a variance.")
    input_tokens: LengthSpec = Field(description="The prompt length.")
    output_tokens: LengthSpec = Field(description="The output length.")


class Load(Strict):
    type: Literal["concurrency"] = Field(description="A closed workload: a fixed number of clients.")
    # an upper bound: 10 million clients hung past --timeout and grew past 3 GB; no GPU serves more than this many
    concurrency: int = Field(ge=1, le=100_000, description="N, the number of clients (at most 100,000).")
    think_time_ms: float = Field(gt=0, description="Mean think time between a response and the client's next "
                                                   "request, in milliseconds; exponential.")


class SteadyState(Strict):
    ttft: float = Field(0.10, gt=0, lt=1, description="Target relative half-width of TTFT's 95 % interval.")
    tpot: float = Field(0.05, gt=0, lt=1, description="Target relative half-width of TPOT's 95 % interval.")
    throughput: float = Field(0.05, gt=0, lt=1, description="Target relative half-width of throughput's interval.")
    e2e: float = Field(0.10, gt=0, lt=1, description="Target relative half-width of the end-to-end latency's interval.")
    ttft_p95: float | None = Field(
        None, gt=0, lt=1, description="Target relative half-width of the 95th percentile of TTFT. Unset: 0.10 with "
        "the steady-state estimator 2026-11; percentiles are not judged by 2026-10.")
    tpot_p95: float | None = Field(
        None, gt=0, lt=1, description="Target relative half-width of the 95th percentile of TPOT. Unset: 0.10 with "
        "the steady-state estimator 2026-11; percentiles are not judged by 2026-10.")


class Stop(Strict):
    steady_state: SteadyState = Field(description="Run until every metric's interval is narrow enough.")


#: the fewest requests a pool holds: `pool_size` draws of synthetic lengths, or the requests of a trace that fit the
#: context length (with fewer, the workload law cannot be fitted). Above it a small pool can still fail the fit, which
#: depends on the draw (`model.run.workload_law` refuses it then, as an input error)
MIN_POOL = 100


def _load_shape(value: Any) -> str:
    """Which of traffic.load's two shapes the manifest wrote: a list of load levels, or one level."""
    return "levels" if isinstance(value, list) else "level"


class Traffic(Strict):
    source: Annotated[TraceSource | SyntheticSource, Field(discriminator="type")] = Field(
        description="Where request lengths come from.")
    load: Annotated[Annotated[Load, Tag("level")] | Annotated[list[Load], Field(min_length=1), Tag("levels")],
                    Discriminator(_load_shape)] = Field(
        description="How requests arrive: one load level, or a list of levels. Each level is predicted and measured "
                    "on its own, at steady state, and reported in the order listed.")
    stop: Stop = Field(description="When a run is long enough.")
    seed: int = Field(0, ge=0, description="Seed of the request pool and of the simulation.")
    pool_size: int = Field(10_000, ge=MIN_POOL, description="Requests in the pool both arms draw from, for synthetic "
                           "lengths; a trace's pool is its requests that fit the context length, at least 100.")

    @field_validator("load")
    @classmethod
    def _each_level_once(cls, load: Load | list[Load]) -> Load | list[Load]:
        """compare pairs a prediction's levels with a measurement's by clients and think time: each is listed once."""
        if isinstance(load, list):
            first: dict[tuple[int, float], int] = {}
            for i, level in enumerate(load):
                j = first.setdefault((level.concurrency, level.think_time_ms), i)
                if j != i:
                    raise ValueError(f"levels {j} and {i} are the same ({level.concurrency} clients, think time "
                                     f"{level.think_time_ms:g} ms); list each level once")
        return load

    def levels(self) -> list[Load]:
        """The load levels in the order listed: the list, or the one level."""
        return list(self.load) if isinstance(self.load, list) else [self.load]


# profile ---------------------------------------------------------------------------------------------------

class ClockLock(Strict):
    lock: int = Field(gt=0, description="Lock the GPU clock at this many MHz, if the machine allows it and the clock "
                                        "holds it under load.")


class Profile(Strict):
    clock: Literal["auto", "measured"] | ClockLock = Field(
        "auto", description="auto locks the GPU clock where this machine allows it (NVML, or sudo -n nvidia-smi): "
        "profile calibration at the highest level the clock holds under load, profile serving at the level its "
        "calibration was made at. Where the clock cannot be locked, or does not hold, it is measured. measured never "
        "locks; {lock: MHz} locks at that level. The clock is always measured and recorded.")
    poweroff_when_safe: bool = Field(False, description="Power the machine off once the results are written and "
                                                        "match their checksums, as --poweroff-when-safe does.")


# the manifest ----------------------------------------------------------------------------------------------

class Manifest(Strict):
    engine: Engine = Field(default_factory=Engine, description="The deployment: LLM, GPU and engine settings.")
    calibration: Calibration = Field(default_factory=Calibration, description="Which calibration, or what to fit.")
    versions: YearMonth | VersionsSpec | None = Field(
        None, description="The components: a year-month (each component's newest entry up to it), or {as_of: ..., "
                          "<component>: <name>}. Unset: each component's default (servecast list components).")
    traffic: Traffic | None = Field(None, description="The workload. Required by predict, profile serving and "
                                                      "workload sample.")
    profile: Profile = Field(default_factory=Profile, description="Read by the profile commands only.")

    @field_validator("versions", mode="before")
    @classmethod
    def _versions_not_a_number(cls, value: Any) -> Any:
        return _no_number(value)


# validation with messages a person can act on --------------------------------------------------------------

UNSUPPORTED_TOP = ("router", "planner", "execution", "evaluation", "optimization", "optimizer")
UNSUPPORTED_ENGINE = ("speculation", "afd", "kv_transfer", "nextn")
UNSUPPORTED_WORKER = ("timing", "startup_seconds")
UNSUPPORTED_KV = ("host_offload", "state_cache", "bytes_per_token", "prefix_match_unit")
OPEN_LOOP = ("poisson", "constant_rate", "trace_timestamps", "kv_capacity_fraction")
FINITE_STOPS = ("requests", "requests_per_load_unit", "sessions", "sessions_per_load_unit", "max_virtual_time_seconds")
KV_CAPACITY_IS_AN_INPUT = (
    "the KV capacity is an input: write {blocks: N}. vLLM logs its own count at start-up as 'GPU KV cache size: T "
    "tokens' (N = T / block_size); or choose N and impose it on the engine with --num-gpu-blocks-override")
PREFIX_CACHING_NOT_MODELLED = ("shared prefixes are not modelled; every prompt is computed in full (set it to false or "
                              "remove it)")
#: what every manifest must give, by path, with how to write it. The schema leaves them out of its own required lists,
#: so that a missing one is reported once, at its full path, with the fix (a missing `engine` would otherwise be
#: reported as `engine`, then `engine.workers`, one level per attempt)
KV_CAPACITY_REQUIRED = ("write {blocks: N}; vLLM logs its own count at start-up as 'GPU KV cache size: T tokens' "
                        "(N = T / block_size), or choose N and impose it with --num-gpu-blocks-override")
GPU_REQUIRED = ("write the GPU's id, one that servecast list gpus lists: the GPU predict simulates, the one profile "
                "calibration and profile serving measure, and the one whose profiling data calibrate fits. A GPU of "
                "your own goes in gpus.json in your data directory, which servecast doctor names")
REQUIRED = {"engine.hardware": GPU_REQUIRED, "engine.workers.aggregated.kv_cache.capacity": KV_CAPACITY_REQUIRED}


class ManifestError(ValueError):
    """A manifest that cannot run: the message names the field and the fix (exit status 2)."""


#: the fields that make a prediction and a measurement the same experiment, by dotted path in the resolved manifest.
#: A key a newer version adds inside one of them (inside `engine.architecture`, say) changes what is compared, so it is
#: refused too.
SEMANTIC_FIELDS = (
    "engine.mode", "engine.backend", "engine.hardware", "engine.context_length", "engine.architecture",
    "engine.workers.aggregated.parallelism.tensor", "engine.workers.aggregated.scheduler.max_batched_tokens",
    "engine.workers.aggregated.scheduler.max_sequences", "engine.workers.aggregated.kv_cache.block_size",
    "engine.workers.aggregated.kv_cache.prefix_caching", "engine.workers.aggregated.kv_cache.capacity.type",
    "engine.workers.aggregated.kv_cache.capacity.blocks", "traffic.seed", "traffic.stop.steady_state",
)
#: the fields left out on purpose: the LLM's name (its architecture stands for it; a different name is flagged), what
#: is observed rather than declared (the vLLM version and the GPU clock, both flagged when they differ), what one side
#: alone reads (memory_fraction, calibration, versions, profile), the load (paired point by point), and where the
#: lengths come from (the pool's digest stands for them). Any other field, such as one a newer version adds, is
#: flagged when the two sides differ on it.
NOT_SEMANTIC = (
    "engine.model", "engine.backend_version", "engine.gpu_clock_mhz",
    "engine.workers.aggregated.kv_cache.capacity.memory_fraction", "traffic.source", "traffic.pool_size",
    "traffic.load", "calibration", "versions", "profile",
)


def _at(doc: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        if not isinstance(doc, dict):
            return None
        doc = doc.get(key)
    if isinstance(doc, dict):
        return {k: v for k, v in doc.items() if v is not None}
    return doc


def semantic_view(resolved: dict[str, Any]) -> dict[str, Any]:
    """The resolved manifest's value at each of SEMANTIC_FIELDS (None where absent; a mapping without its null
    entries). Two resolved manifests describe the same experiment when their views are equal."""
    return {path: _at(resolved, path) for path in SEMANTIC_FIELDS}


@dataclass(frozen=True)
class Problem:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def _get(raw: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(raw, list) and key.isdigit() and int(key) < len(raw):
            raw = raw[int(key)]  # a list's entry, as a path names it: traffic.load.1
            continue
        if not isinstance(raw, dict) or key not in raw:
            return None
        raw = raw[key]
    return raw


def unsupported(raw: dict[str, Any]) -> list[Problem]:
    """Fields another tool accepts that ServeCast does not model: refused by name, with what to do instead."""
    found: list[Problem] = []
    not_modelled = "not supported by ServeCast; remove it"
    for key in UNSUPPORTED_TOP:
        if key in raw:
            found.append(Problem(key, not_modelled))
    for key in UNSUPPORTED_ENGINE:
        if isinstance(raw.get("engine"), dict) and key in raw["engine"]:
            found.append(Problem(f"engine.{key}", not_modelled))
    workers = _get(raw, "engine", "workers")
    if isinstance(workers, dict):
        for role in workers:
            if role != "aggregated":
                found.append(Problem(f"engine.workers.{role}", "only engine.workers.aggregated is supported"))
    worker = _get(raw, "engine", "workers", "aggregated")
    if isinstance(worker, dict):
        for key in UNSUPPORTED_WORKER:
            if key in worker:
                found.append(Problem(f"engine.workers.aggregated.{key}", not_modelled))
        for key in _get(worker, "parallelism") or {}:
            if key != "tensor":
                found.append(Problem(f"engine.workers.aggregated.parallelism.{key}",
                                     "only tensor parallelism is modelled; remove it"))
        for key in UNSUPPORTED_KV:
            if isinstance(worker.get("kv_cache"), dict) and key in worker["kv_cache"]:
                found.append(Problem(f"engine.workers.aggregated.kv_cache.{key}", not_modelled))
        if _get(worker, "kv_cache", "prefix_caching") is True:
            found.append(Problem("engine.workers.aggregated.kv_cache.prefix_caching", PREFIX_CACHING_NOT_MODELLED))
        capacity_type = _get(worker, "kv_cache", "capacity", "type")
        if capacity_type is not None and capacity_type != "fixed":
            found.append(Problem("engine.workers.aggregated.kv_cache.capacity.type", KV_CAPACITY_IS_AN_INPUT))
    if isinstance(raw.get("profile"), dict) and "gpu" in raw["profile"]:
        found.append(Problem("profile.gpu", "not read by ServeCast: a GPU its registry does not list goes in gpus.json "
                             "in your data directory, which servecast doctor names; remove it"))
    source, load, stop = (_get(raw, "traffic", k) for k in ("source", "load", "stop"))
    if isinstance(source, dict):
        if source.get("type") == "synthetic-session":
            found.append(Problem("traffic.source.type", "session workloads are not supported; use synthetic or trace"))
        for key in ("input_tokens", "output_tokens"):
            if isinstance(source.get(key), int | float) and not isinstance(source.get(key), bool):
                found.append(Problem(f"traffic.source.{key}", "fixed lengths are not supported: write {mean: N} "
                                     "(exponential) or {mean: N, variance: V}"))
    for i, level in enumerate(load if isinstance(load, list) else [load]):
        if isinstance(level, dict) and level.get("type") in OPEN_LOOP:
            where = f"traffic.load.{i}.type" if isinstance(load, list) else "traffic.load.type"
            found.append(Problem(where, "closed workloads only: use type: concurrency with concurrency and "
                                 "think_time_ms"))
    if isinstance(stop, dict):
        for key in FINITE_STOPS:
            if key in stop:
                found.append(Problem(f"traffic.stop.{key}",
                                     "finite runs are not supported: use stop: {steady_state: {}}"))
    return found


def missing(raw: dict[str, Any]) -> list[Problem]:
    """The required fields a manifest leaves out, each with what to write."""
    return [Problem(path, f"required: {how}") for path, how in REQUIRED.items() if _get(raw, *path.split(".")) is None]


def unknown_gpu(raw: dict[str, Any]) -> list[Problem]:
    """An engine.hardware that no GPU registry lists: the shipped one, and yours (gpus.json in the data directory)."""
    from servecast.core import gpus  # here, not at the top: core.formats imports this module

    hardware = _get(raw, "engine", "hardware")
    if not isinstance(hardware, str):
        return []
    try:
        known = gpus.registry()
    except ValueError as error:
        return [Problem("engine.hardware", f"your GPU registry cannot be read: {error}")]
    if hardware in known:
        return []
    return [Problem("engine.hardware", f"{hardware}: no GPU of that id; servecast list gpus lists the known ones "
                                       f"({', '.join(sorted(known))}), and a GPU of your own goes in "
                                       f"{gpus.user_registry()}")]


def _without(raw: dict[str, Any], paths: Iterable[str]) -> dict[str, Any]:
    cleaned = copy.deepcopy(raw)
    for path in paths:
        keys = path.split(".")
        parent = _get(cleaned, *keys[:-1]) if len(keys) > 1 else cleaned
        if isinstance(parent, dict):
            parent.pop(keys[-1], None)
    return cleaned


def _models_in(annotation: Any) -> list[type[BaseModel]]:
    """The models a field can hold, through unions, Annotated and lists."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [m for arg in typing.get_args(annotation) for m in _models_in(arg)]


def _member(models: list[type[BaseModel]], node: Any) -> type[BaseModel] | None:
    """The member of a union the input names by its `type` (a synthetic source, not a trace's), else the first."""
    if isinstance(node, dict) and "type" in node:
        for model in models:
            field = model.model_fields.get("type")
            if field is not None and node["type"] in typing.get_args(field.annotation):
                return model
    return next(iter(models), None)


def _model_at(loc: tuple[Any, ...], raw: Any) -> type[BaseModel] | None:
    """The model of the mapping at `loc`, walking the input beside it to choose a union's member."""
    model: Any = Manifest
    node = raw
    for key in loc:
        if str(key).isdigit():  # a list's entry holds the list's model
            node = node[int(key)] if isinstance(node, list) and int(key) < len(node) else None
            continue
        field = getattr(model, "model_fields", {}).get(key)
        if field is None:
            return None
        node = node.get(key) if isinstance(node, dict) else None
        model = _member(_models_in(field.annotation), node)
        if model is None:
            return None
    return model


def _closest(word: str, choices: Iterable[str]) -> str | None:
    """The field a misspelling most likely meant: a transposition first ("modle" is model, not mode), then difflib's
    closest."""
    close = difflib.get_close_matches(word, list(choices), n=3)
    return next((c for c in close if sorted(c) == sorted(word)), close[0] if close else None)


#: a union member's tag in a pydantic location, not a field the user wrote: a model's class name (fields are
#: lowercase), or a constrained, literal or plain value; validators and tagged unions add tags of their own
_ALTERNATIVE = re.compile(r"^(?:[A-Z]\w*|constrained-\w+|literal\[.*\]|int|str|float|bool|none|list\[.*\]|dict\[.*\])$")
_WRAPPER = ("function-", "tagged-union")


def _problems_from(error: ValidationError, raw: Any) -> list[Problem]:
    """The schema's complaints on the paths the user wrote. Where a field takes a value or a mapping, pydantic reports
    each alternative under its tag: the user's input chooses which apply (a mapping, or a value), and the value's
    alternatives make one problem, which says when a mapping would do too."""
    problems: list[Problem] = []
    values: dict[str, list[str]] = {}  # a union field's path -> its value alternatives' messages
    mapping_too: set[str] = set()      # union fields whose input was a value, although a mapping would do
    for item in error.errors():
        parts = [str(part) for part in item["loc"] if not str(part).startswith(_WRAPPER)]
        tag = next((i for i, part in enumerate(parts) if _ALTERNATIVE.match(part)), None)
        loc = tuple(part for part in parts if not _ALTERNATIVE.match(part))
        if len(loc) > 2 and ((loc[:2] == ("traffic", "source") and loc[2] in ("trace", "synthetic"))
                             or (loc[:2] == ("traffic", "load") and loc[2] in ("level", "levels"))):
            loc = loc[:2] + loc[3:]  # the union's tag is not a field the user wrote
        message = item["msg"].removeprefix("Value error, ")
        if item["type"] == "string_pattern_mismatch" and (item.get("ctx") or {}).get("pattern") == YEAR_MONTH:
            message = "Input should be a year-month such as 2026-11"
        if tag is not None:
            field = ".".join(part for part in parts[:tag] if not _ALTERNATIVE.match(part))
            mapping = parts[tag][0].isupper()
            if isinstance(_get(raw, *field.split(".")), dict) != mapping:  # an alternative the input did not choose
                if mapping:
                    mapping_too.add(field)
                continue
            if not mapping:
                values.setdefault(field, []).append(message)
                continue
        if item["type"] == "extra_forbidden" and loc:
            parent = _model_at(loc[:-1], raw)
            close = _closest(loc[-1], parent.model_fields if parent else [])
            message = f"unknown field{f'; did you mean {close}?' if close else ''}"
        problems.append(Problem(".".join(loc) or "(manifest)", message))
    for field, messages in values.items():
        merged = messages[0] + "".join(f", or {m.removeprefix('Input should be ')}" for m in messages[1:])
        if field in mapping_too:
            merged += f", or a mapping (servecast manifest explain {field})"
        problems.append(Problem(field, merged))
    return problems


def validate(raw: Any) -> tuple[Manifest | None, list[Problem]]:
    """Every problem at once: the refused fields first, then the missing ones, then the schema's own complaints."""
    if not isinstance(raw, dict):
        return None, [Problem("(manifest)", "a manifest is a YAML mapping")]
    refused = unsupported(raw)
    absent = missing(raw) + unknown_gpu(raw)
    try:
        manifest = Manifest.model_validate(_without(raw, (p.path for p in refused)))
    except ValidationError as error:
        seen = {p.path for p in refused}
        # a refused type refuses its mapping: the fields that type would have needed or not taken are not problems
        seen |= {p.path.removesuffix(".type") for p in refused if p.path.endswith(".type")}
        rest = [p for p in _problems_from(error, _without(raw, seen))
                if not any(p.path == s or p.path.startswith(s + ".") for s in seen)]
        return None, refused + absent + rest
    return (None, refused + absent) if refused or absent else (manifest, [])


def apply_sets(raw: dict[str, Any], sets: Iterable[str]) -> tuple[dict[str, Any], list[Problem]]:
    """`--set PATH=VALUE`, left to right; VALUE is read as YAML. A whole mapping or sequence replaces the old one."""
    result = copy.deepcopy(raw)
    problems: list[Problem] = []
    for item in sets:
        path, sep, text = item.partition("=")
        if not sep or not path:
            problems.append(Problem("--set", f"expected PATH=VALUE, got {item!r}"))
            continue
        keys = path.split(".")
        trial = copy.deepcopy(result)
        node = trial
        for depth, key in enumerate(keys[:-1]):
            value = node.get(key)
            if value is not None and not isinstance(value, dict):
                if depth == 0 and key == "versions":  # a date: the component overrides keep it, as `as_of`
                    value = {"as_of": value}
                else:
                    where = ".".join(keys[: depth + 1])
                    problems.append(Problem("--set", f"{path}: {where} is {value!r}, not a mapping; set {where} "
                                                     "itself"))
                    break
            node[key] = value if isinstance(value, dict) else {}
            node = node[key]
        else:
            node[keys[-1]] = yaml.safe_load(text)
            result = trial
    return result, problems


class _UniqueKeyLoader(yaml.SafeLoader):
    """yaml.safe_load keeps the last of two equal keys; a manifest refuses them, since the one it keeps is a guess."""


def _mapping_without_duplicates(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    pairs = loader.construct_pairs(node, deep=True)
    seen: set[Any] = set()
    for (key, _), (key_node, _) in zip(pairs, node.value, strict=True):
        if key in seen:
            raise yaml.constructor.ConstructorError("while reading a mapping", node.start_mark,
                                                    f"duplicate key {key!r}", key_node.start_mark)
        seen.add(key)
    return dict(pairs)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping_without_duplicates)


def shipped() -> dict[str, Path]:
    """The manifests that ship with ServeCast, by name (servecast list manifests)."""
    root = Path(str(resources.files("servecast").joinpath("data", "manifests")))
    return {p.stem: p for p in sorted(root.glob("*.yaml"))}


def find(path: Path) -> Path:
    """A manifest file, or the name of a manifest that ships with ServeCast."""
    return path if path.is_file() else shipped().get(str(path), path)


def config_problem(path: Path) -> str | None:
    """Why `-c` cannot read `path` as a manifest, or None. A file that holds no mapping says so when it is read."""
    if str(path) == "-":
        return "-: a manifest is read from a file, not from standard input; save it and give its path"
    if path.is_dir():
        return f"{path}: a directory; -c takes a manifest file"
    if not path.exists():
        return f"{path}: no such file, and no manifest of that name ships with ServeCast ({', '.join(shipped())})"
    return None


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_UniqueKeyLoader)  # a SafeLoader subclass: no arbitrary objects


# explain ---------------------------------------------------------------------------------------------------

def _alternatives(node: dict[str, Any], defs: dict[str, Any]) -> list[dict[str, Any]]:
    if "$ref" in node:
        return _alternatives(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
    options = node.get("anyOf") or node.get("oneOf")
    if options:
        return [alt for option in options for alt in _alternatives(option, defs)]
    return [node]


def _kind(node: dict[str, Any], defs: dict[str, Any]) -> str:
    kinds = []
    for alt in _alternatives(node, defs):
        if "const" in alt:
            kinds.append(repr(alt["const"]))
        elif "enum" in alt:
            kinds += [repr(v) for v in alt["enum"]]
        elif alt.get("type") == "object":
            kinds.append(alt.get("title", "mapping"))
        elif alt.get("type") == "array" and isinstance(alt.get("items"), dict):
            kinds.append(f"list of {_kind(alt['items'], defs)}")
        else:
            kinds.append(str(alt.get("type", "any")))
    return " | ".join(dict.fromkeys(kinds))


def _members(node: dict[str, Any], defs: dict[str, Any]) -> list[dict[str, Any]]:
    """What a dotted path can step into from a field: its alternatives and, for a list, its entries'."""
    out = []
    for alt in _alternatives(node, defs):
        out.append(alt)
        if alt.get("type") == "array" and isinstance(alt.get("items"), dict):
            out += _members(alt["items"], defs)
    return out


def _limits(node: dict[str, Any], defs: dict[str, Any]) -> list[str]:
    signs = {"minimum": ">=", "exclusiveMinimum": ">", "maximum": "<=", "exclusiveMaximum": "<",
             "minLength": "at least", "minItems": "at least", "pattern": "matches"}
    limits = []
    for alt in _alternatives(node, defs):
        limits += [f"{sign} {alt[key]}" for key, sign in signs.items() if key in alt]
    return list(dict.fromkeys(limits))


def explain(field_path: str) -> str:
    """A field's meaning, type, default and limits, read from the manifest's own schema."""
    schema = Manifest.model_json_schema()
    defs = schema.get("$defs", {})
    node: dict[str, Any] = schema
    required = False
    for key in field_path.split("."):
        if key.isdigit() and any(alt.get("type") == "array" for alt in _alternatives(node, defs)):
            continue  # traffic.load.1.concurrency: every entry of a list has the same fields
        holders = [alt for alt in _members(node, defs) if key in alt.get("properties", {})]
        if not holders:
            names = sorted({n for alt in _members(node, defs) for n in alt.get("properties", {})})
            close = difflib.get_close_matches(key, names, n=1)
            hint = f"; did you mean {close[0]}?" if close else f"; fields here: {', '.join(names) or 'none'}"
            raise KeyError(f"{field_path}: no field {key!r}{hint}")
        required = key in holders[0].get("required", [])
        node = holders[0]["properties"][key]
    required = required or field_path in REQUIRED
    lines = [field_path, f"  {node.get('description', '(no description)')}", f"  type: {_kind(node, defs)}"]
    lines.append("  required" if required else f"  default: {node.get('default', 'none')!r}")
    limits = _limits(node, defs)
    if limits:
        lines.append(f"  limits: {', '.join(limits)}")
    return "\n".join(lines)
