"""The files ServeCast writes. They are its only programmatic interface, so each is a published JSON Schema.

Every file carries `schema: servecast-<kind>/<major>`. A reader refuses a major it does not know and ignores fields
it does not know inside a known major, so a minor release can add fields. Units are in field names; component ids
are strings.
"""

from __future__ import annotations

import math
import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from servecast.core.manifest import COMPONENT_NAME, COMPONENT_RELEASE, Manifest


class Part(BaseModel):
    """A part of an output file: unknown fields are ignored, so newer writers stay readable."""

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class Output(Part):
    KIND: ClassVar[str] = ""
    MAJOR: ClassVar[int] = 1
    schema_id: str = Field(alias="schema", description="servecast-<kind>/<major>.")

    @field_validator("schema_id")
    @classmethod
    def _known_major(cls, value: str) -> str:
        expected = f"servecast-{cls.KIND}/{cls.MAJOR}"
        if value != expected:
            raise ValueError(f"this version reads {expected}, not {value}")
        return value


class Interval(Part):
    low: float
    high: float

    @model_validator(mode="after")
    def _a_finite_ordered_interval(self) -> Interval:
        if not (math.isfinite(self.low) and math.isfinite(self.high) and self.low <= self.high):
            raise ValueError(f"[{self.low}, {self.high}] is not an interval: the bounds must be finite, low <= high")
        return self


class Summary(Part):
    mean_ttft_ms: float | None = Field(description="Mean time to first token, ms; null when unavailable.")
    p95_ttft_ms: float | None = Field(description="95th percentile of time to first token, ms.")
    mean_tpot_ms: float | None = Field(description="Mean time per output token, ms: per request, (last token - "
                                                   "first token) / (output tokens - 1).")
    p95_tpot_ms: float | None = Field(description="95th percentile of time per output token, ms.")
    output_throughput_tok_s: float | None = Field(description="Output tokens per second.")
    request_throughput_rps: float | None = Field(description="Completed requests per second.")
    num_ttft_samples: int = Field(ge=0, description="Requests behind the TTFT statistics.")
    num_tpot_samples: int = Field(ge=0, description="Requests behind the TPOT statistics.")
    duration_ms: float = Field(ge=0, allow_inf_nan=False, description="The steady-state window the statistics "
                                                                       "cover, ms.")
    wall_time_ms: float | None = Field(None, ge=0, allow_inf_nan=False, description="Time the computation took on "
                                                                                    "this computer, ms.")
    ci95: dict[str, Interval] = Field(default_factory=dict, description="95 % interval of each metric above, keyed by "
                                                                        "the metric's field name.")

    @field_validator("mean_ttft_ms", "p95_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "output_throughput_tok_s",
                     "request_throughput_rps")
    @classmethod
    def _finite_and_not_negative(cls, value: float | None) -> float | None:
        if value is not None and not (math.isfinite(value) and value >= 0):
            raise ValueError(f"{value} is not a finite non-negative number; write null for a metric not available")
        return value

    @field_validator("ci95")
    @classmethod
    def _known_metrics(cls, value: dict[str, Interval]) -> dict[str, Interval]:
        unknown = sorted(set(value) - set(SUMMARY_METRICS))
        if unknown:
            raise ValueError(f"intervals for {unknown}, which are not metrics; known: {', '.join(SUMMARY_METRICS)}")
        return value


#: the metrics a summary reports, by field name
SUMMARY_METRICS = ("mean_ttft_ms", "p95_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "output_throughput_tok_s",
                   "request_throughput_rps")


class LoadPoint(Part):
    concurrency: int = Field(ge=1, description="Clients.")
    think_time_ms: float = Field(gt=0, allow_inf_nan=False, description="Mean think time, ms.")


class Point(Part):
    load: LoadPoint
    summary: Summary
    converged: bool = Field(True, description="Whether every judged metric met its target; false only when a time "
                                               "budget stopped the run first.")
    stop_reason: str | None = Field(None, description="converged, or what stopped the run.")
    diagnostics: dict[str, Any] = Field(default_factory=dict, description="Model internals, for reading only.")


class MeasuredPoint(Point):
    converged: bool = Field(description="Whether every judged metric met its target before the run was stopped.")
    measured_clock_mhz: float | None = Field(description="Mean GPU clock over the steady-state window, MHz.")
    kv_blocks: int | None = Field(description="The KV-cache blocks the engine ran with, read back from it.")
    kv_blocks_computed: int | None = Field(None, description="The blocks vLLM computed from the GPU's free memory, "
                                                             "before the manifest's capacity overrode them.")


class Estimand(Part):
    rule: Literal["steady_state"]
    ttft: float = Field(description="Target relative half-width for TTFT.")
    tpot: float = Field(description="Target relative half-width for TPOT.")
    throughput: float = Field(description="Target relative half-width for throughput.")
    e2e: float | None = Field(None, description="Target relative half-width for the end-to-end latency.")
    ttft_p95: float | None = Field(None, description="Target relative half-width for TTFT's 95th percentile.")
    tpot_p95: float | None = Field(None, description="Target relative half-width for TPOT's 95th percentile.")


class Validity(Part):
    """Internal to ServeCast: not part of its interface, and never printed."""

    status: Literal["inside", "outside"]
    reasons: list[str] = Field(default_factory=list)


class Provenance(Part):
    components: dict[str, str] = Field(description="Each component that ran, by name (servecast list components).")
    servecast_version: str
    seed: int | None = None
    pool_sha256: str | None = None
    calibration: str | None = Field(None, description="The calibration's name.")
    calibration_sha256: str | None = None
    vllm_version: str | None = Field(None, description="Measurements only.")


class Prediction(Output):
    KIND: ClassVar[str] = "prediction"
    points: list[Point] = Field(min_length=1, description="One per load point; several with --sweep.")
    estimand: Estimand
    validity: Validity = Field(description="Internal to ServeCast; not part of its interface.")
    provenance: Provenance
    resolved_manifest: dict[str, Any] = Field(description="The manifest with every default filled in.")
    warnings: list[str] = Field(default_factory=list, description="What the run noticed but did not refuse.")


class Measurement(Output):
    KIND: ClassVar[str] = "measurement"
    points: list[MeasuredPoint] = Field(min_length=1)
    estimand: Estimand
    provenance: Provenance
    resolved_manifest: dict[str, Any] = Field(description="The manifest with every default filled in, written by the "
                                                          "same resolver as a prediction's, so compare can pair them.")


class Compared(Part):
    predicted: float | None
    measured: float | None
    measured_ci95: Interval | None = None
    signed_error_pct: float | None = Field(description="(predicted - measured) / measured, in percent.")
    within_ci: bool | None = Field(None, description="Whether the prediction falls inside the measured interval.")


class ComparedPoint(Part):
    load: LoadPoint
    metrics: dict[str, Compared]
    measured_certified: bool = Field(description="Whether the measurement met every target of its stop rule.")
    predicted_converged: bool = Field(True, description="Whether the prediction met every target of its stop rule.")


class Checks(Part):
    """The checks compare ran. A pair that fails one is refused and gets no comparison, so each is true; null marks
    a check that does not apply."""

    manifest: bool = Field(description="Same manifest, compared on its meaning.")
    estimand: bool = Field(True, description="Same stop rule, each the one its resolved manifest records.")
    pool: bool = Field(description="Same request pool, by the digest each provenance records.")
    sampler: bool | None = Field(description="Same length-sampler id; null when the lengths come from a trace.")
    estimator: bool = Field(description="Same steady-state estimator id.")


class Comparison(Output):
    KIND: ClassVar[str] = "comparison"
    prediction_sha256: str
    measurement_sha256: str
    checks: Checks
    points: list[ComparedPoint] = Field(min_length=1)
    flags: list[str] = Field(default_factory=list, description="e.g. a clock or KV-capacity mismatch.")


class ProbeCell(Part):
    """One probe cell: a batch composition the engine served, timed. The lists are its repeated measurements; the fit
    takes their mean."""

    q: float = Field(ge=0, description="Prefill tokens computed in the iteration.")
    nd: float = Field(ge=0, description="Decoding requests in the iteration.")
    A: list[float] = Field(min_length=1, description="The prefill attention work, the sum of q (w + q/2) over the "
                                                     "prefill chunks (w: tokens already computed).")
    K: list[float] = Field(min_length=1, description="The decode context: the decoding requests' contexts, summed, "
                                                     "in tokens.")
    n_pf: list[float] = Field(min_length=1, description="Prefill requests in the iteration.")
    T: float = Field(gt=0, description="The measured iteration duration, in ms.")
    w: float = Field(gt=0, description="The cell's weight in the fit, before the procedure's class shares.")
    cls: Literal["decode", "prefill", "mixed"] = Field(description="The probe class: decoding only, prefill only, "
                                                                   "or both.")


class ProbedModel(Part):
    llm: str = Field(description="The synthetic model probed (synthetic/<family>-l<L>-d<D>-f<F>-h<H>-kv<KV>).")
    cells: list[ProbeCell] = Field(min_length=1)


class GemmPoint(Part):
    n: int = Field(ge=1, description="Rows of the matrix multiplications.")
    s_per_iter: float = Field(gt=0, description="Seconds per layer iteration.")
    flops: float = Field(gt=0, description="Floating-point operations per layer iteration.")


class GemmRun(Part):
    model: str = Field(description="The LLM whose layer shapes were timed, or a synthetic layer "
                                   "(synthetic-d<d>-f<d_ff>-k<d_kv>).")
    reps: list[list[GemmPoint]] = Field(min_length=1, description="Repetitions, each a sweep over the row counts.")


class GemmGrid(Part):
    clock_mhz: float = Field(gt=0, description="The SM clock the grid ran at; the probes' clock.")
    runs: list[GemmRun] = Field(min_length=1)


class ProbeBeat(Part):
    prefill: list[list[int]] = Field(description="The prefill chunks, as [tokens, tokens already computed].")
    decode: list[list[int]] = Field(description="The decoding requests, as [context, requests].")
    dur_ms: float = Field(gt=0, description="The engine step's duration on the GPU, ms.")
    t_ms: float = Field(ge=0, description="When it ended, ms from the first beat.")


class ProbeSpan(Part):
    kind: Literal["decode", "prefill", "mixed", "reference"]
    task: str = Field(description="The design's task: <kind>:<index>.")
    beat_from: int = Field(ge=0, description="The span's first beat, an index into beats.")
    beat_to: int = Field(ge=0, description="One past its last beat.")


class ClockSample(Part):
    beat_index: int = Field(ge=0, description="The beat after which the clock was read.")
    sm_clock_mhz: float = Field(description="The SM clock, MHz.")


class ProbeAudit(Part):
    clean: bool = Field(description="No flag: the probe makes profiling data.")
    flags: list[str] = Field(description="Why the probe makes no profiling data, each a sentence.")
    compositions_designed: dict[str, int]
    compositions: int = Field(ge=0, description="Compositions measured.")
    beats_kept: int = Field(ge=0, description="Beats of the composition their cell was made for.")
    reference_drift: float | None = Field(description="max / min - 1 of the reference revisits' mean beat.")
    clock_held: bool
    cells: list[dict[str, Any]] = Field(description="Per composition: beats run and kept, and whether enough.")


class Probe(Output):
    """probe.json: what `profile calibration` measured, beat by beat, and its audit; the profiling data are made
    from it."""

    KIND: ClassVar[str] = "probe"
    gpu: str = Field(description="The GPU's id.")
    llm: str = Field(description="The LLM probed.")
    cost_law: str = Field(description="The cost law the probe was designed for, by name.")
    design: dict[str, Any] = Field(description="The compositions and their order.")
    spans: list[ProbeSpan] = Field(description="One per composition measured, in the order they ran.")
    beats: list[ProbeBeat] = Field(description="Every engine step timed.")
    clock: list[ClockSample] = Field(description="The SM clock, sampled every few beats.")
    audit: ProbeAudit


class GemmTimed(Part):
    n: int = Field(ge=1, description="Rows of the matrix multiplications.")
    s_per_iter: float = Field(gt=0, description="Seconds per layer iteration.")
    iters: int = Field(ge=1, description="Iterations timed.")
    flops: float = Field(gt=0, description="Floating-point operations per layer iteration.")
    sm_mhz: float | None = Field(description="The mean SM clock while the point ran, MHz.")


class GemmTimedRun(Part):
    model: str = Field(description="The layer timed: an LLM's, or a synthetic one (synthetic-d<d>-f<d_ff>-k<d_kv>).")
    reps: list[list[GemmTimed]] = Field(min_length=1, description="Repetitions, each a sweep over the row counts.")


class Gemm(Output):
    """gemm.json: the GEMM grid `profile calibration` timed for a per-GPU law, point by point, and its audit."""

    KIND: ClassVar[str] = "gemm"
    gpu: str = Field(description="The GPU's id.")
    clock_mhz: float | None = Field(description="The grid's SM clock: the lock, else its points' mean, snapped to "
                                                "25 MHz; null when they disagree.")
    seconds: float = Field(gt=0, description="Wall-clock seconds each point was timed for.")
    runs: list[GemmTimedRun]
    clean: bool = Field(description="No flag: the grid enters the profiling data.")
    flags: list[str] = Field(description="Why the grid cannot calibrate the law, each a sentence.")


class ProfilingData(Output):
    """profiling data: what a cost law's calibration is fitted on. A per-LLM law has the probe cells of one LLM; a
    per-GPU law has the probe cells of synthetic models and a grid of synthetic layers."""

    KIND: ClassVar[str] = "profiling-data"
    gpu: str = Field(description="The GPU's id (servecast list gpus).")
    cost_law: str = Field(description="The cost law the probes were run for, by name (in a file written before "
                                      "names, the month it was released).")
    calibration_procedure: str = Field(description="The same entry: the procedure is part of its cost law's.")
    reference_mhz: float = Field(gt=0, description="The SM clock the probes ran at.")
    llm: str | None = Field(None, description="A per-LLM law: the LLM probed.")
    cells: list[ProbeCell] | None = Field(None, description="A per-LLM law: its probe cells.")
    models: list[ProbedModel] | None = Field(None, description="A per-GPU law: the synthetic models probed.")
    gemm_grid: GemmGrid | None = Field(None, description="A per-GPU law: the grid of synthetic layers timed.")

    @model_validator(mode="after")
    def _one_kind(self) -> ProfilingData:
        per_llm = self.llm is not None and self.cells is not None
        per_gpu = self.models is not None and self.gemm_grid is not None
        if per_llm == per_gpu:
            raise ValueError("give either llm and cells (a per-LLM law) or models and gemm_grid (a per-GPU law)")
        return self


class EtaCPoint(Part):
    d_model: int = Field(gt=0, description="The synthetic layer's width.")
    ffn_ratio: float = Field(gt=0, description="Its feed-forward width over its width.")
    eta_c: float = Field(gt=0, description="The fraction of the peak arithmetic rate its matrix multiplications "
                                           "reached.")


class EtaCCurve(Part):
    a: float = Field(description="1/eta_c = a + b / (the LLM's weights per layer).")
    b: float


class EtaCGrid(Part):
    rule: Literal["interpolation", "curve"] = Field(description="How an LLM's eta_c is read: bilinear "
                                                                "interpolation on the points, clamped to the grid, "
                                                                "or the curve when every point lies near it.")
    points: list[EtaCPoint] = Field(min_length=1, description="The GPU's grid of synthetic layers.")
    curve: EtaCCurve | None = None


class CalibratedLaw(Part):
    cost_law: str = Field(pattern=COMPONENT_NAME, description="The cost law, by name (servecast list laws); its "
                                                              "calibration procedure is part of its entry.")
    parameters: dict[str, float] = Field(min_length=1, description="Each of the law's parameters by name, in the "
                                                                   "law's order; components.json gives their "
                                                                   "units.")
    eta_c_grid: EtaCGrid | None = Field(None, description="A per-GPU law reads each LLM's eta_c off this grid.")


class GpuSpec(Part):
    peak_flop_per_s: float = Field(gt=0, description="The registry's peak the fit assumed (gpus.json).")
    memory_bandwidth_bytes_per_s: float = Field(gt=0, description="The registry's bandwidth the fit assumed.")
    boost_clock_mhz: int = Field(gt=0, description="The registry's boost clock, the peak's clock.")


class CalibrationConditions(Part):
    gpu: str = Field(description="The GPU's id (servecast list gpus).")
    gpu_spec: GpuSpec = Field(description="The datasheet numbers the fit assumed; predict refuses a calibration "
                                          "whose numbers are not the registry's.")
    clock_mhz: float = Field(gt=0, description="The SM clock the probes ran at. The law's latencies are in ms at "
                                               "this clock, and it ran at clock_mhz / boost_clock_mhz of the peak.")
    tensor_parallel: int = Field(1, ge=1, description="GPUs per engine; only 1 in this version.")
    fitted_on: str | list[str] = Field(description="The LLM a per-LLM calibration was fitted on, or the synthetic "
                                                   "models of a per-GPU one.")


class TrainingError(Part):
    mape: float = Field(ge=0, description="The paper's training MAPE: the cells' absolute percentage errors "
                                          "averaged with the fit's weights, in %.")
    max: float = Field(ge=0, description="The largest cell error, in %.")


class FitRecord(Part):
    cells: int | None = Field(None, ge=1, description="The probe cells fitted.")
    evaluations: int | None = Field(None, ge=1, description="The cost evaluations the optimizer made.")
    error_pct: TrainingError | None = None
    profiling_sha256: str | None = Field(None, description="The sha256 of the profiling data it was fitted on.")


class CalibrationFile(Output):
    """calibration.json: a cost law's parameters, the conditions they hold for, and how they were fitted."""

    KIND: ClassVar[str] = "calibration"
    MAJOR: ClassVar[int] = 2
    name: str = Field(description="Its name, <gpu>/<cost law>[/<llm>] for a shipped one: a label, which nothing "
                                  "parses.")
    law: CalibratedLaw
    conditions: CalibrationConditions
    fit: FitRecord | None = Field(None, description="How it was fitted: provenance, which predict never reads. The "
                                                    "shipped calibrations, fitted before ServeCast, have none.")


class Moments(Part):
    mean: float
    variance: float


class LengthMoments(Part):
    input: Moments
    output: Moments


class PoolHeader(Output):
    """The first line of pool.jsonl; one {input_tokens, output_tokens} record per line follows."""

    KIND: ClassVar[str] = "pool"
    requests: int = Field(ge=1, description="The requests that follow.")
    sha256: str = Field(description="Their digest, checked when the pool is read.")
    source: Literal["trace", "synthetic"]
    files: list[str] | None = Field(None, description="A trace source: the traces the requests come from.")
    sampler: str | None = Field(None, description="A synthetic source: the length sampler, by name.")
    seed: int | None = Field(None, description="A synthetic source: the seed of the draw.")
    requested: LengthMoments | None = Field(None, description="A synthetic source: the moments the manifest asked "
                                                              "for (a mean alone sets the variance of its law).")
    realized: LengthMoments | None = Field(None, description="A synthetic source: the moments of the requests drawn.")
    laws: dict[str, dict[str, Any]] | None = Field(None, description="A synthetic source: the law each length was "
                                                                     "drawn from.")

    @model_validator(mode="after")
    def _the_sources_fields(self) -> PoolHeader:
        if self.source == "trace" and self.files is None:
            raise ValueError("a trace pool names its files")
        if self.source == "synthetic" and None in (self.sampler, self.seed, self.requested, self.realized):
            raise ValueError("a synthetic pool records its sampler, seed, requested and realized moments")
        return self


class PoolRecord(Part):
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)


class GpuEntry(Part):
    device_name: str = Field(description="The name the NVIDIA driver reports, as `nvidia-smi --query-gpu=name` "
                                         "prints it.")
    memory_gib: int = Field(gt=0, description="The GPU's memory, in GiB.")
    sm_count: int = Field(gt=0, description="Streaming multiprocessors.")
    peak_flop_per_s: float = Field(gt=0, description="The dense bf16 tensor-core peak at the boost clock, in "
                                                     "FLOP/s, from the datasheet.")
    memory_bandwidth_bytes_per_s: float = Field(gt=0, description="The datasheet memory bandwidth, in bytes/s.")
    boost_clock_mhz: int = Field(gt=0, description="The boost clock the peak is quoted at, in MHz.")


class GpuRegistry(Output):
    """gpus.json: the datasheet numbers of each GPU the cost laws divide by."""

    KIND: ClassVar[str] = "gpus"
    gpus: dict[str, GpuEntry] = Field(description="Each GPU by its id (a100_sxm_40gb), as manifests and calibrations "
                                                  "name it.")


class ModelEntry(Part):
    """The config.json fields the cost laws read, under the names config.json gives them."""

    num_hidden_layers: int = Field(gt=0, description="Decoder layers.")
    hidden_size: int = Field(gt=0, description="The hidden width.")
    intermediate_size: int = Field(gt=0, description="The width of the gated feed-forward block.")
    num_attention_heads: int = Field(gt=0, description="Query heads.")
    num_key_value_heads: int | None = Field(None, gt=0, description="KV heads. Unset: the query heads.")
    head_dim: int | None = Field(None, gt=0, description="The head width. Unset: hidden_size / num_attention_heads.")
    vocab_size: int = Field(gt=0, description="The vocabulary.")
    max_position_embeddings: int = Field(gt=0, description="The longest context, in tokens.")
    torch_dtype: str = Field("bfloat16", description="The weights' type: bfloat16 or float16.")


class ModelRegistry(Output):
    """models.json: the LLMs known without downloading their config.json."""

    KIND: ClassVar[str] = "models"
    models: dict[str, ModelEntry] = Field(description="Each LLM by the name engine.model gives it (a Hugging Face id).")


class LawCalibration(Part):
    measures: str = Field(description="What a calibration measures on the GPU.")
    fit: str = Field(description="How the parameters are fitted to it.")
    fitted: list[str] = Field(description="The parameters the fit determines.")
    pinned: dict[str, str] = Field(default_factory=dict,
                                   description="The parameters fixed before the fit, and to what.")


class ComponentEntry(Part):
    summary: str = Field(description="One sentence, printed beside the name.")
    released: str = Field(pattern=COMPONENT_RELEASE, description="The month it was released (2026-11; 2026-11b for a "
                                                                 "second entry that month). `versions: <month>` "
                                                                 "selects each component's newest entry up to it, "
                                                                 "and `versions` accepts it in place of the name.")
    in_paper: bool = Field(description="Whether the paper used it; one entry of each kind at most.")
    default: bool = Field(description="Whether it is used when `versions` does not choose; one entry of each kind.")
    description: str = Field(description="What it computes, in full.")
    paper: str | None = Field(None, description="Where the paper describes it.")
    paper_symbols: dict[str, str] = Field(default_factory=dict, description="Its parameters' names in the paper.")
    code: list[str] = Field(min_length=1, description="The modules that implement it.")


class CostLawEntry(ComponentEntry):
    calibrated_per: Literal["llm_and_gpu", "gpu"] = Field(description="What one calibration covers: one LLM on one "
                                                                       "GPU, or one GPU for any LLM.")
    calibration: LawCalibration = Field(description="Its calibration procedure, part of the same entry.")
    parameters: dict[str, str] = Field(description="Each parameter, with its unit.")


#: the kinds of component, as `versions` and components.json name them
COMPONENT_KINDS = ("cost_law", "population_model", "workload_fitter", "length_sampler", "estimator")


class ComponentRegistry(Output):
    """components.json: every component ServeCast can select, by name, with what it computes."""

    KIND: ClassVar[str] = "components"
    about: list[str] = Field(default_factory=list, description="The rules of the file, in words.")
    cost_law: dict[str, CostLawEntry] = Field(description="The cost laws, each with its calibration procedure.")
    population_model: dict[str, ComponentEntry] = Field(description="The population models.")
    workload_fitter: dict[str, ComponentEntry] = Field(description="The workload fitters.")
    length_sampler: dict[str, ComponentEntry] = Field(description="The length samplers.")
    estimator: dict[str, ComponentEntry] = Field(description="The steady-state estimators.")

    @model_validator(mode="after")
    def _one_name_one_entry(self) -> ComponentRegistry:
        seen: set[str] = set()
        for kind in COMPONENT_KINDS:
            entries: dict[str, ComponentEntry] = dict(getattr(self, kind))
            for name in entries:
                if not re.fullmatch(COMPONENT_NAME, name):
                    raise ValueError(f"{kind}: {name!r} is not a name (eight hexadecimal characters, a letter first)")
                if name in seen:
                    raise ValueError(f"{name} names two entries")
                seen.add(name)
            releases = [e.released for e in entries.values()]
            if len(set(releases)) != len(releases):
                raise ValueError(f"{kind}: two entries released as the same month")
            if sum(e.default for e in entries.values()) != 1:
                raise ValueError(f"{kind}: exactly one entry must be the default")
            if sum(e.in_paper for e in entries.values()) > 1:
                raise ValueError(f"{kind}: at most one entry is the paper's")
        return self


class RunRecord(Output):
    """run.json: what a command ran, from which inputs, and what it wrote beside it."""

    KIND: ClassVar[str] = "run"
    command: str = Field(description="The command line, as typed.")
    servecast: str = Field(description="The version of ServeCast that ran it.")
    started: str = Field(description="When it started: local time with its UTC offset (ISO 8601).")
    finished: str = Field(description="When it wrote this record.")
    inputs: dict[str, dict[str, Any]] = Field(
        description="Each input by its role (manifest, calibration, pool, profiling data): its path or name, and its "
                    "sha256.")
    components: dict[str, str] = Field(default_factory=dict, description="The components that ran, by name.")
    outputs: list[str] = Field(description="The other files the command wrote in this directory.")


SCHEMAS: dict[str, type[BaseModel]] = {
    "manifest": Manifest,
    "pool": PoolHeader,
    "profiling-data": ProfilingData,
    "probe": Probe,
    "gemm": Gemm,
    "calibration": CalibrationFile,
    "prediction": Prediction,
    "measurement": Measurement,
    "comparison": Comparison,
    "run": RunRecord,
    "gpus": GpuRegistry,
    "models": ModelRegistry,
    "components": ComponentRegistry,
}
