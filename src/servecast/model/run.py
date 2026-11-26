"""A manifest, run: from the user's inputs to the prediction.json document.

Resolves the component versions against the calibration, the LLM's geometry, the engine settings and their vLLM
defaults, the request pool and the workload law fitted to it, the stop targets, and runs every load point.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import multiprocessing
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from servecast import __version__
from servecast.core import calibration as calibrations
from servecast.core import deployment as deployments
from servecast.core import estimator as estimators
from servecast.core import gpus, versions
from servecast.core import pool as pools
from servecast.core import sampler as samplers
from servecast.core import summary as summaries
from servecast.core.manifest import Manifest, ManifestError, SyntheticSource, VersionsSpec
from servecast.core.paths import cache_dir
from servecast.model import geometry as geometries
from servecast.model import law as laws
from servecast.model import predict as P
from servecast.model import workload as workloads

#: the components every prediction uses; the length sampler joins them for a mean-and-variance workload
USED = ("cost_law", "calibration_procedure", "population_model", "workload_fitter", "estimator")


@dataclasses.dataclass(frozen=True)
class Plan:
    """Everything a prediction needs, resolved: what `run` executes and what prediction.json records."""

    calibration: dict[str, Any]
    calibration_name: str
    calibration_sha256: str
    components: dict[str, str]
    llm: geometries.Geometry
    engine: P.Engine
    engine_resolved: dict[str, Any]
    deployment: deployments.Deployment
    pool_p: list[int]
    pool_g: list[int]
    pool_sha256: str
    pool_header: dict[str, Any]
    law: dict[str, Any]
    targets: dict[str, float]
    points: list[tuple[int, float]]
    seed: int
    validity: list[str]
    warnings: list[str]


_as_of = deployments.as_of


def resolve_components(m: Manifest, calibration: Mapping[str, Any], synthetic: bool) -> dict[str, str]:
    """The six components' ids: the date's, the explicit overrides, then the calibration's own cost law and
    procedure. A calibration whose law is newer than the date, or that contradicts an explicit id, is refused."""
    try:
        resolved = versions.resolve(m.versions)
    except ValueError as error:
        raise ManifestError(f"versions: {error}") from error
    explicit = m.versions.model_dump(exclude_none=True) if isinstance(m.versions, VersionsSpec) else {}
    law = procedure = versions.canonical("cost_law", str(calibration["law"]["cost_law"]))  # its entry has its procedure
    as_of = _as_of(m)
    if as_of is not None and versions.sort_key(versions.law(law).released)[:2] > versions.sort_key(as_of)[:2]:
        raise ManifestError(f"versions: {as_of} predates cost law {law}, released "
                            f"{versions.law(law).released}, which the calibration "
                            f"{calibration['name']} uses; choose a later date or a calibration of an earlier law")
    if "cost_law" in explicit and versions.canonical("cost_law", str(explicit["cost_law"])) != law:
        raise ManifestError(f"versions.cost_law: {explicit['cost_law']} contradicts the calibration's {law}")
    resolved["cost_law"], resolved["calibration_procedure"] = law, procedure
    used = set(USED) | ({"length_sampler"} if synthetic else set())
    try:
        versions.require(resolved, used, as_of)
    except ValueError as error:
        raise ManifestError(f"versions: {error}") from error
    return {k: v for k, v in resolved.items() if v is not None}


def resolve_llm(model: str | None, *, offline: bool = False) -> geometries.Geometry:
    if not model:
        raise ManifestError("engine.model: required: the LLM to predict (--to), a Hugging Face id or a local "
                            "directory holding config.json")
    try:
        known = model in geometries.known()
    except ValueError as error:
        raise ManifestError(f"engine.model: {error}") from error
    if known or model.startswith("synthetic/"):
        try:
            return geometries.resolve(model)
        except ValueError as error:
            raise ManifestError(f"engine.model: {error}") from error
    path = Path(model)
    try:
        if path.exists():
            cfg = path / "config.json" if path.is_dir() else path
            if not cfg.is_file():
                raise ValueError(f"{model}: no config.json in this directory")
            config, name = json.loads(cfg.read_text(encoding="utf-8")), path.name if path.is_dir() else path.parent.name
        else:
            from servecast.core import hub

            config, name = hub.fetch_config(model, offline=offline), model
        # the refusals the measurement applies too: what the cost laws do not model
        deployments.architecture_from_config(config)
        return geometries.from_config(name, config)
    except ValueError as error:  # the hub's messages name the id already
        said = str(error)
        raise ManifestError(f"engine.model: {said if model in said else f'{model}: {said}'}") from error


def architecture(llm: geometries.Geometry) -> dict[str, int]:
    """The numbers that define the LLM's architecture, as `deployment.architecture_from_config` reads them."""
    return {"layers": llm.n_layers, "hidden": llm.d_model, "ffn": llm.d_ff, "heads": llm.n_heads,
            "kv_heads": llm.h_kv, "kv_width": llm.d_kv, "vocab": llm.vocab, "max_context": llm.max_context}


def resolve_engine(m: Manifest, gpu: gpus.Gpu, llm: geometries.Geometry) -> tuple[P.Engine, dict[str, Any],
                                                                                 deployments.Deployment]:
    d = deployments.resolve(m, gpu, llm.max_context, llm.name, architecture=architecture(llm))
    engine = P.Engine(kv_blocks=d.kv_blocks, block_size=d.block_size, max_batched_tokens=d.max_batched_tokens,
                      max_sequences=d.max_sequences)
    resolved = {"block_size": d.block_size, "max_batched_tokens": d.max_batched_tokens,
                "max_sequences": d.max_sequences, "kv_blocks": d.kv_blocks, "context_length": d.context_length}
    return engine, resolved, d


def resolve_pool(m: Manifest, components: Mapping[str, str], context: int) -> tuple[list[int], list[int],
                                                                                    dict[str, Any], list[str], int]:
    """The request pool (`deployment.build_pool`), and the phase cap the workload law needs for it."""
    pool = deployments.build_pool(m, components, context)
    cap = workloads.v2026_10.DEFAULT_PHASE_CAP
    if pool.output_spec is not None:
        sampler = samplers.get(components["length_sampler"])
        cap = max(cap, sampler.phases_needed(pool.output_spec))
        if cap > workloads.v2026_10.MAX_PHASES:
            raise ManifestError("traffic.source.output_tokens: too regular for the population model's 256 phases")
    return pool.input_tokens, pool.output_tokens, pool.header, pool.warnings, cap


#: what a workload law's cache keeps of the fitter's report
FIT_KEYS = ("prompt_classes", "coupling_groups", "branches", "stages", "phase_cap", "footprint_tol", "reanchored")


class FitTimeoutError(Exception):
    """The workload-law fit did not finish within the run's --timeout."""


def _fit_in_child(conn: Any, fitter_id: str, p: list[int], g: list[int], block_size: int, phase_cap: int) -> None:
    """The fit, in a child process the parent stops at its deadline: sends the law and the report's kept fields."""
    try:
        law, report = workloads.get(fitter_id).fit_law(p, g, block_size=block_size, phase_cap=phase_cap)
        conn.send(("ok", law.to_dict(), {k: report[k] for k in FIT_KEYS}))
    except ValueError as error:  # the fitter refuses this pool: the parent raises it as the same kind of error
        conn.send(("refused", str(error), None))
    except Exception as error:  # the parent re-raises it: a child must not die silently
        conn.send(("error", f"{type(error).__name__}: {error}", None))
    finally:
        conn.close()


def _fit(fitter_id: str, p: Sequence[int], g: Sequence[int], block_size: int, phase_cap: int,
         seconds: float | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The law and the report's kept fields: in this process without a deadline, else in a child stopped at it."""
    if seconds is None:
        law, report = workloads.get(fitter_id).fit_law(p, g, block_size=block_size, phase_cap=phase_cap)
        return law.to_dict(), {k: report[k] for k in FIT_KEYS}
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_fit_in_child, args=(child, fitter_id, list(p), list(g), block_size, phase_cap),
                       daemon=True)
    proc.start()
    child.close()
    try:
        if not parent.poll(seconds):
            raise FitTimeoutError(f"the workload-law fit did not finish in {seconds:g} s ({len(p)} requests); the "
                                  "law is cached once a fit finishes: run once without --timeout, or draw a smaller "
                                  "pool (traffic.pool_size)")
        status, law, kept = parent.recv()
    except EOFError:
        status, law, kept = "error", f"the fitting process stopped (exit code {proc.exitcode})", None
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join()
        parent.close()
    if status == "refused":
        raise ValueError(law)
    if status != "ok":
        raise RuntimeError(f"the workload-law fit failed: {law}")
    return law, kept


def _checksum(doc: Mapping[str, Any]) -> str:
    """The sha256 of a cached workload law, every field but the checksum itself."""
    body = {k: v for k, v in doc.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def fit_warnings(doc: Mapping[str, Any]) -> list[str]:
    """What the workload fitter fell back to: a looser footprint tolerance, or a re-anchored footprint. Each changes
    the law, and both were recorded in the cache only."""
    fit = doc.get("fit") or {}
    tol = fit.get("footprint_tol")
    strict = workloads.v2026_10.FOOTPRINT_TOL
    if fit.get("reanchored"):
        return ["the workload law was re-anchored to the pool's KV footprint: no structure the fitter tried matched "
                f"it within {100 * (tol or strict):g} %"]
    if tol is not None and tol > strict:
        return [f"the workload law matches the pool's KV footprint only within {100 * tol:g} % (no structure the "
                f"fitter tried met {100 * strict:g} %)"]
    return []


def workload_law(p: Sequence[int], g: Sequence[int], fitter_id: str, phase_cap: int, *, block_size: int,
                 progress: Callable[[str], None] | None = None, seconds: float | None = None) -> dict[str, Any]:
    """The workload law fitted to the pool: a shipped law when the pool is a shipped trace, else from the cache, else
    fitted now (minutes on a large pool) and cached by the pool's digest, the fitter's id and the phase cap."""
    digest = pools.digest(p, g)
    fitter_id = versions.canonical("workload_fitter", fitter_id)
    # the shipped workload laws: fitted by the paper's fitter at its default cap, for 16-token blocks
    if (fitter_id == versions.PAPER["workload_fitter"] and phase_cap == workloads.v2026_10.DEFAULT_PHASE_CAP
            and block_size == 16):
        for name in pools.SHIPPED_TRACES:
            doc = pools.shipped_trace(name)
            if pools.digest(doc["input_tokens"], doc["output_tokens"]) == digest:
                return P.load_workload_law(name)
    path = cache_dir() / "workload_laws" / fitter_id / f"cap{phase_cap}-block{block_size}" / f"{digest}.json"
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            cached = None
        if isinstance(cached, dict) and cached.get("sha256") == _checksum(cached) \
                and cached.get("pool_sha256") == digest:
            return cached
        if progress is not None:  # an altered law was read back as it was, and changed the prediction
            progress(f"the cached workload law {path.name} does not match its checksum; fitting it again")
    if progress is not None:
        def is_shipped(name: str) -> bool:
            doc = pools.shipped_trace(name)
            return pools.digest(doc["input_tokens"], doc["output_tokens"]) == digest

        shipped = next((n for n in pools.SHIPPED_TRACES if is_shipped(n)), None) if block_size != 16 else None
        why = f"; the shipped law of {shipped} is for 16-token blocks, not {block_size}" if shipped else ""
        progress(f"fitting the workload law to {len(p)} requests (cached afterwards){why}")
    try:
        law, kept = _fit(fitter_id, p, g, block_size, phase_cap, seconds)
    except ValueError as error:  # some small pools fit no prompt-class count: an input to change, not a crash
        raise ManifestError(f"traffic: the workload law cannot be fitted to these {len(p)} requests ({error}); give "
                            "more requests, with a larger traffic.pool_size (default 10,000) for synthetic lengths, "
                            "or a longer trace") from error
    doc = {"schema": P.WORKLOAD_SCHEMA, "trace": None, "workload_fitter": fitter_id, "law": law, "fit": kept,
           "pool_sha256": digest}
    doc["sha256"] = _checksum(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return doc


resolve_targets = deployments.stop_targets


load_points = deployments.load_points


@dataclasses.dataclass(frozen=True)
class Resolved:
    """What a prediction resolves from its manifest and calibration before it fits the workload law."""

    points: list[tuple[int, float]]
    components: dict[str, str]
    llm: geometries.Geometry
    engine: P.Engine
    engine_resolved: dict[str, Any]
    deployment: deployments.Deployment
    pool_p: list[int]
    pool_g: list[int]
    pool_header: dict[str, Any]
    warnings: list[str]
    phase_cap: int


def check_calibration(calibration: Mapping[str, Any]) -> None:
    """A calibration holds every parameter its cost law reads: one with a misspelled eta_c_grid used to be read, and
    the prediction then ended in a KeyError."""
    cost_law = str(calibration["law"]["cost_law"])
    try:
        calibrations.parameter_values(calibration, laws.get(cost_law).NAMES)
    except ValueError as error:
        raise ManifestError(f"calibration: {error}") from None
    if versions.law(cost_law).calibrated_per == "gpu" and not calibration["law"].get("eta_c_grid"):
        raise ManifestError(f"calibration: {calibration['name']}: cost law {cost_law} reads each LLM's eta_c off "
                            "law.eta_c_grid, which this calibration does not have")


def resolve(m: Manifest, calibration: Mapping[str, Any], *, sweep: tuple[str, list[str]] | None = None,
            offline: bool = False) -> Resolved:
    """Everything a prediction resolves before it fits the workload law, refusing what it cannot run: what
    `servecast manifest validate` checks once the file itself is valid."""
    if m.traffic is None:
        raise ManifestError("traffic: required by predict: the source of the lengths, the load and the stop rule")
    points = load_points(m, sweep)  # first: a load it cannot run is refused before the workload law is fitted
    conditions = calibration["conditions"]
    gpu = gpus.get(conditions["gpu"])
    if m.engine.hardware is not None and m.engine.hardware != gpu.id:
        raise ManifestError(f"engine.hardware: {m.engine.hardware}, but the calibration {calibration['name']} is for "
                            f"{gpu.id}")
    if m.engine.gpu_clock_mhz is not None and m.engine.gpu_clock_mhz != conditions["clock_mhz"]:
        raise ManifestError(f"engine.gpu_clock_mhz: predicting at {m.engine.gpu_clock_mhz} MHz is not supported in "
                            f"this version; the calibration is at {conditions['clock_mhz']} MHz")
    check_calibration(calibration)
    synthetic = isinstance(m.traffic.source, SyntheticSource)
    components = resolve_components(m, calibration, synthetic)
    llm = resolve_llm(m.engine.model, offline=offline)
    deployments.check_architecture(m, architecture(llm))
    check_transfer(m, calibration, llm)
    engine, engine_resolved, deployment = resolve_engine(m, gpu, llm)
    p, g, header, warnings, cap = resolve_pool(m, components, engine_resolved["context_length"])
    return Resolved(points=points, components=components, llm=llm, engine=engine, engine_resolved=engine_resolved,
                    deployment=deployment, pool_p=list(p), pool_g=list(g), pool_header=header,
                    warnings=list(warnings), phase_cap=cap)


def plan(m: Manifest, calibration: Mapping[str, Any], calibration_name: str, calibration_bytes: bytes, *,
         sweep: tuple[str, list[str]] | None = None, offline: bool = False,
         progress: Callable[[str], None] | None = None, fit_seconds: float | None = None) -> Plan:
    """Resolve everything a prediction needs, refusing a manifest it cannot run."""
    r = resolve(m, calibration, sweep=sweep, offline=offline)
    assert m.traffic is not None  # resolve refuses a manifest without it
    law = workload_law(r.pool_p, r.pool_g, r.components["workload_fitter"], r.phase_cap,
                       block_size=r.engine.block_size, progress=progress, seconds=fit_seconds)
    warnings = [*r.warnings, *fit_warnings(law)]
    validity = []
    fitted_on = calibration["conditions"]["fitted_on"]
    if versions.law(calibration["law"]["cost_law"]).calibrated_per == "llm_and_gpu":
        if not _calibrated_on(str(fitted_on), r.llm):
            validity.append(f"a transfer: the calibration was made on {fitted_on}, the prediction is for "
                            f"{m.engine.model}")
    else:
        validity += coverage_reasons(calibration, calibration_name, r.llm)
    return Plan(calibration=dict(calibration), calibration_name=calibration_name,
                calibration_sha256=hashlib.sha256(calibration_bytes).hexdigest(), components=r.components,
                llm=r.llm, engine=r.engine, engine_resolved=r.engine_resolved, deployment=r.deployment,
                pool_p=r.pool_p, pool_g=r.pool_g, pool_sha256=pools.digest(r.pool_p, r.pool_g),
                pool_header=r.pool_header, law=law, targets=resolve_targets(m, r.components["estimator"]),
                points=r.points, seed=m.traffic.seed, validity=validity, warnings=warnings)


#: the LLMs each shipped per-GPU calibration was validated on, beyond the synthetic models it was fitted on: its
#: shipped predictions are for them (tests/goldens/newest-a100.json). The redesign moves this into the calibration file.
VALIDATED_ON: dict[str, tuple[str, ...]] = {
    "a100_sxm_40gb/c62ebe75": ("meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-7B",
                              "Qwen/Qwen2.5-3B", "deepseek-ai/deepseek-coder-1.3b-base",
                              "deepseek-ai/deepseek-llm-7b-base"),
}


@functools.cache
def _shipped_profiling_sha256(name: str) -> str | None:
    path = P.shipped_profiling().get(name)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None else None


def validated_on(calibration: Mapping[str, Any], name: str) -> tuple[str, ...]:
    """The LLMs a calibration was validated on: those of the shipped calibration it is, or whose profiling data it was
    fitted on (the same law, GPU and data give the same numbers, whatever the calibration is called)."""
    if name in VALIDATED_ON:
        return VALIDATED_ON[name]
    sha = (calibration.get("fit") or {}).get("profiling_sha256")
    for shipped, llms in VALIDATED_ON.items():
        if sha and sha == _shipped_profiling_sha256(shipped):
            twin = P.load_calibration(shipped)
            if (twin["law"]["cost_law"], twin["conditions"]["gpu"]) == (calibration["law"]["cost_law"],
                                                                        calibration["conditions"]["gpu"]):
                return llms
    return ()


def coverage_reasons(calibration: Mapping[str, Any], name: str, llm: geometries.Geometry) -> list[str]:
    """Why a per-GPU calibration does not cover the LLM: its layers or width beyond those of the synthetic models the
    calibration was fitted on and the LLMs it was validated on, or its FFN ratio beyond those and the GEMM grid's.
    An empty list is "inside". A calibration of your own has no validated LLMs, so its synthetic models bound it."""
    covered = [geometries.resolve(str(n)) for n in calibration["conditions"]["fitted_on"]]
    covered += [geometries.KNOWN[n] for n in validated_on(calibration, name)]
    grid = [float(pt["ffn_ratio"]) for pt in (calibration["law"].get("eta_c_grid") or {}).get("points") or []]
    ranges: dict[str, tuple[list[float], float, str]] = {
        "layer count": ([float(g.n_layers) for g in covered], float(llm.n_layers), "{:.0f}"),
        "width": ([float(g.d_model) for g in covered], float(llm.d_model), "{:.0f}"),
        "FFN ratio": ([g.d_ff / g.d_model for g in covered] + grid, llm.d_ff / llm.d_model, "{:.2f}")}
    reasons = []
    for what, (values, value, fmt) in ranges.items():
        if values and not min(values) <= value <= max(values):
            reasons.append(f"the LLM's {what} {fmt.format(value)} lies outside {fmt.format(min(values))}-"
                           f"{fmt.format(max(values))}, the range the calibration was fitted or validated on")
    return reasons


def fitted_on_llm(calibration: Mapping[str, Any], llm: geometries.Geometry) -> bool:
    """Whether the calibration was fitted on this LLM: the one LLM of a per-LLM calibration, or one of the synthetic
    models of a per-GPU one, compared by architecture (a fine-tune or a local copy of the same LLM is the same)."""
    fitted_on = calibration["conditions"]["fitted_on"]
    if isinstance(fitted_on, str):
        return _calibrated_on(fitted_on, llm)
    return any(_same_llm(architecture(geometries.resolve(str(name))), architecture(llm)) for name in fitted_on)


def check_transfer(m: Manifest, calibration: Mapping[str, Any], llm: geometries.Geometry) -> None:
    """A prediction for an LLM the calibration was not fitted on is a transfer, and the manifest must say so."""
    if m.calibration.transfer or fitted_on_llm(calibration, llm):
        return
    fitted_on = calibration["conditions"]["fitted_on"]
    made_on = fitted_on if isinstance(fitted_on, str) else f"{len(fitted_on)} synthetic models"
    raise ManifestError(f"calibration.transfer: the calibration {calibration['name']} was fitted on {made_on} and the "
                        f"manifest predicts {m.engine.model}; set calibration.transfer: true to predict it with this "
                        "calibration (a transfer)")


def _calibrated_on(name: str | None, llm: geometries.Geometry) -> bool:
    """Whether a per-LLM calibration was made on this LLM: the same architecture numbers (a fine-tune, a re-upload or
    a local copy computes the same), or the same name for an LLM ServeCast does not know offline."""
    llms = geometries.known()
    if name in llms:
        return _same_llm(architecture(llms[name]), architecture(llm))
    return name == llm.name


def _same_llm(a: Mapping[str, int], b: Mapping[str, int]) -> bool:
    """The same LLM, whatever context length its copy declares: a local copy with another max_position_embeddings
    was a transfer, marked outside the calibration's validity."""
    return {k: v for k, v in a.items() if k != "max_context"} == {k: v for k, v in b.items() if k != "max_context"}


def _point(args: tuple[Plan, int, float, float | None, bool]) -> dict[str, Any]:
    pl, n, z_ms, budget, keep = args
    return P.predict_point(pl.calibration, pl.llm, pl.law, pl.engine, n, z_ms / 1e3, components=pl.components,
                           targets=pl.targets, seed=pl.seed, seconds_budget=budget, keep_requests=keep)


def summary(row: Mapping[str, Any], estimator_id: str) -> dict[str, Any]:
    """A load point's summary in prediction.json's units (`core.summary`, which the measurement uses too)."""
    return summaries.point_summary(row, row["per_metric"], estimators.get(estimator_id), row["wall_s"])


def _finite(v: Any) -> float | None:
    return float(v) if v is not None and math.isfinite(float(v)) else None


def _ms(v: float) -> float | None:
    return v * 1e3 if v is not None and math.isfinite(v) else None


def run(pl: Plan, *, jobs: int = 1, seconds_budget: float | None = None,
        keep_requests: bool = False) -> list[dict[str, Any]]:
    """Every load point; in parallel processes when `jobs` > 1 (the same numbers as one at a time)."""
    tasks = [(pl, n, z, seconds_budget, keep_requests) for n, z in pl.points]
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as ex:
            rows = list(ex.map(_point, tasks))
    else:
        rows = [_point(t) for t in tasks]
    return rows


def prediction_document(pl: Plan, rows: Sequence[Mapping[str, Any]], resolved_manifest: dict[str, Any]) -> dict[
        str, Any]:
    points = []
    for (n, z), row in zip(pl.points, rows, strict=False):
        points.append({"load": {"concurrency": n, "think_time_ms": z},
                       "summary": summary(row, pl.components["estimator"]),
                       "converged": bool(row["converged"]), "stop_reason": row["stopped_because"],
                       "diagnostics": {"preemptions_per_request": _finite(row.get("preemptions_per_request")),
                                       "beats": row.get("n_beats"), "warmup_discarded": row.get("warmup_trunc")}})
    validity = {"status": "outside" if pl.validity else "inside", "reasons": list(pl.validity)}
    return {"schema": "servecast-prediction/1", "points": points,
            "estimand": {"rule": "steady_state", **pl.targets}, "validity": validity,
            "provenance": {"components": pl.components, "servecast_version": __version__, "seed": pl.seed,
                           "pool_sha256": pl.pool_sha256, "calibration": pl.calibration_name,
                           "calibration_sha256": pl.calibration_sha256},
            "resolved_manifest": resolved_manifest, "warnings": list(pl.warnings)}


def resolved_manifest(m: Manifest, pl: Plan) -> dict[str, Any]:
    """The manifest with every default filled in: the engine and traffic sections a measurement of it records too
    (`deployment.resolved_sections`), then what only a prediction reads. Predictions run at the calibration's clock.
    The LLM is recorded under the name the manifest gives it, which a measurement of the same manifest records too."""
    doc = deployments.resolved_sections(m, model=str(m.engine.model), architecture=architecture(pl.llm),
                                        hardware=pl.calibration["conditions"]["gpu"], deployment=pl.deployment,
                                        components=pl.components,
                                        gpu_clock_mhz=float(pl.calibration["conditions"]["clock_mhz"]))
    # profiling data stay profiling data: the calibration fitted from them has a label for people, not a name a run
    # can resolve, and the fit is cached, so running the manifest again reads it back
    doc["calibration"] = ({"data": str(m.calibration.data)} if m.calibration.data is not None
                          else {"use": pl.calibration_name})
    # whether the prediction was a transfer, as ServeCast found it: the flag permits one, and the shipped example
    # sets it, so a prediction for the calibration's own LLM was recorded as a transfer
    doc["calibration"]["transfer"] = not fitted_on_llm(pl.calibration, pl.llm)
    as_of = _as_of(m)
    doc["versions"] = {**({"as_of": as_of} if as_of else {}), **pl.components}  # every component, by name
    return doc


def write_outputs(out_dir: Path, doc: Mapping[str, Any], pl: Plan,
                  rows: Sequence[Mapping[str, Any]] | None = None) -> list[Path]:
    """prediction.json and pool.jsonl; requests.jsonl too when the rows carry their requests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "prediction.json", out_dir / "pool.jsonl"]
    (out_dir / "prediction.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    pools.write(out_dir / "pool.jsonl", pl.pool_p, pl.pool_g, pl.pool_header)
    if rows and any(r.get("requests") for r in rows):
        path = out_dir / "requests.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for (n, z), row in zip(pl.points, rows, strict=False):
                for q in row.get("requests") or ():
                    rec = {"concurrency": n, "think_time_ms": z, "request": q["rid"], "input_tokens": q["p"],
                           "output_tokens": q["g"], "arrival_s": q["arrival_t"], "ttft_ms": _ms(q["ttft"]),
                           "tpot_ms": _ms(q["tpot"]), "e2e_ms": _ms(q["e2e"]), "completion_s": q["completion_t"],
                           "preemptions": q["n_preempt"]}
                    handle.write(json.dumps(rec) + "\n")
        written.append(path)
    return written
