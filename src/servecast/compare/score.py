"""A prediction against a measurement of the same manifest, load point by load point (`servecast-comparison/1`).

Refused unless both describe the same experiment: two files of the right kind, read strictly (a value of the wrong
type is refused, never coerced); the same manifest, compared on its meaning (`manifest.semantic_view`), with every
field a writer records present on both sides and the LLM's architecture well formed; the same stop rule, each side's
being the one its resolved manifest records; the same request pool, by the sha256 each provenance records (a
`pool.jsonl` present in either directory must match it); the same steady-state estimator, and the same length sampler
when the lengths were drawn rather than read from a trace.

Per load point present on both sides (same clients, same think time to float precision) and metric: the prediction,
the measurement and its 95 % interval, the signed error 100 (p - m) / m, and whether the prediction falls inside the
interval. Flagged, never refused: a load point on one side only, a run that did not converge, a certified measurement
without the interval of a metric its stop rule judged, a metric left without an error, an LLM named differently (with
the same architecture), a measurement on another vLLM version than the prediction's or on one ServeCast does not
model, a GPU clock missing or more than `CLOCK_TOLERANCE` away from the clock the prediction was made at, a KV
capacity unrecorded, not the one the engine ran with, or above what vLLM computed from the GPU's memory, a pool
matched on its recorded digest alone, and a field of the resolved manifests that this version does not compare but
that differs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from servecast.core import deployment, formats, manifest
from servecast.core import pool as pools

METRICS = ("mean_ttft_ms", "p95_ttft_ms", "mean_tpot_ms", "p95_tpot_ms", "request_throughput_rps",
           "output_throughput_tok_s")
#: the summary metric each target of a stop rule judges
JUDGED = {"ttft": "mean_ttft_ms", "tpot": "mean_tpot_ms", "throughput": "request_throughput_rps",
          "ttft_p95": "p95_ttft_ms", "tpot_p95": "p95_tpot_ms"}
#: a measured clock further than this, relative, from the prediction's clock is flagged
CLOCK_TOLERANCE = 0.015
#: two think times closer than this, relative, are the same load (a value that went through seconds and back may
#: differ in its last bits); any real difference between two loads is far larger
THINK_TOLERANCE = 1e-9
CAPACITY = "engine.workers.aggregated.kv_cache.capacity.blocks"
FIX = ("compare scores a prediction against a measurement of the same manifest: run `servecast predict` and "
       "`servecast profile serving` with one manifest file")


class NotComparableError(ValueError):
    """Not a prediction and a measurement of the same experiment; the message names what differs."""


def _read(directory: Path, name: str, model: type[formats.Output]) -> tuple[dict[str, Any], str]:
    path = directory / name
    if not path.is_file():
        raise NotComparableError(f"{directory}: no {name}")
    body = path.read_bytes()
    try:
        doc = json.loads(body)
        model.model_validate(doc, strict=True)
    except json.JSONDecodeError as error:
        raise NotComparableError(f"{path}: not JSON ({error})") from error
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'file'}: {e['msg']}" for e in error.errors()[:3])
        raise NotComparableError(f"{path}: not a {model.KIND} ({problems})") from error
    return doc, hashlib.sha256(body).hexdigest()


def _pool_digest(directory: Path) -> str | None:
    """The digest of the directory's pool.jsonl, checked against its own lines; None without the file."""
    path = directory / "pool.jsonl"
    if not path.is_file():
        return None
    try:
        return str(pools.read(path)[2]["sha256"])
    except (ValueError, KeyError, TypeError) as error:
        raise NotComparableError(f"{path}: not a request pool ({error})") from error


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    """Every field path where two JSON values differ, with both values."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            out += differences(a.get(key), b.get(key), f"{path}.{key}" if path else str(key))
        return out
    return [] if a == b else [f"{path}: {a!r} in the prediction, {b!r} in the measurement"]


def _get(doc: Mapping[str, Any], dotted: str) -> Any:
    node: Any = doc
    for key in dotted.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _key(point: Mapping[str, Any]) -> tuple[int, float]:
    """A load point's identity: the clients and the think time."""
    return int(point["load"]["concurrency"]), float(point["load"]["think_time_ms"])


def _same_load(a: tuple[int, float], b: tuple[int, float]) -> bool:
    return a[0] == b[0] and abs(a[1] - b[1]) <= THINK_TOLERANCE * max(abs(a[1]), abs(b[1]))


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def _by_load(doc: Mapping[str, Any], name: str) -> list[tuple[tuple[int, float], dict[str, Any]]]:
    out: list[tuple[tuple[int, float], dict[str, Any]]] = []
    for point in doc["points"]:
        key = _key(point)
        if any(_same_load(key, seen) for seen, _ in out):
            raise NotComparableError(f"{name} lists clients {key[0]}, think time {key[1]:g} ms twice")
        out.append((key, point))
    return out


def compared(p: float | None, m: float | None, ci: Mapping[str, float] | None) -> dict[str, Any]:
    """One metric at one load point: the signed error in percent, and whether p falls inside the measured interval."""
    out: dict[str, Any] = {"predicted": p, "measured": m,
                           "signed_error_pct": 100.0 * (p - m) / m if p is not None and m else None,
                           "within_ci": None}
    if ci is not None:
        out["measured_ci95"] = {"low": ci["low"], "high": ci["high"]}
        out["within_ci"] = None if p is None else bool(ci["low"] <= p <= ci["high"])
    return out


def _rule(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The stop rule a writer applied, without its unset targets."""
    return {k: v for k, v in doc["estimand"].items() if v is not None}


def _synthetic(pred: Mapping[str, Any]) -> bool:
    return _get(pred["resolved_manifest"], "traffic.source.type") == "synthetic"


def _refusals(pred: Mapping[str, Any], meas: Mapping[str, Any], directories: Sequence[Path]) -> list[str]:
    problems = []
    for label, doc in (("prediction", pred), ("measurement", meas)):
        lacks = [path for path, value in manifest.semantic_view(doc["resolved_manifest"]).items()
                 if value is None or value == {}]
        if lacks:
            problems.append(f"the {label}'s resolved manifest lacks {', '.join(lacks)}, which every writer records")
        architecture = _get(doc["resolved_manifest"], "engine.architecture")
        if architecture is not None and architecture != {}:
            try:
                manifest.Architecture.model_validate(architecture, strict=True)
            except ValidationError as error:
                problems.append(f"the {label}'s resolved manifest gives engine.architecture as {architecture!r}, not "
                                f"an LLM's architecture ({error.errors()[0]['msg']})")
        blocks = _get(doc["resolved_manifest"], CAPACITY)
        if blocks is not None and (isinstance(blocks, bool) or not isinstance(blocks, int)):
            problems.append(f"the {label}'s resolved manifest gives {CAPACITY} as {blocks!r}, not a whole number")
        recorded = manifest.semantic_view(doc["resolved_manifest"])["traffic.stop.steady_state"]
        if recorded is not None and {k: v for k, v in _rule(doc).items() if k != "rule"} != recorded:
            problems.append(f"the {label}'s estimand is not the stop rule its resolved manifest records")
    problems += [f"the manifests differ at {d}" for d in differences(
        manifest.semantic_view(pred["resolved_manifest"]), manifest.semantic_view(meas["resolved_manifest"]))]
    problems += [f"the stop rules differ at {d}" for d in differences(_rule(pred), _rule(meas))]
    pp, mp = pred["provenance"], meas["provenance"]
    if not pp.get("pool_sha256") or not mp.get("pool_sha256"):
        problems.append("a provenance records no request-pool digest")
    elif pp["pool_sha256"] != mp["pool_sha256"]:
        problems.append(f"the request pools differ (sha256 {pp['pool_sha256'][:12]} in the prediction, "
                        f"{mp['pool_sha256'][:12]} in the measurement)")
    for directory, recorded_digest in zip(directories, (pp.get("pool_sha256"), mp.get("pool_sha256")), strict=True):
        found = _pool_digest(directory)
        if found is not None and found != recorded_digest:
            problems.append(f"{directory / 'pool.jsonl'} is not the pool its provenance records")
    for component in ("length_sampler", "estimator") if _synthetic(pred) else ("estimator",):
        a, b = pp["components"].get(component), mp["components"].get(component)
        if a != b:
            problems.append(f"the {component.replace('_', ' ')} differs ({a} in the prediction, {b} in the "
                            "measurement)")
    return problems


def _point_flags(label: str, pred: Mapping[str, Any], meas: Mapping[str, Any], clock: float | None,
                 capacity: int | None, metrics: Mapping[str, Mapping[str, Any]],
                 judged: Sequence[str]) -> list[str]:
    flags = []
    missing = [name for name, c in metrics.items() if c["signed_error_pct"] is None]
    if missing:
        flags.append(f"{label}: no error for {', '.join(missing)} (a value missing on one side, or measured as 0)")
    if not meas.get("converged", True):
        flags.append(f"{label}: the measurement did not converge ({meas.get('stop_reason')}); its errors are not "
                     "evidence")
    else:
        without = [name for name in judged if name not in (meas["summary"].get("ci95") or {})]
        if without:
            flags.append(f"{label}: the measurement is certified but records no 95 % interval for "
                         f"{', '.join(without)}, which its stop rule judged")
    if not pred.get("converged", True):
        flags.append(f"{label}: the prediction did not converge ({pred.get('stop_reason')})")
    measured = _finite(meas.get("measured_clock_mhz"))
    if clock is not None and measured is None:
        flags.append(f"{label}: the measurement records no GPU clock")
    elif clock is not None and measured is not None and abs(measured - clock) > CLOCK_TOLERANCE * clock:
        flags.append(f"{label}: GPU clock measured at {measured:.0f} MHz, predicted at {clock:.0f} MHz "
                     f"({100.0 * (measured - clock) / clock:+.1f} %, beyond {100 * CLOCK_TOLERANCE:g} %)")
    ran, computed = meas.get("kv_blocks"), meas.get("kv_blocks_computed")
    if ran is None:
        flags.append(f"{label}: the measurement does not record the KV-cache blocks the engine ran with")
    elif capacity is not None and ran != capacity:
        flags.append(f"{label}: the engine ran with {ran} KV-cache blocks, the manifest gives {capacity}")
    if capacity is not None and computed is not None and computed < capacity:
        flags.append(f"{label}: the manifest's {capacity} KV-cache blocks exceed the {computed} vLLM computed from "
                     "this GPU's free memory")
    return flags


def _run_flags(pred: Mapping[str, Any], meas: Mapping[str, Any], directories: Sequence[Path]) -> list[str]:
    """What concerns the whole pair rather than one load point."""
    flags = []
    names = (_get(pred["resolved_manifest"], "engine.model"), _get(meas["resolved_manifest"], "engine.model"))
    if names[0] != names[1]:
        flags.append(f"the LLM is named {names[0]!r} in the prediction and {names[1]!r} in the measurement (the same "
                     "architecture)")
    version = meas["provenance"].get("vllm_version")
    predicted_for = _get(pred["resolved_manifest"], "engine.backend_version") or deployment.VLLM_VERSION
    if version != predicted_for:
        flags.append(f"measured on vLLM {version}; the prediction is for vLLM {predicted_for}")
    elif version != deployment.VLLM_VERSION:
        flags.append(f"both sides are for vLLM {version}; ServeCast models vLLM {deployment.VLLM_VERSION}")
    if _finite(_get(pred["resolved_manifest"], "engine.gpu_clock_mhz")) is None:
        flags.append("the prediction records no GPU clock, so the measured clocks are not checked")
    for label, directory in zip(("prediction", "measurement"), directories, strict=True):
        if not (directory / "pool.jsonl").is_file():
            flags.append(f"no pool.jsonl beside the {label}: the pool was matched on its recorded digest only")
    owned = (*manifest.SEMANTIC_FIELDS, *manifest.NOT_SEMANTIC)
    for d in differences(pred["resolved_manifest"], meas["resolved_manifest"]):
        path = d.split(":", 1)[0]
        if not any(path == p or path.startswith(p + ".") for p in owned):
            flags.append(f"not compared by this version: {d}")
    return flags


def score(prediction_dir: Path, measurement_dir: Path) -> dict[str, Any]:
    """The comparison document; `NotComparableError` when the two are not of the same experiment."""
    pred, pred_sha = _read(prediction_dir, "prediction.json", formats.Prediction)
    meas, meas_sha = _read(measurement_dir, "measurement.json", formats.Measurement)
    directories = (prediction_dir, measurement_dir)
    problems = _refusals(pred, meas, directories)
    if problems:
        raise NotComparableError("; ".join(problems) + f". {FIX}")
    predicted, measured = _by_load(pred, "prediction.json"), _by_load(meas, "measurement.json")
    common = [(p, m) for p in predicted for m in measured if _same_load(p[0], m[0])]
    if not common:
        raise NotComparableError(f"no load point in common (prediction: {[k for k, _ in predicted]}, measurement: "
                                 f"{[k for k, _ in measured]}); `--sweep` takes the same values on both commands")
    paired_p, paired_m = {id(p[1]) for p, _ in common}, {id(m[1]) for _, m in common}
    flags = [f"clients {n}, think time {z:g} ms: predicted only" for (n, z), point in predicted
             if id(point) not in paired_p]
    flags += [f"clients {n}, think time {z:g} ms: measured only" for (n, z), point in measured
              if id(point) not in paired_m]
    flags += _run_flags(pred, meas, directories)
    clock = _finite(_get(pred["resolved_manifest"], "engine.gpu_clock_mhz"))
    capacity = _get(meas["resolved_manifest"], CAPACITY)
    judged = [JUDGED[k] for k in _rule(meas) if k in JUDGED]
    points = []
    for (key, p), (_, m) in common:
        ci = m["summary"].get("ci95") or {}
        metrics = {name: compared(p["summary"].get(name), m["summary"].get(name), ci.get(name)) for name in METRICS}
        points.append({"load": m["load"], "metrics": metrics, "measured_certified": bool(m["converged"]),
                       "predicted_converged": bool(p.get("converged", True))})
        flags += _point_flags(f"clients {key[0]}, think time {key[1]:g} ms", p, m, clock, capacity, metrics, judged)
    checks = {"manifest": True, "estimand": True, "pool": True, "sampler": True if _synthetic(pred) else None,
              "estimator": True}
    doc = {"schema": "servecast-comparison/1", "prediction_sha256": pred_sha, "measurement_sha256": meas_sha,
           "checks": checks, "points": points, "flags": flags}
    formats.Comparison.model_validate(doc)
    return doc
