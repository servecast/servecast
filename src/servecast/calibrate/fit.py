"""Profiling data to a calibration: the procedure of the data's cost law, and the calibration document.

Profiling data (`servecast-profiling-data/1`) is a file, or a directory holding `profiling.json`: the probe cells (and,
for cost law 2026-11, the GEMM grid and the synthetic models' cells) measured at one clock on one GPU.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from servecast.calibrate import procedure as procedures
from servecast.core import calibration as calibrations
from servecast.core import deployment, formats, gpus, versions
from servecast.core.paths import cache_dir, data_dir
from servecast.model import geometry

PROFILING_SCHEMA = "servecast-profiling-data/1"
#: the probe classes every procedure weights (its CLASS_SHARES): a fit needs a cell of each. With one each, it has at
#: least as many cells as the three parameters either law fits
PROBE_CLASSES = ("decode", "prefill", "mixed")
#: the law's form each procedure prices a cell with, by the cost law's name: the switches it passes to the shared
#: `cell_ms` (servecast list laws)
FORM_OF_LAW: dict[str, dict[str, bool]] = {
    "d9cbf1d3": {},
    "c62ebe75": {"tie_memory": False, "kappa_per_layer": True, "token_cost": True},
}


def read_profiling(path: Path) -> tuple[dict[str, Any], str]:
    """The profiling data and the sha256 of its bytes."""
    if not path.exists():  # the name of shipped profiling data, which is its calibration's name
        from servecast.model.predict import shipped_profiling

        path = shipped_profiling().get(path.as_posix(), path)
    file = path / "profiling.json" if path.is_dir() else path
    if not file.is_file():
        raise ValueError(f"{path}: no profiling data (a {PROFILING_SCHEMA} file, a directory holding "
                         "profiling.json, or the name of a shipped calibration whose profiling data ships: "
                         "servecast list calibrations)")
    body = file.read_bytes()
    try:
        doc = json.loads(body)
        formats.ProfilingData.model_validate(doc)
    except (ValueError, ValidationError) as error:
        raise ValueError(f"{file}: not profiling data ({PROFILING_SCHEMA}): {error}") from None
    return versions.named(doc, file), hashlib.sha256(body).hexdigest()


def slug(llm: str) -> str:
    return llm.rstrip("/").split("/")[-1].lower().removeprefix("meta-")


def default_name(profiling: dict[str, Any]) -> str:
    """local/<gpu>/<cost law>[/<llm>], as the shipped calibrations are named: a calibration of your own never takes
    a shipped calibration's name."""
    law = versions.canonical("cost_law", str(profiling["cost_law"]))
    base = f"local/{profiling['gpu']}/{law}"
    return f"{base}/{slug(profiling['llm'])}" if versions.law(law).calibrated_per == "llm_and_gpu" else base


#: one segment of a calibration name, which may have folders: local/a100_sxm_40gb/c62ebe75
NAME_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+")


def check_name(name: str) -> str:
    """A calibration name: segments of letters, digits, '.', '_' and '-' joined by '/', never absolute and never '.' or
    '..', so the calibration is saved inside the data directory (`../../x` was written outside it)."""
    parts = name.split("/")
    if not name or any(p in ("", ".", "..") or not NAME_SEGMENT.fullmatch(p) for p in parts):
        raise ValueError(f"--name: {name!r} is not a calibration name: use segments of letters, digits, '.', '_' and "
                         "'-', joined by '/', such as my-a100 or team/a100/llama")
    return name


def check_probe(profiling: Mapping[str, Any]) -> None:
    """Refuse a probe the fit cannot calibrate from: without a cell of every class it weights, the optimizer returns
    its starting values as if it had converged (a probe with no cell did, and predict used them)."""
    models = profiling.get("models")
    if models is not None:
        empty = [str(m["llm"]) for m in models if not m["cells"]]
        if empty:
            raise ValueError(f"the profiling data's synthetic model(s) {', '.join(empty)} have no probe cell")
        cells = [c for m in models for c in m["cells"]]
    else:
        cells = list(profiling["cells"])
    present = {str(c["cls"]) for c in cells}
    missing = [k for k in PROBE_CLASSES if k not in present]
    if missing:
        names = missing[0] if len(missing) == 1 else f"{', '.join(missing[:-1])} or {missing[-1]}"
        raise ValueError(f"the profiling data has no {names} cell: the fit weights the {', '.join(PROBE_CLASSES)} "
                         "probe classes and needs at least one cell of each")


def _fit_cells(profiling: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[Any], list[float]]:
    """The probe cells with the fit's weights (task and class), each with its LLM's geometry and GQA group."""
    if versions.law(profiling["cost_law"]).calibrated_per == "llm_and_gpu":
        base = procedures.get(profiling["calibration_procedure"])
        spec = geometry.resolve(profiling["llm"])
        cells = base.reweight(profiling["cells"], base.shares())
        return cells, [spec] * len(cells), [float(spec.gqa_group)] * len(cells)
    cells, specs, gs = procedures.get(str(profiling["calibration_procedure"])).pooled(profiling["models"])
    return cells, specs, gs


def fit_errors(profiling: Mapping[str, Any], values: Sequence[float], peak_fraction: float,
               gpu: gpus.Gpu) -> list[float]:
    """Each probe cell's relative error, (predicted - measured) / measured, priced as the fit prices it."""
    cells, specs, gs = _fit_cells(profiling)
    price = procedures.CELL_MS
    form = FORM_OF_LAW[versions.canonical("cost_law", str(profiling["cost_law"]))]
    return [(price(values, c, s, gpu, peak_fraction, g, **form) - float(c["T"])) / float(c["T"])
            for c, s, g in zip(cells, specs, gs, strict=True)]


def training_error(profiling: Mapping[str, Any], values: Sequence[float], peak_fraction: float,
                   gpu: gpus.Gpu) -> dict[str, float]:
    """The paper's training MAPE, the absolute percentage errors averaged over the cells with the fit's task and class
    weights (its Table 5 reports their median and range per GPU), and the largest cell error, both in %."""
    errors = [abs(e) for e in fit_errors(profiling, values, peak_fraction, gpu)]
    weights = [float(c["w"]) for c in _fit_cells(profiling)[0]]
    mape = sum(w * e for w, e in zip(weights, errors, strict=True)) / sum(weights)
    return {"mape": round(100 * mape, 3), "max": round(100 * max(errors), 3)}


def calibrate(profiling: dict[str, Any], sha256: str, *, cost_law: str | None = None,
              name: str | None = None) -> dict[str, Any]:
    """The calibration of the profiling data's cost law; `cost_law`, when given, must be the data's."""
    law = versions.canonical("cost_law", str(profiling["cost_law"]))
    if cost_law is not None and versions.canonical("cost_law", cost_law) != law:
        raise ValueError(f"the versions select cost law {cost_law}; this profiling data was measured for {law}")
    if name is not None:  # before the fit, which takes seconds to minutes
        check_name(name)
    check_probe(profiling)
    proc = procedures.get(profiling["calibration_procedure"])
    gpu = gpus.get(profiling["gpu"])
    if versions.law(law).calibrated_per == "llm_and_gpu":
        local = Path(str(profiling["llm"]))
        if local.exists():  # the refusals predict applies, which the geometry's own reader does not
            config = local / "config.json" if local.is_dir() else local
            deployment.architecture_from_config(json.loads(config.read_text(encoding="utf-8")))
        result = proc.fit(profiling, geometry.resolve(profiling["llm"]), gpu)
        calibrated_on: Any = profiling["llm"]
    else:
        result = proc.fit(profiling, gpu)
        calibrated_on = [m["llm"] for m in profiling["models"]]
    if not result["success"]:
        raise ValueError("the fit did not converge; the profiling data cannot calibrate the law")
    error = training_error(profiling, result["values"], result["peak_fraction"], gpu)
    if float(result["reference_mhz"]) / float(gpu.boost_sm_mhz) != float(result["peak_fraction"]):
        raise ValueError("the procedure fitted at a peak fraction other than its clock over the boost clock")
    fitted: dict[str, Any] = {"cost_law": law,
                              "parameters": dict(zip(result["names"], result["values"], strict=True))}
    if "eta_c_grid" in result:
        fitted["eta_c_grid"] = result["eta_c_grid"]
    doc = {"schema": calibrations.SCHEMA,
           "name": check_name(name) if name is not None else default_name(profiling),
           "law": fitted,
           "conditions": {"gpu": gpu.id,
                          "gpu_spec": {"peak_flop_per_s": gpu.peak_flops,
                                       "memory_bandwidth_bytes_per_s": gpu.bandwidth,
                                       "boost_clock_mhz": gpu.boost_sm_mhz},
                          "clock_mhz": float(result["reference_mhz"]), "tensor_parallel": 1,
                          "fitted_on": calibrated_on},
           "fit": {"cells": result["cells"], "evaluations": result["evaluations"], "error_pct": error,
                   "profiling_sha256": sha256}}
    return calibrations.parse(doc, doc["name"])  # what predict reads, checked as predict checks it


def user_path(name: str) -> Path:
    return data_dir() / "calibrations" / f"{name}.json"


class NameTakenError(ValueError):
    """One of the user's calibrations has the name, and it is not to be replaced: a refused resource."""


def check_free(name: str, shipped: set[str], overwrite: bool = False) -> None:
    """A name a calibration can be saved under: not a shipped calibration's (ValueError), and not one of the user's
    unless it is to be replaced (NameTakenError). calibrate checks it before the fit, which takes seconds to minutes."""
    if name in shipped:
        raise ValueError(f"{name} is the name of a shipped calibration; choose another with --name")
    path = user_path(name)
    if path.exists() and not overwrite:
        raise NameTakenError(f"a calibration named {name} exists ({path}); choose another name, or replace it with "
                             "--overwrite")


def save(doc: dict[str, Any], shipped: set[str], overwrite: bool = False) -> Path:
    """Saved under the user's data directory, by name; a shipped calibration's name is refused."""
    name = doc["name"]
    check_free(name, shipped, overwrite)
    path = user_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


def cached(profiling_path: Path, cost_law: str | None) -> tuple[dict[str, Any], bytes]:
    """The calibration of profiling data, fitted once and cached by the data's sha256 and the procedure's id (what
    `predict` uses when its manifest gives `calibration.data`)."""
    profiling, sha = read_profiling(profiling_path)
    key = cache_dir() / "calibrations" / profiling["calibration_procedure"] / f"{sha}.json"
    if key.is_file():
        body = key.read_bytes()
        doc = calibrations.parse(json.loads(body), key)
        if cost_law is not None and doc["law"]["cost_law"] != versions.canonical("cost_law", cost_law):
            raise ValueError(f"the versions select cost law {cost_law}; this profiling data was measured for "
                             f"{doc['law']['cost_law']}")
        return doc, body
    doc = calibrate(profiling, sha, cost_law=cost_law)
    body = (json.dumps(doc, indent=1) + "\n").encode()
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_bytes(body)
    return doc, body
