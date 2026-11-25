"""A calibration file, servecast-calibration/2: a cost law's parameters, the conditions they hold for, and how they
were fitted.

    law         the cost law by name, its parameters by name, and the grid a per-GPU law reads eta_c off
    conditions  the GPU, the registry numbers the fit assumed, the clock the probes ran at, the tensor parallelism,
                and the LLM or the synthetic models it was fitted on
    fit         how it was obtained (cells, evaluations, training error, the profiling data's sha256): provenance,
                which predict never reads

A file of the first format, servecast-calibration/1, is read by converting it with `upgrade`, as the shipped
calibrations were converted (tools/convert_calibrations.py): the same numbers, moved.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from servecast.core import gpus, versions
from servecast.core.formats import CalibrationFile

SCHEMA = "servecast-calibration/2"
V1 = "servecast-calibration/1"


def upgrade(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The servecast-calibration/2 form of a /1 calibration. Its peak fraction was its clock over the registry's boost
    clock, which /2 records instead; a /1 file whose fraction is anything else is refused."""
    gpu = gpus.get(str(doc["gpu"]))
    clock = float(doc["clock"]["reference_mhz"])
    if clock / float(gpu.boost_sm_mhz) != float(doc["clock"]["peak_fraction"]):
        raise ValueError(f"its peak fraction {doc['clock']['peak_fraction']} is not its clock over {gpu.id}'s boost "
                         f"clock ({clock:g} / {gpu.boost_sm_mhz})")
    law: dict[str, Any] = {"cost_law": versions.canonical("cost_law", str(doc["cost_law"])),
                           "parameters": dict(zip(doc["parameters"]["names"], doc["parameters"]["values"],
                                                  strict=True))}
    if doc.get("eta_c_grid") is not None:
        law["eta_c_grid"] = doc["eta_c_grid"]
    out: dict[str, Any] = {
        "schema": SCHEMA, "name": doc["name"], "law": law,
        "conditions": {"gpu": gpu.id,
                       "gpu_spec": {"peak_flop_per_s": doc["datasheet"]["peak_flops"],
                                    "memory_bandwidth_bytes_per_s": doc["datasheet"]["bandwidth"],
                                    "boost_clock_mhz": gpu.boost_sm_mhz},
                       "clock_mhz": clock, "tensor_parallel": int(doc.get("tensor_parallel", 1)),
                       "fitted_on": doc["calibrated_on"]}}
    if "fit" in doc or "profiling_sha256" in doc:
        out["fit"] = {**dict(doc.get("fit") or {}), "profiling_sha256": doc.get("profiling_sha256")}
    return out


def parse(doc: Any, source: object) -> dict[str, Any]:
    """A calibration as ServeCast reads it: servecast-calibration/2, checked against its format, its cost law by name.
    A /1 file is converted first. ValueError names `source` and the problem."""
    if not isinstance(doc, dict):
        raise ValueError(f"{source}: not a calibration ({SCHEMA})")
    try:
        if doc.get("schema") == V1:
            doc = upgrade(doc)
        CalibrationFile.model_validate(doc)
        versions.law(doc["law"]["cost_law"])  # a law this version of ServeCast has
    except (KeyError, TypeError, ValueError, ValidationError) as error:
        raise ValueError(f"{source}: not a calibration ({SCHEMA}): {error}") from None
    return doc


def peak_fraction(calibration: Mapping[str, Any]) -> float:
    """The fraction of the GPU's peak the probes ran at: their clock over the boost clock."""
    conditions = calibration["conditions"]
    return float(conditions["clock_mhz"]) / float(conditions["gpu_spec"]["boost_clock_mhz"])


def parameter_values(calibration: Mapping[str, Any], names: tuple[str, ...]) -> list[float]:
    """The parameters in the law's order; a calibration with other parameters is refused."""
    parameters = calibration["law"]["parameters"]
    if tuple(parameters) != tuple(names):
        raise ValueError(f"{calibration['name']}: parameters {list(parameters)}, cost law "
                         f"{calibration['law']['cost_law']} has {list(names)}")
    return [float(parameters[n]) for n in names]
