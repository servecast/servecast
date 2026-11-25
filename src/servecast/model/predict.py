"""The prediction path: a calibration, an LLM, a workload law and the engine's settings, to one load point's numbers.

Not a component: it only assembles the released components by their ids, so every number it returns is a function
of those ids and of its inputs.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from servecast.core import calibration as calibrations
from servecast.core import estimator as estimators
from servecast.core import gpus, versions
from servecast.core import summary as summaries
from servecast.model import law as laws
from servecast.model import population as populations
from servecast.model import workload as workloads
from servecast.model.geometry import Geometry

#: the stop rule's metrics: manifest name -> the estimator's
#: the paper's stop rule: relative half-widths of the 95 % intervals
PAPER_TARGETS = {"throughput": 0.05, "e2e": 0.10, "ttft": 0.10, "tpot": 0.05}
WORKLOAD_SCHEMA = "servecast-workload-law/1"


@dataclasses.dataclass(frozen=True)
class Engine:
    """The engine settings the model needs: the KV capacity in blocks and the batching limits."""

    kv_blocks: int
    block_size: int
    max_batched_tokens: int
    max_sequences: int | None


def _data(*parts: str) -> Path:
    return Path(str(resources.files("servecast").joinpath("data", *parts)))


def shipped_profiling() -> dict[str, Path]:
    """The profiling data each shipped calibration was fitted on, by that calibration's name
    (a100_sxm_40gb/d9cbf1d3/llama-3-8b, a100_sxm_40gb/c62ebe75): `servecast calibrate --data <name>` refits it."""
    root = _data("profiling")
    return {p.relative_to(root).with_suffix("").as_posix(): p for p in sorted(root.rglob("*.json"))}


def shipped_calibrations() -> dict[str, Path]:
    """Name -> path of every calibration the package ships."""
    root = _data("calibrations")
    return {p.relative_to(root).with_suffix("").as_posix(): p for p in sorted(root.rglob("*.json"))}


def load_calibration(name_or_path: str) -> dict[str, Any]:
    """A shipped calibration by name, or a calibration file by path."""
    path = shipped_calibrations().get(name_or_path) or Path(name_or_path)
    if not path.is_file():
        raise ValueError(f"no calibration {name_or_path!r}; shipped: {', '.join(shipped_calibrations())}")
    return calibrations.parse(json.loads(path.read_text(encoding="utf-8")), path)


def load_workload_law(trace_or_path: str) -> dict[str, Any]:
    path = _data("workload_laws", f"{trace_or_path}.json")
    if not path.is_file():
        path = Path(trace_or_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != WORKLOAD_SCHEMA:
        raise ValueError(f"{path}: not a workload law ({WORKLOAD_SCHEMA})")
    return doc


def clock(calibration: Mapping[str, Any], llm: Geometry) -> Callable[[Any], float]:
    """The beat clock of `calibration` (servecast-calibration/2) for `llm`. The datasheet numbers it was fitted with
    must be the registry's."""
    conditions, cost_law = calibration["conditions"], calibration["law"]["cost_law"]
    gpu = gpus.get(conditions["gpu"])
    spec = conditions["gpu_spec"]
    if (float(spec["peak_flop_per_s"]) != gpu.peak_flops or float(spec["memory_bandwidth_bytes_per_s"]) != gpu.bandwidth
            or int(spec["boost_clock_mhz"]) != gpu.boost_sm_mhz):
        raise ValueError(f"{calibration['name']}: fitted with datasheet numbers other than the registry's {gpu.id}")
    if int(conditions.get("tensor_parallel", 1)) != 1:
        raise ValueError("tensor parallelism above 1 is not supported yet")
    law = laws.get(cost_law)
    values = calibrations.parameter_values(calibration, law.NAMES)
    kw: dict[str, Any] = {"reference_peak": float(spec["peak_flop_per_s"]),
                          "peak_fraction": calibrations.peak_fraction(calibration)}
    if versions.law(cost_law).calibrated_per == "gpu":  # its eta_c is read off the GPU's grid
        kw["eta_c_grid"] = calibration["law"]["eta_c_grid"]
    return law.clock(values, llm, gpu, **kw)


def predict_point(calibration: Mapping[str, Any], llm: Geometry, workload_law: Mapping[str, Any], engine: Engine,
                  concurrency: int, think_time_s: float, *, components: Mapping[str, str],
                  targets: Mapping[str, float], seed: int = 0, checkpoint_ratio: float | None = None,
                  max_completions: int | None = None, seconds_budget: float | None = None,
                  progress: Callable[[dict[str, Any]], None] | None = None,
                  keep_requests: bool = False) -> dict[str, Any]:
    """One load point: N clients, mean think time Z. `components` maps each component to its resolved id."""
    est = estimators.get(components["estimator"])
    pop = populations.get(components["population_model"])
    wl = workloads.get(components["workload_fitter"])
    resolved_law = versions.canonical("cost_law", str(components["cost_law"]))  # a name, or a month that stands for it
    if resolved_law != versions.canonical("cost_law", str(calibration["law"]["cost_law"])):
        raise ValueError(f"cost law {resolved_law} resolved, the calibration's is {calibration['law']['cost_law']}")
    if any(k.endswith("_p95") for k in targets) and not hasattr(est, "quantile_ci"):
        raise ValueError(f"percentile targets need a newer steady-state estimator than {components['estimator']}")
    law = wl.JointLaw.from_dict(workload_law["law"])
    cfg = pop.config(engine.kv_blocks, engine.block_size, engine.max_batched_tokens, engine.max_sequences)
    row = pop.run_cell(cfg, law, int(concurrency), float(think_time_s), clock(calibration, llm), est,
                       estimators.series_targets(targets), seed=int(seed),
                       n_ceiling=pop.UNBOUNDED_CEILING if max_completions is None else int(max_completions),
                       checkpoint_ratio=checkpoint_ratio, seconds_budget=seconds_budget, progress=progress,
                       keep_requests=keep_requests)
    row["p95_ttft_s"], row["p95_tpot_s"] = summaries.p95(row["ttft"]), summaries.p95(row["tpot"])
    return row
