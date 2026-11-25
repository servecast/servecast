"""`servecast replicate`: the paper, with its pinned components, calibrations, LLMs, traces and settings.

Every prediction here uses the paper's lockfile (all six components at 2026-10, no length sampler), the paper's
calibration of the GPU and LLM, the campaign's own engine settings and stop rule, and the paper's trace; every output
names those components. Vidur is never run: its predictions ship frozen.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from servecast.core import versions
from servecast.model import geometry
from servecast.model import predict as P
from servecast.paper import data, refit, tables

#: the paper's lockfile: every component it used, at the version it used
COMPONENTS = dict(versions.PAPER)  # by name


@dataclasses.dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    tables: tuple[str, ...]
    recompute: str | None          # "diagonal", "transfers", "refits", "workload", or None (printed only)


EXPERIMENTS = {e.key: e for e in (
    Experiment("rq1", "accuracy: ServeCast and Vidur, each calibrated on the LLM and GPU it predicts, overall and "
                      "by load band",
               ("rq1-accuracy", "load-bands"),
               "diagonal"),
    Experiment("rq2", "transfer: calibrated on one LLM, predicting another on the same GPU",
               ("rq2-matrix-ttft", "rq2-matrix-throughput", "rq2-matrix-tpot"), "transfers"),
    Experiment("rq3", "cost: profiling GPU time and simulation time, against Vidur's", ("rq3-cost",), "refits"),
    Experiment("ablation", "what the model's parts are worth: residual, clock, length model, workload law",
               ("workload-fit", "length-ablation", "residual-ablation", "clock-ablation"), "workload"),
    Experiment("appendix", "the appendix: calibration coefficients and simulation run times",
               ("calibration-coefficients", "appendix-runtime"), "refits"),
)}


def predict_one(gpu: str, calibrated_on: str, llm: str, trace: str, concurrency: int, *,
                progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """One of the paper's load points, predicted with the paper's pipeline."""
    camp = data.campaign(gpu, llm, trace)
    cell = next((c for c in camp["cells"] if c["concurrency"] == int(concurrency)), None)
    if cell is None:
        raise ValueError(f"the campaign of {llm} on {gpu} with {trace} has no load point at {concurrency} clients; it "
                         f"has {', '.join(str(c['concurrency']) for c in camp['cells'])}")
    s = camp["prediction"]
    calibration = P.load_calibration(f"{gpu}/{versions.PAPER['cost_law']}/{data.slug(calibrated_on)}")
    row = P.predict_point(calibration, geometry.resolve(llm), P.load_workload_law(trace),
                          P.Engine(s["kv_blocks"], s["block_size"], s["max_batched_tokens"], s["max_sequences"]),
                          cell["concurrency"], cell["think_time_s"], components=COMPONENTS, targets=s["targets"],
                          seed=s["seed"], checkpoint_ratio=s["checkpoint_ratio"], max_completions=s["max_completions"],
                          progress=progress)
    return {"throughput_rps": row["throughput_rps"], "mean_ttft_s": row["ttft_mean"], "mean_tpot_s": row["tpot_mean"],
            "converged": row["converged"], "wall_s": row["wall_s"]}


def _job(args: tuple[str, str, str, str, int]) -> tuple[tuple[str, str, str, str, int], dict[str, Any]]:
    return args, predict_one(*args)


def jobs_of(experiment: Experiment, gpu: str | None) -> list[tuple[str, str, str, str, int]]:
    """(gpu, calibrated on, llm, trace, clients) of every load point the experiment predicts."""
    out = []
    for c in data.campaigns():
        if gpu is not None and c["gpu"] != gpu:
            continue
        sources = [c["llm"]] if experiment.recompute == "diagonal" else list(data.LLMS)
        for cell in c["cells"]:
            for src in sources:
                if src in (cell.get("predicted") or {}):
                    out.append((c["gpu"], src, c["llm"], c["trace"], cell["concurrency"]))
    return out


def recompute(experiment: Experiment, gpu: str | None, jobs: int,
              report: Callable[[int, int, tuple, dict[str, Any]], None] | None = None) -> dict[tuple, dict[str, Any]]:
    todo = jobs_of(experiment, gpu)
    out: dict[tuple, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        futures = [ex.submit(_job, t) for t in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            key, row = fut.result()
            out[key] = row
            if report is not None:
                report(i, len(todo), key, row)
    return out


def predicted_from(recomputed: Mapping[tuple, Mapping[str, Any]]) -> tables.Predicted:
    """A `predicted` function for the tables that reads recomputed predictions, Vidur's from the shipped data."""
    def predicted(campaign: dict[str, Any], cell: dict[str, Any], method: str) -> Mapping[str, Any] | None:
        if method == "vidur":
            return cell["vidur"]
        return recomputed.get((campaign["gpu"], method, campaign["llm"], campaign["trace"], cell["concurrency"]))
    return predicted


def compute_tables(experiment: Experiment, recomputed: Mapping[tuple, Mapping[str, Any]] | None,
                   gpu: str | None = None,
                   refitted: tables.Refitted | None = None,
                   laws: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, list[list[Any]]]:
    """The experiment's tables: computed for rq1 and rq2 (from recomputed or shipped predictions), computed from the
    refitted calibrations when `refitted` is given (the coefficients, and the cost table's training MAPE), the
    ablations from their shipped per-load-point predictions and the workload laws (`laws`, refitted, or the shipped
    ones), as printed for the others."""
    out: dict[str, list[list[Any]]] = {}
    published = data.tables()
    for name in experiment.tables:
        if name in tables.REFITTED and refitted is not None:
            fits = {k: v for k, v in refitted.items() if gpu is None or k[0] == gpu}
            out[name] = (tables.coefficients(fits) if name == "calibration-coefficients"
                         else tables.training_mape(fits, published[name]))
        elif name in tables.COMPUTED and tables.COMPUTED[name][0] in ("residual", "length", "clock", "workload"):
            out[name] = tables.compute(name, data.campaigns(), published=published[name], laws=laws)
        elif name in tables.COMPUTED:
            camps = [c for c in data.campaigns() if gpu is None or c["gpu"] == gpu]
            pred = tables.frozen if recomputed is None else predicted_from(recomputed)
            if tables.COMPUTED[name][0] == "bands":
                out[name] = tables.compute(name, camps, pred, published[name])
            elif gpu is not None and tables.COMPUTED[name][0] == "accuracy":
                out[name] = tables.accuracy(camps, pred)
            elif gpu is not None:
                rows = tables.matrix(data.campaigns(), str(tables.COMPUTED[name][1]), pred)
                label = data.GPUS[gpu]
                start = next(i for i, r in enumerate(rows) if r[0] == label)
                out[name] = rows[start:start + len(data.LLMS) + 2]
            else:
                out[name] = tables.compute(name, camps, pred)
        else:
            out[name] = [list(r) for r in published[name]["rows"]]
    return out


def verify(experiment_keys: list[str], jobs: int = 1) -> tuple[int, list[str]]:
    """The computed tables against the printed ones: rq1 and rq2 from the shipped predictions, the coefficients and
    the training MAPE from the paper's calibrations fitted again from their shipped profiling data."""
    compared, gaps = 0, []
    for key in experiment_keys:
        exp = EXPERIMENTS[key]
        for name in exp.tables:
            if name in tables.REFITTED:
                rows = compute_tables(exp, None, refitted=refit.refits(jobs))[name]
                n, g = tables.compare(name, rows, data.tables()[name], tables.REFITTED[name])
            elif name in tables.COMPUTED:
                published = data.tables()[name]
                n, g = tables.compare(name, tables.compute(name, data.campaigns(), published=published), published,
                                      tables.COLUMNS.get(name))
            else:
                continue
            compared += n
            gaps += g
    return compared, gaps


def now() -> float:
    return time.time()
