"""The paper's tables that ServeCast computes: the accuracy table (RQ1) and the transfer matrices (RQ2) from
per-load-point predictions, and the appendix's coefficients and the cost table's training MAPE from the paper's
calibrations fitted again (`paper.refit`).

The rules are the paper's: a campaign's MAPE is the mean absolute signed error 100 (p - m) / m over its scored load
points; the accuracy question scores the PRIMARY load points (a certified measurement, a converged prediction on the
diagonal, a certified Vidur run) and weighs every campaign with one equally; the transfer matrices average a
calibration's campaign MAPEs over the target LLM's traces, over the EVALUATED load points.
"""

from __future__ import annotations

import math
import statistics as st
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from servecast.paper.data import GPUS, LLMS, METRICS, TRACES

Predicted = Callable[[dict[str, Any], dict[str, Any], str], Mapping[str, Any] | None]


def err_paper(pred: float, meas: float) -> float:
    if not (math.isfinite(pred) and math.isfinite(meas)) or meas == 0:
        return math.nan
    return 100.0 * (pred - meas) / meas


def _nan(x: float | None) -> float:
    return math.nan if x is None else x


def frozen(campaign: dict[str, Any], cell: dict[str, Any], method: str) -> Mapping[str, Any] | None:
    """The shipped prediction of `method` (an LLM's calibration, or "vidur") for one load point."""
    return cell["vidur"] if method == "vidur" else (cell.get("predicted") or {}).get(method)


def campaign_mape(campaign: dict[str, Any], method: str, flag: str, predicted: Predicted = frozen) -> dict[
        str, float] | None:
    cells = [x for x in campaign["cells"] if x[flag]]
    if not cells:
        return None
    out = {}
    for m, key in METRICS.items():
        errs = []
        for c in cells:
            pred = predicted(campaign, c, method)
            errs.append(abs(err_paper(_nan(None if pred is None else pred[key]), _nan(c["measured"][key]))))
        out[m] = st.fmean(errs)
    return out


def accuracy(campaigns: Sequence[dict[str, Any]], predicted: Predicted = frozen) -> list[list[Any]]:
    """The accuracy table's rows: [group, label, ServeCast X, TTFT, TPOT, Vidur X, TTFT, TPOT] in percent."""
    def pooled(field: str | None, value: str | None) -> list[float] | None:
        rows: list[tuple[dict[str, float], dict[str, float]]] = []
        for c in campaigns:
            if field is not None and c[field] != value:
                continue
            ours, vidur = campaign_mape(c, c["llm"], "primary", predicted), campaign_mape(c, "vidur", "primary")
            if ours is not None and vidur is not None:
                rows.append((ours, vidur))
        if not rows:  # a group the selection leaves out, as the other GPUs under --gpu
            return None
        return ([st.fmean(r[0][m] for r in rows) for m in METRICS]
                + [st.fmean(r[1][m] for r in rows) for m in METRICS])

    out: list[list[Any]] = [["", "All", *(pooled(None, None) or [])]]
    for group, field, labels in (("GPU", "gpu", GPUS), ("Model", "llm", LLMS), ("Workload", "trace", TRACES)):
        first = True
        for key, label in labels.items():
            values = pooled(field, key)
            if values is not None:
                out.append([group if first else "", label, *values])
                first = False
    return out


def matrix(campaigns: Sequence[dict[str, Any]], metric: str, predicted: Predicted = frozen) -> list[list[Any]]:
    """A transfer matrix's rows, per GPU: one row per LLM calibrated on (its MAPE on each target LLM and the mean over
    the other LLMs), then the transfers' mean per target and Vidur's row."""
    camp = {(c["gpu"], c["llm"], c["trace"]): c for c in campaigns}
    order = list(LLMS)
    rows: list[list[Any]] = []
    for gpu, gpu_label in GPUS.items():
        def cell(src: str, tgt: str, g: str = gpu) -> float:
            per = [campaign_mape(camp[(g, tgt, tr)], src, "evaluated", predicted)
                   for tr in TRACES if (g, tgt, tr) in camp]
            return st.fmean(p[metric] for p in per if p is not None)

        for i, src in enumerate(order):
            got = [cell(src, tgt) for tgt in order]
            got.append(st.fmean(v for tgt, v in zip(order, got, strict=False) if tgt != src))
            rows.append([gpu_label if i == 0 else "", LLMS[src], *got])
        cols = [[cell(src, tgt) for src in order if src != tgt] for tgt in order]
        rows.append(["", "ServeCast, transferred", *[st.fmean(col) for col in cols],
                     st.fmean(v for col in cols for v in col)])
        per_target = []
        for tgt in order:
            per = [campaign_mape(camp[(gpu, tgt, tr)], "vidur", "evaluated") for tr in TRACES if (gpu, tgt, tr) in camp]
            per_target.append(st.fmean(p[metric] for p in per if p is not None))
        rows.append(["", "Vidur, profiled", *per_target, st.fmean(per_target)])
    return rows


#: a load point's band: its place among its campaign's six designed client counts, sorted
BANDS = ((0, 1), (2, 3), (4, 5))


def load_bands(campaigns: Sequence[dict[str, Any]], labels: Sequence[str],
               predicted: Predicted = frozen) -> list[list[Any]]:
    """The accuracy table by load band, with the paper's rule, declared before any band error was computed: low is the
    first two of a campaign's six designed client counts, mid the next two, high the last two; a band's campaign MAPE
    is over the campaign's primary load points in that band, for ServeCast and for Vidur, and the band's value is the
    mean over the campaigns that have one. `labels` name the bands as the paper prints them."""
    rows: list[list[Any]] = []
    for label, positions in zip(labels, BANDS, strict=True):
        ours, vidur = [], []
        for c in campaigns:
            ladder = sorted(x["concurrency"] for x in c["cells"])
            band = {ladder[i] for i in positions}
            within = {**c, "cells": [x for x in c["cells"] if x["concurrency"] in band]}
            o, v = campaign_mape(within, c["llm"], "primary", predicted), campaign_mape(within, "vidur", "primary")
            if o is not None and v is not None:
                ours.append(o)
                vidur.append(v)
        if ours:
            rows.append([label, *[st.fmean(r[m] for r in ours) for m in METRICS],
                         *[st.fmean(r[m] for r in vidur) for m in METRICS]])
    return rows


#: the arms of the three ablations as the paper prints them; "full" is the campaign's own diagonal prediction
RESIDUAL_ARMS = {"full": "Full", "constant_only": "Constant only", "no_correction": "No correction"}
LENGTH_ARMS = {"full": "Original joint distribution", "independent_lengths": "Independent lengths",
               "fixed_mean_lengths": "Fixed mean lengths"}
CLOCK_ARMS = {"adjust_f_and_residual": "Adjust F and residual", "adjust_f_only": "Adjust F only",
              "no_clock_adjustment": "No clock adjustment", "refit_at_target_clock": "Refit at target clock"}
#: the traces as the length ablation's groups name them
TRACE_SHORT = {"burstgpt": "BurstGPT", "azure-conversation": "Azure", "servegen": "ServeGen"}


def err_ratio(pred: float, meas: float) -> float:
    """The signed error as the residual ablation's degradation was counted, 100 (p / m - 1): equal to `err_paper` up
    to the last bits, and those bits decide a strict comparison between two arms."""
    return 100.0 * (pred / meas - 1.0)


def _paired(campaigns: Sequence[dict[str, Any]], ablated: Sequence[dict[str, Any]]) -> list[tuple[dict, list]]:
    """Each ablated campaign with its load points paired to the campaign's own (measurement, full prediction)."""
    camp = {(c["gpu"], c["llm"], c["trace"]): c for c in campaigns}
    out = []
    for a in ablated:
        by_n = {x["concurrency"]: x for x in camp[(a["gpu"], a["llm"], a["trace"])]["cells"]}
        out.append((a, [(x, by_n[x["concurrency"]]) for x in a["cells"]]))
    return out


def _arm(arm: str, ablated: Mapping[str, Any], cell: Mapping[str, Any], llm: str) -> Mapping[str, Any]:
    return cell["predicted"][llm] if arm == "full" else ablated[arm]


def residual_ablation(campaigns: Sequence[dict[str, Any]], ablations: Mapping[str, Any]) -> list[list[Any]]:
    """The residual correction's ablation as the paper's table was made, from the predictions alone: per GPU and over
    all GPUs, the MAPE of the full law, of its constant alone and of no correction, averaged over the campaigns with
    equal weights; beside each reduced law, the percentage of the scored load points where its absolute error is
    strictly larger than the full law's (unrounded; a tie is no degradation)."""
    paired = _paired(campaigns, ablations["residual"])
    rows: list[list[Any]] = []
    for gpu, label in (*GPUS.items(), (None, "All GPUs")):
        group = [(a, cells) for a, cells in paired if gpu is None or a["gpu"] == gpu]
        if not group:
            continue
        rows.append([label])
        for arm, name in RESIDUAL_ARMS.items():
            row: list[Any] = [name]
            for key in METRICS.values():
                mape = st.fmean(st.fmean(abs(err_paper(_arm(arm, x, c, a["llm"])[key], c["measured"][key]))
                                         for x, c in cells) for a, cells in group)
                if arm == "full":
                    row.append(mape)
                    continue
                points = [(x, c, a["llm"]) for a, cells in group for x, c in cells]
                worse = sum(abs(err_ratio(x[arm][key], c["measured"][key]))
                            > abs(err_ratio(c["predicted"][llm][key], c["measured"][key])) for x, c, llm in points)
                row.append(f"{printed(mape)} ({printed(100.0 * worse / len(points))})")
            rows.append(row)
    return rows


def length_ablation(campaigns: Sequence[dict[str, Any]], ablations: Mapping[str, Any]) -> list[list[Any]]:
    """The length model's ablation: on the three campaigns the paper selected, the MAPE of the fitted joint law (the
    campaign's own prediction), of independent lengths and of fixed mean lengths, over the campaign's primary load
    points; an arm that did not converge on one loses it from its mean."""
    rows: list[list[Any]] = []
    for a, cells in _paired(campaigns, ablations["length"]):
        rows.append([f"{GPUS[a['gpu']]} / {LLMS[a['llm']]} / {TRACE_SHORT[a['trace']]}"])
        for arm, name in LENGTH_ARMS.items():
            ok = [(x, c) for x, c in cells if _arm(arm, x, c, a["llm"])["converged"]]
            rows.append([name, *[st.fmean(abs(err_paper(_arm(arm, x, c, a["llm"])[key], c["measured"][key]))
                                          for x, c in ok) for key in METRICS.values()]])
    return rows


def clock_ablation(ablations: Mapping[str, Any]) -> list[list[Any]]:
    """The clock ablation at 900 and 750 MHz: each treatment's MAPE over the certified load points measured at that
    clock that every treatment converged on."""
    rows: list[list[Any]] = []
    for c in ablations["clock"]:
        rows.append([f"{int(c['sm_clock_lock_mhz'])} MHz"])
        cells = [x for x in c["cells"]
                 if x["measured"]["certified"] and all(x.get(arm) and x[arm]["converged"] for arm in CLOCK_ARMS)]
        for arm, name in CLOCK_ARMS.items():
            rows.append([name, *[st.fmean(abs(err_paper(x[arm][key], x["measured"][key])) for x in cells)
                                 for key in METRICS.values()]])
    return rows


def workload_fit(laws: Mapping[str, Mapping[str, Any]]) -> list[list[Any]]:
    """The workload law fitted to each trace: its prompt classes, its branches and each branch's phases."""
    stages = [list(laws[t]["fit"]["stages"]) for t in TRACE_SHORT]
    return [["Prompt classes", *[int(laws[t]["fit"]["prompt_classes"]) for t in TRACE_SHORT]],
            ["Branches", *[len(s) for s in stages]],
            ["Phases per branch", *[", ".join(str(n) for n in s) for s in stages]],
            ["Total phases L_g", *[sum(s) for s in stages]]]


def shipped_laws() -> dict[str, Mapping[str, Any]]:
    """The workload laws that ship with ServeCast, one per trace of the paper."""
    from servecast.model import predict as P

    return {t: P.load_workload_law(t) for t in TRACE_SHORT}


#: table name -> (computed by, metric)
COMPUTED = {"rq1-accuracy": ("accuracy", None), "load-bands": ("bands", None),
            "rq2-matrix-throughput": ("matrix", "X"), "rq2-matrix-ttft": ("matrix", "TTFT"),
            "rq2-matrix-tpot": ("matrix", "TPOT"), "residual-ablation": ("residual", None),
            "length-ablation": ("length", None), "clock-ablation": ("clock", None), "workload-fit": ("workload", None)}
#: the columns compared for a computed table that has one label column, not two
COLUMNS = {"load-bands": (1, 2, 3, 4, 5, 6), "residual-ablation": (1, 2, 3), "length-ablation": (1, 2, 3),
           "clock-ablation": (1, 2, 3), "workload-fit": (1, 2, 3)}


def compute(name: str, campaigns: Sequence[dict[str, Any]], predicted: Predicted = frozen,
            published: Mapping[str, Any] | None = None,
            laws: Mapping[str, Mapping[str, Any]] | None = None) -> list[list[Any]]:
    """A computed table: `laws` are the workload laws of the workload-fit table (default: the shipped ones)."""
    from servecast.paper import data

    how, metric = COMPUTED[name]
    if how == "bands":
        assert published is not None, "the load bands are named as the paper prints them"
        return load_bands(campaigns, [r[0] for r in published["rows"]], predicted)
    if how == "residual":
        return residual_ablation(campaigns, data.ablations())
    if how == "length":
        return length_ablation(campaigns, data.ablations())
    if how == "clock":
        return clock_ablation(data.ablations())
    if how == "workload":
        return workload_fit(laws if laws is not None else shipped_laws())
    return accuracy(campaigns, predicted) if how == "accuracy" else matrix(campaigns, str(metric), predicted)


def printed(x: Any) -> str:
    return f"{x:.1f}" if isinstance(x, float) else str(x)


Refitted = Mapping[tuple[str, str], Mapping[str, Any]]

#: the tables computed from the paper's calibrations fitted again (`paper.refit`), and the columns of each that the fits
#: give; the cost table's other columns are measurements of the paper's machines, printed as the paper gives them
REFITTED = {"calibration-coefficients": (2, 3, 4), "rq3-cost": (1, 2)}


def coefficients(refitted: Refitted) -> list[list[Any]]:
    """The appendix's coefficients of each calibration: the compute efficiency eta_c in %, the constant residual kappa
    in ms and the residual per decode request beta_req in microseconds (the table's c0/f0 and cd/f0)."""
    rows: list[list[Any]] = []
    for gpu, gpu_label in GPUS.items():
        for i, (llm, label) in enumerate(LLMS.items()):
            if (gpu, llm) in refitted:
                p = refitted[(gpu, llm)]["parameters"]
                rows.append([gpu_label if i == 0 else "", label, 100.0 * p["eta_c"], p["kappa"],
                             1000.0 * p["beta_req"]])
    return rows


def training_mape(refitted: Refitted, published: Mapping[str, Any]) -> list[list[Any]]:
    """The cost table: per GPU, the median and the range of the training MAPE over its six calibrations, computed; the
    profiling minutes and the simulation times beside them are the paper's measurements, as printed."""
    rows: list[list[Any]] = []
    for gpu, gpu_label in GPUS.items():
        mapes = sorted(float(refitted[(gpu, llm)]["mape"]) for llm in LLMS if (gpu, llm) in refitted)
        if mapes:
            recorded = next(r for r in published["rows"] if r[0] == gpu_label)
            rows.append([gpu_label, st.median(mapes), f"{printed(mapes[0])}--{printed(mapes[-1])}", *recorded[3:]])
    return rows


def compare(name: str, rows: Sequence[Sequence[Any]], published: Mapping[str, Any],
            columns: Sequence[int] | None = None) -> tuple[int, list[str]]:
    """Cells compared at the printed precision, and the differences: `columns` of every row, or every column after
    the two that name the row."""
    want = published["rows"]
    gaps = []
    compared = 0
    if len(rows) != len(want):
        return 0, [f"{name}: {len(rows)} rows computed, {len(want)} printed"]
    group = ""
    for got, row in zip(rows, want, strict=False):
        group = row[0] or group
        which = columns if columns is not None else range(2, min(len(got), len(row)))
        if columns is not None and len(row) <= max(columns):  # a group's one-cell row: its name, nothing to compare
            if list(got) != list(row):
                gaps.append(f"{name}: a group printed as {row}, computed as {[printed(c) for c in got]}")
            continue
        label = f"{group}/{row[1]}" if min(which, default=2) >= 2 else group
        for col in which:
            compared += 1
            if printed(got[col]) != row[col]:
                gaps.append(f"{name} {label} column {published['columns'][col]}: printed {row[col]}, computed "
                            f"{printed(got[col])}")
    return compared, gaps
