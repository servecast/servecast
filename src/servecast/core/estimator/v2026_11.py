"""Steady-state estimator 2026-11: the 2026-10 rule, plus the 95th percentiles of TTFT and TPOT as judged metrics.
FROZEN: never edit; a change is a new id.

A `<metric>_p95` target makes that metric's 95th percentile an estimand like the means: its interval is Woodruff's,
with a batch-means variance. The indicator 1{x_i <= q} goes through the same 32-batch interval and mixing rule, and its
half-width h on the probability scale maps to [Q(p - h), Q(p + h)] on the empirical quantile function of the window.
Without such a target the rule is exactly 2026-10's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from servecast.core.estimator.v2026_10 import (BATCHES, CHECKPOINT_RATIO, LITTLE_TOL, METRICS, MSER_CAP,
                                               batch_ci, checkpoints, first_checkpoint, little_projection,
                                               mser5_truncation, series_of, window_stats)

__all__ = ["BATCHES", "CHECKPOINT_RATIO", "LITTLE_TOL", "METRICS", "MSER_CAP", "P95", "assess", "checkpoints",
           "converged", "first_checkpoint", "quantile_ci", "stop_decision"]

#: the quantile a `<metric>_p95` target judges
P95 = 0.95
SUFFIX = "_p95"


def quantile_ci(x: Any, p: float = P95, k: int = BATCHES) -> dict[str, Any]:
    """The p-quantile of x with an autocorrelation-aware 95 % interval (Woodruff, batch-means variance)."""
    x = np.asarray(x, dtype=float)
    out: dict[str, Any] = {"n": int(x.size), "k": int(k), "b": int(x.size // max(k, 1)), "q": math.nan,
                           "mean": math.nan, "lo": math.nan, "hi": math.nan, "hw": math.nan, "rel_hw": math.inf,
                           "rho1": math.nan, "usable": False}
    if x.size == 0:
        return out
    q = float(np.quantile(x, p))
    b = batch_ci((x <= q).astype(float), k)
    out.update({"q": q, "mean": q, "rho1": b["rho1"]})
    if not b["usable"]:
        return out
    lo = float(np.quantile(x, max(0.0, p - b["hw"])))
    hi = float(np.quantile(x, min(1.0, p + b["hw"])))
    hw = (hi - lo) / 2.0
    out.update({"lo": lo, "hi": hi, "hw": hw, "rel_hw": hw / q if q > 0 else math.inf, "usable": True})
    return out


def converged(series: Mapping[str, Any], targets: Mapping[str, float], k: int = BATCHES) -> dict[str, Any]:
    """2026-10's rule over the means, then every `<metric>_p95` target in the order `targets` gives them."""
    tgt = {name: float(targets[name]) for name in series}
    per = {name: batch_ci(x, k) for name, x in series.items()}
    for name, t in targets.items():
        base = name.removesuffix(SUFFIX)
        if name != base and base in series:
            tgt[name] = float(t)
            per[name] = quantile_ci(series[base], P95, k)
    if not per:
        return {"converged": False, "worst": None, "worst_rel_hw": math.inf, "worst_ratio": math.inf,
                "per_metric": {}, "not_usable": [], "targets": tgt}
    worst_name, worst = max(per.items(), key=lambda kv: kv[1]["rel_hw"] / tgt[kv[0]])
    ok = all(v["usable"] and v["rel_hw"] <= tgt[n] for n, v in per.items())
    return {"converged": bool(ok), "worst": worst_name, "worst_rel_hw": float(worst["rel_hw"]),
            "worst_ratio": float(worst["rel_hw"]) / tgt[worst_name], "per_metric": per,
            "not_usable": [n for n, v in per.items() if not v["usable"]], "targets": tgt}


def stop_decision(*, n_done: int, n_ceiling: int, series: Mapping[str, Any], targets: Mapping[str, float],
                  little_res: float, elapsed_s: float = 0.0, seconds_budget: float | None = None) -> dict[str, Any]:
    """As 2026-10's, with this module's `converged`."""
    conv = converged(series, targets)
    little_ok = math.isfinite(little_res) and abs(little_res) <= LITTLE_TOL
    n_need = (math.nan if not math.isfinite(conv["worst_ratio"]) else float(n_done) * conv["worst_ratio"] ** 2)
    out = {"n_done": int(n_done), "converged": False, "why": None, "stop": False, "ci_ok": bool(conv["converged"]),
           "little_ok": bool(little_ok), "little_res": float(little_res), "worst": conv["worst"],
           "worst_rel_hw": float(conv["worst_rel_hw"]), "targets": conv["targets"],
           "not_usable": conv["not_usable"], "n_needed": n_need,
           "n_needed_little": little_projection(little_res, n_done),
           "per_metric": {k: {kk: v[kk] for kk in ("k", "b", "rho1", "mean", "hw", "rel_hw", "usable")}
                          for k, v in conv["per_metric"].items()}}
    if conv["converged"] and little_ok:
        out.update(stop=True, converged=True, why="converged")
        return out
    if n_done >= n_ceiling:
        out.update(stop=True, converged=False, why="n_ceiling")
        return out
    if seconds_budget and elapsed_s >= float(seconds_budget):
        out.update(stop=True, converged=False, why="seconds_budget")
        return out
    return out


def assess(records: Sequence[Mapping[str, Any]], n_at: int, n_users: int, targets: Mapping[str, float], *,
           n_ceiling: int, elapsed_s: float = 0.0, seconds_budget: float | None = None) -> dict[str, Any]:
    """As 2026-10's, with this module's decision."""
    done = sorted((q for q in records if np.isfinite(q["completion_t"])), key=lambda q: (q["completion_t"], q["rid"]))
    pref = done[:n_at + 1]
    e2e = np.array([q["e2e"] for q in pref], float)
    trunc = int(min(mser5_truncation(e2e), int(MSER_CAP * e2e.size)))
    w = window_stats(np.array([q["completion_t"] for q in pref], float), np.array([q["e2e"] for q in pref], float),
                     np.array([q.get("think_t", 0.0) for q in pref], float), n_users, trunc)
    kept = pref[trunc + 1:]
    d = stop_decision(n_done=w["n_kept"], n_ceiling=n_ceiling, series=series_of(kept), targets=targets,
                      little_res=w["little_res"], elapsed_s=elapsed_s, seconds_budget=seconds_budget)
    return {"pref": pref, "trunc": trunc, "window": w, "kept": kept, "decision": d, "n_at": int(n_at)}
