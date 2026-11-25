"""Steady-state estimator 2026-10: the paper's stopping rule for a closed-loop run. FROZEN: never edit; a change is a
new id.

A closed-loop run (the simulation or the measured engine) is a steady-state simulation. At each checkpoint of a
geometric schedule, the rule

1. removes the initial transient with MSER-5, capped at 40 % of the completions;
2. builds the window: from the completion of the last discarded request to the last completion, with the throughput,
   the mean response time and the mean think time of the requests kept;
3. estimates every metric's mean with a batch-means 95 % interval over 32 non-overlapping batches, usable only when
   the batch means are uncorrelated enough;
4. stops when every metric's relative half-width is under its target AND Little's law closes on the window
   (|X (R + Z) / N - 1| <= 1 %).

The metrics judged: the inverse throughput (inter-completion times), the response time, TTFT and TPOT.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

#: number of batches of the batch-means interval; the batch width grows with the run
BATCHES = 32
#: the one-sided 95 % normal quantile of the mixing test: only a POSITIVE correlation between adjacent batch means
#: narrows the interval
Z95 = 1.645
#: the mixing threshold never goes below this
RHO1_FLOOR = 0.10
#: nor above this: a residual lag-1 correlation of 0.30 already understates the half-width by about a third
RHO1_CAP = 0.30
#: Little's law must close within this relative residual on the window
LITTLE_TOL = 0.01
#: the smallest batch width at which the interval can say anything: the first checkpoint is BATCHES * this
MIN_BATCH_WIDTH = 20
#: the checkpoint schedule's ratio: each checkpoint is this many times the previous one
CHECKPOINT_RATIO = 1.2
#: MSER-5 may discard at most this fraction of the completions
MSER_CAP = 0.40
#: the targets: relative half-widths of the 95 % intervals. `inv_X` is the throughput's (its inverse is judged)
METRICS = ("inv_X", "R", "TTFT", "TPOT")


def mser5_truncation(series: Any) -> int:
    """MSER-5's truncation point, as a raw index: batch the series into means of 5 and pick the truncation d that
    minimises sum_{i>d} (z_i - mean_d)^2 / (M - d)^2 over the batched series, searching the first half only."""
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 10:
        return 0
    m = n // 5
    z = x[: m * 5].reshape(m, 5).mean(axis=1)
    m_half = max(1, m // 2)
    s1 = np.cumsum(z[::-1])[::-1]
    s2 = np.cumsum((z * z)[::-1])[::-1]
    cnt = (m - np.arange(m)).astype(float)
    g = (s2 - s1 * s1 / cnt) / (cnt * cnt)
    g = np.maximum(g[:m_half], 0.0)
    best_d = int(np.argmin(g))
    return best_d * 5


def mix_threshold(k: int) -> float:
    """The largest lag-1 correlation of k batch means still called mixed: min(0.30, max(0.10, 1.645 / sqrt(k)))."""
    if k <= 0:
        return RHO1_FLOOR
    return min(RHO1_CAP, max(RHO1_FLOOR, Z95 / math.sqrt(k)))


def is_mixed(rho1: float, k: int) -> bool:
    """Whether k batch means with lag-1 correlation rho1 are uncorrelated enough. A negative correlation passes."""
    return bool(math.isfinite(rho1) and rho1 <= mix_threshold(k))


def t975(dof: int) -> float:
    """Student's t at 0.975 with `dof` degrees of freedom; nan for dof <= 0."""
    if dof <= 0:
        return math.nan
    return float(stats.t.ppf(0.975, dof))


def batch_means(x: Any, k: int = BATCHES) -> np.ndarray:
    """The k non-overlapping batch means of x, dropping the tail that does not fill a batch."""
    x = np.asarray(x, dtype=float)
    b = x.size // k if k > 0 else 0
    if b < 1:
        return np.array([])
    return x[: k * b].reshape(k, b).mean(axis=1)


def lag1(bm: Any) -> float:
    """The lag-1 correlation of the batch means; nan with fewer than 3."""
    bm = np.asarray(bm, dtype=float)
    if bm.size < 3:
        return math.nan
    d = bm - bm.mean()
    den = float(np.dot(d, d))
    return float(np.dot(d[:-1], d[1:]) / den) if den > 0 else 0.0


def batch_ci(x: Any, k: int = BATCHES) -> dict[str, Any]:
    """Mean, 95 % half-width and relative half-width over k batches. Not usable (mean and half-width nan, relative
    half-width infinite) when the batch means are correlated or fewer than 3."""
    x = np.asarray(x, dtype=float)
    k = int(k)
    out: dict[str, Any] = {"n": int(x.size), "k": k, "b": x.size // max(k, 1), "rho1": math.nan,
                           "mean": math.nan, "hw": math.nan, "rel_hw": math.inf, "usable": False}
    bm = batch_means(x, k)
    if bm.size < 3:
        return out
    r1 = lag1(bm)
    out["rho1"] = float(r1)
    if not is_mixed(r1, bm.size):
        return out
    m = float(bm.mean())
    hw = t975(bm.size - 1) * float(bm.std(ddof=1)) / math.sqrt(bm.size)
    out.update({"mean": m, "hw": float(hw), "rel_hw": float(abs(hw / m)) if m else math.inf, "usable": True})
    return out


def judge(series: Mapping[str, Any], targets: Mapping[str, float], k: int = BATCHES) -> tuple[
        dict[str, dict[str, Any]], dict[str, float]]:
    """Each metric's interval and its target. Every metric of `series` is judged."""
    tgt = {name: float(targets[name]) for name in series}
    per = {name: batch_ci(x, k) for name, x in series.items()}
    return per, tgt


def converged(series: Mapping[str, Any], targets: Mapping[str, float], k: int = BATCHES) -> dict[str, Any]:
    """Converged only when EVERY metric is usable and under its target; the worst is the farthest from its target."""
    per, tgt = judge(series, targets, k)
    if not per:
        return {"converged": False, "worst": None, "worst_rel_hw": math.inf, "worst_ratio": math.inf,
                "per_metric": {}, "not_usable": [], "targets": tgt}
    worst_name, worst = max(per.items(), key=lambda kv: kv[1]["rel_hw"] / tgt[kv[0]])
    ok = all(v["usable"] and v["rel_hw"] <= tgt[n] for n, v in per.items())
    return {"converged": bool(ok), "worst": worst_name, "worst_rel_hw": float(worst["rel_hw"]),
            "worst_ratio": float(worst["rel_hw"]) / tgt[worst_name], "per_metric": per,
            "not_usable": [n for n, v in per.items() if not v["usable"]], "targets": tgt}


def little_residual(x_rate: float, r_mean: float, z_mean: float, n_users: int) -> float:
    """X (R + Z) / N - 1: Little's law on the closed loop. Zero in any stationary window, so a residual is a window
    or bookkeeping defect, never information about the model."""
    if n_users <= 0 or not math.isfinite(x_rate) or not math.isfinite(r_mean):
        return math.nan
    if not math.isfinite(z_mean):
        return math.nan
    return float(x_rate * (r_mean + z_mean) / float(n_users) - 1.0)


def little_projection(residual: float, n_done: int, tol: float = LITTLE_TOL) -> float:
    """How many completions in total bring Little's residual under `tol`: it decays as 1/n."""
    if not math.isfinite(residual) or n_done <= 0 or tol <= 0:
        return math.nan
    return float(n_done) * abs(float(residual)) / float(tol)


def first_checkpoint() -> int:
    """The first completion count at which the interval can say anything: BATCHES * MIN_BATCH_WIDTH."""
    return int(BATCHES) * int(MIN_BATCH_WIDTH)


def checkpoints(n_floor: int, n_ceiling: int, ratio: float = CHECKPOINT_RATIO) -> list[int]:
    """The geometric checkpoint schedule n_floor, ratio * n_floor, ... up to n_ceiling (which closes it)."""
    if n_floor < 1:
        raise ValueError(f"n_floor={n_floor} must be >= 1")
    if not (1.0 < float(ratio) <= 4.0):
        raise ValueError(f"ratio={ratio} must be in (1, 4]")
    out: list[int] = []
    n = float(n_floor)
    while int(n) < int(n_ceiling):
        if not out or int(n) > out[-1]:
            out.append(int(n))
        n *= float(ratio)
    out.append(int(n_ceiling))
    return out


def window_stats(completion_t: Any, e2e: Any, think_t: Any, n_users: int, trunc: int = 0) -> dict[str, Any]:
    """X, mean R, mean Z and Little's residual on the window (t_d, t_last], t_d the completion of the last discarded
    request (the first completion when nothing is discarded, which opens the window and is not counted in it)."""
    ct = np.asarray(completion_t, float)
    r = np.asarray(e2e, float)
    z = np.asarray(think_t, float)
    n = int(ct.size)
    out: dict[str, Any] = {"n_kept": 0, "x": math.nan, "r_mean": math.nan, "z_mean": math.nan, "t0": math.nan,
                           "t1": math.nan, "span": math.nan, "little_res": math.nan, "trunc": int(trunc)}
    d = max(0, int(trunc))
    if n - d < 3:
        return out
    t0 = float(ct[d])
    t1 = float(ct[-1])
    span = t1 - t0
    if not (span > 0):
        return out
    n_kept = n - d - 1
    r_mean = float(r[d + 1:].mean())
    z_mean = float(z[d + 1:].mean())
    x = n_kept / span
    out.update({"n_kept": n_kept, "x": x, "r_mean": r_mean, "z_mean": z_mean, "t0": t0, "t1": t1, "span": span,
                "little_res": little_residual(x, r_mean, z_mean, n_users)})
    return out


def stop_decision(*, n_done: int, n_ceiling: int, series: Mapping[str, Any], targets: Mapping[str, float],
                  little_res: float, elapsed_s: float = 0.0, seconds_budget: float | None = None) -> dict[str, Any]:
    """Stop or continue, and why. Only precision AND Little can declare convergence; the ceiling and the time budget
    stop a run without converging."""
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


def series_of(kept: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """The judged per-request series of the window, in completion order."""
    ct = np.array([q["completion_t"] for q in kept], float)
    return {"inv_X": np.diff(ct) if ct.size > 1 else np.array([]),
            "R": np.array([q["e2e"] for q in kept], float),
            "TTFT": np.array([float(q["ttft"]) for q in kept if np.isfinite(q["ttft"])]),
            "TPOT": np.array([float(q["tpot"]) for q in kept if np.isfinite(q["tpot"])])}


def assess(records: Sequence[Mapping[str, Any]], n_at: int, n_users: int, targets: Mapping[str, float], *,
           n_ceiling: int, elapsed_s: float = 0.0, seconds_budget: float | None = None) -> dict[str, Any]:
    """One checkpoint: the first n_at + 1 completions, MSER-5, the window, the decision. `records` are every request
    of the run, finished or not, each with completion_t, e2e, think_t, ttft, tpot and rid."""
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
