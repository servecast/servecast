"""A load point's summary, from the steady-state estimator's last assessment. The prediction and the measurement both
call this module, so both report the same quantities over the same window: the requests the estimator kept after
removing the warm-up, and the window it computed the throughput on.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any

import numpy as np


def p95(values: Sequence[float]) -> float:
    """The 95th percentile (numpy's default, linear interpolation); nan without values."""
    return float(np.quantile(np.asarray(values, dtype=float), 0.95)) if len(values) else math.nan


def window_quantities(kept: Sequence[Mapping[str, Any]], window: Mapping[str, Any]) -> dict[str, Any]:
    """The kept requests' TTFT and TPOT (finite values, in completion order), their means and 95th percentiles, the
    throughput and the output tokens per second over the window."""
    ttfts = [float(q["ttft"]) for q in kept if np.isfinite(q["ttft"])]
    tpots = [float(q["tpot"]) for q in kept if np.isfinite(q["tpot"])]
    span = window["span"]
    out_tokens = sum(int(q["g"]) for q in kept)
    return {"throughput_rps": window["x"], "span_s": span, "n_kept": window["n_kept"],
            "output_tokens_per_s": (out_tokens / span) if span and np.isfinite(span) else math.nan,
            "ttft": ttfts, "tpot": tpots,
            "ttft_mean": float(np.mean(ttfts)) if ttfts else math.nan,
            "tpot_mean": float(np.mean(tpots)) if tpots else math.nan,
            "p95_ttft_s": p95(ttfts), "p95_tpot_s": p95(tpots)}


def interval(per: Mapping[str, Any], scale: float, invert: bool = False) -> dict[str, float] | None:
    """A metric's 95 % interval in the summary's units; None when it is not usable or, inverted, unbounded."""
    if not per or not per.get("usable"):
        return None
    m, hw = float(per["mean"]), float(per["hw"])
    if invert:
        return {"low": 1.0 / (m + hw), "high": 1.0 / (m - hw)} if m > hw else None
    return {"low": (m - hw) * scale, "high": (m + hw) * scale}


def _finite(v: Any) -> float | None:
    return float(v) if v is not None and math.isfinite(float(v)) else None


def _ms(v: float | None) -> float | None:
    return v * 1e3 if v is not None and math.isfinite(v) else None


def point_summary(q: Mapping[str, Any], per_metric: Mapping[str, Any], estimator: ModuleType,
                  wall_s: float | None) -> dict[str, Any]:
    """The summary in the output files' units (ms, requests and tokens per second); `q` is `window_quantities`'s
    result (or the population model's equivalent row), `per_metric` the estimator's last decision's."""
    x = q["throughput_rps"]
    ci: dict[str, Any] = {}
    for key, metric, scale, invert in (("mean_ttft_ms", "TTFT", 1e3, False), ("mean_tpot_ms", "TPOT", 1e3, False),
                                       ("request_throughput_rps", "inv_X", 1.0, True)):
        iv = interval(per_metric.get(metric, {}), scale, invert)
        if iv is not None:
            ci[key] = iv
    if hasattr(estimator, "quantile_ci"):
        for key, samples in (("p95_ttft_ms", q["ttft"]), ("p95_tpot_ms", q["tpot"])):
            r = estimator.quantile_ci(samples)
            if r["usable"]:
                ci[key] = {"low": r["lo"] * 1e3, "high": r["hi"] * 1e3}
    return {"mean_ttft_ms": _ms(q["ttft_mean"]), "p95_ttft_ms": _ms(q["p95_ttft_s"]),
            "mean_tpot_ms": _ms(q["tpot_mean"]), "p95_tpot_ms": _ms(q["p95_tpot_s"]),
            "output_throughput_tok_s": _finite(q.get("output_tokens_per_s")),
            "request_throughput_rps": x if math.isfinite(x) else None,
            "num_ttft_samples": len(q["ttft"]), "num_tpot_samples": len(q["tpot"]),
            "duration_ms": (q["n_kept"] / x * 1e3) if x and math.isfinite(x) else 0.0,
            "wall_time_ms": wall_s * 1e3 if wall_s is not None else None, "ci95": ci}
