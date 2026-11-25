"""Calibration procedure 2026-10: the paper's fit of cost law 2026-10 to one LLM's probe on one GPU. FROZEN: never
edit; a change is a new id.

The probe runs synthetic prefill, decode and mixed batches at a locked clock; its CELLS are compositions with their
mean measured beat time. The fit is weighted nonlinear least squares on the relative errors (p - T) / T: each probe
class carries a fixed share of the weight (decode 0.4749, prefill 0.0110, mixed 0.5141), eta_w is pinned to 1, the KV
read runs at eta_w, and eta_c (the arithmetic efficiency), kappa (ms per beat) and beta (ms per decoding request) are
fitted within physical bounds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from servecast.core.gpus import Gpu
from servecast.model.geometry import Geometry
from servecast.model.law import v2026_10 as law

NAMES = law.NAMES
#: each probe class's share of the fit's weight
CLASS_SHARES = {"decode": 0.4749, "prefill": 0.0110, "mixed": 0.5141}
#: the cells give the LEVEL; without clock legs the residual weight is sqrt((1 - 0.5) w)
LEG_WEIGHT = 0.5
SEED = (0.746, 0.842, 0.75, 1.0, 0.03)
LOWER = (0.05, 0.05, 0.05, 0.0, 0.0)
UPPER = (1.0, 1.0, 1.0, 20.0, 1.0)
PIN_ETA_W = 1.0
PIN_HALF_WIDTH = 5e-7
#: the unused eta_kv slot, pinned so the fit does not wander in a flat direction
TIED_ETA_KV = 0.5
TIED_HALF_WIDTH = 5e-9
MAX_EVALUATIONS = 4000


def shares(raw: Mapping[str, float] = CLASS_SHARES) -> dict[str, float]:
    tot = sum(raw.values())
    return {k: v / tot for k, v in raw.items()}


def reweight(cells: Sequence[Mapping[str, Any]], class_shares: Mapping[str, float]) -> list[dict[str, Any]]:
    """Each class's cells rescaled so that the class carries its share of the total weight."""
    tot: dict[str, float] = {}
    for c in cells:
        tot[c["cls"]] = tot.get(c["cls"], 0.0) + float(c["w"])
    missing = set(tot) - set(class_shares)
    if missing:
        raise ValueError(f"no weight share for the probe class(es) {sorted(missing)}")
    return [{**c, "w": float(c["w"]) / tot[c["cls"]] * class_shares[c["cls"]]} for c in cells]


def _mean(a: np.ndarray) -> float:
    return float(a.flat[0]) if a.size == 1 else float(np.mean(a))


def cell_ms(theta: Sequence[float], cell: Mapping[str, Any], spec: Geometry, gpu: Gpu, peak_fraction: float,
            g_mix: float, **form: bool) -> float:
    """A probe cell's predicted beat time (ms): the law at the cell's mean composition."""
    return law.cell_ms(theta, spec, float(cell["q"]), float(cell["nd"]), _mean(np.asarray(cell["A"], float)),
                       _mean(np.asarray(cell["K"], float)), _mean(np.asarray(cell.get("n_pf", 0.0), float)),
                       bandwidth=float(gpu.bandwidth), peak=float(gpu.peak_flops),
                       reference_peak=float(gpu.peak_flops), peak_fraction=float(peak_fraction), g_mix=g_mix, **form)


def solve(cells: Sequence[Mapping[str, Any]], specs: Sequence[Geometry], g_mixes: Sequence[float], gpu: Gpu,
          peak_fraction: float, seed: Sequence[float], lower: Sequence[float], upper: Sequence[float],
          **form: bool) -> Any:
    """Weighted least squares on the relative errors; each cell priced with its own LLM's counts."""
    w = np.array([c["w"] for c in cells], float)
    w = w / w.sum()
    t = np.array([float(c["T"]) for c in cells], float)

    def resid(th: np.ndarray) -> np.ndarray:
        p = np.array([cell_ms(th, c, s, gpu, peak_fraction, g, **form) for c, s, g in zip(cells, specs, g_mixes)])
        return np.asarray(list(np.sqrt((1.0 - LEG_WEIGHT) * w) * (p - t) / t), float)

    return least_squares(resid, np.asarray(seed, float), bounds=(np.asarray(lower, float), np.asarray(upper, float)),
                         max_nfev=MAX_EVALUATIONS)


def fit(profiling: Mapping[str, Any], spec: Geometry, gpu: Gpu) -> dict[str, Any]:
    """The calibration of one LLM on one GPU from its probe cells."""
    cells = reweight(profiling["cells"], shares())
    peak_fraction = float(profiling["reference_mhz"]) / float(gpu.boost_sm_mhz)
    seed, lower, upper = list(SEED), list(LOWER), list(UPPER)
    seed[1], lower[1], upper[1] = TIED_ETA_KV, TIED_ETA_KV - TIED_HALF_WIDTH, TIED_ETA_KV + TIED_HALF_WIDTH
    seed[0], lower[0], upper[0] = PIN_ETA_W, PIN_ETA_W - PIN_HALF_WIDTH, PIN_ETA_W + PIN_HALF_WIDTH
    res = solve(cells, [spec] * len(cells), [float(spec.gqa_group)] * len(cells), gpu, peak_fraction, seed, lower,
                upper)
    values = [float(x) for x in res.x]
    values[1] = values[0]               # the KV read runs at eta_w: the tied slot reports it
    return {"names": list(NAMES), "values": values, "success": bool(res.success),
            "evaluations": int(res.nfev), "cells": len(cells), "reference_mhz": float(profiling["reference_mhz"]),
            "peak_fraction": peak_fraction}
