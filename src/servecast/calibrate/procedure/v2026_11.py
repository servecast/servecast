"""Calibration procedure 2026-11: cost law 2026-11 fitted once per GPU, with no profiling of the LLM to predict.
FROZEN: never edit; a change is a new id.

Two measurements on the GPU, both at one locked clock:

1. a GEMM GRID: one synthetic transformer layer per (width d, FFN ratio d_ff / d) timed at one row and at 2,048
   rows. The one-row times against the layer's bytes give eta_w (a line's slope, 1 / (slope B)); the 2,048-row times
   give each shape's arithmetic efficiency eta_c, and the grid keeps them for interpolation (or the curve
   1 / eta_c = a + b / params, when every shape lies within 0.03 of it);
2. probes of SYNTHETIC MODELS (random weights, chosen geometries) served by the real engine, as the 2026-10 procedure
   probes one LLM.

The fit is the 2026-10 procedure's least squares over the synthetic models' cells pooled, each model carrying the same
total weight and each cell priced with its own model's counts; eta_w is pinned to the grid's, eta_c to the mean of
the grid's values for the synthetic models, beta to 0; the KV efficiency, the per-layer latency kappa and the
per-token cost tau are fitted. An LLM's own eta_c is read off the grid at prediction time.
"""

from __future__ import annotations

import statistics as st
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from servecast.core.gpus import Gpu
from servecast.model import geometry
from servecast.model.law import v2026_11 as law
from servecast.calibrate.procedure import v2026_10 as base

NAMES = law.NAMES
#: the rows at which a GEMM-grid shape's arithmetic efficiency is read
GRID_ROWS = 2048
#: beyond this residual from the curve, the grid is interpolated instead
CURVE_TOLERANCE = 0.03
SEED_KAPPA_LAYERS = 32
SEED_TAU = 5.0
TAU_UPPER = 1000.0
PIN_BETA = 0.0
PIN_DECIMALS = 6


def grid(gemm: Mapping[str, Any], gpu: Gpu) -> dict[str, Any]:
    """eta_w, the shapes' eta_c, and the rule that reads an LLM's eta_c off them."""
    mhz = float(gemm["clock_mhz"])
    peak = float(gpu.peak_flops) * mhz / float(gpu.boost_sm_mhz)
    syn = []
    for run in gemm["runs"]:
        if not run["model"].startswith("synthetic-"):
            continue
        pts = [p for rep in run["reps"] for p in rep]
        eta = st.fmean(p["flops"] / p["s_per_iter"] / peak for p in pts if int(p["n"]) == GRID_ROWS)
        s = geometry.layer(run["model"])
        one = [p for p in pts if int(p["n"]) == 1]
        syn.append({"d": s.d_model, "ratio": s.d_ff / s.d_model, "params": s.w_params, "eta_c": eta,
                    "bytes1": st.fmean(p["flops"] for p in one), "t1": st.fmean(p["s_per_iter"] for p in one)})
    slope, _floor = np.polyfit([g["bytes1"] for g in syn], [g["t1"] for g in syn], 1)
    b, a = np.polyfit([1 / g["params"] for g in syn], [1 / g["eta_c"] for g in syn], 1)
    res = [g["eta_c"] - 1 / (a + b / g["params"]) for g in syn]
    return {"clock_mhz": mhz, "eta_w": float(1 / (slope * float(gpu.bandwidth))),
            "rule": "interpolation" if max(map(abs, res)) > CURVE_TOLERANCE else "curve",
            "points": [{"d_model": int(g["d"]), "ffn_ratio": round(float(g["ratio"]), 4), "eta_c": float(g["eta_c"])}
                       for g in syn],
            "curve": {"a": float(a), "b": float(b)}}


def pooled(models: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[geometry.Geometry], list[float]]:
    """The synthetic models' cells as one fit: each model's classes reweighted, then each model brought to the first
    model's total weight; every cell keeps its model's geometry and GQA group."""
    shares = base.shares()
    cells, specs, gs = [], [], []
    ref_tot = 0.0
    for m in models:
        spec = geometry.resolve(m["llm"])
        cm = base.reweight(m["cells"], shares)
        tot = sum(float(c["w"]) for c in cm)
        ref_tot = ref_tot or tot
        for c in cm:
            cells.append({**c, "w": float(c["w"]) * (ref_tot / tot)})
            specs.append(spec)
            gs.append(float(spec.gqa_group))
    return cells, specs, gs


def fit(profiling: Mapping[str, Any], gpu: Gpu) -> dict[str, Any]:
    """The GPU's calibration from its GEMM grid and its synthetic models' probes."""
    g = grid(profiling["gemm_grid"], gpu)
    if float(profiling["reference_mhz"]) != g["clock_mhz"]:
        raise ValueError(f"the GEMM grid ran at {g['clock_mhz']} MHz, the probes at {profiling['reference_mhz']}")
    pin_w = float(f"{g['eta_w']:.{PIN_DECIMALS}f}")
    pin_c = float(f"{st.fmean(law.eta_c_for(geometry.resolve(m['llm']), g) for m in profiling['models']):.{PIN_DECIMALS}f}")
    seed = [*base.SEED, SEED_TAU]
    lower, upper = [*base.LOWER, 0.0], [*base.UPPER, TAU_UPPER]
    seed[3] = seed[3] / SEED_KAPPA_LAYERS
    for i, v, hw in ((0, pin_w, base.PIN_HALF_WIDTH), (2, pin_c, base.PIN_HALF_WIDTH),
                     (4, PIN_BETA, base.TIED_HALF_WIDTH)):
        seed[i], lower[i], upper[i] = v, v - hw, v + hw
    cells, specs, gs = pooled(profiling["models"])
    peak_fraction = float(profiling["reference_mhz"]) / float(gpu.boost_sm_mhz)
    res = base.solve(cells, specs, gs, gpu, peak_fraction, seed, lower, upper, tie_memory=False, kappa_per_layer=True,
                     token_cost=True)
    return {"names": list(NAMES), "values": [float(x) for x in res.x], "success": bool(res.success),
            "evaluations": int(res.nfev), "cells": len(cells), "reference_mhz": float(profiling["reference_mhz"]),
            "peak_fraction": peak_fraction, "eta_c_grid": {k: g[k] for k in ("rule", "points", "curve")},
            "pins": {"eta_w": pin_w, "eta_c": pin_c, "beta_req": PIN_BETA}}
