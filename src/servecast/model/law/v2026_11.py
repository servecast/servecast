"""Cost law 2026-11: calibrated once per GPU, without profiling the LLM to predict. FROZEN: never edit; a change is a
new id.

The 2026-10 law's form, with three changes, and the arithmetic efficiency read from the GPU instead of fitted per LLM:

* the KV read has its own efficiency eta_kv, fitted, instead of running at eta_w;
* the fixed latency is per LAYER (kappa x L), since the chain of small kernels runs in every layer;
* one more kind of kernel in series, the per-token elementwise work, tau x M x d x L (tau in picoseconds), which
  does not follow the clock;
* eta_c, the fraction of the peak the layer GEMMs reach, depends on the GEMM shapes: the calibration holds a grid of
  synthetic layers (width d, FFN ratio d_ff / d) measured on the GPU, and the LLM's eta_c is read off it, bilinear
  in (d, d_ff / d) and clamped to the grid (or, when the grid follows the curve 1 / eta_c = a + b / (parameters per
  layer) closely, from that curve).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from servecast.core.gpus import Gpu
from servecast.model.geometry import Geometry
from servecast.model.law.v2026_10 import beat_ms

NAMES = ("eta_w", "eta_kv", "eta_c", "kappa", "beta_req", "tau_tok")


def eta_c_for(spec: Geometry, grid: Mapping[str, Any]) -> float:
    """The LLM's arithmetic efficiency read off the GPU's grid of synthetic layers."""
    if grid["rule"] == "curve":
        return 1.0 / (grid["curve"]["a"] + grid["curve"]["b"] / (spec.w_params / spec.n_layers))
    if grid["rule"] != "interpolation":
        raise ValueError(f"unknown grid rule {grid['rule']!r}")
    pts = {(int(p["d_model"]), float(p["ffn_ratio"])): float(p["eta_c"]) for p in grid["points"]}
    ds = sorted({d for d, _ in pts})

    def lerp(xs: Sequence[float], x: float, f: Callable[[Any], float]) -> float:
        x = min(max(x, xs[0]), xs[-1])
        for x0, x1 in zip(xs, xs[1:]):
            if x0 <= x <= x1:
                w = (x - x0) / (x1 - x0)
                return (1 - w) * f(x0) + w * f(x1)
        return f(xs[0])

    def at_d(d: Any) -> float:
        return lerp(sorted(r for dd, r in pts if dd == d), spec.d_ff / spec.d_model, lambda r: pts[(d, r)])

    return lerp(ds, float(spec.d_model), at_d)


def clock(theta: Sequence[float], spec: Geometry, gpu: Gpu, *, reference_peak: float, peak_fraction: float,
          eta_c_grid: Mapping[str, Any]) -> Callable[[Any], float]:
    """The beat clock (seconds per beat) of the GPU's calibration for the LLM `spec`."""
    th = [float(x) for x in theta]
    if len(th) != len(NAMES):
        raise ValueError(f"cost law 2026-11 has {len(NAMES)} parameters ({', '.join(NAMES)}), got {len(th)}")
    th[NAMES.index("eta_c")] = eta_c_for(spec, eta_c_grid)
    g_mix = float(spec.gqa_group)

    def seconds(beat: Any) -> float:
        return beat_ms(th, spec, beat.prefill or (), beat.decode or (), bandwidth=float(gpu.bandwidth),
                       peak=float(gpu.peak_flops), reference_peak=float(reference_peak),
                       peak_fraction=float(peak_fraction), g_mix=g_mix, tie_memory=False, kappa_per_layer=True,
                       token_cost=True) * 1e-3

    return seconds
