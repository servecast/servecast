"""Cost law 2026-10: the arXiv paper's beat-time law. FROZEN: never edit; a change is a new id.

A beat runs three kinds of kernels one after another, and each costs the larger of its memory time and its arithmetic
time (no overlap between kinds, full overlap within one):

    layer GEMMs     max(W / (eta_w B), 2 P M / (eta_c F))
    vocabulary      max(H / (eta_w B), 2 d V R / (eta_c F))
    attention       max(G_mix kv K / (eta_w B), 4 d L A / (eta_c F))

then a fixed latency per beat and a cost per decoding request, both following the SM clock:

    T = sum of the three maxima + (kappa + beta n_d) F_ref / F

with W the decoder's weight bytes, H the vocabulary matrix's, P the decoder's parameters, M the rows the GEMMs run
(the CUDA-graph size above n_d + q when it is captured), d the width, V the vocabulary, R the logit rows (prefill
requests plus decoding requests), kv the KV bytes per token of context, K the decode context, A the prefill attention
work sum q (w + q / 2), L the layers, B the datasheet bandwidth, F = peak x (clock / boost) the peak at the calibration
clock, G_mix the query heads per KV head on a beat that mixes prefill and decode (the engine reads each KV head once
per query head there) and 1 otherwise. Fitted per LLM and GPU: eta_c, kappa, beta (eta_w is pinned to 1, and the KV
read runs at eta_w: the eta_kv slot is not used).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from servecast.core.gpus import Gpu
from servecast.model.geometry import Geometry

NAMES = ("eta_w", "eta_kv", "eta_c", "kappa", "beta_req")

#: vLLM 0.25.1's CUDA-graph capture sizes at its default maximum (512): a batch runs the graph of the smallest size
#: at or above it; above the largest nothing is padded
CAPTURE_SIZES: tuple[int, ...] = tuple([1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16)))
MAX_CAPTURE = max(CAPTURE_SIZES)


def executed_rows(q: float, nd: float, sizes: Sequence[int] = CAPTURE_SIZES, s_max: int = MAX_CAPTURE) -> float:
    """The rows the GEMMs actually run: padded up to the captured graph size below the largest one."""
    m = q + nd
    if m <= s_max:
        for s in sizes:
            if m <= s:
                return float(s)
        return float(m)
    return float(m)


def cell_ms(theta: Sequence[float], spec: Geometry, q: float, nd: float, a_work: float, k_ctx: float, n_pf: float, *,
            bandwidth: float, peak: float, reference_peak: float, peak_fraction: float, g_mix: float,
            tie_memory: bool = True, kappa_per_layer: bool = False, token_cost: bool = False) -> float:
    """The beat's duration in milliseconds from its counts: prefill tokens q, decoding requests nd, the prefill
    attention work sum q (w + q / 2), the decode context K and the prefill requests. This law reads the KV at eta_w
    (`tie_memory`); a free eta_kv, the per-layer latency and the token cost belong to later laws."""
    eta_w, eta_kv, eta_c, kappa = (float(x) for x in theta[:4])
    if tie_memory:
        eta_kv = eta_w
    r_rows = n_pf + nd
    byte_kv = float(spec.kv_bytes_per_token) * k_ctx
    if q > 0 and nd > 0:
        byte_kv = byte_kv * g_mix
    p_dense = float(spec.w_bytes) / float(spec.bytes_per_elt)
    m_rows = executed_rows(q, nd)
    if kappa_per_layer:
        kappa = kappa * float(spec.n_layers)
    t_lat = kappa * (reference_peak / peak)
    n_req = float(theta[4]) * nd
    t_lat = t_lat + n_req * (reference_peak / peak)
    bw, fp = eta_w * bandwidth, eta_c * peak * peak_fraction
    head = (float(spec.head_bytes) / bw, 2.0 * spec.d_model * spec.vocab * r_rows / fp)
    parts: tuple[tuple[float, float], ...] = (
        (float(spec.w_bytes) / bw, 2.0 * p_dense * m_rows / fp), head,
        (byte_kv / (eta_kv * bandwidth), 4.0 * spec.d_model * spec.n_layers * a_work / fp))
    if token_cost:
        parts = (*parts, (float(theta[5]) * 1e-12 * m_rows * float(spec.d_model) * float(spec.n_layers), 0.0))
    t = sum(max(m, c) for m, c in parts) * 1e3 + t_lat
    return float(t)


def beat_ms(theta: Sequence[float], spec: Geometry, prefill: Sequence[tuple[int, int]],
            decode: Sequence[tuple[int, int]], *, bandwidth: float, peak: float, reference_peak: float,
            peak_fraction: float, g_mix: float, tie_memory: bool = True, kappa_per_layer: bool = False,
            token_cost: bool = False) -> float:
    """The beat's duration in milliseconds from its composition: prefill chunks (tokens, tokens done) and decode
    classes (context, count)."""
    pf = tuple(prefill)
    dc = tuple(decode)
    q = float(sum(a for a, _ in pf))
    nd = float(sum(c for _, c in dc))
    a_work = float(sum(a * (kv + a / 2.0) for a, kv in pf))
    k_ctx = float(sum(kv * c for kv, c in dc))
    n_pf = float(len(pf))
    return cell_ms(theta, spec, q, nd, a_work, k_ctx, n_pf, bandwidth=bandwidth, peak=peak,
                   reference_peak=reference_peak, peak_fraction=peak_fraction, g_mix=g_mix, tie_memory=tie_memory,
                   kappa_per_layer=kappa_per_layer, token_cost=token_cost)


def clock(theta: Sequence[float], spec: Geometry, gpu: Gpu, *, reference_peak: float,
          peak_fraction: float) -> Callable[[Any], float]:
    """The beat clock (seconds per beat) of a calibration for the LLM `spec` on `gpu`. A calibration made on another
    LLM moves to `spec` by its counts alone: the fitted parameters do not change."""
    th = [float(x) for x in theta]
    if len(th) != len(NAMES):
        raise ValueError(f"cost law 2026-10 has {len(NAMES)} parameters ({', '.join(NAMES)}), got {len(th)}")
    g_mix = float(spec.gqa_group)

    def seconds(beat: Any) -> float:
        return beat_ms(th, spec, beat.prefill or (), beat.decode or (), bandwidth=float(gpu.bandwidth),
                       peak=float(gpu.peak_flops), reference_peak=float(reference_peak),
                       peak_fraction=float(peak_fraction), g_mix=g_mix) * 1e-3

    return seconds
