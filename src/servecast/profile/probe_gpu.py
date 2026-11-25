"""The calibration probe of the per-GPU cost law (c62ebe75): the synthetic models it probes and the GEMM grid it times.

Ported unchanged from the probe behind the shipped per-GPU calibration. Two measurements, at one clock:

1. the paper's probe (`probe_design`, `probe`) on each of 8 SYNTHETIC MODELS: a public base model's config.json
   (DeepSeek-Coder-1.3B, a Llama-family model) with the geometry replaced, and its tokenizer, served by vLLM with
   random weights, since an engine step's time depends on the shapes, not on what the weights compute;
2. a GEMM GRID: one transformer layer's five matrix multiplications (QKV, O, gate, up, down) on random bf16 matrices,
   for 24 layer shapes (21 synthetic: width 1024 to 8192 times three FFN ratios, KV width a quarter of the width; and
   the layers of three LLMs), at 8 row counts, 3 times, each point timed for 1.5 s.

Pure: the GPU side is in `probe_vllm`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from importlib import resources
from typing import Any

from servecast.core import deployment
from servecast.profile.probe import ProbeError, Step, build_cells, records, reference_clock

#: the synthetic models the per-GPU law is calibrated on, in the order they are probed
POOL = ("synthetic/llama-l12-d2048-f5504-h16-kv2", "synthetic/llama-l12-d3072-f8256-h24-kv24",
        "synthetic/llama-l24-d1024-f2752-h8-kv1", "synthetic/llama-l24-d2048-f5504-h16-kv16",
        "synthetic/llama-l24-d3072-f8256-h24-kv6", "synthetic/llama-l36-d1024-f2752-h8-kv8",
        "synthetic/llama-l36-d2048-f5504-h16-kv4", "synthetic/llama-l36-d3072-f8256-h24-kv3")
#: the base model of the synthetic family: its config.json, with the geometry replaced, and its tokenizer
BASE = "deepseek-ai/deepseek-coder-1.3b-base"
BASE_ARCHITECTURE = "LlamaForCausalLM"
BASE_VOCAB = 32256
BASE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "tokenizer.model")
HEAD_DIM = 128
SYNTHETIC_MODEL = re.compile(r"synthetic/llama-l(\d+)-d(\d+)-f(\d+)-h(\d+)-kv(\d+)")
SYNTHETIC_LAYER = re.compile(r"synthetic-d(\d+)-f(\d+)-k(\d+)")
#: the grid's layers of real LLMs, read from the shipped model registry
GRID_LLMS = ("deepseek-ai/deepseek-coder-1.3b-base", "Qwen/Qwen2.5-3B", "meta-llama/Llama-3.2-3B")
#: the grid's synthetic layers: width 1024 to 8192, the FFN ratios 2.75, 3.5 and 5.4 (rounded to 256), KV width d / 4
GRID_SYNTHETIC = tuple(f"synthetic-d{d}-f{256 * round(r * d / 256)}-k{d // 4}"
                       for d in (1024, 2048, 3072, 4096, 5120, 6144, 8192) for r in (2.75, 3.5, 5.4))
GRID_ROWS = (1, 64, 128, 256, 512, 1024, 2048, 4096)
GRID_REPEATS = 3
GRID_SECONDS = 1.5
#: a grid run passes when its times over the rows >= 64 lie on a line and no point beats the GPU's peak
GRID_SLOPE_FROM = 64
R2_MIN = 0.98
PEAK_CEILING = 1.02


def synthetic_geometry(name: str) -> dict[str, int]:
    """The geometry a synthetic model's name declares: layers, width, FFN width, query heads and KV heads."""
    m = SYNTHETIC_MODEL.fullmatch(name)
    if not m:
        raise ProbeError(f"{name!r} is not a synthetic model: synthetic/llama-l<L>-d<D>-f<F>-h<H>-kv<KV>")
    layers, d, f, h, kv = (int(x) for x in m.groups())
    if d != h * HEAD_DIM or h % kv:
        raise ProbeError(f"{name}: the head width must be {HEAD_DIM} and the KV heads must divide the query heads")
    return {"layers": layers, "hidden": d, "ffn": f, "heads": h, "kv_heads": kv}


def synthetic_config(name: str, base_config: Mapping[str, Any]) -> dict[str, Any]:
    """The synthetic model's config.json: the base model's, with the geometry replaced."""
    if base_config.get("architectures") != [BASE_ARCHITECTURE] or int(base_config.get("vocab_size", -1)) != BASE_VOCAB:
        raise ProbeError(f"{BASE}: config.json is not the base model's ({BASE_ARCHITECTURE}, vocabulary {BASE_VOCAB})")
    g = synthetic_geometry(name)
    return {**base_config, "hidden_size": g["hidden"], "intermediate_size": g["ffn"],
            "num_hidden_layers": g["layers"], "num_attention_heads": g["heads"], "num_key_value_heads": g["kv_heads"],
            "head_dim": HEAD_DIM, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
            "_synthetic": {"name": name, "base": BASE}}


def layer_shapes(d: int, d_ff: int, d_kv: int) -> tuple[tuple[str, int, int], ...]:
    """(name, inputs, outputs) of one layer's five matrix multiplications."""
    return (("qkv", d, d + 2 * d_kv), ("o", d, d), ("gate", d, d_ff), ("up", d, d_ff), ("down", d_ff, d))


def layer_flops(shapes: Sequence[tuple[str, int, int]], n: int) -> float:
    """Floating-point operations of the layer's matrix multiplications at n rows."""
    return 2.0 * n * sum(k * m for _, k, m in shapes)


def grid_shapes() -> list[tuple[str, tuple[tuple[str, int, int], ...]]]:
    """The 24 layer shapes of the grid, by name, in the order they are timed."""
    registry = json.loads(resources.files("servecast").joinpath("data", "registry", "models.json")
                          .read_text(encoding="utf-8"))["models"]
    out = []
    for name in GRID_LLMS:
        a = deployment.architecture_from_config(registry[name])
        out.append((name, layer_shapes(a["hidden"], a["ffn"], a["kv_width"])))
    for name in GRID_SYNTHETIC:
        m = SYNTHETIC_LAYER.fullmatch(name)
        assert m is not None, name
        d, f, k = (int(x) for x in m.groups())
        out.append((name, layer_shapes(d, f, k)))
    return out


def slope_fit(xs: Sequence[float], ys: Sequence[float]) -> dict[str, float]:
    """Ordinary least squares: slope, intercept and r2 (nan when undetermined)."""
    n = len(xs)
    nan = float("nan")
    if n < 2:
        return {"slope": nan, "intercept": nan, "r2": nan}
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return {"slope": nan, "intercept": nan, "r2": nan}
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    return {"slope": b, "intercept": a, "r2": (1.0 - sse / sst) if sst > 0 else nan}


def grid_flags(runs: Sequence[Mapping[str, Any]], peak_flop_s: float) -> list[str]:
    """Why the grid cannot calibrate the law: a repetition whose times do not lie on a line over the rows, or a point
    faster than the GPU's peak at the grid's clock."""
    out = []
    for run in runs:
        for i, rep in enumerate(run["reps"]):
            big = [p for p in rep if int(p["n"]) >= GRID_SLOPE_FROM] or list(rep)
            f = slope_fit([float(p["n"]) for p in big], [float(p["s_per_iter"]) for p in big])
            if not (f["r2"] == f["r2"] and f["r2"] >= R2_MIN) or f["slope"] <= 0:
                out.append(f"{run['model']}, repetition {i + 1}: the times do not grow on a line with the rows "
                           f"(r2 {f['r2']:.4f}, at least {R2_MIN})")
            best = max(float(p["flops"]) / float(p["s_per_iter"]) / peak_flop_s for p in rep)
            if best > PEAK_CEILING:
                out.append(f"{run['model']}, repetition {i + 1}: {100 * best:.0f} % of the GPU's peak, which is not "
                           "possible: the clock or the timing is wrong")
    return out


def profiling_document(probed: Sequence[tuple[str, Sequence[Step]]], runs: Sequence[Mapping[str, Any]], *,
                       gpu: str, cost_law: str, declared_mhz: float | None,
                       grid_clock_mhz: float) -> dict[str, Any]:
    """The profiling data (`servecast-profiling-data/1`) of the per-GPU law: each synthetic model's cells, at one
    reference clock that the GEMM grid shares, and the grid's points."""
    clocks = {}
    models = []
    for name, steps in probed:
        recs = records(steps)
        fref = reference_clock(recs, declared_mhz)
        if not fref:
            raise ProbeError(f"{name}: no composition ran at a steady clock")
        clocks[name] = fref
        models.append({"llm": name, "cells": build_cells([r for r in recs if r["f_mhz"] == fref])})
    if len(set(clocks.values())) != 1:
        raise ProbeError("the synthetic models ran at different clocks: "
                         + ", ".join(f"{n} {f:g} MHz" for n, f in clocks.items()))
    fref = next(iter(clocks.values()))
    if float(grid_clock_mhz) != fref:
        raise ProbeError(f"the GEMM grid ran at {grid_clock_mhz:g} MHz, the synthetic models at {fref:g} MHz")
    return {"schema": "servecast-profiling-data/1", "gpu": gpu, "cost_law": cost_law, "calibration_procedure": cost_law,
            "reference_mhz": float(fref), "models": models,
            "gemm_grid": {"clock_mhz": float(grid_clock_mhz),
                          "runs": [{"model": str(r["model"]),
                                    "reps": [[{"n": int(p["n"]), "s_per_iter": float(p["s_per_iter"]),
                                               "flops": float(p["flops"])} for p in rep] for rep in r["reps"]]}
                                   for r in runs]}}
