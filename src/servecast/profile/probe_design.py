"""The calibration probe of the paper's cost law (d9cbf1d3): which batch compositions it times, and in what order.

Pure: the engine's KV pool, its context length, its token budget and its sequence cap are the only inputs, so a 1.3B
and a 7B LLM get the same design, cut to what fits. Ported unchanged from the probe behind the paper's calibrations.

The compositions sit on three axes: decoding requests (nd), the context of each (L), and the prefill chunk (q).
Each is held for several engine steps, all on one engine, with the serving configuration:

- decode cells: nd requests of an l0-token prompt, every one admitted (and decoding) before anything is measured,
  then `DECODE_SKIP` steps dropped and `DECODE_GEN` measured;
- prefill cells: one q-token prompt at a time, `PREFILL_REPEATS` times, after one untimed repetition;
- mixed cells: a background of nd decoding requests, into which q-token prompts are injected one at a time, the
  budget-saturated one (q + nd = budget) included, after one untimed injection per size;
- a reference decode cell, revisited at the start, at the end and evenly between, to measure drift.

Every class runs in one seeded random order, so a drift during the run is not aliased with a class.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

from servecast.core import deployment, gpus
from servecast.core.manifest import Manifest

#: the engine every probe of the paper ran on, whatever deployment it calibrated: a context of 4,096 tokens, 512
#: sequences, 2,048 batched tokens, blocks of 16 tokens in 0.9 of the GPU's memory, and 9,965 KV blocks; an LLM whose
#: KV cache did not fit ran with vLLM's own, smaller count (DeepSeek-LLM-7B on an A100: 2,870)
ENGINE_CONTEXT = 4096
ENGINE_SEQUENCES = 512
ENGINE_BUDGET = 2048
ENGINE_BLOCK_SIZE = 16
ENGINE_KV_BLOCKS = 9965
ENGINE_MEMORY_FRACTION = 0.9

DECODE_ND = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DECODE_L = (128, 256, 512, 1024, 2048, 4096)
DECODE_GEN = 32           # decode steps measured per cell, once every request decodes
DECODE_SKIP = 2           # decode steps dropped at the start of a cell (the first steps of a new shape)
PREFILL_Q = (64, 128, 256, 512, 1024, 2048)
PREFILL_REPEATS = 8
MIXED_ND = (1, 2, 4, 8, 16, 32, 64, 128)
MIXED_L = (128, 512, 2048)
MIXED_Q = PREFILL_Q
MIXED_REPEATS = 4
INJECT_EVERY = 2          # steps between two injections: the previous prompt has left before the next joins
INJECT_STEPS = INJECT_EVERY + 2   # steps one injection can take, with the scheduler's one-step admission lag
REFERENCE = (512, 16)     # (l0, nd) of the reference decode cell, revisited to measure drift
REFERENCE_GEN = 16
REFERENCE_VISITS = 4
KV_FILL = 0.9             # never ask for more than this fraction of the pool: no preemption inside a cell
MARGIN = 8                # extra outputs per request, beyond what the cell needs
ORDER_SEED = 20261014


def admission_beats(nd: int, l0: int, budget: int) -> int:
    """Steps until every one of nd prompts of l0 tokens has been prefilled, when the requests already decoding take
    one budget token each first (vLLM schedules running requests first): at most ceil(nd l0 / (budget - nd)), plus
    the admission lag."""
    return math.ceil(nd * l0 / max(1, budget - nd)) + 2


def decode_max_tokens(nd: int, l0: int, budget: int, gen: int = DECODE_GEN) -> int:
    """Outputs per request: the first admitted must still be decoding when the cell's last measured step runs."""
    return admission_beats(nd, l0, budget) + DECODE_SKIP + gen + MARGIN


def mixed_injections(nd: int, budget: int, max_model_len: int) -> list[list[int]]:
    """The injected prompt sizes of one background, the budget-saturated one included (q + nd = budget)."""
    qs = [q for q in MIXED_Q if q + nd <= budget and q + 1 <= max_model_len]
    sat = budget - nd
    if sat + 1 <= max_model_len and sat not in qs:
        qs.append(sat)
    return [[q, MIXED_REPEATS] for q in sorted(qs)]


def mixed_max_tokens(nd: int, l0: int, budget: int, inject: list[list[int]]) -> int:
    """Outputs per background request: admission, then every injection (one untimed warm-up per size, then the
    repeats), each up to INJECT_STEPS steps."""
    n_inj = sum(r + 1 for _, r in inject)
    return admission_beats(nd, l0, budget) + n_inj * INJECT_STEPS + MARGIN


def design(kv_tokens: int, max_model_len: int, budget: int = 2048, max_num_seqs: int = 512,
           seed: int = ORDER_SEED) -> dict[str, Any]:
    """The probe's cells and the order they run in, cut to what this engine can hold."""
    cap = KV_FILL * kv_tokens
    decode = []
    for nd in DECODE_ND:
        for l0 in DECODE_L:
            mt = decode_max_tokens(nd, l0, budget)
            if nd <= max_num_seqs and l0 + mt <= max_model_len and nd * (l0 + mt) <= cap:
                decode.append({"l0": l0, "nd": nd, "max_tokens": mt})
    prefill = [[q, PREFILL_REPEATS] for q in PREFILL_Q if q <= budget and q + 1 <= max_model_len]
    mixed = []
    for nd in MIXED_ND:
        for l0 in MIXED_L:
            inj = mixed_injections(nd, budget, max_model_len)
            if not inj:
                continue
            mt = mixed_max_tokens(nd, l0, budget, inj)
            if (nd + 1 <= max_num_seqs and l0 + mt <= max_model_len
                    and nd * (l0 + mt) + max(q for q, _ in inj) + 1 <= cap):
                mixed.append({"l0": l0, "nd": nd, "max_tokens": mt, "inject": inj})
    rl0, rnd = REFERENCE
    reference = {"l0": rl0, "nd": rnd, "max_tokens": decode_max_tokens(rnd, rl0, budget, REFERENCE_GEN),
                 "gen": REFERENCE_GEN}
    tasks = ([f"decode:{i}" for i in range(len(decode))] + [f"prefill:{i}" for i in range(len(prefill))]
             + [f"mixed:{i}" for i in range(len(mixed))])
    random.Random(seed).shuffle(tasks)
    # the reference at the start, at the end, and evenly in between
    cuts = sorted({round(j * len(tasks) / (REFERENCE_VISITS - 1)) for j in range(REFERENCE_VISITS)})
    order, visit = [], 0
    for i in range(len(tasks) + 1):
        if i in cuts:
            order.append(f"reference:{visit}")
            visit += 1
        if i < len(tasks):
            order.append(tasks[i])
    return {"version": 2, "decode": decode, "prefill": prefill, "mixed": mixed, "reference": reference,
            "order": order, "order_seed": seed, "decode_gen": DECODE_GEN, "decode_skip": DECODE_SKIP,
            "inject_every": INJECT_EVERY, "kv_tokens": int(kv_tokens), "max_model_len": int(max_model_len),
            "budget": int(budget), "max_num_seqs": int(max_num_seqs)}


def n_cells(d: Mapping[str, Any]) -> dict[str, int]:
    """Compositions per class (a mixed background with k injected sizes is k compositions)."""
    return {"decode": len(d["decode"]), "prefill": len(d["prefill"]),
            "mixed": sum(len(m["inject"]) for m in d["mixed"])}


def gates(d: Mapping[str, Any]) -> list[str]:
    """Why a design cannot calibrate the law, each reason a sentence; empty when it can."""
    out = []
    n = n_cells(d)
    for cls in ("decode", "prefill", "mixed"):
        if n[cls] == 0:
            out.append(f"no {cls} composition fits this engine")
    if n["mixed"] < 24:
        out.append(f"only {n['mixed']} mixed compositions fit this engine; the paper's calibrations had 113 or more")
    if len({c["nd"] for c in d["decode"]}) < 4:
        out.append("fewer than 4 decode batch sizes fit this engine, too few to tell the law's decode terms apart")
    sat = [m for m in d["mixed"] if any(q + m["nd"] == d["budget"] for q, _ in m["inject"])]
    if len(sat) < 4:
        out.append(f"only {len(sat)} backgrounds with a budget-saturated injection fit this engine; the law's "
                   "compute term needs mixed steps near the token budget")
    if sum(1 for o in d["order"] if o.startswith("reference")) < 3:
        out.append("fewer than 3 reference revisits: drift is not measurable")
    return out


def engine(gpu: gpus.Gpu, architecture: Mapping[str, Any], name: str, *,
           kv_blocks: int = ENGINE_KV_BLOCKS) -> deployment.Deployment:
    """The paper's probe engine for the model `name` on `gpu`, with `kv_blocks` KV blocks: refused, as a deployment
    is, when the model's context is shorter than the engine's or its KV cache and weights do not fit the GPU."""
    m = Manifest.model_validate({"engine": {"context_length": ENGINE_CONTEXT, "workers": {"aggregated": {
        "scheduler": {"max_batched_tokens": ENGINE_BUDGET, "max_sequences": ENGINE_SEQUENCES},
        "kv_cache": {"block_size": ENGINE_BLOCK_SIZE,
                     "capacity": {"blocks": int(kv_blocks), "memory_fraction": ENGINE_MEMORY_FRACTION}}}}}})
    return deployment.resolve(m, gpu, int(architecture["max_context"] or 0), name, architecture=architecture)
