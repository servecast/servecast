"""The synthetic inputs of the component goldens, rebuilt from a seed (numpy's generators are platform-stable).

The goldens in `goldens/` were computed by the reference implementation on exactly these inputs; the functions here
must stay line for line what produced them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

GOLDENS = Path(__file__).parent / "goldens"


def load(name: str) -> Any:
    def restore(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: restore(v) for k, v in x.items()}
        if isinstance(x, list):
            return [restore(v) for v in x]
        if x in ("nan", "inf", "-inf"):
            return float(x)
        return x

    return restore(json.loads((GOLDENS / f"{name}.json").read_text(encoding="utf-8")))


def same(a: Any, b: Any) -> bool:
    """Exact equality, with NaN equal to NaN, through dicts, lists and numpy values."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=False))
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b) and type(bool(a)) is type(bool(b))
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        fa, fb = float(a), float(b)
        return (math.isnan(fa) and math.isnan(fb)) or fa == fb
    return a == b


def series(kind: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "iid":
        return 1.0 + 0.1 * rng.standard_normal(n)
    if kind == "lognormal":
        return rng.lognormal(mean=-2.0, sigma=1.0, size=n)
    if kind == "ar1":
        e = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + e[i]
        return 5.0 + x
    if kind == "transient":
        t = np.arange(n, dtype=float)
        return 2.0 + 3.0 * np.exp(-t / (n / 10.0)) + 0.2 * rng.standard_normal(n)
    raise ValueError(kind)


def closed_records(n: int, seed: int, transient: bool = False) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(0.05, n)
    ct = np.cumsum(gaps)
    e2e = rng.lognormal(-1.0, 0.5, n)
    think = rng.exponential(1.0, n)
    ttft = rng.lognormal(-3.0, 0.6, n)
    tpot = rng.lognormal(-4.0, 0.2, n)
    one_token = rng.random(n) < 0.05
    if transient:
        e2e[: int(0.45 * n)] += 5.0
    recs = []
    for i in range(n):
        recs.append({"rid": int(i), "completion_t": float(ct[i]), "e2e": float(e2e[i]), "think_t": float(think[i]),
                     "ttft": float(ttft[i]), "tpot": (math.nan if one_token[i] else float(tpot[i]))})
    recs.append({"rid": n, "completion_t": math.nan, "e2e": math.nan, "think_t": 0.3, "ttft": math.nan,
                 "tpot": math.nan})
    recs.append({"rid": n + 1, "completion_t": math.nan, "e2e": math.nan, "think_t": 0.1, "ttft": 0.02,
                 "tpot": math.nan})
    order = np.random.default_rng(seed + 1).permutation(len(recs))
    return [recs[int(i)] for i in order]


def fitter_sample() -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(41)
    n = 3000
    long_ = rng.random(n) < 0.3
    p = np.where(long_, rng.integers(1500, 3000, n), rng.integers(40, 600, n))
    g = np.where(long_, 1 + rng.geometric(1 / 60.0, n), 1 + rng.negative_binomial(3, 3 / 250.0, n))
    return [int(x) for x in p], [int(x) for x in g]
