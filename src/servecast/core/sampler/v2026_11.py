"""Length sampler 2026-11: integer lengths >= 1 from a mean and, optionally, a variance. FROZEN: never edit; a change
is a new id.

The distribution is the discrete phase-type match of the two moments, in the family the population model represents
(sums of geometric stages, each at least one token):

* the mean alone: a geometric on {1, 2, ...} with that mean (variance m (m - 1), the discrete exponential);
* a larger variance (coefficient of variation above the geometric's): two geometrics with balanced means
  (p1 m1 = p2 m2 = m / 2), mixed with weights p1 and p2;
* a smaller variance: a mixture of sums of k - 1 and k geometric stages with a common success probability, k the
  fewest stages that reach it.

Input and output lengths are independent. A pair whose input plus output exceeds the context is drawn again (no atom
at the limit), so the realised moments can differ from the requested ones; both are reported. A coefficient of
variation below 1/16 is refused: the population model cannot represent it within its 256 phases.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

#: the smallest coefficient of variation the population model can represent (256 phases)
MIN_CV = 1.0 / 16.0
#: the seed stream of the pool, after the user's seed
STREAM = 7
#: rounds of redrawing pairs that exceed the context before giving up
MAX_REDRAW_ROUNDS = 1000


@dataclasses.dataclass(frozen=True)
class Spec:
    """One length's distribution: `kind` geometric, two-geometric (balanced means) or stages (k - 1 / k mixture)."""

    kind: str
    mean: float
    variance: float
    p1: float = 1.0          # two-geometric: weight of branch 1; stages: weight of the (k - 1)-stage branch
    m1: float = 0.0          # two-geometric: branch means
    m2: float = 0.0
    k: int = 1               # stages: k
    q: float = 1.0           # stages: the stages' common success probability

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def match(mean: float, variance: float | None = None) -> Spec:
    """The two-moment match of a length with this mean and variance (None: the geometric's)."""
    m = float(mean)
    if not (m >= 1.0) or not math.isfinite(m):
        raise ValueError(f"a mean length must be >= 1 token, got {mean}")
    v_geo = m * (m - 1.0)
    if variance is None or float(variance) == v_geo:
        return Spec("geometric", m, v_geo)
    v = float(variance)
    if not (v > 0.0) or not math.isfinite(v):
        raise ValueError(f"a variance must be > 0, got {variance}")
    if math.sqrt(v) / m < MIN_CV:
        raise ValueError(f"a coefficient of variation of {math.sqrt(v) / m:.4f} is below {MIN_CV}: the population "
                         "model cannot represent it")
    if v > v_geo:
        kk = m * m / (2.0 * (v + m * m + m))      # p1 p2
        p1 = (1.0 + math.sqrt(1.0 - 4.0 * kk)) / 2.0
        return Spec("two_geometric", m, v, p1=p1, m1=m / (2.0 * p1), m2=m / (2.0 * (1.0 - p1)))
    k = max(2, math.ceil(m * m / (v + m)))
    if k > m:
        raise ValueError(f"a variance of {v} is too small for a mean of {m} tokens")
    # Var(u) = 2 k m u - k (k - 1) u^2 - m - m^2, u = 1 / q, with u in [m / k, m / (k - 1)]
    a, b, c = k * (k - 1.0), -2.0 * k * m, m + m * m + v
    disc = b * b - 4.0 * a * c
    if disc < 0:
        raise ValueError(f"no {k}-stage match for mean {m} and variance {v}")
    lo, hi = m / k, m / (k - 1.0)
    roots = [(-b - math.sqrt(disc)) / (2.0 * a), (-b + math.sqrt(disc)) / (2.0 * a)]
    inside = [u for u in roots if lo * (1 - 1e-12) <= u <= hi * (1 + 1e-12)]
    if not inside:
        raise ValueError(f"no {k}-stage match for mean {m} and variance {v}")
    q = 1.0 / inside[0]
    w = min(1.0, max(0.0, k - m * q))
    return Spec("stages", m, v, p1=w, k=k, q=q)


def phases_needed(spec: Spec) -> int:
    """The phases the population model needs for this output law: about 1 / CV^2 stages below CV 1."""
    if spec.kind != "stages":
        return 1
    return spec.k


def draw(spec: Spec, rng: np.random.Generator, n: int) -> np.ndarray:
    """n lengths from `spec`."""
    if spec.kind == "geometric":
        return rng.geometric(1.0 / spec.mean, size=n).astype(np.int64)
    if spec.kind == "two_geometric":
        first = rng.random(n) < spec.p1
        a = rng.geometric(1.0 / spec.m1, size=n)
        b = rng.geometric(1.0 / spec.m2, size=n)
        return np.where(first, a, b).astype(np.int64)
    if spec.kind == "stages":
        short = rng.random(n) < spec.p1
        stages = np.where(short, spec.k - 1, spec.k)
        return (stages + rng.negative_binomial(stages, spec.q)).astype(np.int64)
    raise ValueError(f"unknown length law {spec.kind!r}")


def realized(x: np.ndarray) -> dict[str, float]:
    return {"mean": float(x.mean()), "variance": float(x.var())}


def pool(input_spec: Spec, output_spec: Spec, *, size: int, seed: int,
         context_length: int | None = None) -> dict[str, Any]:
    """`size` independent (input, output) pairs; a pair exceeding `context_length` tokens in all is redrawn."""
    if size < 1:
        raise ValueError("a pool needs at least one request")
    rng = np.random.default_rng([int(seed), STREAM])
    p = draw(input_spec, rng, size)
    g = draw(output_spec, rng, size)
    if context_length is not None:
        for _ in range(MAX_REDRAW_ROUNDS):
            bad = np.flatnonzero(p + g > int(context_length))
            if bad.size == 0:
                break
            p[bad] = draw(input_spec, rng, bad.size)
            g[bad] = draw(output_spec, rng, bad.size)
        else:
            raise ValueError(f"the lengths requested rarely fit a context of {context_length} tokens")
    return {"input_tokens": p.tolist(), "output_tokens": g.tolist(),
            "requested": {"input": {"mean": input_spec.mean, "variance": input_spec.variance},
                          "output": {"mean": output_spec.mean, "variance": output_spec.variance}},
            "realized": {"input": realized(p), "output": realized(g)},
            "laws": {"input": input_spec.as_dict(), "output": output_spec.as_dict()}}
