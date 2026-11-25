"""The length sampler 2026-11: the two moments are matched exactly by the law, and a pool realises them."""

from __future__ import annotations

import math

import numpy as np
import pytest

from servecast.core.sampler import v2026_11 as S


def law_moments(spec: S.Spec) -> tuple[float, float]:
    """The matched law's exact mean and variance, from its parameters."""
    if spec.kind == "geometric":
        return spec.mean, spec.mean * (spec.mean - 1.0)
    if spec.kind == "two_geometric":
        p1, p2 = spec.p1, 1.0 - spec.p1
        mean = p1 * spec.m1 + p2 * spec.m2
        ex2 = p1 * (2 * spec.m1 ** 2 - spec.m1) + p2 * (2 * spec.m2 ** 2 - spec.m2)
        return mean, ex2 - mean ** 2
    w, k, q = spec.p1, spec.k, spec.q
    def mom(j: int) -> tuple[float, float]:
        return j / q, j * (1 - q) / q ** 2 + (j / q) ** 2
    (m1, s1), (m2, s2) = mom(k - 1), mom(k)
    mean = w * m1 + (1 - w) * m2
    return mean, w * s1 + (1 - w) * s2 - mean ** 2


@pytest.mark.parametrize("mean, variance", [(250.0, None), (250.0, 250.0 ** 2 * 4), (1200.0, 1200.0 ** 2 * 0.25),
                                            (250.0, 40000.0), (100.0, 100.0 ** 2 * 0.01), (3.0, 1.0)])
def test_the_law_matches_both_moments_exactly(mean: float, variance: float | None) -> None:
    spec = S.match(mean, variance)
    m, v = law_moments(spec)
    want_v = mean * (mean - 1) if variance is None else variance
    assert math.isclose(m, mean, rel_tol=1e-9) and math.isclose(v, want_v, rel_tol=1e-9)


@pytest.mark.parametrize("mean, variance, kind", [(250.0, None, "geometric"), (250.0, 250000.0, "two_geometric"),
                                                  (250.0, 2500.0, "stages")])
def test_a_pool_realises_the_moments(mean: float, variance: float | None, kind: str) -> None:
    spec = S.match(mean, variance)
    assert spec.kind == kind
    x = S.draw(spec, np.random.default_rng(5), 400_000).astype(float)
    m, v = law_moments(spec)
    assert x.min() >= 1
    assert abs(x.mean() - m) < 5 * math.sqrt(v / x.size)
    assert abs(x.var() / v - 1) < 0.05


def test_pairs_beyond_the_context_are_redrawn_and_the_pool_is_seeded() -> None:
    a = S.pool(S.match(1200.0), S.match(250.0), size=5000, seed=3, context_length=4096)
    assert max(p + g for p, g in zip(a["input_tokens"], a["output_tokens"], strict=False)) <= 4096
    b = S.pool(S.match(1200.0), S.match(250.0), size=5000, seed=3, context_length=4096)
    assert a == b
    c = S.pool(S.match(1200.0), S.match(250.0), size=5000, seed=4, context_length=4096)
    assert a["input_tokens"] != c["input_tokens"]


@pytest.mark.parametrize("mean, variance, message", [(0.5, None, "mean length"), (1000.0, 1000.0, "below"),
                                                     (250.0, -1.0, "variance must be > 0")])
def test_impossible_lengths_are_refused(mean: float, variance: float | None, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        S.match(mean, variance)


def test_a_regular_output_needs_more_phases() -> None:
    assert S.phases_needed(S.match(250.0, 250.0 ** 2 * 0.25 ** 2)) >= 16
    assert S.phases_needed(S.match(250.0)) == 1


@pytest.mark.exact
def test_the_released_draws_are_pinned() -> None:
    from tests.components.inputs import load

    want = load("sampler")
    for i, case in enumerate(want["draws"]):
        spec = S.match(case["mean"], case["variance"])
        assert spec.as_dict() == case["law"]
        assert S.draw(spec, np.random.default_rng(100 + i), 200).tolist() == case["first"]
    pool = S.pool(S.match(1200.0), S.match(250.0, 40000.0), size=500, seed=7, context_length=2048)
    assert (pool["input_tokens"], pool["output_tokens"]) == (want["pool"]["input_tokens"],
                                                             want["pool"]["output_tokens"])
