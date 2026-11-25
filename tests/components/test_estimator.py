"""The steady-state estimators 2026-10 and 2026-11 against the reference implementation, bit for bit."""

from __future__ import annotations

import math

import numpy as np
import pytest

from servecast.core import estimator
from servecast.core.estimator import v2026_10 as e10
from servecast.core.estimator import v2026_11 as e11
from tests.components.inputs import closed_records, load, same, series

G = load("estimator")
PAPER = {"inv_X": 0.05, "R": 0.10, "TTFT": 0.10, "TPOT": 0.05}
P95 = {**PAPER, "TTFT_p95": 0.10, "TPOT_p95": 0.10}


@pytest.mark.exact
def test_t975_and_the_mixing_threshold() -> None:
    assert all(same(e10.t975(int(d)), v) for d, v in G["t975"].items())
    assert all(same(e10.mix_threshold(int(k)), v) for k, v in G["mix_threshold"].items())


@pytest.mark.exact
def test_the_checkpoint_schedule() -> None:
    assert e10.checkpoints(640, 2_621_440) == G["checkpoints"]["paper_ceiling"]
    assert e10.checkpoints(640, 10 ** 12)[:80] == G["checkpoints"]["unbounded_head"]
    assert e10.first_checkpoint() == 640


@pytest.mark.exact
@pytest.mark.parametrize("case", G["series"], ids=lambda c: f"{c['kind']}-{c['n']}")
def test_one_series(case: dict) -> None:
    x = series(case["kind"], case["n"], case["seed"])
    assert e10.mser5_truncation(x) == case["mser5"]
    assert same(e10.batch_ci(x), case["batch_ci"])
    assert same(e11.quantile_ci(x), case["quantile_ci"])
    assert same(e10.lag1(e10.batch_means(x)), case["lag1_of_batch_means"])


@pytest.mark.exact
@pytest.mark.parametrize("case", G["closed"], ids=lambda c: f"n{c['n']}-N{c['n_users']}")
def test_closed_loop_checkpoints(case: dict) -> None:
    recs = closed_records(case["n"], case["seed"], case["transient"])
    capped: list[bool] = []
    for want in case["checkpoints"]:
        for name, module, targets, ceiling in (("paper", e10, PAPER, 2_621_440), ("p95", e11, P95, 10 ** 12),
                                               ("ceiling", e10, PAPER, want["n_at"] // 2)):
            got = module.assess(recs, want["n_at"], case["n_users"], targets, n_ceiling=ceiling)
            assert got["trunc"] == want["trunc"]
            capped.append(got["trunc"] == int(e10.MSER_CAP * (want["n_at"] + 1)))
            assert same(got["window"], want["window"])
            d = got["decision"]
            assert same({k: d[k] for k in want["decisions"][name]}, want["decisions"][name]), name
    if case["transient"]:
        assert any(capped), "this input exists to make MSER-5's 40 % cap bind"


def test_2026_11_without_a_percentile_target_is_2026_10() -> None:
    recs = closed_records(4000, 11)
    a = e10.assess(recs, 3999, 40, PAPER, n_ceiling=10 ** 12)["decision"]
    b = e11.assess(recs, 3999, 40, PAPER, n_ceiling=10 ** 12)["decision"]
    assert same(a, b)


def test_only_precision_and_little_together_declare_convergence() -> None:
    flat = {name: series("iid", 6400, i) for i, name in enumerate(PAPER)}
    d = e10.stop_decision(n_done=6400, n_ceiling=10 ** 12, series=flat, targets=PAPER, little_res=0.5)
    assert d["ci_ok"] and not d["little_ok"] and not d["converged"] and not d["stop"]
    d = e10.stop_decision(n_done=6400, n_ceiling=6400, series=flat, targets=PAPER, little_res=0.5)
    assert d["stop"] and not d["converged"] and d["why"] == "n_ceiling"
    d = e10.stop_decision(n_done=6400, n_ceiling=10 ** 12, series=flat, targets=PAPER, little_res=0.0)
    assert d["stop"] and d["converged"] and d["why"] == "converged"


def test_a_correlated_series_is_not_usable_and_says_so() -> None:
    trend = np.linspace(0.0, 1.0, 3200) + 0.01 * series("iid", 3200, 9)
    ci = e10.batch_ci(trend)
    assert not ci["usable"] and math.isinf(ci["rel_hw"]) and math.isnan(ci["mean"])


def test_released_ids() -> None:
    # by name, and by the month each was released, which stands for the same entry
    assert estimator.get("cabeb968") is estimator.get("2026-10") is e10
    assert estimator.get("c3ad4006") is estimator.get("2026-11") is e11
    with pytest.raises(ValueError, match=r"known: cabeb968 \(2026-10\), c3ad4006 \(2026-11\)"):
        estimator.get("2026-12")
