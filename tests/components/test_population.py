"""The population model 2026-10 against the reference implementation: closed-loop runs on a synthetic law
and a toy clock, request by request, evictions included."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from servecast.model.population import v2026_10 as M
from servecast.model.workload import v2026_10 as W
from tests.components.inputs import load, same

pytestmark = pytest.mark.exact

G = load("population")
LAW = W.JointLaw.from_dict(load("workload")["law"])


def toy_clock(beat: M.Beat) -> float:
    pf = sum(q for q, _ in beat.prefill)
    nd = sum(c for _, c in beat.decode)
    k = sum(kv * c for kv, c in beat.decode)
    return 0.002 + 2e-5 * (pf + nd) + 1e-9 * k


@pytest.mark.parametrize("case", G["cases"],
                         ids=lambda c: f"N{c['n_users']}-C{c['capacity_c']}-B{c['b_max']}-R{c['n_recent']}")
def test_a_closed_run_is_the_reference_one(case: dict) -> None:
    cfg = M.Config(capacity_c=case["capacity_c"], n_max=2048, b_max=case["b_max"], block_size=16,
                   n_recent=case["n_recent"])
    sd = [case["seed"], case["n_users"], int(1e6 * case["think_z"]), 0]
    r = M.run_closed(cfg, W.SeededTypes(LAW, [*sd, 1]), case["n_users"], case["think_z"],
                     np.random.default_rng([*sd, 2]), clock=toy_clock, n_complete=3000,
                     rng_tags=np.random.default_rng([*sd, 3]), rng_context=np.random.default_rng([*sd, 4]),
                     rng_evict=np.random.default_rng([*sd, 5]), invariant_every=1)
    recs = r["requests"]
    assert (r["n_beats"], len(recs), r["n_evicted"], r["seq_cap_blocked"], r["t_end"]) == (
        case["n_beats"], case["n_requests"], case["n_evicted"], case["seq_cap_blocked"], case["t_end"])
    assert same([{k: q[k] for k in case["records"][0]} for q in recs[:300]], case["records"])
    keys = ("rid", "p", "g", "j_entry", "arrival_t", "think_t", "ttft", "tpot", "completion_t", "e2e")
    assert hashlib.sha256(json.dumps([[q[k] for k in keys] for q in recs]).encode()).hexdigest() == \
        case["records_sha256"], "some request differs beyond the first 300"
    assert float(sum(q["e2e"] for q in recs if np.isfinite(q["e2e"]))) == case["sum_e2e"]


def test_the_goldens_exercise_eviction_the_sequence_cap_and_the_pooled_cohort() -> None:
    evicting = [c for c in G["cases"] if c["n_evicted"] >= 100]
    assert len(evicting) >= 2, "two cases must evict often"
    assert any(c["seq_cap_blocked"] > 0 for c in G["cases"]), "a case must hit the sequence cap"
    assert any(c["n_recent"] < 8 and c["n_evicted"] >= 100 for c in G["cases"]), \
        "a case must evict with few tracked cohorts, so the pooled cohort holds victims"


def test_the_engines_null_block_is_not_kv_capacity() -> None:
    assert M.config(9965, 16, 2048, 512).capacity_c == 9964 * 16
