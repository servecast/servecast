"""The closed workload driver, on a fake engine: structure, not numbers (the timing is real, so only what does not
depend on the machine's speed is asserted)."""

from __future__ import annotations

import asyncio
import itertools
import math
from collections.abc import AsyncIterator, Sequence

import numpy as np
import pytest

from servecast.core import estimator as estimators
from servecast.profile import closed_loop as C

#: the stop rule as the manifest names it, which is what `profile serving` passes
TARGETS = {"throughput": 0.05, "e2e": 0.10, "ttft": 0.10, "tpot": 0.05}


class FakeEngine:
    """Prefill then one token per step; records what it was asked and what was aborted."""

    def __init__(self, per_token_s: float = 0.0002) -> None:
        self.per_token_s = per_token_s
        self.asked: list[tuple[str, int, int]] = []
        self.aborted: list[str] = []

    async def generate(self, request_id: str, prompt_ids: Sequence[int], max_tokens: int) -> AsyncIterator[int]:
        self.asked.append((request_id, len(prompt_ids), max_tokens))
        await asyncio.sleep(self.per_token_s)
        for k in range(1, max_tokens + 1):
            await asyncio.sleep(self.per_token_s)
            yield k

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def serve(engine: FakeEngine | StalledEngine | FailingEngine, *, estimator: str = "2026-10",
          targets: dict[str, float] = TARGETS, guard_s: float = 60, **kw):
    pool_p, pool_g = [5, 9, 13, 2], [3, 1, 4, 6]
    run = C.serve(engine, list(range(20)), C.lengths(pool_p, pool_g, 0), kw.pop("n", 6), 0.001,
                  estimators.get(estimator), targets, seed=0, **kw)
    return asyncio.run(asyncio.wait_for(run, timeout=guard_s))


class StalledEngine(FakeEngine):
    """An engine that accepts every request and never produces a token, as a hung server would."""

    async def generate(self, request_id: str, prompt_ids: Sequence[int], max_tokens: int) -> AsyncIterator[int]:
        self.asked.append((request_id, len(prompt_ids), max_tokens))
        await asyncio.Event().wait()
        yield 0


class FailingEngine(FakeEngine):
    """An engine that fails on its fifth request."""

    async def generate(self, request_id: str, prompt_ids: Sequence[int], max_tokens: int) -> AsyncIterator[int]:
        if int(request_id) == 4:
            raise RuntimeError("the engine failed")
        async for k in super().generate(request_id, prompt_ids, max_tokens):
            yield k


def test_the_pool_is_served_in_seeded_permutations() -> None:
    first = list(itertools.islice(C.lengths([1, 2, 3, 4], [5, 6, 7, 8], 3), 8))
    assert sorted(first[:4]) == sorted(first[4:]) == [(1, 5), (2, 6), (3, 7), (4, 8)]
    assert first == list(itertools.islice(C.lengths([1, 2, 3, 4], [5, 6, 7, 8], 3), 8))
    assert first != list(itertools.islice(C.lengths([1, 2, 3, 4], [5, 6, 7, 8], 4), 8))


def test_think_times_are_exponential_and_seeded_per_client() -> None:
    a = list(itertools.islice(C.think_times(0, 1, 2.0), 20000))
    assert a[:5] == list(itertools.islice(C.think_times(0, 1, 2.0), 5))
    assert a[:5] != list(itertools.islice(C.think_times(0, 2, 2.0), 5))
    assert abs(np.mean(a) - 2.0) < 0.06 and abs(np.std(a) - 2.0) < 0.08


def test_a_load_point_records_every_request_and_stops_by_the_rule() -> None:
    engine = FakeEngine()
    out = serve(engine, n_ceiling=800)
    records, last = out["records"], out["assessment"]
    done = [q for q in records if math.isfinite(q["completion_t"])]
    assert len(done) >= 641 and out["stopped_because"] in ("converged", "n_ceiling")
    for q in done:
        assert 0 <= q["arrival_t"] <= q["first_t"] <= q["completion_t"]
        assert q["ttft"] == q["first_t"] - q["arrival_t"] and q["e2e"] == q["completion_t"] - q["arrival_t"]
        assert q["n_out"] == q["g"]
        if q["g"] > 1:
            assert q["tpot"] == (q["completion_t"] - q["first_t"]) / (q["g"] - 1)
        else:
            assert math.isnan(q["tpot"])
    assert [r[1] for r in engine.asked] == [q["p"] for q in records]
    assert sorted(engine.aborted) == sorted(str(q["rid"]) for q in records if not math.isfinite(q["completion_t"]))
    assert last["n_at"] + 1 <= len(done) and last["kept"]


def test_every_client_sends_at_once_and_thinks_between_its_requests() -> None:
    out = serve(FakeEngine(), n=4, n_ceiling=700)
    by_slot: dict[int, list[dict]] = {}
    for q in out["records"]:
        by_slot.setdefault(q["slot"], []).append(q)
    assert sorted(by_slot) == [0, 1, 2, 3]
    for qs in by_slot.values():
        assert qs[0]["think_t"] == 0.0
        for prev, nxt in itertools.pairwise(qs):
            assert nxt["arrival_t"] >= prev["completion_t"] + nxt["think_t"] * 0.5


def test_the_last_checkpoint_ends_the_run() -> None:
    out = serve(FakeEngine(), n_ceiling=700)
    assert out["assessment"]["n_at"] <= 700 and out["stopped_because"] in ("converged", "n_ceiling")


def test_a_time_budget_stops_a_run_before_its_first_checkpoint() -> None:
    # two clients on a slow engine reach the first checkpoint (640 completions) only after about 20 s
    out = serve(FakeEngine(per_token_s=0.01), n=2, seconds_budget=0.5)
    assert out["stopped_because"] == "seconds_budget" and not out["converged"]
    assert out["wall_s"] < 3


def test_the_manifests_target_names_reach_the_estimator_as_its_series() -> None:
    assert estimators.series_targets({"throughput": 0.05, "ttft_p95": 0.1}) == {"inv_X": 0.05, "TTFT_p95": 0.1}
    with pytest.raises(ValueError, match="unknown stop targets"):
        estimators.series_targets({"TTFT": 0.1})


def test_the_newest_estimator_judges_the_percentiles_too() -> None:
    out = serve(FakeEngine(), estimator="2026-11", targets={**TARGETS, "ttft_p95": 0.10, "tpot_p95": 0.10},
                n_ceiling=800, guard_s=20)
    assert out["stopped_because"] in ("converged", "n_ceiling") and "TTFT_p95" in str(out["assessment"])


def test_an_engine_that_fails_fails_the_run_instead_of_hanging() -> None:
    with pytest.raises(RuntimeError, match="the engine failed"):
        serve(FailingEngine(), n_ceiling=800, guard_s=10)


def test_an_engine_that_stalls_is_stopped_by_the_time_budget() -> None:
    with pytest.raises(RuntimeError, match="before its first checkpoint"):
        serve(StalledEngine(), seconds_budget=0.5, guard_s=10)
