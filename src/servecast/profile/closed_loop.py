"""The closed workload served on an engine, and the stop rule applied to what it measures.

N clients each send a request from the pool, wait for the whole response, think for an exponential time of mean Z,
and send the next. Every client sends its first request at once. The engine is anything with `generate(request_id,
prompt_ids, max_tokens)`, an async iterator of the number of output tokens so far, and `abort(request_id)`: vLLM's
AsyncLLM on the GPU host (`vllm_engine`), a fake in the tests.

Timing follows the paper's measurements. A request's submit time is taken just before it is handed to the engine, its
first-token time at the first output that holds a token, its completion time when the engine's stream ends. TTFT =
first - submit, e2e = done - submit, and TPOT = (done - first) / (g - 1) for a request of g > 1 output tokens (the
count requested and enforced with ignore_eos), none otherwise.

The run stops by the steady-state estimator's rule, evaluated at the checkpoints the model uses: when the (c + 1)-th
request completes, for each c of the estimator's schedule, the estimator sees the first c + 1 completions. Requests
still running when the rule stops the run are aborted and never enter a window.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from types import ModuleType
from typing import Any, Protocol

import numpy as np

from servecast.core import estimator as estimators

#: no ceiling on completions: the estimator's rule (or a time budget) alone stops a run, as in the paper
UNBOUNDED_CEILING = 2 ** 62


class Engine(Protocol):
    def generate(self, request_id: str, prompt_ids: Sequence[int], max_tokens: int) -> AsyncIterator[int]: ...

    async def abort(self, request_id: str) -> None: ...


def lengths(pool_p: Sequence[int], pool_g: Sequence[int], seed: int) -> Iterator[tuple[int, int]]:
    """The pool's requests in a seeded random order: one permutation after another, each from its own stream."""
    n = len(pool_p)
    for k in range(2 ** 31):
        for i in np.random.default_rng([int(seed), 424242, k]).permutation(n):
            yield int(pool_p[i]), int(pool_g[i])


def think_times(seed: int, slot: int, mean_s: float) -> Iterator[float]:
    """One client's think times, exponential with mean `mean_s`, from the client's own stream."""
    rng = np.random.default_rng([int(seed), 991, int(slot)])
    while True:
        yield float(rng.exponential(mean_s))


def record(rid: int, slot: int, p: int, g: int, think_t: float, submit: float, first: float | None,
           done: float | None, n_out: int) -> dict[str, Any]:
    """One request, in the estimator's fields (seconds from the start of the load point)."""
    ttft = (first - submit) if first is not None else math.inf
    e2e = (done - submit) if done is not None else math.inf
    tpot = ((done - first) / (g - 1)) if (done is not None and first is not None and g > 1) else math.nan
    return {"rid": rid, "slot": slot, "p": p, "g": g, "think_t": think_t, "arrival_t": submit,
            "first_t": first if first is not None else math.inf,
            "completion_t": done if done is not None else math.inf, "ttft": ttft, "tpot": tpot, "e2e": e2e,
            "n_out": n_out}


async def serve(engine: Engine, prompt_ids: Sequence[int], stream: Iterator[tuple[int, int]], n_clients: int,
                think_mean_s: float, estimator: ModuleType, targets: Mapping[str, float], *, seed: int,
                n_ceiling: int = UNBOUNDED_CEILING, checkpoint_ratio: float | None = None,
                seconds_budget: float | None = None, now: Callable[[], float] | None = None,
                progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """One load point, served until the estimator's rule stops it. Returns the records, the last assessment and `t0`,
    the clock's reading when the run started (the records' times count from it). `targets` is the stop rule as the
    manifest names it (`deployment.stop_targets`), translated here as `predict` translates it. The clock is the event
    loop's unless `now` is given: real time on the GPU host, simulated time on the GPU-free mock's loop.
    `seconds_budget`, on that clock, is a timer: it stops a run whose engine stalls as well as a slow one. A failure in
    any client (the engine's, or the estimator's) stops the run and is raised here."""
    ratio = estimator.CHECKPOINT_RATIO if checkpoint_ratio is None else float(checkpoint_ratio)
    ladder = [int(c) for c in estimator.checkpoints(min(estimator.first_checkpoint(), n_ceiling), n_ceiling, ratio)]
    series = estimators.series_targets(targets)
    records: list[dict[str, Any]] = []
    running: dict[str, int] = {}
    state: dict[str, Any] = {"next": 0, "done": 0, "last": None, "stop": asyncio.Event(), "rid": 0, "error": None}
    clock = now if now is not None else asyncio.get_running_loop().time
    t0 = clock()

    def assess_at(c: int) -> None:
        res = estimator.assess(records, c, n_clients, series, n_ceiling=n_ceiling, elapsed_s=clock() - t0,
                               seconds_budget=seconds_budget)
        state["last"] = res
        if progress is not None:
            d = res["decision"]
            progress({"n_at": res["n_at"], "worst": d["worst"], "worst_rel_hw": d["worst_rel_hw"],
                      "target": (d.get("targets") or {}).get(d["worst"]), "stop": bool(d["stop"]),
                      "elapsed_s": clock() - t0})
        if res["decision"]["stop"]:
            state["stop"].set()

    def completed() -> None:
        state["done"] += 1
        while state["next"] < len(ladder) and state["done"] >= ladder[state["next"]] + 1:
            assess_at(ladder[state["next"]])
            state["next"] += 1
            if state["stop"].is_set():
                return
        if state["next"] >= len(ladder):
            # the schedule's last checkpoint is the run's last, as in the model (the estimator's ceiling counts the
            # requests it keeps, which never reach it)
            state["stop"].set()
            return
        if seconds_budget is not None and clock() - t0 >= seconds_budget and not state["stop"].is_set():
            assess_at(state["done"] - 1)
            state["stop"].set()

    async def client(slot: int) -> None:
        thinks = think_times(seed, slot, think_mean_s)
        think_t = 0.0
        while not state["stop"].is_set():
            p, g = next(stream)
            rid = state["rid"]
            state["rid"] += 1
            index = len(records)
            submit = clock() - t0
            records.append(record(rid, slot, p, g, think_t, submit, None, None, 0))
            first = None
            n_out = 0
            running[str(rid)] = index
            async for n_out in engine.generate(str(rid), prompt_ids[:p], g):
                if first is None and n_out > 0:
                    first = clock() - t0
            running.pop(str(rid), None)
            records[index] = record(rid, slot, p, g, think_t, submit, first, clock() - t0, n_out)
            completed()
            if state["stop"].is_set():
                return
            think_t = next(thinks)
            try:
                await asyncio.wait_for(state["stop"].wait(), timeout=think_t)
                return
            except TimeoutError:
                pass

    async def budget(limit: float) -> None:
        while clock() - t0 < limit:
            await asyncio.sleep(max(limit - (clock() - t0), 0.001))
        if not state["stop"].is_set():
            if state["done"] > 0:
                assess_at(state["done"] - 1)
            state["stop"].set()

    async def guarded(work: Awaitable[None]) -> None:
        """Any failure ends the run, and serve raises it, instead of leaving the others waiting forever."""
        try:
            await work
        except Exception as error:
            if state["error"] is None:
                state["error"] = error
            state["stop"].set()

    tasks = [asyncio.create_task(guarded(client(i))) for i in range(int(n_clients))]
    if seconds_budget is not None:
        tasks.append(asyncio.create_task(guarded(budget(seconds_budget))))
    await state["stop"].wait()
    for rid in list(running):
        await engine.abort(rid)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if state["error"] is not None:
        raise state["error"]
    last = state["last"]
    if last is None:
        raise RuntimeError("the run stopped before its first checkpoint: no window to report")
    decision = last["decision"]
    return {"records": records, "assessment": last, "t0": t0, "wall_s": clock() - t0,
            "converged": bool(decision["converged"]),
            "stopped_because": decision["why"] or "n_ceiling"}
