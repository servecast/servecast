"""A GPU-free stand-in for vLLM under the calibration probe, ported unchanged from the probe behind the paper.

It schedules as vLLM's engine does: running requests first (a decoding one takes one budget token, a partly prefilled
one its next chunk), then waiting ones admitted first come first served with the budget left; a request whose prompt
completes emits its first token in that step. A request added is schedulable from the second step after (the
one-step admission lag of asynchronous scheduling). Every step is priced by `price(composition) -> seconds`, with
a 0.5 % seeded jitter, recorded as a beat, and the SM clock is sampled every CLOCK_EVERY beats.

Its numbers are evidence of nothing about vLLM: it tests the driver, the audit and the profiling data around them,
and, priced by a calibrated cost law, that a fit of its profiling data gives that law's parameters back.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from servecast.model.population.v2026_10 import Beat as Composition
from servecast.profile.probe import CLOCK_EVERY, Beat


class SimEngine:
    def __init__(self, price: Callable[[Any], float], rng: np.random.Generator, budget: int, max_seqs: int,
                 clock_mhz: float, lag: int = 1) -> None:
        self.price, self.rng, self.budget, self.max_seqs, self.clock_mhz, self.lag = \
            price, rng, budget, max_seqs, clock_mhz, lag
        self.req: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []          # admission order (first come, first served)
        self.beats: list[Beat] = []
        self.clock: list[dict[str, Any]] = []
        self.t = 0
        self.shut_down = False

    def add(self, rid: str, n_tokens: int, max_tokens: int) -> None:
        self.req[rid] = {"p": n_tokens, "done": 0, "out": 0, "max": max_tokens, "ready": self.t + self.lag,
                         "running": False}
        self.order.append(rid)

    def abort(self, ids: list[str]) -> None:
        for rid in ids:
            self.req.pop(rid, None)
        self.order = [r for r in self.order if r in self.req]

    def unfinished(self) -> bool:
        return bool(self.req)

    def step(self) -> list[tuple[str, bool, int]]:
        self.t += 1
        budget, prefill, ctx, outs = self.budget, [], {}, []
        running = [r for r in self.order if self.req[r]["running"]]
        waiting = [r for r in self.order if not self.req[r]["running"] and self.req[r]["ready"] <= self.t]
        for rid in running + waiting:
            if budget <= 0:
                break
            r = self.req[rid]
            if not r["running"]:
                if len([x for x in self.req.values() if x["running"]]) >= self.max_seqs:
                    break
                r["running"] = True
            if r["done"] < r["p"]:
                chunk = min(r["p"] - r["done"], budget)
                prefill.append((chunk, r["done"]))
                r["done"] += chunk
                budget -= chunk
                if r["done"] == r["p"]:
                    r["out"] += 1
                    outs.append([rid, r["out"] >= r["max"], r["out"]])
            else:
                c = r["p"] + r["out"]
                ctx[c] = ctx.get(c, 0) + 1
                budget -= 1
                r["out"] += 1
                outs.append([rid, r["out"] >= r["max"], r["out"]])
        for rid, fin, _ in outs:
            if fin:
                self.req.pop(rid, None)
        self.order = [r for r in self.order if r in self.req]
        if prefill or ctx:
            comp = Composition(prefill=tuple(prefill), decode=tuple(sorted(ctx.items())))
            self.beats.append(Beat(prefill=[list(p) for p in prefill], decode=[list(x) for x in sorted(ctx.items())],
                                   dur_ms=max(1e-3, self.price(comp) * 1e3 * float(self.rng.normal(1.0, 0.005)))))
            if len(self.beats) % CLOCK_EVERY == 1:
                self.clock.append({"sm_clock_mhz": self.clock_mhz, "beat_index": len(self.beats) - 1})
        return [(o[0], o[1], o[2]) for o in outs]

    def shutdown(self) -> None:
        self.shut_down = True
