"""A GPU-free stand-in for vLLM, for the plumbing of `profile serving`.

It serves in outline as the engine does. A request that arrives while a beat runs waits one more beat before it can
be admitted, as vLLM's asynchronous scheduling makes it and as the population model stages it (when the engine
empties, every waiting arrival is released). Requests are admitted first come first served while the sequence cap and
the KV blocks allow (a request reserves the blocks of its whole context at admission, so nothing is evicted). Each
beat's token budget goes first to one token per decoding request, then to the prompts still to compute, in chunks;
a prompt's first output token comes at the end of the beat that completes it. Each beat lasts what a calibration's
cost law prices its composition at, on the event loop's clock. Run on `VirtualTimeLoop`, that clock is simulated: a
measurement takes as long as the computer needs to compute it, its numbers do not depend on the computer's speed, and
the same manifest measures to the same numbers every time. Its numbers are evidence of nothing about vLLM; only the
plumbing around them is tested.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from servecast.core.deployment import Deployment
from servecast.model.population.v2026_10 import Beat


class MockEngine:
    def __init__(self, beat_seconds: Callable[[Beat], float], d: Deployment) -> None:
        self.beat_seconds, self.d = beat_seconds, d
        self.waiting: deque[dict[str, Any]] = deque()
        self.arriving: list[dict[str, Any]] = []      # arrived during the beat that runs
        self.held: list[dict[str, Any]] = []          # arrived during the beat before
        self.running: list[dict[str, Any]] = []
        self.free_blocks = d.kv_blocks
        self.loop_task: asyncio.Task[None] | None = None
        self.beats = 0
        self.shut_down = False

    def kv_blocks(self) -> int:
        return self.d.kv_blocks

    def shutdown(self) -> None:
        self.shut_down = True

    def _blocks(self, r: dict[str, Any]) -> int:
        return math.ceil((r["p"] + r["g"]) / self.d.block_size)

    async def generate(self, request_id: str, prompt: Sequence[int], max_tokens: int) -> AsyncIterator[int]:
        r: dict[str, Any] = {"rid": request_id, "p": len(prompt), "g": int(max_tokens), "w": 0, "n": 0,
                             "q": asyncio.Queue()}
        if self.loop_task is None or self.loop_task.done():
            self.waiting.append(r)       # an idle engine admits it at its next beat
            self.loop_task = asyncio.get_running_loop().create_task(self._loop())
        else:
            self.arriving.append(r)
        while True:
            n = await r["q"].get()
            if n is None:
                return
            yield n
            if n >= r["g"]:
                return

    async def abort(self, request_id: str) -> None:
        for queue in (self.waiting, self.arriving, self.held, self.running):
            for r in list(queue):
                if r["rid"] == request_id:
                    queue.remove(r)
                    if queue is self.running:
                        self.free_blocks += self._blocks(r)
                    r["q"].put_nowait(None)

    async def _loop(self) -> None:
        while self.waiting or self.running:
            while (self.waiting and len(self.running) < self.d.max_sequences
                   and self._blocks(self.waiting[0]) <= self.free_blocks):
                r = self.waiting.popleft()
                self.free_blocks -= self._blocks(r)
                self.running.append(r)
            budget = self.d.max_batched_tokens
            decoding = [r for r in self.running if r["w"] >= r["p"]][:budget]
            budget -= len(decoding)
            chunks = []
            for r in self.running:
                if r["w"] < r["p"] and budget > 0:
                    q = min(budget, r["p"] - r["w"])
                    chunks.append((r, q))
                    budget -= q
            if not decoding and not chunks:
                return  # nothing fits: the requests wait, and a measurement's time budget ends the run
            contexts: dict[int, int] = {}
            for r in decoding:
                contexts[r["p"] + r["n"]] = contexts.get(r["p"] + r["n"], 0) + 1
            beat = Beat(prefill=tuple(sorted(((q, r["w"]) for r, q in chunks), reverse=True)),
                        decode=tuple(sorted(contexts.items())))
            await asyncio.sleep(self.beat_seconds(beat))
            self.beats += 1
            for r in decoding:
                r["n"] += 1
                r["q"].put_nowait(r["n"])
            for r, q in chunks:
                r["w"] += q
                if r["w"] >= r["p"]:
                    r["n"] = 1
                    r["q"].put_nowait(1)
            for r in [r for r in self.running if r["n"] >= r["g"]]:
                self.running.remove(r)
                self.free_blocks += self._blocks(r)
            # the arrivals held one beat become admissible; when the engine empties, every arrival is released
            self.waiting.extend(self.held)
            if self.running:
                self.held, self.arriving = self.arriving, []
            else:
                self.waiting.extend(self.arriving)
                self.held, self.arriving = [], []


class FixedClock:
    """A clock sampler that reads `mhz` at every sample, on the run's own clock."""

    def __init__(self, mhz: float) -> None:
        self.mhz = mhz

    def __call__(self, now: Callable[[], float]) -> FixedClock:
        return self

    def __enter__(self) -> FixedClock:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def window(self, t0: float, t1: float) -> dict[str, Any]:
        return {"mean_mhz": self.mhz, "min_mhz": self.mhz, "max_mhz": self.mhz, "n": max(1, int(t1 - t0))}


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock moves only by the waits it is asked for: whenever nothing is ready to run, it jumps to
    the next timer instead of waiting for it. Engine steps and think times then take their simulated duration, and the
    computer's own work takes none. (It reads the loop's queue of timers, a private part of asyncio that has not changed
    since Python 3.4.)"""

    def __init__(self) -> None:
        super().__init__()
        self._simulated = 0.0

    def time(self) -> float:
        return self._simulated

    def _run_once(self) -> None:
        if not self._ready and self._scheduled:  # type: ignore[attr-defined]
            self._simulated = max(self._simulated, self._scheduled[0].when())  # type: ignore[attr-defined]
        super()._run_once()  # type: ignore[misc]
