"""The calibration probe of the paper's cost law (d9cbf1d3): the driver that runs the design on an engine, the audit of
what it measured, and the profiling data the calibration is fitted on.

Ported unchanged from the probe behind the paper's calibrations (the aggregation reproduces the shipped profiling
data byte for byte from the research measurements).

THE DRIVER runs the design's tasks (`probe_design.design`) in its order on an engine that offers four calls:
`add(request_id, n_tokens, max_tokens)`, `step()` (one engine step; the (request_id, finished, outputs so far) of
the requests that produced a token), `abort(request_ids)` and `unfinished()`. On the GPU it is vLLM's in-process
engine (`probe_vllm`); in the tests, a scheduler with vLLM's policy priced by a cost law. The engine step timer
records one beat per step: its composition and its duration.

THE AUDIT says, per composition, how many beats were run and kept (a beat is kept when it is the composition its
cell was made for), whether that is enough (`MIN_KEPT`), the drift of the reference revisits (`DRIFT_MAX`), and
whether the clock held. A probe whose audit has flags is written, but no profiling data is made from it.

THE PROFILING DATA. Each composition is a step: its kept beats and the SM clock samples taken during it (or the two
around it). A step's clock is the mean of its samples, snapped to 25 MHz, and unknown when they differ by more than
2 %. Only steps at the reference clock enter: the lock when the clock was locked, else the clock most beats ran at.
Within a step the beats of one composition are sorted by their decode context and cut into up to 10 cells of
consecutive beats; each cell carries its beats' values and their mean duration, and the cells of a step share a
total weight of 1.
"""

from __future__ import annotations

import dataclasses
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from servecast.profile.probe_design import REFERENCE_GEN, n_cells

#: the fewest kept beats a composition needs to enter the profiling data
MIN_KEPT = {"decode": 28, "reference": REFERENCE_GEN - 4, "prefill": 8, "mixed": 3}
DRIFT_MAX = 0.02          # the reference revisits may differ by at most 2 % (mean beat)
CLOCK_TOL = 0.015         # with a lock, every clock sample within 1.5 % of it
CLOCK_EVERY = 10          # beats between two clock samples
CLOCK_SNAP = 25.0         # a step's clock is snapped to this many MHz
CLOCK_SPREAD = 0.02       # a step whose samples differ by more than this has no clock
CELLS_PER_STEP = 10       # the decode-context bins of one composition
#: the engine-step budget of one wait
STEP_LIMIT = 20000


class ProbeError(RuntimeError):
    """The engine did not reach the state a cell needs."""


@dataclasses.dataclass(frozen=True)
class Beat:
    """One engine step: the prefill chunks as [tokens, tokens already computed] (largest first), the decoding
    requests as [context, requests] (by context), its duration and its start on the probe's clock (ms)."""

    prefill: list[list[int]]
    decode: list[list[int]]
    dur_ms: float
    t_ms: float = 0.0


class Engine(Protocol):
    def add(self, request_id: str, n_tokens: int, max_tokens: int) -> None: ...

    def step(self) -> list[tuple[str, bool, int]]: ...

    def abort(self, request_ids: list[str]) -> None: ...

    def unfinished(self) -> bool: ...


def intended(span: Mapping[str, Any], beat: Beat) -> bool:
    """Is this beat the composition its cell was made for?"""
    q = sum(a for a, _ in beat.prefill)
    nd = sum(c for _, c in beat.decode)
    if span["kind"] in ("decode", "reference"):
        return not beat.prefill and nd == span["nd"]
    if span["kind"] == "prefill":
        return len(beat.prefill) == 1 and q == span["q"] and nd == 0
    return len(beat.prefill) == 1 and q == span["q"] and nd == span["nd"]


# the driver ------------------------------------------------------------------------------------------------

def _step_until(eng: Engine, done: Callable[[list[tuple[str, bool, int]]], bool], limit: int = STEP_LIMIT) -> None:
    for _ in range(limit):
        if done(eng.step()):
            return
    raise ProbeError("the engine did not reach the expected state within the step limit")


def _start_all(eng: Engine, ids: list[str], n_tokens: int, max_tokens: int) -> None:
    """Add the requests and step until every one has emitted a token: from then on the batch is whole."""
    for rid in ids:
        eng.add(rid, n_tokens, max_tokens)
    started: set[str] = set()

    def all_started(outs: list[tuple[str, bool, int]]) -> bool:
        started.update(rid for rid, _, n in outs if rid in ids and n >= 1)
        return len(started) == len(ids)
    _step_until(eng, all_started)


def _finish(eng: Engine, ids: list[str]) -> None:
    """Abort the cell's requests and let the engine empty."""
    eng.abort(ids)
    for _ in range(STEP_LIMIT):
        if not eng.unfinished():
            return
        eng.step()
    raise ProbeError("the engine did not empty after an abort")


def _inject(eng: Engine, rid: str, q: int, every: int) -> None:
    """One q-token prompt with one output: step until it is done, then `every - 1` more steps."""
    eng.add(rid, q, 1)
    _step_until(eng, lambda outs: any(r == rid and fin for r, fin, _ in outs))
    for _ in range(every - 1):
        eng.step()


def drive(eng: Engine, d: Mapping[str, Any], beats_len: Callable[[], int], spans: list[dict[str, Any]],
          progress: Callable[[int, int], None] | None = None) -> None:
    """Run the design's tasks in its order on `eng`, appending one span per composition to `spans` as it is
    measured (so a failure keeps every span already measured). `beats_len()` is the number of beats recorded."""
    for done, task in enumerate(d["order"]):
        if progress is not None:
            progress(done, len(d["order"]))
        kind, index = task.split(":")
        i = int(index)
        if kind in ("decode", "reference"):
            c = d["decode"][i] if kind == "decode" else d["reference"]
            gen = d["decode_gen"] if kind == "decode" else c["gen"]
            ids = [f"{task}-{k}" for k in range(c["nd"])]
            _start_all(eng, ids, c["l0"], c["max_tokens"])
            for _ in range(d["decode_skip"]):
                eng.step()
            start = beats_len()
            for _ in range(gen):
                eng.step()
            spans.append({"kind": kind, "task": task, "l0": c["l0"], "nd": c["nd"],
                          **({"visit": i} if kind == "reference" else {}),
                          "beat_from": start, "beat_to": beats_len()})
            _finish(eng, ids)
        elif kind == "prefill":
            q, rep = d["prefill"][i]
            _inject(eng, f"{task}-warm", q, 1)
            start = beats_len()
            for r in range(rep):
                _inject(eng, f"{task}-{r}", q, 1)
            spans.append({"kind": "prefill", "task": task, "q": q, "repeats": rep,
                          "beat_from": start, "beat_to": beats_len()})
        else:
            bg = d["mixed"][i]
            ids = [f"{task}-bg{k}" for k in range(bg["nd"])]
            _start_all(eng, ids, bg["l0"], bg["max_tokens"])
            for q, rep in bg["inject"]:
                _inject(eng, f"{task}-q{q}-warm", q, d["inject_every"])
                start = beats_len()
                for r in range(rep):
                    _inject(eng, f"{task}-q{q}-{r}", q, d["inject_every"])
                spans.append({"kind": "mixed", "task": task, "l0": bg["l0"], "nd": bg["nd"], "q": q,
                              "repeats": rep, "beat_from": start, "beat_to": beats_len()})
            _finish(eng, ids)


# the audit -------------------------------------------------------------------------------------------------

def span_clock(clock: Sequence[Mapping[str, Any]], beat_from: int, beat_to: int) -> list[Mapping[str, Any]]:
    """The clock samples taken during a span or, for a span shorter than the sampling interval, the last one before
    it and the first one after it."""
    inside = [t for t in clock if beat_from <= t.get("beat_index", -1) < beat_to]
    if inside:
        return inside
    before = [t for t in clock if t.get("beat_index", -1) < beat_from]
    after = [t for t in clock if t.get("beat_index", -1) >= beat_to]
    return ([before[-1]] if before else []) + ([after[0]] if after else [])


def audit(beats: Sequence[Beat], spans: Sequence[Mapping[str, Any]], design: Mapping[str, Any],
          clock: Sequence[Mapping[str, Any]], *, lock_mhz: float | None, aborted: str | None = None,
          degraded: bool = False) -> dict[str, Any]:
    """What the probe measured, and the flags that keep it from making profiling data (none: `clean`)."""
    cells = []
    ref_means = []
    for sp in spans:
        run = beats[sp["beat_from"]:sp["beat_to"]]
        kept = [b for b in run if intended(sp, b)]
        need = MIN_KEPT[sp["kind"]]
        cells.append({**{k: v for k, v in sp.items() if k not in ("beat_from", "beat_to")},
                      "beats_run": len(run), "beats_kept": len(kept), "min_kept": need,
                      "complete": len(kept) >= need})
        if sp["kind"] == "reference" and kept:
            ref_means.append(statistics.fmean(b.dur_ms for b in kept))
    designed = n_cells(design)
    measured = [c for c in cells if c["kind"] != "reference"]
    drift = (max(ref_means) / min(ref_means) - 1) if len(ref_means) >= 2 else None
    sm = [float(t["sm_clock_mhz"]) for t in clock if t.get("sm_clock_mhz")]
    clock_ok = bool(sm) and (lock_mhz is None or all(abs(f / lock_mhz - 1) <= CLOCK_TOL for f in sm))
    flags = []
    if aborted:
        flags.append(f"aborted: {aborted}")
    if len(measured) != sum(designed.values()):
        flags.append(f"{len(measured)} compositions measured of {sum(designed.values())} designed")
    if any(not c["complete"] for c in cells):
        flags.append(f"{sum(not c['complete'] for c in cells)} compositions below their minimum of kept beats")
    if drift is None or drift > DRIFT_MAX:
        flags.append(f"reference drift {'unmeasured' if drift is None else f'{100 * drift:.2f} %'} (at most "
                     f"{100 * DRIFT_MAX:.0f} %)")
    if not clock_ok:
        flags.append(f"the clock did not hold within {100 * CLOCK_TOL:.1f} % of {lock_mhz:g} MHz on every sample"
                     if lock_mhz else "no clock sample: without a lock the clock must be measured")
    if degraded:
        flags.append("the composition of some steps could not be read from the scheduler and was guessed")
    return {"compositions_designed": designed, "compositions": len(measured),
            "with_beats": sum(1 for c in measured if c["beats_kept"]),
            "beats_kept": sum(c["beats_kept"] for c in measured), "reference_drift": drift,
            "clock_samples": len(sm), "clock_held": clock_ok, "clean": not flags, "flags": flags,
            "beats_run": len(beats), "beat_gpu_s": round(sum(b.dur_ms for b in beats) / 1000.0, 3), "cells": cells}


# the profiling data ----------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Step:
    """One composition as the profiling data reads it: its kept beats and its clock samples, in MHz."""

    run: str
    beats: Sequence[Beat]
    clock_mhz: Sequence[float]


def steps(beats: Sequence[Beat], spans: Sequence[Mapping[str, Any]],
          clock: Sequence[Mapping[str, Any]]) -> list[Step]:
    """The probe's compositions with at least one kept beat, in the order they ran; the reference revisits measure
    drift and are left out."""
    out = []
    for i, sp in enumerate(spans):
        if sp["kind"] == "reference":
            continue
        kept = [b for b in beats[sp["beat_from"]:sp["beat_to"]] if intended(sp, b)]
        if kept:
            samples = span_clock(clock, sp["beat_from"], sp["beat_to"])
            out.append(Step(f"step{i:03d}", kept, [float(t["sm_clock_mhz"]) for t in samples
                                                   if t.get("sm_clock_mhz")]))
    return out


def step_clock(samples_mhz: Sequence[float]) -> float:
    """The step's clock: the mean of its samples snapped to CLOCK_SNAP, 0 when there is none or they disagree."""
    sm = [float(f) for f in samples_mhz if f]
    if sm and max(sm) > 0 and (max(sm) - min(sm)) / max(sm) <= CLOCK_SPREAD:
        return round(sum(sm) / len(sm) / CLOCK_SNAP) * CLOCK_SNAP
    return 0.0


def reference_clock(records: Sequence[Mapping[str, Any]], declared: float | None) -> float:
    """The clock the profiling data are read at: the step clock nearest the declared one (within half a snap), else
    the clock most beats ran at (the lower on a tie); 0 when there is none."""
    have = Counter(r["f_mhz"] for r in records if r.get("f_mhz"))
    if not have:
        return 0.0
    if declared:
        near = min(have, key=lambda f: abs(f - declared))
        return float(near) if abs(near - declared) <= CLOCK_SNAP / 2 else 0.0
    return float(max(sorted(have), key=lambda f: (have[f], -f)))


def records(all_steps: Sequence[Step]) -> list[dict[str, Any]]:
    """Every kept beat as the cells read it: q, nd, the decode context K, the prefill attention work A, the prefill
    requests n_pf, the duration T (ms), its step and its step's clock."""
    out = []
    for st in all_steps:
        f_mhz = step_clock(st.clock_mhz)
        for b in st.beats:
            q = float(sum(a for a, _ in b.prefill))
            nd = float(sum(c for _, c in b.decode))
            out.append({"run": st.run, "cls": "mixed" if (q and nd) else ("prefill" if q else "decode"), "q": q,
                        "nd": nd, "K": float(sum(kv * c for kv, c in b.decode)),
                        "A": float(sum(a * (kv + a / 2.0) for a, kv in b.prefill)), "n_pf": float(len(b.prefill)),
                        "T": float(b.dur_ms), "f_mhz": f_mhz})
    return out


def build_cells(recs: Sequence[Mapping[str, Any]], n_kbin: int = CELLS_PER_STEP) -> list[dict[str, Any]]:
    """The beats of each (step, nd, q), sorted by K and cut into up to `n_kbin` cells of consecutive beats; the cells
    of a step share a total weight of 1."""
    grp: dict[tuple[str, float, float], list[Mapping[str, Any]]] = defaultdict(list)
    for r in recs:
        grp[(r["run"], r["nd"], r["q"])].append(r)
    cells: list[dict[str, Any]] = []
    for (run, nd, q), rs in grp.items():
        rs = sorted(rs, key=lambda r: r["K"])
        b = min(n_kbin, max(1, len(rs)))
        edges = np.linspace(0, len(rs), b + 1).astype(int)
        for i in range(b):
            sub = rs[edges[i]:edges[i + 1]]
            if not sub:
                continue
            cells.append({"run": run, "q": float(q), "nd": float(nd), "A": [float(s["A"]) for s in sub],
                          "K": [float(s["K"]) for s in sub], "n_pf": [float(s["n_pf"]) for s in sub],
                          "T": float(np.mean([s["T"] for s in sub])), "cls": str(sub[0]["cls"])})
    per_run: dict[str, int] = defaultdict(int)
    for c in cells:
        per_run[c["run"]] += 1
    return [{"q": c["q"], "nd": c["nd"], "A": c["A"], "K": c["K"], "n_pf": c["n_pf"], "T": c["T"],
             "w": 1.0 / per_run[c["run"]], "cls": c["cls"]} for c in cells]


def profiling_document(all_steps: Sequence[Step], *, gpu: str, llm: str, cost_law: str,
                       declared_mhz: float | None) -> dict[str, Any]:
    """The profiling data (`servecast-profiling-data/1`) of a per-LLM law from the probe's compositions."""
    recs = records(all_steps)
    fref = reference_clock(recs, declared_mhz)
    if not fref:
        raise ProbeError("no composition ran at a steady clock" + (f" near the lock ({declared_mhz:g} MHz)"
                                                                  if declared_mhz else ""))
    return {"schema": "servecast-profiling-data/1", "gpu": gpu, "cost_law": cost_law,
            "calibration_procedure": cost_law, "llm": llm, "reference_mhz": float(fref),
            "cells": build_cells([r for r in recs if r["f_mhz"] == fref])}
