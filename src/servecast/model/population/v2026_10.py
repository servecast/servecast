"""Population model 2026-10: the population model with eviction. FROZEN: never edit; a change is a new id.

The engine serves iteration by iteration (a "beat"). The state between two beats is a population census:

    states   the clients thinking; the requests staged (arrived during the current or the previous beat); the FIFO of
             requests waiting for admission, with at most one BLOCKED candidate at its head; the prefill head (the one
             request whose prompt is only partly computed, with its progress w); the decode census, the requests
             generating, each in a phase j of its output-length chain and with its exact context.
    jumps    one synchronous beat jump: evictions until the beat's growth fits the KV capacity, then admissions
             (budget first, then memory), then the beat runs for T(composition) seconds, during which thinking
             clients submit; at its end every decoding request emits a token and moves phase (or finishes), the
             prefill head advances, and the staged arrivals become admissible one beat later.
    rates    the beat's duration T is the cost law's, a function of the composition; a thinking client submits after
             an exponential think time of mean Z.

Eviction follows the engine: when the beat's KV growth does not fit, the prefill head is evicted first (its progress
restarts), otherwise a decoding request from the YOUNGEST occupied admission cohort, drawn by counts. A victim goes
back to the head of the FIFO with its output so far, and must recompute its whole context before emitting again.

Random streams are separate, so a change in one decision never shifts the draws of another: the dynamics (think
times, arrivals, phase moves), the bookkeeping (which client submits, arrival instants), which members of a phase
move, the victims, and (through `SeededTypes`) the arriving types.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

from servecast.model.workload.v2026_10 import BLOCK_SIZE, JointLaw, SeededTypes, sigma_blk

#: admission cohorts kept distinct for the victim's draw; older ones are pooled
N_RECENT = 128
#: the census invariants are checked every this many beats
INVARIANT_EVERY = 1000
#: a run with no ceiling stops only on the estimator's precision rule; this only closes the checkpoint ladder
UNBOUNDED_CEILING = 10 ** 12


class Beat(NamedTuple):
    """A beat's composition: prefill chunks (tokens, tokens already computed), descending; decode classes
    (context, count), ascending by context."""

    prefill: tuple[tuple[int, int], ...]
    decode: tuple[tuple[int, int], ...]


Clock = Callable[[Beat], float]


@dataclasses.dataclass(frozen=True)
class Config:
    capacity_c: int                 # usable KV capacity in tokens (the null block already removed)
    n_max: int                      # token budget per beat (max_num_batched_tokens)
    b_max: int | None = None        # most requests resident at once (max_num_seqs); None = no limit
    block_size: int = BLOCK_SIZE
    n_recent: int = N_RECENT

    def __post_init__(self) -> None:
        if self.capacity_c < 1 or self.n_max < 1 or self.block_size < 1:
            raise ValueError("Config: capacity_c, n_max and block_size must be >= 1")
        if self.b_max is not None and self.b_max < 1:
            raise ValueError("Config: b_max must be >= 1 or None")
        if self.n_recent < 0:
            raise ValueError("Config: n_recent must be >= 0")

    @property
    def sigma_blk(self) -> float:
        return sigma_blk(self.block_size)

    @property
    def n_cohorts(self) -> int:
        """n_recent + 2: cohort 0 is the current beat's admissions (never a victim), the last one pools the rest."""
        return self.n_recent + 2


def config(blocks: int, block_size: int, max_batched_tokens: int, max_sequences: int | None) -> Config:
    """The model's configuration from the engine's: the KV capacity is the block count less the engine's null
    block, in tokens."""
    return Config(capacity_c=int(blocks) * int(block_size) - int(block_size), n_max=int(max_batched_tokens),
                  b_max=None if max_sequences is None else int(max_sequences), block_size=int(block_size))


@dataclasses.dataclass(frozen=True)
class ReplayMark:
    """What a victim carries while it waits and rebuilds: its original prompt a, frozen phase j, tokens generated k,
    and x = a + k, the context to recompute."""

    a: int
    j: int
    k: int
    x: int


@dataclasses.dataclass(slots=True)
class Tag:
    """A request's passive record: its type (p, j), its instants and its own context."""

    rid: int
    p: int
    j: int
    arrival_t: float
    think_t: float
    w: int = 0
    k: int = 0
    first_token_t: float = math.nan
    completion_t: float = math.nan
    itl_sum: float = 0.0
    n_dec: int = 0
    r: int = 0
    last_tok_t: float = math.nan
    returning: bool = False
    n_preempt: int = 0
    recompute_interval_sum: float = 0.0
    mark: ReplayMark | None = None

    @property
    def adm_len(self) -> int:
        """The tokens admission must grant: the prompt, or the whole context to rebuild after an eviction."""
        return self.mark.x if self.mark is not None else self.p

    @property
    def k_gen(self) -> int:
        """Tokens generated: the first token enters with the prompt, so it is n_dec + 1."""
        return self.n_dec + 1

    def record(self) -> dict[str, Any]:
        done = not math.isnan(self.completion_t)
        return {"rid": self.rid, "p": self.p, "g": self.n_dec + 1 if done else -1, "j_entry": self.j,
                "arrival_t": self.arrival_t, "think_t": self.think_t,
                "ttft": (self.first_token_t - self.arrival_t) if not math.isnan(self.first_token_t) else math.nan,
                "tpot": (self.itl_sum / self.n_dec) if self.n_dec > 0 else math.nan, "itl_n": self.n_dec,
                "completion_t": self.completion_t, "e2e": (self.completion_t - self.arrival_t) if done else math.nan,
                "n_preempt": self.n_preempt}


@dataclasses.dataclass(slots=True)
class Head:
    tag: Tag
    w: int

    @property
    def p(self) -> int:
        return self.tag.adm_len

    @property
    def j(self) -> int:
        return self.tag.mark.j if self.tag.mark is not None else self.tag.j


class State:
    """The state between two beats; derived views are properties."""

    def __init__(self, n_phases: int, n_users: int, t: float = 0.0) -> None:
        self.L = int(n_phases)
        self.N = np.zeros(self.L + 1, dtype=np.int64)
        self.M = np.zeros(self.L + 1, dtype=float)
        self.phase_tags: dict[int, list[Tag]] = {}
        self.head: Head | None = None
        self.blocked: Tag | None = None
        self.fifo: deque[Tag] = deque()
        self.S: list[Tag] = []
        self.S_hat: list[Tag] = []
        self.thinkers: list[float] = [0.0] * int(n_users)
        self.n_in_beat = 0
        self.t = float(t)
        self.n_users = int(n_users)

    @property
    def N_D(self) -> int:  # noqa: N802
        return int(np.add.reduce(self.N))

    @property
    def K(self) -> float:  # noqa: N802
        return float(np.add.reduce(self.M))

    @property
    def h(self) -> int:
        return 1 if self.head is not None else 0

    @property
    def b(self) -> int:
        return 1 if self.blocked is not None else 0

    def occ(self, cfg: Config) -> float:
        """KV occupancy: K + N_D sigma + h (w_H + sigma)."""
        s = cfg.sigma_blk
        return self.K + self.N_D * s + (self.head.w + s if self.head is not None else 0.0)

    @property
    def busy(self) -> bool:
        return self.N_D > 0 or self.head is not None or self.blocked is not None or bool(self.fifo)

    def population(self) -> int:
        return (len(self.thinkers) + len(self.S) + len(self.S_hat) + len(self.fifo) + self.b + self.h + self.N_D
                + self.n_in_beat)

    def check_invariants(self, cfg: Config) -> None:
        pop = self.population()
        if pop != self.n_users:
            raise RuntimeError(f"population lost: {pop} of {self.n_users}")
        if (self.N < 0).any() or (self.M < -1e-9).any():
            raise RuntimeError("negative census")
        occ = np.flatnonzero(self.N)
        if ((self.N == 0) & (np.abs(self.M) >= 1e-9)).any() or ((self.N > 0) & (np.abs(self.M) < 1e-9)).any():
            raise RuntimeError("M_j == 0 <=> N_j == 0 violated")
        if set(int(j) for j in occ) != set(self.phase_tags) or any(
                len(self.phase_tags[int(j)]) != int(self.N[j]) for j in occ):
            raise RuntimeError("the tags per phase do not count N_j")
        if self.head is not None and not (1 <= self.head.w < self.head.p):
            raise RuntimeError(f"prefill head with progress {self.head.w} outside [1, p={self.head.p})")
        if self.N_D > cfg.n_max:
            raise RuntimeError(f"N_D={self.N_D} > n_max={cfg.n_max}")
        if cfg.b_max is not None and self.N_D + self.h > cfg.b_max:
            raise RuntimeError(f"N_D + h = {self.N_D + self.h} > b_max={cfg.b_max}")


# the victim's law ------------------------------------------------------------------------------------------------

Cell = tuple[int, int, int, int]        # (cohort r, original prompt a, phase j, tokens generated k)


def _cell_of(tag: Tag, j: int) -> Cell:
    return (tag.r, tag.p, j, tag.k_gen)


def _draw_victim_cell(phase_tags: Mapping[int, list[Tag]], rng: np.random.Generator) -> Cell:
    """The victim's cell, from the youngest occupied cohort, with probability proportional to the cell's count."""
    r_star: int | None = None
    for tags in phase_tags.values():
        for tg in tags:
            if r_star is None or tg.r < r_star:
                r_star = tg.r
    if r_star is None:
        raise ValueError("no resident request to evict")
    n: dict[Cell, int] = defaultdict(int)
    for j, tags in phase_tags.items():
        for tg in tags:
            if tg.r == r_star:
                n[_cell_of(tg, j)] += 1
    cells = sorted(c for c in n if c[0] == r_star)
    counts = np.array([n[c] for c in cells], dtype=float)
    return cells[int(rng.choice(len(cells), p=counts / counts.sum()))]


def _member_of(phase_tags: Mapping[int, list[Tag]], cell: Cell) -> Tag:
    """A member of the cell; members are exchangeable, so the first one is taken (no random draw)."""
    r, a, j, k = cell
    for tg in phase_tags.get(j, ()):
        if tg.r == r and tg.p == a and tg.k_gen == k:
            return tg
    raise KeyError(f"cell {cell} not populated: census and tags out of sync")


def _age_residents(phase_tags: Mapping[int, list[Tag]], head: Head | None, n_cohorts: int) -> None:
    """Once per beat, at its end: every resident moves to the next cohort; the oldest tracked one merges into the
    pooled cohort."""
    top = n_cohorts - 1
    for tags in phase_tags.values():
        for tg in tags:
            if tg.r < top:
                tg.r += 1
    if head is not None:
        head.tag.r = min(head.tag.r + 1, n_cohorts - 1)


# one beat ----------------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Growth:
    """The beat's demand, from the epoch state: the head's grant, the residents' growth, the budget left, and the
    KV room left after the growth (negative: a deficit, which triggers eviction)."""

    q_h: int
    delta: int
    r0: int
    free_rem: float


@dataclasses.dataclass
class Admission:
    q_h: int
    delta: int
    r0: int
    free_rem: float
    admitted: list[tuple[Tag, int]]
    seq_cap_hit: bool = False

    @property
    def n_adm(self) -> int:
        return len(self.admitted)


def _growth(st: State, cfg: Config) -> Growth:
    n_d = st.N_D
    if n_d > cfg.n_max:
        raise RuntimeError(f"N_D={n_d} > n_max={cfg.n_max}")
    head = st.head
    q_h = min(head.p - head.w, cfg.n_max - n_d) if head is not None else 0
    return Growth(q_h=q_h, delta=n_d + q_h, r0=cfg.n_max - n_d - q_h,
                  free_rem=cfg.capacity_c - st.occ(cfg) - (n_d + q_h))


def _admit(st: State, cfg: Config, g: Growth) -> Admission:
    """Candidates in order: the blocked one, then the FIFO's head. Budget first (P_{i-1} < r0), then memory
    (P_i + i sigma <= free_rem; a refusal makes the candidate the blocked one). Only the last admitted may receive less
    than its prompt. The sequence limit caps residents."""
    n_d = st.N_D
    sigma = cfg.sigma_blk
    out = Admission(q_h=g.q_h, delta=g.delta, r0=g.r0, free_rem=g.free_rem, admitted=[])
    p_cum = 0
    i = 0
    while True:
        cand = st.blocked if st.blocked is not None else (st.fifo[0] if st.fifo else None)
        if cand is None or not (p_cum < g.r0):
            break
        if cfg.b_max is not None and n_d + st.h + len(out.admitted) >= cfg.b_max:
            out.seq_cap_hit = True
            break
        i += 1
        p_i = p_cum + cand.adm_len
        if p_i + i * sigma > g.free_rem:
            if cand is not st.blocked:
                st.fifo.popleft()
                st.blocked = cand
            break
        grant = min(cand.adm_len, g.r0 - p_cum)
        if cand is st.blocked:
            st.blocked = None
        else:
            st.fifo.popleft()
        cand.r = 0
        out.admitted.append((cand, grant))
        p_cum = p_i
        if grant < cand.adm_len:
            break
    st.n_in_beat = out.n_adm
    return out


def _requeue(st: State, tg: Tag) -> None:
    """A victim goes back to the head of the FIFO; a blocked candidate moves behind it."""
    if st.blocked is not None:
        st.fifo.appendleft(st.blocked)
        st.blocked = None
    st.fifo.appendleft(tg)


def _evict_to_fit(st: State, cfg: Config, rng_evict: np.random.Generator) -> int:
    """Evict one request at a time until the beat's growth fits, recomputing the demand after each. Returns the
    number of victims."""
    n_evict = 0
    while True:
        g = _growth(st, cfg)
        if g.free_rem >= 0:
            return n_evict
        if st.head is not None:
            tg = st.head.tag
            tg.n_preempt += 1
            tg.returning = True
            st.head = None
            _requeue(st, tg)
        elif st.N_D > 0:
            cell = _draw_victim_cell(st.phase_tags, rng_evict)
            tg = _member_of(st.phase_tags, cell)
            _r, a, j, k = cell
            x = a + k
            st.phase_tags[j].remove(tg)
            if not st.phase_tags[j]:
                del st.phase_tags[j]
            st.N[j] -= 1
            st.M[j] -= float(x)
            tg.mark = ReplayMark(a=a, j=j, k=k, x=x)
            tg.returning = True
            tg.n_preempt += 1
            _requeue(st, tg)
        else:
            raise RuntimeError(f"a deficit of {max(0.0, -g.free_rem):.1f} tokens with nothing to evict: a KV "
                               f"capacity of {cfg.capacity_c} tokens cannot hold one prompt and its growth")
        n_evict += 1


def _decode_classes(n_by_phase: np.ndarray, m_by_phase: np.ndarray) -> tuple[tuple[int, int], ...]:
    """The decode classes (context, count), the exact integer split of M_j over N_j rows, ascending by context."""
    out: dict[int, int] = {}
    nz = n_by_phase.nonzero()[0]
    for n, total in zip(n_by_phase[nz].tolist(), m_by_phase[nz].tolist()):
        base = int(total // n)
        r = int(round(total - base * n))
        r = min(max(r, 0), n)
        if r == n:
            base, r = base + 1, 0
        if n - r > 0:
            out[base] = out.get(base, 0) + (n - r)
        if r > 0:
            out[base + 1] = out.get(base + 1, 0) + r
    return tuple(sorted(out.items()))


def _composition(head_grant: int, head_w: int, admitted_grants: Sequence[int], n_by_phase: np.ndarray,
                 m_by_phase: np.ndarray) -> Beat:
    pf = []
    if head_grant >= 1:
        pf.append((int(head_grant), int(head_w)))
    pf.extend((int(q), 0) for q in admitted_grants if q >= 1)
    return Beat(prefill=tuple(sorted(pf, reverse=True)), decode=_decode_classes(n_by_phase, m_by_phase))


def _decode_step(st: State, adm: Admission, dt: float, now: float, rng: np.random.Generator,
                 rng_context: np.random.Generator, law: Any, complete: Callable[[Tag, float], None],
                 check_context: bool) -> None:
    """The end of the beat: the prefill head and the admitted requests that finish their prompt emit their first
    token and enter decoding; every decoding request emits a token, then moves phase or finishes."""
    gen = law.gen
    head = st.head
    entrants: list[Tag] = []
    new_head: Head | None = None
    if head is not None:
        head.tag.w = head.w + adm.q_h
        if head.w + adm.q_h >= head.p:
            entrants.append(head.tag)
            st.head = None
        else:
            head.w += adm.q_h
    for tg, grant in adm.admitted:
        tg.w = grant
        if grant >= tg.adm_len:
            entrants.append(tg)
        else:
            new_head = Head(tg, grant)
    if new_head is not None:
        if st.head is not None:
            raise RuntimeError("two prefill heads in one beat")
        st.head = new_head
    st.n_in_beat = 0

    n_new = np.zeros(st.L + 1, dtype=np.int64)
    m_new = np.zeros(st.L + 1, dtype=float)
    tags_new: dict[int, list[Tag]] = {}
    for j in (int(v) for v in np.flatnonzero(st.N)):
        n = int(st.N[j])
        ids = st.phase_tags[j]
        for tg in ids:
            tg.itl_sum += dt
            tg.n_dec += 1
            tg.k += 1
            tg.last_tok_t = now
        counts = rng.multinomial(n, gen.phase_probs[j])
        order = [ids[int(i)] for i in rng_context.permutation(n)] if n > 1 else list(ids)
        pos = int(counts[0])
        for tg in order[:pos]:
            complete(tg, now)
        for jp in (int(v) for v in np.flatnonzero(counts[1:]) + 1):
            c = int(counts[jp])
            n_new[jp] += c
            m_new[jp] += sum(tg.p + 1 + tg.n_dec for tg in order[pos:pos + c])
            tags_new.setdefault(jp, []).extend(order[pos:pos + c])
            pos += c
    for tg in entrants:
        if tg.mark is not None:
            # back from recomputing an evicted context: one token, and the whole stall since the last token is ONE
            # inter-token interval (what the client measures); the TTFT was paid the first time
            mk = tg.mark
            stall = now - tg.last_tok_t
            tg.itl_sum += stall
            tg.recompute_interval_sum += stall
            tg.n_dec += 1
            tg.k += 1
            tg.last_tok_t = now
            tg.mark, tg.returning = None, False
            n_new[mk.j] += 1
            m_new[mk.j] += float(mk.x + 1)
            tags_new.setdefault(mk.j, []).append(tg)
            continue
        tg.first_token_t = now
        tg.last_tok_t = now
        tg.returning = False
        if tg.j == 0:
            complete(tg, now)
        else:
            n_new[tg.j] += 1
            m_new[tg.j] += tg.p + 1.0
            tags_new.setdefault(tg.j, []).append(tg)
    st.N, st.M, st.phase_tags = n_new, m_new, tags_new
    if check_context:
        for j, members in tags_new.items():
            if st.M[j] != sum(tg.p + 1 + tg.n_dec for tg in members):
                raise RuntimeError("the decode census lost a member's context")


def _stage(st: State) -> None:
    """Arrivals held one beat become admissible; when the engine empties, everything is released."""
    if st.N_D > 0 or st.head is not None:
        st.fifo.extend(st.S_hat)
        st.S_hat, st.S = st.S, []
    else:
        st.fifo.extend(st.S_hat)
        st.fifo.extend(st.S)
        st.S_hat, st.S = [], []


# the closed loop ---------------------------------------------------------------------------------------------------

def run_closed(cfg: Config, law: Any, n_users: int, think_z: float, rng: np.random.Generator, *, clock: Clock,
               n_complete: int, rng_tags: np.random.Generator, rng_context: np.random.Generator,
               rng_evict: np.random.Generator, checkpoints: Sequence[int] = (),
               on_checkpoint: Callable[[int, list[dict[str, Any]]], bool] | None = None,
               invariant_every: int = INVARIANT_EVERY) -> dict[str, Any]:
    """`n_users` clients with exponential think time of mean `think_z` > 0, until `n_complete` completions or until
    `on_checkpoint` (called at each checkpoint with every request's record) returns True."""
    if int(n_users) != n_users or n_users < 1:
        raise ValueError(f"n_users={n_users} must be a positive integer")
    if not (think_z > 0.0) or math.isinf(think_z):
        raise ValueError(f"think_z={think_z} must be finite and > 0")
    if n_complete < 1:
        raise ValueError("n_complete must be >= 1")
    n_users = int(n_users)
    gen = law.gen
    st = State(gen.L, n_users)
    recs: list[Tag] = []
    next_rid = 0
    n_done = 0
    n_beats = 0
    n_evicted = 0
    seq_cap_blocked = 0

    def new_tag(p: int, j: int, arrival_t: float, think_t: float) -> Tag:
        nonlocal next_rid
        tg = Tag(rid=next_rid, p=int(p), j=int(j), arrival_t=arrival_t, think_t=think_t)
        next_rid += 1
        recs.append(tg)
        return tg

    def submit_from_think(epochs: list[float], types: list[tuple[int, int]]) -> list[Tag]:
        k = len(epochs)
        idx = [int(i) for i in rng_tags.choice(len(st.thinkers), size=k, replace=False)]
        starts = [st.thinkers[i] for i in idx]
        for i in sorted(idx, reverse=True):
            st.thinkers.pop(i)
        return [new_tag(p, j, ep, ep - start) for start, ep, (p, j) in zip(starts, sorted(epochs), types)]

    def complete(tg: Tag, now: float) -> None:
        nonlocal n_done
        tg.completion_t = now
        n_done += 1
        st.thinkers.append(now)

    pending = sorted(int(c) for c in checkpoints)
    while n_done < n_complete:
        check_now = n_beats % invariant_every == 0
        if check_now:
            st.check_invariants(cfg)

        if not st.busy:
            if st.S or st.S_hat:
                raise RuntimeError("staged requests with an empty engine")
            if not st.thinkers:
                raise RuntimeError("population lost: nobody thinks and the engine is empty")
            dt = float(rng.exponential(think_z / len(st.thinkers)))
            st.t += dt
            st.fifo.extend(submit_from_think([st.t], law.sample_types(rng, 1)))
            continue

        g = _growth(st, cfg)
        n_evict = _evict_to_fit(st, cfg, rng_evict)
        if n_evict:
            # a beat with a victim admits nobody; the head keeps growing, recomputed after the evictions
            n_evicted += n_evict
            g = _growth(st, cfg)
            adm = Admission(q_h=g.q_h, delta=g.delta, r0=g.r0, free_rem=g.free_rem, admitted=[])
            st.n_in_beat = 0
        else:
            adm = _admit(st, cfg, g)
        if adm.seq_cap_hit:
            seq_cap_blocked += 1
        beat = _composition(adm.q_h, st.head.w if st.head is not None else 0, [q for _, q in adm.admitted], st.N,
                            st.M)
        if not beat.prefill and not beat.decode:
            raise RuntimeError("an empty beat with a busy engine: an admission defect")
        dt = float(clock(beat))
        if not (dt > 0.0):
            raise RuntimeError(f"a non-positive beat: {dt}")
        n_beats += 1

        if st.thinkers:
            p_fire = -math.expm1(-dt / think_z)
            k_fire = int(rng.binomial(len(st.thinkers), p_fire))
            if k_fire:
                types = law.sample_types(rng, k_fire)
                u = rng_tags.random(k_fire)
                eps = -think_z * np.log1p(-u * p_fire)
                st.S.extend(submit_from_think([st.t + float(e) for e in eps], types))
        st.t += dt

        _decode_step(st, adm, dt, st.t, rng, rng_context, law, complete, check_now)
        _age_residents(st.phase_tags, st.head, cfg.n_cohorts)
        _stage(st)

        if pending and n_done >= pending[0]:
            n_at = pending[0]
            while pending and n_done >= pending[0]:
                n_at = pending.pop(0)
            if on_checkpoint is not None and on_checkpoint(n_at, [tg.record() for tg in recs]):
                break

    return {"requests": [tg.record() for tg in recs], "n_beats": n_beats, "n_evicted": n_evicted,
            "seq_cap_blocked": seq_cap_blocked, "t_end": st.t}


def run_cell(cfg: Config, law: JointLaw, n_users: int, think_z: float, clock: Clock, estimator: ModuleType,
             targets: Mapping[str, float], *, seed: int, rep: int = 0, n_ceiling: int = UNBOUNDED_CEILING,
             checkpoint_ratio: float | None = None, seconds_budget: float | None = None,
             progress: Callable[[dict[str, Any]], None] | None = None,
             keep_requests: bool = False) -> dict[str, Any]:
    """One load point (N clients, mean think time Z), run until the estimator's rule stops it.

    Seeding: the run's streams are numpy generators seeded with [seed, N, round(1e6 Z), rep] plus the stream's
    index: 1 the arriving types, 2 the dynamics, 3 the bookkeeping, 4 who moves within a phase, 5 the victims."""
    n_users, z = int(n_users), float(think_z)
    seed_list = [int(seed), n_users, int(1e6 * z), int(rep)]
    t_start = time.time()
    ratio = estimator.CHECKPOINT_RATIO if checkpoint_ratio is None else float(checkpoint_ratio)
    ladder = [int(c) for c in estimator.checkpoints(min(estimator.first_checkpoint(), n_ceiling), n_ceiling, ratio)]
    last: dict[str, Any] = {}

    def at_checkpoint(n_reached: int, records: list[dict[str, Any]]) -> bool:
        res = estimator.assess(records, int(n_reached) - 1, n_users, targets, n_ceiling=n_ceiling,
                               elapsed_s=time.time() - t_start, seconds_budget=seconds_budget)
        last.clear()
        last.update(res)
        if progress is not None:
            d = res["decision"]
            progress({"n_at": res["n_at"], "worst": d["worst"], "worst_rel_hw": d["worst_rel_hw"],
                      "stop": bool(d["stop"]), "elapsed_s": time.time() - t_start})
        return bool(res["decision"]["stop"])

    r = run_closed(cfg, SeededTypes(law, seed_list + [1]), n_users, z, np.random.default_rng(seed_list + [2]),
                   clock=clock, n_complete=int(ladder[-1]) + 1, rng_tags=np.random.default_rng(seed_list + [3]),
                   rng_context=np.random.default_rng(seed_list + [4]),
                   rng_evict=np.random.default_rng(seed_list + [5]), checkpoints=[c + 1 for c in ladder],
                   on_checkpoint=at_checkpoint)
    if not last:
        raise RuntimeError("no checkpoint was reached: the run produced no window")
    kept, w, d = last["kept"], last["window"], last["decision"]
    ttfts = [float(q["ttft"]) for q in kept if np.isfinite(q["ttft"])]
    tpots = [float(q["tpot"]) for q in kept if np.isfinite(q["tpot"])]
    span = w["span"]
    out_tokens = sum(int(q["g"]) for q in kept)
    preemptions = sum(int(q["n_preempt"]) for q in kept)
    return {"seed": seed_list, "throughput_rps": w["x"], "r_mean": w["r_mean"], "z_mean": w["z_mean"],
            "output_tokens_per_s": (out_tokens / span) if span and np.isfinite(span) else math.nan,
            "span_s": span,
            "ttft_mean": float(np.mean(ttfts)) if ttfts else math.nan,
            "tpot_mean": float(np.mean(tpots)) if tpots else math.nan,
            "ttft": ttfts, "tpot": tpots, "n_kept": w["n_kept"], "warmup_trunc": last["trunc"],
            "n_prefix": int(last["n_at"]), "little_res": w["little_res"], "converged": bool(d["converged"]),
            "stopped_because": d["why"] or "n_ceiling", "per_metric": d["per_metric"], "worst": d["worst"] or "",
            "worst_rel_hw": d["worst_rel_hw"], "n_beats": r["n_beats"],
            # evictions of the requests completed in the window, per completed request: each eviction counts once
            "preemptions_per_request": (preemptions / len(kept)) if kept else math.nan,
            "requests": ([{k: q[k] for k in ("rid", "p", "g", "arrival_t", "ttft", "tpot", "e2e", "completion_t",
                                             "n_preempt")} for q in kept] if keep_requests else None),
            "wall_s": time.time() - t_start}
