"""Workload law and fitter 2026-10: prompt classes, generation phases, their joint law, and how a trace becomes one.
FROZEN: never edit; a change is a new id.

The population model never draws a request's lengths; it draws its TYPE (a, j): a prompt class a with a deterministic
length p_a, and the phase j it enters decoding in. The output length then emerges from the phase moves: one decode
step from phase j ends the request with probability omega_j or moves it to phase k with probability Omega[j, k].

* `PromptLaw`: l_p prompt lengths p_a (never a multiple of the KV block size: lengths on the block lattice would make
  the block-slack term exact where the engine's is not) with weights rho_a.
* `GenLaw`: the phase chain over D = g - 1 (g the output length): Omega sub-stochastic on rows 1..L, exit
  omega = 1 - Omega 1. Branches of geometric stages in series (a hyper-Erlang): branch b has k_b stages, each kept
  with probability s_b = 1 - k_b / mu_b.
* `JointLaw`: the joint pi[a, j] over types, j = 0 being the atom g = 1. Its rows sum to rho, its columns are the
  entry law alpha.

Moment identities (D the number of decode steps after the first token):

    E[D | j]   = ((I - Omega)^-1 1)_j
    E[D^2 | j] = ((I + Omega)(I - Omega)^-2 1)_j
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

#: the engine's KV-cache block size this law is written for
BLOCK_SIZE = 16


def sigma_blk(block_size: int = BLOCK_SIZE) -> float:
    """The mean block-quantisation slack per resident request, (B - 1) / 2."""
    return (block_size - 1) / 2.0


def off_lattice(x: float, block_size: int) -> int:
    """The nearest integer >= 1 that is not a multiple of the block size."""
    v = int(round(x))
    if v < 1:
        v = 1
    if v % block_size == 0:
        v += 1
    return v


@dataclasses.dataclass(frozen=True)
class PromptLaw:
    p: tuple[int, ...]
    rho: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.p) != len(self.rho) or not self.p:
            raise ValueError("PromptLaw: p and rho must have the same length > 0")
        if any(int(x) != x or x < 1 for x in self.p):
            raise ValueError(f"PromptLaw: integer lengths >= 1 required, got {self.p}")
        if any(r < 0 for r in self.rho) or abs(sum(self.rho) - 1.0) > 1e-9:
            raise ValueError(f"PromptLaw: rho must be a distribution, got {self.rho}")

    @property
    def ell_p(self) -> int:
        return len(self.p)

    def mean(self) -> float:
        return float(sum(r * x for r, x in zip(self.rho, self.p)))

    def second_moment(self) -> float:
        return float(sum(r * x * x for r, x in zip(self.rho, self.p)))


@dataclasses.dataclass(frozen=True)
class GenLaw:
    """The phase chain over D = g - 1. `Omega` is (L+1)x(L+1), its row and column 0 inert."""

    Omega: np.ndarray
    shift: bool = False

    def __post_init__(self) -> None:
        om = np.asarray(self.Omega, float)
        if om.ndim != 2 or om.shape[0] != om.shape[1] or om.shape[0] < 2:
            raise ValueError("GenLaw: Omega must be (L+1)x(L+1) with L >= 1")
        if (om < -1e-12).any() or (om[1:].sum(axis=1) > 1.0 + 1e-12).any():
            raise ValueError("GenLaw: Omega must be sub-stochastic on rows 1..L")
        if om[0].any() or om[:, 0].any():
            raise ValueError("GenLaw: row and column 0 (the atom) must be zero")
        if self.shift:
            raise ValueError("GenLaw: the countdown chain is not part of this law")
        object.__setattr__(self, "Omega", om)

    @staticmethod
    def hyper_erlang(mu: Sequence[float], k: Sequence[int]) -> GenLaw:
        """Parallel branches, branch b a chain of k_b geometric stages in series with mean mu_b steps in all."""
        mu_ = np.asarray(mu, float)
        ks = np.asarray(k, int)
        if ks.ndim != 1 or len(ks) != len(mu_):
            raise ValueError(f"GenLaw.hyper_erlang: one stage count per branch required, got {ks}")
        if (ks < 1).any():
            raise ValueError(f"GenLaw.hyper_erlang: k_b >= 1 required, got {ks}")
        if (mu_ < ks - 1e-9).any():
            raise ValueError(f"GenLaw.hyper_erlang: mu_b >= k_b required, got mu={mu_} k={ks}")
        s = 1.0 - ks / np.maximum(mu_, ks)
        if (s < -1e-12).any() or (s >= 1.0).any():
            raise ValueError(f"GenLaw.hyper_erlang: s in [0, 1) required, got {s}")
        n_phases = int(ks.sum())
        om = np.zeros((n_phases + 1, n_phases + 1))
        base = 1
        for sj, kb in zip(s, ks):
            for t in range(int(kb)):
                om[base + t, base + t] = sj
                if t < kb - 1:
                    om[base + t, base + t + 1] = 1.0 - sj
            base += int(kb)
        return GenLaw(om)

    @staticmethod
    def entry_phases(k: Sequence[int]) -> np.ndarray:
        """The phase index of the first stage of each branch."""
        ks = np.asarray(k, int)
        return 1 + np.concatenate(([0], np.cumsum(ks)[:-1]))

    @property
    def L(self) -> int:  # noqa: N802 - the phase count, as in the model's equations
        return int(self.Omega.shape[0] - 1)

    @property
    def omega(self) -> np.ndarray:
        """The exit probability of each phase, 1 - Omega 1 (index 0 inert)."""
        w = 1.0 - self.Omega.sum(axis=1)
        w[0] = 0.0
        return w

    @functools.cached_property
    def phase_probs(self) -> tuple[np.ndarray, ...]:
        """Per phase j: (omega_j, Omega[j, 1:]), the multinomial of one decode step."""
        om = self.omega
        return tuple(np.concatenate(([om[j]], self.Omega[j, 1:])) for j in range(self.L + 1))

    def mean_D_by_phase(self) -> np.ndarray:  # noqa: N802
        """E[D | entry phase j]; index 0 (the atom) is 0."""
        n = self.L
        a = np.eye(n) - self.Omega[1:, 1:]
        m = np.zeros(n + 1)
        m[1:] = np.linalg.solve(a, np.ones(n))
        return m

    def second_moment_D_by_phase(self) -> np.ndarray:  # noqa: N802
        """E[D^2 | entry phase j]; index 0 is 0."""
        n = self.L
        om = self.Omega[1:, 1:]
        a = np.eye(n) - om
        v = np.linalg.solve(a, np.linalg.solve(a, np.ones(n)))
        s = np.zeros(n + 1)
        s[1:] = (np.eye(n) + om) @ v
        return s

    def pmf_D(self, alpha: Any, i_max: int) -> np.ndarray:  # noqa: N802
        """P(D = i), i = 0..i_max, from the entry law alpha (index 0 the atom): alpha Omega^(i-1) omega."""
        alpha = np.asarray(alpha, float)
        out = np.zeros(i_max + 1)
        out[0] = alpha[0]
        v = alpha[1:].copy()
        om_, w = self.Omega[1:, 1:], self.omega[1:]
        for i in range(1, i_max + 1):
            out[i] = float(v @ w)
            v = v @ om_
        return out


@dataclasses.dataclass(frozen=True)
class JointLaw:
    """pi[a, j]: rows the prompt classes, columns the entry phases (0 = the atom g = 1)."""

    prompt: PromptLaw
    gen: GenLaw
    pi: np.ndarray

    def __post_init__(self) -> None:
        pi = np.asarray(self.pi, float)
        if pi.shape != (self.prompt.ell_p, self.gen.L + 1):
            raise ValueError(f"JointLaw: pi must be ({self.prompt.ell_p}, {self.gen.L + 1}), got {pi.shape}")
        if (pi < -1e-12).any() or abs(pi.sum() - 1.0) > 1e-9:
            raise ValueError("JointLaw: pi must be a non-negative distribution")
        if np.abs(pi.sum(axis=1) - np.asarray(self.prompt.rho)).max() > 1e-9:
            raise ValueError("JointLaw: the rows of pi must sum to rho")
        object.__setattr__(self, "pi", np.clip(pi, 0.0, None))

    @property
    def alpha(self) -> np.ndarray:
        return self.pi.sum(axis=0)

    @property
    def rho(self) -> np.ndarray:
        return self.pi.sum(axis=1)

    def moments(self) -> dict[str, float]:
        """E[p], E[p^2], E[g], E[g^2], E[pg], P(g=1), E[D], E[D^2] of the law (never of data)."""
        p = np.asarray(self.prompt.p, float)
        m_d = self.gen.mean_D_by_phase()
        s_d = self.gen.second_moment_D_by_phase()
        mg = 1.0 + m_d
        sg = s_d + 2.0 * m_d + 1.0
        eg = float(self.pi.sum(axis=0) @ mg)
        eg2 = float(self.pi.sum(axis=0) @ sg)
        epg = float(p @ (self.pi @ mg))
        return {"E_p": self.prompt.mean(), "E_p2": self.prompt.second_moment(), "E_g": eg, "E_g2": eg2,
                "E_pg": epg, "P_g1": float(self.alpha[0]), "E_D": eg - 1.0, "E_D2": eg2 - 2.0 * eg + 1.0,
                "cv2_g": (eg2 - eg * eg) / (eg * eg)}

    def phi_joint(self, block_size: int = BLOCK_SIZE) -> float:
        """The duration-weighted KV footprint of a resident request: E[pg]/E[g] + (E[g^2]/E[g] + 1)/2 + sigma."""
        m = self.moments()
        return m["E_pg"] / m["E_g"] + (m["E_g2"] / m["E_g"] + 1.0) / 2.0 + sigma_blk(block_size)

    def residence_cap(self, capacity_c: float, b_max: int | None = None, block_size: int = BLOCK_SIZE) -> float:
        """How many requests the KV capacity holds at once, on average, capped by the sequence limit."""
        c = capacity_c / self.phi_joint(block_size)
        return min(c, float(b_max)) if b_max else c

    def sample_types(self, rng: np.random.Generator, k: int) -> list[tuple[int, int]]:
        """k i.i.d. types (p, j): the prompt LENGTH and the entry phase."""
        if k <= 0:
            return []
        flat = self.pi.ravel()
        idx = rng.choice(flat.size, size=k, p=flat / flat.sum())
        ncol = self.pi.shape[1]
        return [(int(self.prompt.p[i // ncol]), int(i % ncol)) for i in idx]

    def to_dict(self) -> dict[str, Any]:
        return {"prompt": {"p": list(self.prompt.p), "rho": list(self.prompt.rho)},
                "gen": {"Omega": self.gen.Omega.tolist()}, "pi": self.pi.tolist()}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> JointLaw:
        return JointLaw(PromptLaw(tuple(int(x) for x in d["prompt"]["p"]), tuple(float(x) for x in d["prompt"]["rho"])),
                        GenLaw(np.asarray(d["gen"]["Omega"], float)), np.asarray(d["pi"], float))


class SeededTypes:
    """A law whose types come from their OWN random stream, so the k-th arrival has the same type whatever else the
    run consumes (common random numbers between runs that differ elsewhere)."""

    def __init__(self, law: JointLaw, seed: Any) -> None:
        self.law = law
        self.rng = np.random.default_rng(seed)
        self.gen = law.gen

    def sample_types(self, _rng: np.random.Generator, k: int) -> list[tuple[int, int]]:
        return self.law.sample_types(self.rng, k)

    def moments(self) -> dict[str, float]:
        return self.law.moments()


# ================================================================================================================
# The fitter: a trace's (prompt, output) pairs -> a joint law. Reads the trace only.
#
# * prompt classes: Lloyd quantisation of the prompts with re-anchored moments; the fewest classes of
#   PROMPT_CLASS_GRID whose law is within PROMPT_TOL of the observed prompts in W1 / E[p];
# * the output law: B parallel branches, branch b a chain of k_b geometric stages (a hyper-Erlang), shared by all
#   prompt classes; each COUPLING GROUP of adjacent classes has its own branch weights and its own empirical g = 1
#   atom (the rows of pi). Maximum likelihood on binned (group, length) counts by EM;
# * the stages: per branch, from STAGE_LADDER, by greedy ascent of the training likelihood, at most `phase_cap`
#   phases in all (default 12);
# * the selection: grid (coupling groups, B); each point scored by 5 seeded half/half splits, both ways, on held-out
#   S = sum_a rho_a W1(g | a) / E[g]; the 1-SE rule (fewest phases, then fewest groups, then fewest branches);
#   admissible only when the KV footprint is within 1 % of the trace's (2.5 % if no point is), every reachable
#   phase can exit, and EM converged. The law is the chosen structure refitted on the whole trace.
# ================================================================================================================

PROMPT_CLASS_GRID = (1, 2, 3, 4, 6, 8, 12, 16)
STAGE_LADDER = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233)
#: the engine's phase cap: the most phases a law may have
MAX_PHASES = 256
DEFAULT_PHASE_CAP = 12
DEFAULT_BRANCHES = (1, 2, 3, 4)
CV_SEEDS = (0, 1, 2, 3, 4)
FOOTPRINT_TOL = 0.01
FOOTPRINT_TOL_RELAXED = 0.025
PROMPT_TOL = 0.10
#: output lengths are binned per KV block above this many blocks; below, every length is its own bin
EXACT_BLOCKS = 4
EM_MAX_ITER = 2000
EM_TOL = 1e-9


def moments_of_pairs(p: Sequence[int], g: Sequence[int]) -> dict[str, Any]:
    """The trace's moments: E[p], E[p^2], E[g], E[g^2], E[pg], P(g = 1)."""
    p_ = np.asarray(p, float)
    g_ = np.asarray(g, float)
    if p_.size != g_.size or p_.size == 0 or (g_ < 1).any() or (p_ < 1).any():
        raise ValueError("moments_of_pairs: pairs (p >= 1, g >= 1) required")
    d = g_ - 1.0
    eg, eg2 = float(g_.mean()), float((g_ * g_).mean())
    return {"n": int(p_.size), "E_p": float(p_.mean()), "E_p2": float((p_ * p_).mean()), "E_g": eg, "E_g2": eg2,
            "E_pg": float((p_ * g_).mean()), "P_g1": float((g_ == 1).mean()), "E_D": float(d.mean()),
            "E_D2": float((d * d).mean())}


def exact_footprint(p: Sequence[int], g: Sequence[int], block_size: int = BLOCK_SIZE) -> float:
    """The trace's duration-weighted KV footprint: sum g (p + (g + 1) / 2) / sum g + (B - 1) / 2."""
    p_ = np.asarray(p, float)
    g_ = np.asarray(g, float)
    return float((g_ * (p_ + (g_ + 1.0) / 2.0)).sum() / g_.sum() + sigma_blk(block_size))


def fit_prompt_classes(p_emp: Sequence[int], ell_p: int, *, block_size: int = BLOCK_SIZE,
                       max_iter: int = 200) -> PromptLaw:
    """`ell_p` deterministic prompt classes: Lloyd quantisation (1D k-means, quantile init), lengths rescaled about
    the mean so that E[p^2] is exact and rounded off the block lattice, weights re-solved so that E[p] (and, from 3
    classes, E[p^2]) is exact."""
    from scipy.optimize import lsq_linear, minimize

    x = np.asarray(p_emp, float)
    if ell_p < 1 or x.size < ell_p:
        raise ValueError(f"fit_prompt_classes: {ell_p} classes are not possible on {x.size} points")
    m, var = float(x.mean()), float(x.var())
    c = np.quantile(x, (np.arange(ell_p) + 0.5) / ell_p)
    for _ in range(max_iter):
        edges = (c[1:] + c[:-1]) / 2.0
        idx = np.searchsorted(edges, x)
        c_new = np.array([x[idx == k].mean() if np.any(idx == k) else c[k] for k in range(ell_p)])
        if np.allclose(c_new, c, rtol=1e-12, atol=1e-9):
            c = c_new
            break
        c = c_new
    idx = np.searchsorted((c[1:] + c[:-1]) / 2.0, x)
    rho0 = np.array([(idx == k).mean() for k in range(ell_p)])
    var_between = float(rho0 @ (c - m) ** 2)
    s = math.sqrt(var / var_between) if var_between > 0 else 1.0
    p_scaled = m + s * (c - m)
    p_int = np.array([off_lattice(v, block_size) for v in p_scaled], float)
    if len(set(p_int.tolist())) != ell_p:
        raise ValueError(f"fit_prompt_classes: classes coincide after rounding {p_int.tolist()}")
    rows = [np.ones(ell_p), p_int]
    rhs = [1.0, m]
    if ell_p >= 3:
        rows.append(p_int ** 2)
        rhs.append(m * m + var)
    a_mat, b_vec = np.array(rows), np.array(rhs)
    res = minimize(lambda r: float(((r - rho0) ** 2).sum()), rho0, method="SLSQP", bounds=[(0.0, 1.0)] * ell_p,
                   constraints=[{"type": "eq", "fun": (lambda r, A=a_mat, b=b_vec: A @ r - b)}],
                   options={"maxiter": 500, "ftol": 1e-14})
    rho = np.clip(res.x, 0.0, None)
    rho /= rho.sum()
    if np.abs(a_mat @ rho - b_vec).max() > 1e-6 * max(1.0, abs(b_vec[-1])):
        res2 = lsq_linear(np.vstack([np.ones(ell_p), p_int]), np.array([1.0, m]), bounds=(0, 1))
        rho = np.clip(res2.x, 0.0, None)
        rho /= rho.sum()
    order = np.argsort(p_int)
    return PromptLaw(tuple(int(v) for v in p_int[order]), tuple(float(v) for v in rho[order]))


def negbin_pmf_matrix(mu: Any, ks: Any, i_max: int) -> np.ndarray:
    """P(D = i | branch b), i = 1..i_max per row, one branch per column, k_b stages per branch."""
    from scipy.special import gammaln

    mu = np.asarray(mu, float)
    ks = np.asarray(ks, int)
    i = np.arange(1, i_max + 1)[:, None]
    k = ks[None, :].astype(float)
    p = (ks / mu)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = gammaln(i) - gammaln(k) - gammaln(np.maximum(i - k + 1, 1))
        lp = logc + k * np.log(p) + (i - k) * np.log1p(-p)
        return np.where(i >= k, np.exp(lp), 0.0)


def class_of(prompt: PromptLaw, p: Any) -> np.ndarray:
    """The nearest prompt class of each prompt."""
    c = np.asarray(prompt.p, float)
    if len(c) == 1:
        return np.zeros(len(p), int)
    return np.searchsorted((c[1:] + c[:-1]) / 2.0, np.asarray(p, float))


@dataclasses.dataclass
class Counts:
    """(class, D) cells with multiplicities: EM works on unique values, not on rows."""

    cls: np.ndarray
    d: np.ndarray
    w: np.ndarray
    n_class: np.ndarray
    atom: np.ndarray
    ell_p: int

    @staticmethod
    def build(cls: Any, g: Any, ell_p: int, bin_size: int = BLOCK_SIZE) -> Counts:
        """Cells (class, D bin) of width `bin_size` above EXACT_BLOCKS blocks, each carrying its multiplicity and
        its mean D rounded to an integer; below, one cell per length."""
        cls = np.asarray(cls, int)
        g = np.asarray(g, int)
        n_class = np.bincount(cls, minlength=ell_p).astype(float)
        atom = np.bincount(cls[g == 1], minlength=ell_p).astype(float)
        m = g > 1
        d = (g[m] - 1).astype(np.int64)
        bs = max(int(bin_size), 1)
        exact = EXACT_BLOCKS * bs
        b = np.where(d <= exact, d - 1, exact + (d - exact - 1) // bs)
        nb = int(b.max()) + 1 if b.size else 1
        key = cls[m].astype(np.int64) * nb + b
        uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        dsum = np.bincount(inv, weights=d.astype(float), minlength=len(uniq))
        return Counts(cls=(uniq // nb).astype(int), d=np.maximum(np.rint(dsum / cnt), 1).astype(int),
                      w=cnt.astype(float), n_class=n_class, atom=atom, ell_p=ell_p)


@dataclasses.dataclass
class HyperErlangFit:
    mu: np.ndarray           # branch means, in steps of D
    ks: np.ndarray           # stages per branch
    wts: np.ndarray          # (classes, B): branch weights per class over the non-atom mass
    loglik: float
    iters: int
    converged: bool

    @property
    def L(self) -> int:  # noqa: N802
        return int(self.ks.sum())


def _log_pmf_const(ks: np.ndarray, d: Any) -> tuple[np.ndarray, np.ndarray]:
    from scipy.special import gammaln

    dd = np.asarray(d, float)[:, None]
    k = ks[None, :].astype(float)
    logc = gammaln(dd) - gammaln(k) - gammaln(np.maximum(dd - k + 1, 1))
    return logc, dd >= k


def _log_pmf_from_const(mu: np.ndarray, ks: np.ndarray, d: Any, logc: np.ndarray, ok: np.ndarray) -> np.ndarray:
    dd = np.asarray(d, float)[:, None]
    k = ks[None, :].astype(float)
    p = (ks / mu)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = logc + k * np.log(p) + (dd - k) * np.log1p(-p)
    return np.where(ok, lp, -np.inf)


def em_hyper_erlang(counts: Counts, ks: Any, *, mu0: Any = None, wts0: Any = None, max_iter: int = EM_MAX_ITER,
                    tol: float = EM_TOL) -> HyperErlangFit:
    """Maximum likelihood of (mu_b, w_ab) at fixed stages, closed-form M-step; initialised at quantiles of D."""
    ks = np.asarray(ks, int)
    nb = len(ks)
    d, w, cls = counts.d, counts.w, counts.cls
    ell_p = counts.ell_p
    if mu0 is None:
        qs = np.quantile(np.repeat(d, w.astype(int)) if w.sum() < 5e6 else d,
                         (np.arange(nb) + 0.5) / nb) if nb > 1 else np.array([float((w * d).sum() / w.sum())])
        mu0 = np.maximum(np.asarray(qs, float), ks + 0.5)
    mu = np.maximum(np.asarray(mu0, float), ks + 1e-3)
    wts = np.full((ell_p, nb), 1.0 / nb) if wts0 is None else np.asarray(wts0, float).copy()
    wts = np.maximum(wts, 1e-12)
    wts /= wts.sum(axis=1, keepdims=True)
    ll_old = -np.inf
    converged = False
    it = 0
    logc, ok = _log_pmf_const(ks, d)
    for it in range(1, max_iter + 1):
        lp = _log_pmf_from_const(mu, ks, d, logc, ok) + np.log(wts[cls])
        mx = lp.max(axis=1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        ex = np.exp(lp - mx)
        den = ex.sum(axis=1, keepdims=True)
        ll = float((w * (np.log(np.maximum(den[:, 0], 1e-300)) + mx[:, 0])).sum())
        r = ex / np.maximum(den, 1e-300)
        rw = r * w[:, None]
        n_b = rw.sum(axis=0)
        s_d = (rw * d[:, None]).sum(axis=0)
        mu = np.where(n_b > 0, s_d / np.maximum(n_b, 1e-300), mu)
        mu = np.maximum(mu, ks + 1e-6)
        for a in range(ell_p):
            sel = cls == a
            tot = rw[sel].sum()
            wts[a] = (rw[sel].sum(axis=0) / tot) if tot > 0 else np.full(nb, 1.0 / nb)
        wts = np.maximum(wts, 1e-300)
        if abs(ll - ll_old) <= tol * max(1.0, abs(ll)):
            converged = True
            ll_old = ll
            break
        ll_old = ll
    atom_ll = 0.0
    for a in range(ell_p):
        n_a, at = counts.n_class[a], counts.atom[a]
        if n_a > 0 and 0 < at < n_a:
            atom_ll += at * math.log(at / n_a) + (n_a - at) * math.log(1 - at / n_a)
    return HyperErlangFit(mu=mu, ks=ks, wts=wts, loglik=ll_old + atom_ll, iters=it, converged=converged)


def fit_stages_greedy(counts: Counts, n_branches: int, *, ladder: Sequence[int] = STAGE_LADDER,
                      phase_cap: int = MAX_PHASES) -> HyperErlangFit:
    """Stages per branch by greedy ascent of the training likelihood, from k_b = 1, one ladder step at a time,
    within the phase cap; the branches are then ordered by mean."""
    ladder = list(ladder)
    ks = np.ones(n_branches, int)
    best = em_hyper_erlang(counts, ks)
    while True:
        cand = None
        for b in range(n_branches):
            pos = ladder.index(int(ks[b]))
            if pos + 1 >= len(ladder):
                continue
            trial = ks.copy()
            trial[b] = ladder[pos + 1]
            if trial.sum() > phase_cap:
                continue
            f = em_hyper_erlang(counts, trial, mu0=np.maximum(best.mu, trial + 0.5), wts0=best.wts)
            if f.loglik > best.loglik + 1e-9 * abs(best.loglik) and (cand is None or f.loglik > cand.loglik):
                cand = f
        if cand is None:
            break
        best, ks = cand, cand.ks
    order = np.argsort(best.mu, kind="stable")
    return HyperErlangFit(mu=best.mu[order], ks=best.ks[order], wts=best.wts[:, order], loglik=best.loglik,
                          iters=best.iters, converged=best.converged)


def law_from_fit(prompt: PromptLaw, counts: Counts, fit: HyperErlangFit, *, p_atom: Any = None) -> JointLaw:
    """The joint law of a fit; `p_atom` (per prompt class) overrides the atom read from the counts."""
    gen = GenLaw.hyper_erlang(fit.mu, fit.ks)
    entry = GenLaw.entry_phases(fit.ks)
    pi = np.zeros((prompt.ell_p, gen.L + 1))
    rho = np.asarray(prompt.rho, float)
    for a in range(prompt.ell_p):
        if p_atom is not None:
            pa = float(p_atom[a])
        else:
            n_a = counts.n_class[a]
            pa = (counts.atom[a] / n_a) if n_a > 0 else 0.0
        pi[a, 0] = rho[a] * pa
        pi[a, entry] = rho[a] * (1.0 - pa) * fit.wts[a]
    return JointLaw(prompt, gen, pi)


def _cdf_full(pmf_branch: np.ndarray, atom: float, wts: np.ndarray) -> np.ndarray:
    pm = (1.0 - atom) * (pmf_branch @ wts)
    return np.concatenate(([atom], atom + np.cumsum(pm)))


def conditional_w1(fit: HyperErlangFit, atom_fit: Any, cls: Any, g: Any, ell_p: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    """Per class: the full-tail W1 between the fitted output law and the empirical one, KS, and the class's share."""
    g = np.asarray(g, int)
    cls = np.asarray(cls, int)
    gmax = int(g.max())
    horizon = int(max(gmax, (fit.mu + 12 * np.sqrt(fit.mu ** 2 / np.maximum(fit.ks, 1))).max()) + 2)
    pmf = negbin_pmf_matrix(fit.mu, fit.ks, horizon)
    w1 = np.zeros(ell_p)
    ks_ = np.zeros(ell_p)
    share = np.zeros(ell_p)
    for a in range(ell_p):
        ga = g[cls == a]
        share[a] = ga.size / g.size
        if ga.size == 0:
            continue
        cdf_fit = _cdf_full(pmf, float(atom_fit[a]), fit.wts[a])
        cnt = np.bincount(ga, minlength=horizon + 2)[1:horizon + 2]
        cdf_emp = np.cumsum(cnt) / ga.size
        diff = np.abs(cdf_fit - cdf_emp)
        w1[a] = float(diff.sum() + (1.0 - cdf_fit[-1]))
        ks_[a] = float(diff.max())
    return w1, ks_, share


def score_heldout(law_prompt: PromptLaw, fit: HyperErlangFit, atom_fit: Any, p_te: Any, g_te: Any) -> float:
    cls = class_of(law_prompt, p_te)
    w1, _ks, share = conditional_w1(fit, atom_fit, cls, g_te, law_prompt.ell_p)
    return float((share * w1).sum() / float(np.mean(g_te)))


def select_structure(points: list[dict[str, Any]], *, size: str = "ell_c") -> dict[str, Any]:
    """1-SE: among admissible points, the fewest phases (median over the splits), then the fewest groups, then the
    fewest branches, within the minimum score plus its standard error."""
    adm = [q for q in points if q.get("admissible", True) and math.isfinite(q["S_mean"])]
    if not adm:
        raise ValueError("select_structure: no admissible point")
    best = min(adm, key=lambda q: q["S_mean"])
    band = best["S_mean"] + best["S_se"]
    within = [q for q in adm if q["S_mean"] <= band]
    return min(within, key=lambda q: (q["L_median"], q[size], q["B"]))


def reachable_phases_absorb(law: JointLaw) -> bool:
    """Whether every phase reachable from the entry law can exit."""
    omega = law.gen.omega
    transitions = law.gen.Omega
    active = set(int(j) for j in np.flatnonzero(law.alpha[1:] > 0) + 1)
    if not active:
        return True
    reachable = set(active)
    pending = list(active)
    reverse: dict[int, set[int]] = {}
    while pending:
        j = pending.pop()
        for successor in np.flatnonzero(transitions[j, 1:] > 0) + 1:
            successor = int(successor)
            reverse.setdefault(successor, set()).add(j)
            if successor not in reachable:
                reachable.add(successor)
                pending.append(successor)
    can_exit = {j for j in reachable if omega[j] > 0}
    pending_exit = deque(can_exit)
    while pending_exit:
        for predecessor in reverse.get(pending_exit.popleft(), ()):
            if predecessor not in can_exit:
                can_exit.add(predecessor)
                pending_exit.append(predecessor)
    return reachable <= can_exit


def admissibility(law: JointLaw, p: Any, g: Any, fit: HyperErlangFit, *, block_size: int = BLOCK_SIZE,
                  tol: float = FOOTPRINT_TOL) -> dict[str, Any]:
    phi_fit = law.phi_joint(block_size)
    phi_ex = exact_footprint(p, g, block_size)
    rel = phi_fit / phi_ex - 1.0
    ok_phi = abs(rel) <= tol
    ok_abs = reachable_phases_absorb(law)
    return {"footprint_rel_err": float(rel), "reachable_phases_absorb": bool(ok_abs),
            "em_converged": bool(fit.converged), "ok": bool(ok_phi and ok_abs and fit.converged)}


def reanchor_footprint(prompt: PromptLaw, counts: Counts, fit: HyperErlangFit, p: Any, g: Any, *,
                       block_size: int = BLOCK_SIZE, p_atom: Any = None) -> tuple[JointLaw, HyperErlangFit, float]:
    """The declared fallback when no point is admissible: one common factor on every branch mean, by bisection, so
    that the footprint is exact."""
    phi_ex = exact_footprint(p, g, block_size)

    def law_at(c: float) -> tuple[JointLaw, HyperErlangFit]:
        f = HyperErlangFit(mu=np.maximum(fit.mu * c, fit.ks + 1e-6), ks=fit.ks, wts=fit.wts, loglik=fit.loglik,
                           iters=fit.iters, converged=fit.converged)
        return law_from_fit(prompt, counts, f, p_atom=p_atom), f

    lo, hi = 0.5, 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        law, f = law_at(mid)
        if law.phi_joint(block_size) < phi_ex:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    c = 0.5 * (lo + hi)
    law, f = law_at(c)
    return law, f, c


def prompt_law(p: Any, ell_p: int, *, block_size: int = BLOCK_SIZE) -> PromptLaw:
    p = np.asarray(p, int)
    if ell_p == 1:
        return PromptLaw((int(off_lattice(float(p.mean()), block_size)),), (1.0,))
    return fit_prompt_classes(p, ell_p, block_size=block_size)


def prompt_w1(prompt: PromptLaw, p: Any) -> float:
    """W1 in tokens between the discrete prompt law and the empirical prompt lengths."""
    xs = np.sort(np.asarray(p, float))
    c = np.asarray(prompt.p, float)
    w = np.asarray(prompt.rho, float)
    o = np.argsort(c)
    c, w = c[o], w[o]
    grid = np.unique(np.concatenate([xs, c]))
    f_emp = np.searchsorted(xs, grid, side="right") / xs.size
    f_fit = np.cumsum(w)[np.searchsorted(c, grid, side="right") - 1] * (grid >= c[0])
    return float(np.sum(np.abs(f_emp - f_fit)[:-1] * np.diff(grid)))


def prompt_classes_by_tolerance(p: Any, *, tol: float = PROMPT_TOL, grid: Sequence[int] = PROMPT_CLASS_GRID,
                                block_size: int = BLOCK_SIZE) -> int:
    """The fewest prompt classes of `grid` whose law is within `tol` of the observed prompts in W1 / E[p]."""
    p = np.asarray(p, int)
    for k in grid:
        try:
            r = prompt_w1(prompt_law(p, k, block_size=block_size), p) / float(p.mean())
        except ValueError:
            continue
        if r <= tol:
            return k
    raise ValueError(f"no prompt-class count in {list(grid)} is within {tol:.0%} of the prompts")


def coupling_groups(prompt: PromptLaw, ell_c: int) -> np.ndarray:
    """The group of each prompt class: `ell_c` groups of ADJACENT classes, by Lloyd on the class lengths weighted by
    their probabilities."""
    c = np.asarray(prompt.p, float)
    w = np.asarray(prompt.rho, float)
    n = c.size
    if ell_c >= n:
        return np.arange(n)
    if ell_c <= 1:
        return np.zeros(n, int)
    o = np.argsort(c)
    cs, ws = c[o], w[o]
    centres = np.interp((np.arange(ell_c) + 0.5) / ell_c, np.cumsum(ws) - ws / 2, cs)
    for _ in range(200):
        idx = np.searchsorted((centres[1:] + centres[:-1]) / 2.0, cs)
        new = np.array([np.average(cs[idx == k], weights=ws[idx == k]) if np.any(idx == k) else centres[k]
                        for k in range(ell_c)])
        if np.allclose(new, centres, rtol=1e-12, atol=1e-9):
            break
        centres = new
    idx = np.searchsorted((centres[1:] + centres[:-1]) / 2.0, cs)
    used = np.unique(idx)
    if used.size != ell_c:
        raise ValueError(f"coupling_groups: {ell_c} groups requested, {used.size} non-empty")
    grp = np.empty(n, int)
    grp[o] = np.searchsorted(used, idx)
    return grp


def fit_point(p: Any, g: Any, ell_p: int, ell_c: int, n_branches: int, *, block_size: int = BLOCK_SIZE,
              phase_cap: int = MAX_PHASES) -> tuple[PromptLaw, np.ndarray, Counts, HyperErlangFit, np.ndarray,
                                                     HyperErlangFit, np.ndarray]:
    """One structure: the prompt classes, the coupling groups, the hyper-Erlang fitted by EM on the group counts,
    and its expansion to every prompt class."""
    p = np.asarray(p, int)
    g = np.asarray(g, int)
    prompt = prompt_law(p, ell_p, block_size=block_size)
    grp = coupling_groups(prompt, ell_c)
    n_grp = int(grp.max()) + 1
    counts = Counts.build(grp[class_of(prompt, p)], g, n_grp, bin_size=block_size)
    fit = fit_stages_greedy(counts, n_branches, phase_cap=phase_cap)
    atom_g = np.where(counts.n_class > 0, counts.atom / np.maximum(counts.n_class, 1), 0.0)
    fit_c = HyperErlangFit(mu=fit.mu, ks=fit.ks, wts=fit.wts[grp], loglik=fit.loglik, iters=fit.iters,
                           converged=fit.converged)
    return prompt, grp, counts, fit, atom_g, fit_c, atom_g[grp]


def cv_point(p: Any, g: Any, ell_p: int, ell_c: int, n_branches: int, *, seeds: Sequence[int] = CV_SEEDS,
             block_size: int = BLOCK_SIZE, phase_cap: int = MAX_PHASES) -> dict[str, Any]:
    """The 10 held-out scores of one structure: per seed, half A / half B, both ways."""
    p = np.asarray(p, int)
    g = np.asarray(g, int)
    n = p.size
    scores, phases = [], []
    for s in seeds:
        perm = np.random.default_rng(int(s)).permutation(n)
        a_half, b_half = perm[: n // 2], perm[n // 2:]
        for tr, te in ((a_half, b_half), (b_half, a_half)):
            prompt, _, _, fit, _, fit_c, atom_c = fit_point(p[tr], g[tr], ell_p, ell_c, n_branches,
                                                            block_size=block_size, phase_cap=phase_cap)
            scores.append(score_heldout(prompt, fit_c, atom_c, p[te], g[te]))
            phases.append(fit.L)
    arr = np.asarray(scores)
    return {"ell_p": ell_p, "ell_c": ell_c, "B": n_branches, "S_mean": float(arr.mean()),
            "S_se": float(arr.std(ddof=1) / math.sqrt(arr.size)), "L_median": float(np.median(phases))}


def resolve_phase_cap(phase_cap: int | None) -> int:
    """None -> the default (12); 0 -> unlimited (the engine's 256)."""
    if phase_cap is None:
        return DEFAULT_PHASE_CAP
    phase_cap = int(phase_cap)
    if phase_cap < 0:
        raise ValueError(f"the phase cap must be >= 0 (0 = unlimited), got {phase_cap}")
    return MAX_PHASES if phase_cap == 0 else min(phase_cap, MAX_PHASES)


def fit_law(p: Any, g: Any, *, block_size: int = BLOCK_SIZE, prompt_tol: float = PROMPT_TOL,
            branches: Sequence[int] = DEFAULT_BRANCHES, seeds: Sequence[int] = CV_SEEDS,
            phase_cap: int | None = DEFAULT_PHASE_CAP) -> tuple[JointLaw, dict[str, Any]]:
    """A trace's joint law: prompt classes by tolerance, then the coupled selection, then the refit."""
    cap = resolve_phase_cap(phase_cap)
    p = np.asarray(p, int)
    g = np.asarray(g, int)
    ell_p = prompt_classes_by_tolerance(p, tol=prompt_tol, block_size=block_size)
    points = []
    for ell_c in [k for k in PROMPT_CLASS_GRID if k <= ell_p]:
        for nb in branches:
            try:
                q = cv_point(p, g, ell_p, ell_c, nb, seeds=seeds, block_size=block_size, phase_cap=cap)
                prompt, _grp, counts, fit, _atom_g, fit_c, atom_c = fit_point(p, g, ell_p, ell_c, nb,
                                                                               block_size=block_size, phase_cap=cap)
            except ValueError:
                points.append({"ell_p": ell_p, "ell_c": ell_c, "B": nb, "admissible": False, "S_mean": math.inf,
                               "S_se": math.inf, "L_median": math.inf})
                continue
            law = law_from_fit(prompt, counts, fit_c, p_atom=atom_c)
            adm = admissibility(law, p, g, fit, block_size=block_size, tol=FOOTPRINT_TOL)
            q.update({"admissibility": adm, "admissible": adm["ok"]})
            points.append(q)
    tol_used, reanchored = FOOTPRINT_TOL, False
    if not any(q["admissible"] for q in points):
        tol_used = FOOTPRINT_TOL_RELAXED
        for q in points:
            a = q.get("admissibility")
            if a:
                q["admissible"] = bool(abs(a["footprint_rel_err"]) <= tol_used and a["reachable_phases_absorb"]
                                       and a["em_converged"])
    try:
        chosen = select_structure(points)
    except ValueError:
        reanchored = True
        chosen = select_structure([{**q, "admissible": math.isfinite(q["S_mean"])} for q in points])
    ell_c, nb = chosen["ell_c"], chosen["B"]
    prompt, grp, counts, fit, _atom_g, fit_c, atom_c = fit_point(p, g, ell_p, ell_c, nb, block_size=block_size,
                                                                 phase_cap=cap)
    scale = 1.0
    if reanchored:
        law, fit_c, scale = reanchor_footprint(prompt, counts, fit_c, p, g, block_size=block_size, p_atom=atom_c)
    else:
        law = law_from_fit(prompt, counts, fit_c, p_atom=atom_c)
    return law, {"prompt_classes": ell_p, "coupling_groups": ell_c, "branches": nb, "stages": fit.ks.tolist(),
                 "groups": grp.tolist(), "phase_cap": cap, "footprint_tol": tol_used, "reanchored": reanchored,
                 "reanchor_scale": scale, "points": points}
