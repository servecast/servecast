"""The paper's fits, done again from the data that ship: its 18 calibrations (its cost law on 3 GPUs for 6 LLMs), from
their profiling data by the fit `servecast calibrate` runs, and the workload law of each of its three traces, from the
trace by the paper's workload fitter. The appendix's calibration coefficients, the cost question's training MAPE and
the workload-law table come from these fits. Each fit gives back the shipped calibration or law bit for bit on the
reference platform, so these tables are recomputed from the measurements, not copied.
"""

from __future__ import annotations

import functools
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from servecast.core import versions
from servecast.paper import data


def name(gpu: str, llm: str) -> str:
    """The calibration, and its profiling data, of the paper's law for `llm` on `gpu`."""
    return f"{gpu}/{versions.PAPER['cost_law']}/{data.slug(llm)}"


def refit_one(gpu: str, llm: str) -> dict[str, Any]:
    """The coefficients and the training MAPE (%) of one calibration, fitted from its shipped profiling data."""
    from servecast.calibrate import fit
    from servecast.model import predict as P

    # the shipped file itself: the name, read as a path, found a folder of the same name in the working directory first
    profiling, sha256 = fit.read_profiling(P.shipped_profiling()[name(gpu, llm)])
    doc = fit.calibrate(profiling, sha256)
    return {"parameters": dict(doc["law"]["parameters"]), "mape": float(doc["fit"]["error_pct"]["mape"])}


def _job(key: tuple[str, str]) -> tuple[tuple[str, str], dict[str, Any]]:
    return key, refit_one(*key)


def refit_law(trace: str) -> dict[str, Any]:
    """The workload law of one of the paper's traces, fitted again from the shipped trace by the paper's fitter at its
    default phase cap, for 16-token blocks (minutes: about 20 s for Azure, 1 min for BurstGPT, 4 min for ServeGen on an
    Apple M1 Pro)."""
    from servecast.core import pool as pools
    from servecast.model import run as R
    from servecast.model import workload as W

    doc = pools.shipped_trace(trace)
    law, kept = R._fit(versions.PAPER["workload_fitter"], doc["input_tokens"], doc["output_tokens"], 16,
                       W.v2026_10.DEFAULT_PHASE_CAP, None)
    return {"law": law, "fit": kept}


def _law_job(trace: str) -> tuple[str, dict[str, Any]]:
    return trace, refit_law(trace)


def law_refits(traces: tuple[str, ...], jobs: int = 1) -> dict[str, dict[str, Any]]:
    """The workload law of each trace, fitted again; `jobs` processes fit them in parallel without changing them."""
    if jobs <= 1:
        return {t: refit_law(t) for t in traces}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return dict(ex.map(_law_job, traces))


@functools.cache
def refits(jobs: int = 1) -> dict[tuple[str, str], dict[str, Any]]:
    """Every calibration of the paper, by (GPU, LLM); `jobs` processes fit them in parallel without changing them."""
    keys = [(gpu, llm) for gpu in data.GPUS for llm in data.LLMS]
    if jobs <= 1:
        return {key: refit_one(*key) for key in keys}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return dict(ex.map(_job, keys))
