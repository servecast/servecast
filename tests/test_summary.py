"""The summary both writers share: on the model's own requests it reproduces the frozen population model's numbers."""

from __future__ import annotations

import math

import pytest

from servecast.core import estimator as estimators
from servecast.core import gpus
from servecast.core import summary as S
from servecast.core.manifest import Manifest
from servecast.model import predict as P
from servecast.model import run as R
from tests.test_cli_predict import BASE

COMPONENTS = {"cost_law": "2026-10", "calibration_procedure": "2026-10", "population_model": "2026-10",
              "workload_fitter": "2026-10", "length_sampler": None, "estimator": "2026-10"}


@pytest.fixture(scope="module")
def row() -> dict:
    calibration = P.load_calibration("a100_sxm_40gb/d9cbf1d3/llama-3-8b")
    m = Manifest.model_validate(BASE)
    llm = R.resolve_llm(m.engine.model)
    engine, _, _ = R.resolve_engine(m, gpus.get("a100_sxm_40gb"), llm)
    return P.predict_point(calibration, llm, P.load_workload_law("azure-conversation"), engine, 8, 1.0,
                           components=COMPONENTS, seed=0, keep_requests=True,
                           targets=P.PAPER_TARGETS)


def test_window_quantities_reproduce_the_population_models_numbers(row: dict) -> None:
    window = {"x": row["throughput_rps"], "span": row["span_s"], "n_kept": row["n_kept"]}
    q = S.window_quantities(row["requests"], window)
    for key in ("ttft", "tpot", "ttft_mean", "tpot_mean", "output_tokens_per_s", "p95_ttft_s", "p95_tpot_s"):
        assert q[key] == row[key], key


def test_the_point_summary_is_the_predictions(row: dict) -> None:
    est = estimators.get("2026-10")
    window = {"x": row["throughput_rps"], "span": row["span_s"], "n_kept": row["n_kept"]}
    from_requests = S.point_summary(S.window_quantities(row["requests"], window), row["per_metric"], est, 1.0)
    from_row = R.summary({**row, "wall_s": 1.0}, "2026-10")
    assert from_requests == from_row


def test_an_unbounded_throughput_interval_is_left_out() -> None:
    assert S.interval({"usable": True, "mean": 0.01, "hw": 0.02}, 1.0, invert=True) is None
    iv = S.interval({"usable": True, "mean": 0.1, "hw": 0.01}, 1.0, invert=True)
    assert iv is not None and iv["low"] == pytest.approx(1 / 0.11) and iv["high"] == pytest.approx(1 / 0.09)
    assert math.isnan(S.p95([]))
