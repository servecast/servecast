"""A prediction for an LLM other than the one the calibration was fitted on is a transfer, and the manifest says so
(calibration.transfer). Unset, it is false, and predict refuses a transfer naming the field; --to sets it."""

from __future__ import annotations

import copy

import pytest

from servecast.core.manifest import Manifest
from servecast.model import predict as P
from servecast.model import run as R

BASE = {"engine": {"model": "meta-llama/Meta-Llama-3-8B", "workers": {"aggregated": {"kv_cache": {
    "capacity": {"blocks": 9965}}}}},
    "traffic": {"source": {"type": "trace", "paths": ["burstgpt"]},
                "load": {"type": "concurrency", "concurrency": 4, "think_time_ms": 1000}, "stop": {"steady_state": {}}}}


def _plan(calibration: str, transfer: bool | None, model: str = "meta-llama/Meta-Llama-3-8B") -> R.Plan:
    doc = copy.deepcopy(BASE)
    doc["engine"]["model"] = model
    doc["calibration"] = {"use": calibration, **({} if transfer is None else {"transfer": transfer})}
    return R.plan(Manifest.model_validate(doc), P.load_calibration(calibration), calibration, b"{}")


def test_a_per_llm_calibration_on_its_own_llm_is_no_transfer() -> None:
    assert _plan("a100_sxm_40gb/d9cbf1d3/llama-3-8b", None).llm.name == "meta-llama/Meta-Llama-3-8B"


@pytest.mark.parametrize(("calibration", "made_on"), [
    ("a100_sxm_40gb/d9cbf1d3/llama-3-8b", "meta-llama/Meta-Llama-3-8B"),
    ("a100_sxm_40gb/c62ebe75", "8 synthetic models")])
def test_an_undeclared_transfer_is_refused_naming_the_field(calibration: str, made_on: str) -> None:
    with pytest.raises(R.ManifestError, match=rf"calibration\.transfer: the calibration {calibration} was fitted on "
                                              rf"{made_on} and the manifest predicts Qwen/Qwen2\.5-7B; set "
                                              r"calibration\.transfer: true"):
        _plan(calibration, None, model="Qwen/Qwen2.5-7B")


@pytest.mark.parametrize("calibration", ["a100_sxm_40gb/d9cbf1d3/llama-3-8b", "a100_sxm_40gb/c62ebe75"])
def test_a_declared_transfer_predicts(calibration: str) -> None:
    assert _plan(calibration, True, model="Qwen/Qwen2.5-7B").llm.name == "Qwen/Qwen2.5-7B"


def test_a_per_gpu_calibration_needs_a_transfer_even_for_a_paper_llm() -> None:
    """Fitted on synthetic models, it predicts every real LLM by transfer, the paper's six included."""
    with pytest.raises(R.ManifestError, match=r"calibration\.transfer"):
        _plan("a100_sxm_40gb/c62ebe75", False)


def test_the_manifest_as_it_ran_keeps_the_transfer() -> None:
    """manifest.yaml must run again: it dropped calibration.transfer, and a transfer then refused to run."""
    doc = copy.deepcopy(BASE)
    doc["engine"]["model"] = "Qwen/Qwen2.5-7B"
    doc["calibration"] = {"use": "a100_sxm_40gb/c62ebe75", "transfer": True}
    m = Manifest.model_validate(doc)
    pl = R.plan(m, P.load_calibration("a100_sxm_40gb/c62ebe75"), "a100_sxm_40gb/c62ebe75", b"{}")
    assert R.resolved_manifest(m, pl)["calibration"] == {"use": "a100_sxm_40gb/c62ebe75", "transfer": True}
