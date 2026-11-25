"""The workload law 2026-10 against the reference implementation, and the shipped laws."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from servecast.model.workload import v2026_10 as W
from tests.components.inputs import load, same

G = load("workload")
DATA = Path(__file__).resolve().parents[2] / "src" / "servecast" / "data"


@pytest.mark.exact
def test_the_law_is_the_reference_one() -> None:
    law = W.JointLaw.from_dict(G["law"])
    assert same(law.moments(), G["moments"])
    assert law.phi_joint(16) == G["phi_joint"]
    assert law.residence_cap(40000.0, 256, 16) == G["residence_cap"]
    assert law.gen.mean_D_by_phase().tolist() == G["mean_D_by_phase"]
    assert law.gen.second_moment_D_by_phase().tolist() == G["second_moment_D_by_phase"]
    # pmf_D is a matrix-vector product, which numpy hands to the platform's BLAS: its summation order is the
    # library's, and a hosted macOS runner's differs from the reference machine's in the last bit (measured in CI).
    # It is a diagnostic, never on the prediction path, so it alone is compared to a relative 1e-12.
    assert np.allclose(law.gen.pmf_D(law.alpha, 50), G["pmf_D"], rtol=1e-12, atol=0.0)
    assert [list(t) for t in law.sample_types(np.random.default_rng(31), 500)] == G["types"]


@pytest.mark.parametrize("trace", ["azure-conversation", "burstgpt", "servegen"])
def test_the_shipped_laws_load_and_are_off_the_block_lattice(trace: str) -> None:
    doc = json.loads((DATA / "workload_laws" / f"{trace}.json").read_text(encoding="utf-8"))
    law = W.JointLaw.from_dict(doc["law"])
    assert doc["workload_fitter"] == "2026-10" and doc["trace"] == trace
    assert all(p % W.BLOCK_SIZE for p in law.prompt.p)
    assert abs(law.pi.sum() - 1.0) < 1e-9


F = load("fitter")


@pytest.mark.exact
def test_the_fitter_is_the_reference_one_on_a_synthetic_sample() -> None:
    from tests.components.inputs import fitter_sample

    law, report = W.fit_law(*fitter_sample())
    d = law.to_dict()
    assert d["prompt"] == F["law"]["prompt"] and d["gen"]["Omega"] == F["law"]["gen"]["Omega"]
    assert d["pi"] == F["law"]["pi"]
    assert (report["prompt_classes"], report["coupling_groups"], report["branches"], report["stages"]) == (
        F["ell_p"], F["ell_c"], F["B"], F["stages"])


@pytest.mark.exact
@pytest.mark.parametrize("trace", ["azure-conversation", "burstgpt", "servegen"])
def test_the_fitter_refits_each_shipped_trace_to_its_shipped_law(trace: str) -> None:
    import os

    if trace != "azure-conversation" and os.environ.get("SERVECAST_SLOW") != "1":
        pytest.skip("minutes: runs with SERVECAST_SLOW=1 (nightly)")
    t = json.loads((DATA / "traces" / f"{trace}.json").read_text(encoding="utf-8"))
    want = json.loads((DATA / "workload_laws" / f"{trace}.json").read_text(encoding="utf-8"))["law"]
    got = W.fit_law(t["input_tokens"], t["output_tokens"])[0].to_dict()
    assert (got["prompt"], got["gen"]["Omega"], got["pi"]) == (want["prompt"], want["gen"]["Omega"], want["pi"])
