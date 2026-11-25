"""The cost laws 2026-10 and 2026-11 against the reference implementation, beat by beat."""

from __future__ import annotations

import pytest

from servecast.core import versions
from servecast.model import geometry
from servecast.model import predict as P
from servecast.model.population.v2026_10 import Beat
from tests.components.inputs import load

pytestmark = pytest.mark.exact

G = load("law")
BEATS = [Beat(prefill=tuple(map(tuple, c["prefill"])), decode=tuple(map(tuple, c["decode"])))
         for c in G["compositions"]]


def _slug(hf: str) -> str:
    return hf.split("/")[-1].lower().removeprefix("meta-")


@pytest.mark.parametrize("case", G["cases"], ids=lambda c: f"{c['gpu']}-{c['law']}-{c['llm'].split('/')[-1]}")
def test_beat_times_are_the_reference_ones(case: dict) -> None:
    law = versions.canonical("cost_law", case["law"])  # the reference names laws by their month
    name = (f"{case['gpu']}/{law}/{_slug(case['calibrated_on'])}" if versions.law(law).calibrated_per == "llm_and_gpu"
            else f"{case['gpu']}/{law}")
    clock = P.clock(P.load_calibration(name), geometry.resolve(case["llm"]))
    got = [clock(b) for b in BEATS]
    assert got == case["seconds"]


def test_a_mixed_beat_reads_the_kv_once_per_query_head() -> None:
    cal = P.load_calibration("a100_sxm_40gb/d9cbf1d3/llama-3-8b")
    clock = P.clock(cal, geometry.resolve("meta-llama/Meta-Llama-3-8B"))
    decode_only = clock(Beat(prefill=(), decode=((4000, 64),)))
    mixed = clock(Beat(prefill=((1, 0),), decode=((4000, 64),)))
    assert mixed > decode_only  # the KV term grows by the GQA group (4) once a prefill chunk joins


def test_a_calibration_refuses_datasheet_numbers_other_than_the_registrys() -> None:
    cal = P.load_calibration("a100_sxm_40gb/d9cbf1d3/llama-3-8b")
    bad = {**cal, "conditions": {**cal["conditions"], "gpu_spec": {**cal["conditions"]["gpu_spec"],
                                                                   "memory_bandwidth_bytes_per_s": 2.0e12}}}
    with pytest.raises(ValueError, match="datasheet"):
        P.clock(bad, geometry.resolve("meta-llama/Meta-Llama-3-8B"))


E = load("eta_c")


@pytest.mark.parametrize("rule", ["interpolation", "curve"])
def test_eta_c_is_read_off_the_grid_as_the_reference_reads_it(rule: str) -> None:
    from servecast.model.law import v2026_11

    grid = {**E["grid"], "rule": rule}
    got = {name: v2026_11.eta_c_for(geometry.resolve(name), grid) for name in E[rule]}
    assert got == E[rule]


def test_the_eta_c_goldens_reach_the_clamps() -> None:
    widths = sorted({p["d_model"] for p in E["grid"]["points"]})
    names = [geometry.resolve(n) for n in E["interpolation"]]
    assert any(g.d_model < widths[0] for g in names) and any(g.d_model > widths[-1] for g in names)
