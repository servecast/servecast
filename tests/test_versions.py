"""The components by name: how `versions` resolves them, and the one registry they come from."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from servecast.calibrate import fit
from servecast.calibrate import procedure as procedures
from servecast.core import estimator as estimators
from servecast.core import sampler as samplers
from servecast.core import versions as v
from servecast.model import law as laws
from servecast.model import population as populations
from servecast.model import predict as P
from servecast.model import workload as workloads

#: every entry of components.json, pinned: a name never changes meaning, so neither its kind, its month nor its code
#: may move (the golden tests pin the numbers each one gives)
ENTRIES = {
    "d9cbf1d3": ("cost_law", "2026-10", ["servecast.model.law.v2026_10", "servecast.calibrate.procedure.v2026_10"]),
    "c62ebe75": ("cost_law", "2026-11", ["servecast.model.law.v2026_11", "servecast.calibrate.procedure.v2026_11"]),
    "f2f5d726": ("population_model", "2026-10", ["servecast.model.population.v2026_10"]),
    "cd88bc61": ("workload_fitter", "2026-10", ["servecast.model.workload.v2026_10"]),
    "fe3db391": ("length_sampler", "2026-11", ["servecast.core.sampler.v2026_11"]),
    "cabeb968": ("estimator", "2026-10", ["servecast.core.estimator.v2026_10"]),
    "c3ad4006": ("estimator", "2026-11", ["servecast.core.estimator.v2026_11"]),
}
PACKAGES = {"cost_law": laws, "calibration_procedure": procedures, "population_model": populations,
            "workload_fitter": workloads, "length_sampler": samplers, "estimator": estimators}


def test_a_name_never_changes_meaning() -> None:
    found = {name: (kind, e.released, e.code) for kind in ("cost_law", "population_model", "workload_fitter",
                                                           "length_sampler", "estimator")
             for name, e in v.entries(kind).items()}
    assert found == ENTRIES


@pytest.mark.parametrize("key", list(PACKAGES))
def test_each_package_has_a_module_for_every_entry_and_an_entry_for_every_module(key: str) -> None:
    package = PACKAGES[key]
    assert set(package.RELEASED) == set(v.entries(key))
    on_disk = {p.stem for p in Path(str(package.__file__)).parent.glob("v20*.py")}
    assert {m.__name__.rsplit(".", 1)[1] for m in package.RELEASED.values()} == on_disk


def test_every_shipped_calibration_is_named_after_its_cost_law() -> None:
    for name, path in P.shipped_calibrations().items():
        law = json.loads(path.read_text(encoding="utf-8"))["law"]["cost_law"]
        assert name.split("/")[1] == law, name


def test_every_cost_law_has_its_form_in_the_fit() -> None:
    assert set(fit.FORM_OF_LAW) == set(v.entries("cost_law"))


def test_a_file_written_before_names_reads_as_the_same_entries() -> None:
    doc = v.named({"cost_law": "2026-11", "calibration_procedure": "2026-11"}, "old.json")
    assert doc == {"cost_law": "c62ebe75", "calibration_procedure": "c62ebe75"}
    with pytest.raises(ValueError, match=r"old\.json: no cost law 2027-01"):
        v.named({"cost_law": "2027-01", "calibration_procedure": "2027-01"}, "old.json")


def test_there_are_six_components_and_no_kv_capacity_estimator() -> None:
    # The KV capacity is an input of the manifest, as in the paper (decision of 2026-11-18): nothing estimates it.
    assert len(v.COMPONENTS) == 6 and "kv_capacity" not in v.COMPONENTS


def test_omitted_means_the_defaults_whatever_the_date() -> None:
    assert v.resolve(None) == {"cost_law": "c62ebe75", "calibration_procedure": "c62ebe75",
                               "population_model": "f2f5d726", "workload_fitter": "cd88bc61",
                               "length_sampler": "fe3db391", "estimator": "c3ad4006"}


def test_the_papers_date_gives_exactly_the_papers_components() -> None:
    assert v.PAPER == {"cost_law": "d9cbf1d3", "calibration_procedure": "d9cbf1d3", "population_model": "f2f5d726",
                       "workload_fitter": "cd88bc61", "estimator": "cabeb968"}
    resolved = v.resolve("2026-10")
    assert {key: resolved[key] for key in v.PAPER} == v.PAPER
    assert resolved["length_sampler"] is None


def test_a_date_before_any_version_resolves_nothing() -> None:
    assert all(value is None for value in v.resolve("2025-01").values())


@pytest.mark.parametrize("law", ["2026-11", "c62ebe75"])
def test_an_override_wins_by_name_or_month_and_the_procedure_follows_the_law(law: str) -> None:
    resolved = v.resolve({"as_of": "2026-10", "cost_law": law})
    assert resolved["cost_law"] == resolved["calibration_procedure"] == "c62ebe75"
    assert resolved["estimator"] == "cabeb968"


def test_a_procedure_that_contradicts_the_law_is_refused() -> None:
    with pytest.raises(ValueError, match="part of its cost law"):
        v.resolve({"cost_law": "2026-10", "calibration_procedure": "2026-11"})


@pytest.mark.parametrize("unknown", ["2026-12", "abcdef12"])
def test_an_unknown_component_is_refused_with_the_known_ones(unknown: str) -> None:
    with pytest.raises(ValueError, match=f"no steady-state estimator {unknown}; known: cabeb968 \\(2026-10\\), "
                                         "c3ad4006 \\(2026-11\\)"):
        v.resolve({"estimator": unknown})


def test_months_order_by_month_then_letter() -> None:
    ordered = ["2026-10", "2026-11", "2026-11b", "2026-12"]
    assert sorted(reversed(ordered), key=v.sort_key) == ordered


def test_a_same_month_letter_is_the_newer_entry() -> None:
    estimator = v.Component("steady-state estimator", {"cabeb968": "2026-10", "c3ad4006": "2026-11",
                                                       "e0000001": "2026-11b"}, "c3ad4006")
    assert v.resolve("2026-11", registry={**v.COMPONENTS, "estimator": estimator})["estimator"] == "e0000001"


def test_a_component_used_but_absent_at_the_date_is_refused() -> None:
    paper = v.resolve("2026-10")
    with pytest.raises(ValueError, match="length sampler has no version up to 2026-10"):
        v.require(paper, {"length_sampler"}, as_of="2026-10")
