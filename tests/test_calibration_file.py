"""The calibration file, servecast-calibration/2: every shipped one follows its documented format, a file of the first
format reads as its conversion with the same numbers, and what predict refuses."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from servecast.core import calibration as C
from servecast.core import formats, gpus
from servecast.model import geometry
from servecast.model import predict as P


def _shipped() -> dict[str, dict[str, Any]]:
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in P.shipped_calibrations().items()}


def test_every_shipped_calibration_follows_the_documented_format() -> None:
    shipped = _shipped()
    assert len(shipped) == 19
    for name, doc in shipped.items():
        formats.CalibrationFile.model_validate(doc)
        assert doc["schema"] == C.SCHEMA and doc["name"] == name


def _first_format(doc: dict[str, Any]) -> dict[str, Any]:
    """The servecast-calibration/1 file a /2 one was converted from, written out by hand (the inverse of upgrade)."""
    law, cond = doc["law"], doc["conditions"]
    old = {"schema": C.V1, "name": doc["name"], "gpu": cond["gpu"], "tensor_parallel": cond["tensor_parallel"],
           "cost_law": law["cost_law"], "calibration_procedure": law["cost_law"], "calibrated_on": cond["fitted_on"],
           "parameters": {"names": list(law["parameters"]), "values": list(law["parameters"].values())},
           "clock": {"reference_mhz": cond["clock_mhz"],
                     "peak_fraction": cond["clock_mhz"] / cond["gpu_spec"]["boost_clock_mhz"]},
           "datasheet": {"peak_flops": cond["gpu_spec"]["peak_flop_per_s"],
                         "bandwidth": cond["gpu_spec"]["memory_bandwidth_bytes_per_s"]}}
    if "eta_c_grid" in law:
        old["eta_c_grid"] = law["eta_c_grid"]
    return old


@pytest.mark.parametrize("name", ["a100_sxm_40gb/c62ebe75", "h100_sxm/d9cbf1d3/llama-3-8b"])
def test_a_first_format_file_reads_as_its_conversion_with_the_same_numbers(name: str) -> None:
    doc = _shipped()[name]
    assert C.parse(_first_format(doc), "old.json") == doc


def test_a_first_format_file_with_months_reads_by_name() -> None:
    old = {**_first_format(_shipped()["a100_sxm_40gb/c62ebe75"]), "cost_law": "2026-11",
           "calibration_procedure": "2026-11"}
    assert C.parse(old, "old.json")["law"]["cost_law"] == "c62ebe75"


def test_a_first_format_file_whose_fraction_is_not_its_clock_over_the_boost_is_refused() -> None:
    old = _first_format(_shipped()["a100_sxm_40gb/c62ebe75"])
    old["clock"]["peak_fraction"] = 0.85
    with pytest.raises(ValueError, match=r"old\.json: .*not its clock over a100_sxm_40gb's boost clock"):
        C.parse(old, "old.json")


@pytest.mark.parametrize(("path", "value", "refusal"), [
    (("law", "cost_law"), "abcdef12", "no cost law abcdef12"),
    (("law", "cost_law"), "2026-11", "law.cost_law"),
    (("conditions", "clock_mhz"), 0, "conditions.clock_mhz"),
    (("schema",), "servecast-calibration/3", "reads servecast-calibration/2"),
])
def test_a_file_that_is_not_a_calibration_is_refused_naming_it(path: tuple[str, ...], value: object,
                                                              refusal: str) -> None:
    doc = copy.deepcopy(_shipped()["a100_sxm_40gb/c62ebe75"])
    target = doc
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=rf"(?s)bad\.json: not a calibration .*{refusal}"):
        C.parse(doc, "bad.json")


@pytest.mark.parametrize("field", ["peak_flop_per_s", "memory_bandwidth_bytes_per_s", "boost_clock_mhz"])
def test_a_calibration_fitted_with_other_registry_numbers_is_refused(field: str) -> None:
    doc = copy.deepcopy(_shipped()["a100_sxm_40gb/c62ebe75"])
    doc["conditions"]["gpu_spec"][field] *= 2
    with pytest.raises(ValueError, match="datasheet numbers other than the registry's"):
        P.clock(doc, geometry.KNOWN["Qwen/Qwen2.5-7B"])


def test_parameters_in_another_order_are_refused() -> None:
    doc = copy.deepcopy(_shipped()["a100_sxm_40gb/c62ebe75"])
    doc["law"]["parameters"] = dict(reversed(list(doc["law"]["parameters"].items())))
    with pytest.raises(ValueError, match="cost law c62ebe75 has"):
        P.clock(doc, geometry.KNOWN["Qwen/Qwen2.5-7B"])


def test_the_peak_fraction_is_the_clock_over_the_boost_clock() -> None:
    for doc in _shipped().values():
        gpu = gpus.get(doc["conditions"]["gpu"])
        assert C.peak_fraction(doc) == doc["conditions"]["clock_mhz"] / gpu.boost_sm_mhz


def test_the_converter_changes_nothing_on_converted_files(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("convert", Path(__file__).resolve().parents[1] / "tools"
                                                  / "convert_calibrations.py")
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    old = tmp_path / "old.json"
    old.write_text(json.dumps(_first_format(_shipped()["a100_sxm_40gb/c62ebe75"]), indent=1) + "\n", encoding="utf-8")
    assert tool.convert(old) is True
    assert json.loads(old.read_text(encoding="utf-8")) == _shipped()["a100_sxm_40gb/c62ebe75"]
    assert tool.convert(old) is False
