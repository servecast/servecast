"""The calibration procedures: how profiling data become a calibration. One frozen module per released id; a cost
law has exactly one procedure, with the same id."""

from __future__ import annotations

from types import ModuleType

from servecast.calibrate.procedure import v2026_10, v2026_11
from servecast.core import versions

#: each released module by the name of its entry in components.json
RELEASED: dict[str, ModuleType] = versions.modules("calibration_procedure", (v2026_10, v2026_11))


def get(component_id: str) -> ModuleType:
    """The calibration procedure's module, by its name or the month it was released."""
    name = versions.canonical("calibration_procedure", component_id)
    try:
        return RELEASED[name]
    except KeyError:
        raise ValueError(f"no module for calibration procedure {name} in this version of ServeCast") from None


#: the price of a probe cell, which every procedure shares (it lives in the paper's procedure)
CELL_MS = v2026_10.cell_ms
