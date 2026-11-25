"""The cost laws: how long one engine iteration (a beat) lasts, given what it computes.

One frozen module per released id. Every term is a count of bytes or FLOPs from the LLM's geometry over a fitted
efficiency, or a fitted latency, so a calibration moves to another LLM by changing the counts only.
"""

from __future__ import annotations

from types import ModuleType

from servecast.core import versions
from servecast.model.law import v2026_10, v2026_11

#: each released module by the name of its entry in components.json
RELEASED: dict[str, ModuleType] = versions.modules("cost_law", (v2026_10, v2026_11))


def get(component_id: str) -> ModuleType:
    """The cost law's module, by its name or the month it was released."""
    name = versions.canonical("cost_law", component_id)
    try:
        return RELEASED[name]
    except KeyError:
        raise ValueError(f"no module for cost law {name} in this version of ServeCast") from None
