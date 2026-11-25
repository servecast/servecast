"""The workload law: how the population model draws the type of each arriving request.

One frozen module per released id of the workload fitter.
"""

from __future__ import annotations

from types import ModuleType

from servecast.core import versions
from servecast.model.workload import v2026_10

#: each released module by the name of its entry in components.json
RELEASED: dict[str, ModuleType] = versions.modules("workload_fitter", (v2026_10,))


def get(component_id: str) -> ModuleType:
    """The workload fitter's module, by its name or the month it was released."""
    name = versions.canonical("workload_fitter", component_id)
    try:
        return RELEASED[name]
    except KeyError:
        raise ValueError(f"no module for workload fitter {name} in this version of ServeCast") from None
