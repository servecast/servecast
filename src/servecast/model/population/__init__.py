"""The population model with eviction: continuous batching as a population process over the engine's iterations.

One frozen module per released id.
"""

from __future__ import annotations

from types import ModuleType

from servecast.core import versions
from servecast.model.population import v2026_10

#: each released module by the name of its entry in components.json
RELEASED: dict[str, ModuleType] = versions.modules("population_model", (v2026_10,))


def get(component_id: str) -> ModuleType:
    """The population model's module, by its name or the month it was released."""
    name = versions.canonical("population_model", component_id)
    try:
        return RELEASED[name]
    except KeyError:
        raise ValueError(f"no module for population model {name} in this version of ServeCast") from None
