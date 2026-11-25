"""The length sampler: a mean-and-variance workload becomes a concrete pool of (input, output) lengths.

One frozen module per released id. The prediction fits its workload law to the pool exactly as the paper fits a trace,
and the measurement draws its requests from the same pool.
"""

from __future__ import annotations

from types import ModuleType

from servecast.core import versions
from servecast.core.sampler import v2026_11

#: each released module by the name of its entry in components.json
RELEASED: dict[str, ModuleType] = versions.modules("length_sampler", (v2026_11,))


def get(component_id: str) -> ModuleType:
    """The length sampler's module, by its name or the month it was released."""
    name = versions.canonical("length_sampler", component_id)
    try:
        return RELEASED[name]
    except KeyError:
        raise ValueError(f"no module for length sampler {name} in this version of ServeCast") from None
