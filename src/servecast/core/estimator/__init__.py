"""The steady-state estimator: when a closed-loop run is long enough, and the window its numbers come from.

One frozen module per released id. The prediction and the measurement both stop by this rule, so a comparison between
them never compares two stopping rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType

from servecast.core import versions
from servecast.core.estimator import v2026_10, v2026_11

#: each released module by the name of its entry in components.json
RELEASED: dict[str, ModuleType] = versions.modules("estimator", (v2026_10, v2026_11))
#: each target of the stop rule, by the manifest's name, and the series the estimators judge against it
SERIES = {"throughput": "inv_X", "e2e": "R", "ttft": "TTFT", "tpot": "TPOT", "ttft_p95": "TTFT_p95",
          "tpot_p95": "TPOT_p95"}


def series_targets(targets: Mapping[str, float]) -> dict[str, float]:
    """The stop rule's targets under the series names the estimators judge, the means first, in a fixed order. The
    prediction and the measurement both translate through this, so both judge the same series against the same
    targets."""
    unknown = set(targets) - set(SERIES)
    if unknown:
        raise ValueError(f"unknown stop targets {sorted(unknown)}; known: {', '.join(SERIES)}")
    return {SERIES[k]: float(targets[k]) for k in SERIES if k in targets}


def get(component_id: str) -> ModuleType:
    """The steady-state estimator's module, by its name or the month it was released."""
    name = versions.canonical("estimator", component_id)
    try:
        return RELEASED[name]
    except KeyError:
        raise ValueError(f"no module for steady-state estimator {name} in this version of ServeCast") from None
