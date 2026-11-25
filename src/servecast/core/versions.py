"""The components of the predictor, by name.

Every component ServeCast can select is an entry of data/registry/components.json, the one place a name is tied to
its code and to what it computes. A name is eight hexadecimal characters starting with a letter (c62ebe75); it never
changes meaning, and an improvement is a new entry with a new name. Each entry also has the month it was released
(2026-11, or 2026-11b for a second one that month): `versions: <month>` selects every component at its newest entry up
to that month (the paper is 2026-10), and the month stands for the entry wherever a name is expected, as in files
written before names. Unset, `versions` selects each component's default entry.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from types import ModuleType
from typing import Any

from servecast.core.formats import COMPONENT_KINDS, ComponentEntry, ComponentRegistry, CostLawEntry

RELEASE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])([a-z]?)$")

#: components.json, as shipped
REGISTRY: ComponentRegistry = ComponentRegistry.model_validate(json.loads(
    resources.files("servecast").joinpath("data", "registry", "components.json").read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Component:
    name: str                                                  # as printed: "cost law"
    releases: dict[str, str] = field(default_factory=dict)    # each entry's name -> its month, oldest first
    default: str | None = None

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.releases)


#: the registry kind each key of `versions` takes its entries from: a calibration procedure is part of its cost law
KIND_OF = {"cost_law": "cost_law", "calibration_procedure": "cost_law", "population_model": "population_model",
           "workload_fitter": "workload_fitter", "length_sampler": "length_sampler", "estimator": "estimator"}
_SHOWN = {"cost_law": "cost law", "calibration_procedure": "calibration procedure",
          "population_model": "population model", "workload_fitter": "workload fitter",
          "length_sampler": "length sampler", "estimator": "steady-state estimator"}
assert set(KIND_OF.values()) == set(COMPONENT_KINDS)


def sort_key(release: str) -> tuple[int, int, str]:
    match = RELEASE.match(release)
    if match is None:
        raise ValueError(f"not a month: {release!r} (expected e.g. 2026-10, or 2026-11b)")
    return int(match.group(1)), int(match.group(2)), match.group(3)


def entries(key: str) -> dict[str, ComponentEntry]:
    """The entries a key of `versions` selects from, by name, oldest first."""
    found: dict[str, ComponentEntry] = dict(getattr(REGISTRY, KIND_OF[key]))
    return dict(sorted(found.items(), key=lambda item: sort_key(item[1].released)))


COMPONENTS: dict[str, Component] = {
    key: Component(_SHOWN[key], {name: e.released for name, e in entries(key).items()},
                   next(name for name, e in entries(key).items() if e.default))
    for key in KIND_OF}

#: the arXiv paper's components. The length sampler did not exist: the paper used traces. The KV capacity is not a
#: component: it is an input of the manifest, as in the paper
PAPER: dict[str, str] = {key: name for key in KIND_OF for name, e in entries(key).items() if e.in_paper}


def entry(key: str, name: str) -> ComponentEntry:
    return entries(key)[canonical(key, name)]


def law(name: str) -> CostLawEntry:
    found = entry("cost_law", name)
    assert isinstance(found, CostLawEntry)
    return found


def canonical(key: str, value: str, registry: Mapping[str, Component] = COMPONENTS) -> str:
    """The name of the entry `value` stands for: its name, or the month it was released."""
    component = registry[key]
    if value in component.releases:
        return value
    for name, month in component.releases.items():
        if month == value:
            return name
    known = ", ".join(f"{name} ({month})" for name, month in component.releases.items())
    raise ValueError(f"no {component.name} {value}; known: {known}")


def _newest_up_to(component: Component, as_of: str) -> str | None:
    limit = sort_key(as_of)[:2]
    eligible = [(sort_key(month), name) for name, month in component.releases.items() if sort_key(month)[:2] <= limit]
    return max(eligible)[1] if eligible else None


def resolve(versions: str | Mapping[str, Any] | Any | None,
            registry: Mapping[str, Component] = COMPONENTS) -> dict[str, str | None]:
    """Each component's name: the defaults, or the newest up to `as_of`; then the explicit choices."""
    spec: dict[str, Any] = {}
    if isinstance(versions, str):
        spec = {"as_of": versions}
    elif isinstance(versions, Mapping):
        spec = dict(versions)
    elif versions is not None:  # a pydantic VersionsSpec
        spec = versions.model_dump(exclude_none=True)
    as_of = spec.pop("as_of", None)
    resolved = {key: (c.default if as_of is None else _newest_up_to(c, as_of)) for key, c in registry.items()}
    for key, value in spec.items():
        if value is not None:
            resolved[key] = canonical(key, str(value), registry)
    cost_law = resolved.get("cost_law")
    if cost_law is not None:
        procedure = spec.get("calibration_procedure")
        if procedure is not None and canonical("calibration_procedure", str(procedure), registry) != cost_law:
            raise ValueError(f"the calibration procedure is part of its cost law's entry: cost law {cost_law} has its "
                             f"own, not {procedure}")
        resolved["calibration_procedure"] = cost_law
    return resolved


def require(resolved: Mapping[str, str | None], used: Iterable[str], as_of: str | None = None) -> None:
    """Refuse a run that needs a component which has no entry at the chosen month."""
    missing = [COMPONENTS[key].name for key in used if resolved.get(key) is None]
    if missing:
        when = f" up to {as_of}" if as_of else ""
        raise ValueError(f"{', '.join(missing)} has no version{when}; choose a later date in `versions`")


def named(doc: dict[str, Any], source: object) -> dict[str, Any]:
    """A calibration or profiling data with its cost law and procedure by name: a file written before names gives the
    month each was released, which stands for the same entry."""
    try:
        for key in ("cost_law", "calibration_procedure"):
            doc[key] = canonical(key, str(doc[key]))
    except (KeyError, ValueError) as error:
        raise ValueError(f"{source}: {error}") from None
    return doc


def modules(key: str, released: Iterable[ModuleType]) -> dict[str, ModuleType]:
    """A component package's modules by entry name, as components.json's `code` ties them."""
    by_path = {m.__name__: m for m in released}
    return {name: by_path[path] for name, e in entries(key).items() for path in e.code if path in by_path}


PREDICT_TEMPLATE = """\
# A ServeCast manifest. Explain any field with: servecast manifest explain <path>
# Check it with: servecast manifest validate -c <this file>
engine:
  model: Qwen/Qwen2.5-7B             # the LLM to predict (--to)
  hardware: a100_sxm_40gb            # the GPU id; must match the calibration
  workers:
    aggregated:
      parallelism: {{tensor: 1}}       # only 1 in this version
      scheduler: {{}}                  # unset fields take vLLM 0.25.1's defaults
      kv_cache:
        capacity: {{blocks: 23488}}    # KV-cache blocks: vLLM logs its own count at start-up as 'GPU KV
                                     # cache size: T tokens' (T / block_size); or impose one with
                                     # --num-gpu-blocks-override
calibration:
  use: a100_sxm_40gb/{law}         # a calibration name or path (--with); or `data: <profiling-data dir>`
  transfer: true                     # the LLM is not the one the calibration was fitted on (here, synthetic models)
versions: {month}                    # every component at its newest version up to this date
traffic:
  source: {{type: trace, paths: [azure-conversation]}}   # a shipped trace (azure-conversation, burstgpt,
                                                     # servegen) or your files, with format: mooncake or lengths
  # source: {{type: synthetic, input_tokens: {{mean: 1200}}, output_tokens: {{mean: 250, variance: 40000}}}}
  load: {{type: concurrency, concurrency: 64, think_time_ms: 1000}}   # N clients, mean think time
  stop: {{steady_state: {{}}}}            # until every metric's 95 % interval is narrow enough
"""

PROFILE_TEMPLATE = """\
# A ServeCast manifest for the GPU host:
#   servecast profile calibration -c <this file>    (what the cost law needs on this GPU)
#   servecast profile serving -c <this file>        (the workload, served and measured)
engine:
  model: Qwen/Qwen2.5-7B             # the LLM served; also the LLM probed by the per-LLM cost law
  hardware: a100_sxm_40gb
  workers:
    aggregated:
      parallelism: {{tensor: 1}}
      kv_cache: {{capacity: {{blocks: 23488}}}}   # imposed on vLLM with --num-gpu-blocks-override
versions: {month}                    # selects the cost law, hence what `profile calibration` measures
traffic:
  source: {{type: trace, paths: [azure-conversation]}}   # the same requests the prediction draws from
  load: {{type: concurrency, concurrency: 64, think_time_ms: 1000}}
  stop: {{steady_state: {{}}}}
profile:
  clock: auto                        # locked where the machine allows it; measured never locks; or {{lock: <MHz>}}
  poweroff_when_safe: false
"""


def newest_release() -> str:
    """The newest month any entry was released in. A new manifest pins it, so the manifest means the same, and names a
    calibration that ships, on any later day (it named the current month until 2026-11, and would have named a
    calibration that does not exist on 1 December)."""
    return max((e.released for key in KIND_OF for e in entries(key).values()), key=sort_key)[:7]


def template(kind: str) -> str:
    """A starter manifest: the newest release, and the A100 calibration of the default cost law (calibrations are
    named <gpu>/<cost law>[/<llm>])."""
    text = {"predict": PREDICT_TEMPLATE, "profile": PROFILE_TEMPLATE}[kind]
    return text.format(month=newest_release(), law=COMPONENTS["cost_law"].default)
