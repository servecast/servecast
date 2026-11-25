"""The GPU registry: the datasheet numbers the cost laws divide by. Never a measurement; a calibration says what
fraction of them a GPU reaches.

The GPUs ServeCast ships calibrations for are in data/registry/gpus.json. A gpus.json in your data directory (same
format, `servecast manifest schema --kind gpus`) adds GPUs of your own; it cannot change a shipped one, whose numbers
the shipped calibrations were fitted with.
"""

from __future__ import annotations

import dataclasses
import json
from importlib import resources
from pathlib import Path

from pydantic import ValidationError

from servecast.core.formats import GpuRegistry
from servecast.core.paths import data_dir


@dataclasses.dataclass(frozen=True)
class Gpu:
    id: str
    device_name: str            # as the driver reports it
    memory_gib: int
    n_sm: int
    peak_flops: float           # dense bf16 tensor-core peak at the boost clock, FLOP/s
    bandwidth: float            # HBM/GDDR datasheet bandwidth, B/s
    boost_sm_mhz: int


def read(path: Path) -> dict[str, Gpu]:
    """The GPUs a gpus.json lists, by id; ValueError names the file and the problem."""
    try:
        doc = GpuRegistry.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as error:
        raise ValueError(f"{path}: not a GPU registry: {error}") from None
    return {gpu_id: Gpu(gpu_id, e.device_name, memory_gib=e.memory_gib, n_sm=e.sm_count, peak_flops=e.peak_flop_per_s,
                        bandwidth=e.memory_bandwidth_bytes_per_s, boost_sm_mhz=e.boost_clock_mhz)
            for gpu_id, e in doc.gpus.items()}


#: the GPUs ServeCast ships calibrations for
SHIPPED: dict[str, Gpu] = read(Path(str(resources.files("servecast").joinpath("data", "registry", "gpus.json"))))


def user_registry() -> Path:
    return data_dir() / "gpus.json"


def registry() -> dict[str, Gpu]:
    """The shipped GPUs, then yours."""
    path = user_registry()
    if not path.is_file():
        return dict(SHIPPED)
    yours = read(path)
    clash = sorted(set(yours) & set(SHIPPED))
    if clash:
        raise ValueError(f"{path}: {', '.join(clash)} ship with ServeCast and cannot be changed; give your GPU "
                         "another id")
    return {**SHIPPED, **yours}


def get(gpu_id: str) -> Gpu:
    known = registry()
    try:
        return known[gpu_id]
    except KeyError:
        raise ValueError(f"unknown GPU {gpu_id!r}; known: {', '.join(known)} (add one in {user_registry()})") from None
