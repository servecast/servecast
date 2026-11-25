"""`servecast doctor`: what this computer can run and, for `profile`, what is missing.

The GPU is found through the same interface `profile` uses (`profile.gpu`), so doctor refuses what profile would
refuse, for the same reason.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import sys
from typing import Any

import typer

from servecast import __version__
from servecast.core.deployment import VLLM_VERSION


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def gpu() -> tuple[dict[str, Any] | None, str | None]:
    """The GPU vLLM would run on, and why it cannot be profiled (None when it can)."""
    from servecast.core import gpus
    from servecast.profile import gpu as G

    try:
        backend = G.detect()
    except G.CannotProfileError as error:
        return None, str(error)
    described = backend.describe()
    described["processes"] = len(backend.busy())
    why = backend.unsupported(described)
    if why is None and described["id"] is None:
        why = f"{described['name']} is not in ServeCast's GPU registry; add it to {gpus.user_registry()}"
    return described, why


def serving_blocked(gpu_problem: str | None, vllm: str | None) -> str | None:
    """Why `profile serving` cannot run here, or None."""
    if gpu_problem is not None:
        return gpu_problem
    if vllm is None:
        return f"vLLM {VLLM_VERSION} is not installed; install the profile extra: pip install 'servecast[profile]'"
    if vllm != VLLM_VERSION:
        return f"vLLM {vllm} is installed; the measurement runs {VLLM_VERSION}, the version every calibration uses"
    return None


def main(output_format: str) -> None:
    reference = (platform.system() == "Darwin" and platform.machine() == "arm64" and sys.version_info[:2] == (3, 12)
                 and _version("numpy") == "2.5.0" and _version("scipy") == "1.18.0")
    device, problem = gpu()
    vllm = _version("vllm")
    blocked = serving_blocked(problem, vllm)
    doc: dict[str, Any] = {
        "servecast": __version__, "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}", "numpy": _version("numpy"),
        "scipy": _version("scipy"), "reference_environment": reference, "gpu": device, "vllm": vllm,
        "torch": _version("torch"),
        "profile": {"serving": {"ready": blocked is None, "why_not": blocked},
                    "calibration": {"ready": blocked is None, "why_not": blocked}}}
    if output_format == "json":
        typer.echo(json.dumps(doc, indent=1))
        return
    typer.echo(f"servecast {doc['servecast']} on Python {doc['python']} ({doc['platform']}), numpy {doc['numpy']}, "
               f"scipy {doc['scipy']}")
    if device is None:
        typer.echo("GPU: none that profile can use")
    else:
        busy = f"; in use by {device['processes']} process(es)" if device["processes"] else ""
        known = f" ({device['id']})" if device["id"] else ""
        typer.echo(f"GPU: {device['name']}{known}, {device['memory_bytes'] / 2**30:.0f} GiB, compute capability "
                   f"{device['compute_capability']}, driver {device['driver']}{busy}")
    typer.echo(f"vLLM: {vllm or 'not installed'}; torch: {doc['torch'] or 'not installed'}")
    typer.echo(f"profile serving: {'ready' if blocked is None else 'cannot run here: ' + blocked}")
    typer.echo(f"profile calibration: {'ready' if blocked is None else 'cannot run here: ' + blocked}")
    typer.echo("every other command runs on this computer")
