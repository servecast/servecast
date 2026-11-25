"""`servecast workload describe | sample`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import typer

from servecast.core import pool as pools


def _read_any(path: Path) -> tuple[list[int], list[int], str]:
    if not path.exists() and str(path) in pools.SHIPPED_TRACES:
        doc = pools.shipped_trace(str(path))
        return doc["input_tokens"], doc["output_tokens"], f"{doc['source']} ({doc['license']})"
    if not path.is_file():
        typer.echo(f"servecast workload describe: {path}: no such file (shipped traces: "
                   f"{', '.join(pools.SHIPPED_TRACES)})", err=True)
        raise typer.Exit(2)
    first = path.read_text(encoding="utf-8").lstrip()[:2000]
    fmt = "mooncake" if '"input_length"' in first else "lengths"
    try:
        p, g, warnings = pools.read_trace(path, fmt)
    except ValueError as error:
        typer.echo(f"servecast workload describe: {error}", err=True)
        raise typer.Exit(2) from error
    for w in warnings:
        typer.echo(f"warning: {w}", err=True)
    return p, g, f"{path.name} ({fmt})"


def _stats(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(x, [0.5, 0.9, 0.99])
    return {"mean": float(x.mean()), "variance": float(x.var()), "cv": float(x.std() / x.mean()),
            "p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), "max": float(x.max())}


def describe(path: Path, output_format: str) -> None:
    p, g, label = _read_any(path)
    a, b = np.asarray(p, float), np.asarray(g, float)
    doc: dict[str, Any] = {"trace": label, "requests": len(p), "input_tokens": _stats(a), "output_tokens": _stats(b),
           "correlation": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None}
    if output_format == "json":
        typer.echo(json.dumps(doc, indent=1))
        return
    typer.echo(f"{label}: {len(p)} requests")
    typer.echo(f"{'':14}{'mean':>10}{'variance':>14}{'cv':>7}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>8}")
    for name in ("input_tokens", "output_tokens"):
        s = doc[name]
        typer.echo(f"{name:14}{s['mean']:>10.1f}{s['variance']:>14.1f}{s['cv']:>7.2f}{s['p50']:>8.0f}"
                   f"{s['p90']:>8.0f}{s['p99']:>8.0f}{s['max']:>8.0f}")
    if doc["correlation"] is not None:
        typer.echo(f"correlation of input and output lengths: {doc['correlation']:.3f}")


def sample(config: Path, sets: list[str], out: Path, output_format: str) -> None:
    from servecast.cli.predict_cmd import load_manifest
    from servecast.core import versions
    from servecast.model import run as R

    m = load_manifest(config, sets, [])
    if m.traffic is None:
        typer.echo("servecast workload sample: traffic: required", err=True)
        raise typer.Exit(2)
    try:
        resolved = {k: v for k, v in versions.resolve(m.versions).items() if v is not None}
    except ValueError as error:
        typer.echo(f"servecast workload sample: versions: {error}", err=True)
        raise typer.Exit(2) from None
    try:
        llm = R.resolve_llm(m.engine.model) if m.engine.model else None
        context = (llm.max_context if (llm and m.engine.context_length == "max")
                   else (int(m.engine.context_length) if m.engine.context_length != "max" else 1 << 30))
        p, g, header, warnings, _cap = R.resolve_pool(m, resolved, context)
    except (R.ManifestError, KeyError) as error:
        message = str(error) if not isinstance(error, KeyError) else "versions: no length sampler at this date"
        typer.echo(f"servecast workload sample: {message}", err=True)
        raise typer.Exit(2) from error
    for w in warnings:
        typer.echo(f"warning: {w}", err=True)
    sha = pools.write(out, p, g, header)
    if output_format == "json":
        typer.echo(json.dumps({"pool": str(out), "requests": len(p), "sha256": sha, **header}, indent=1))
        return
    typer.echo(f"wrote {out}: {len(p)} requests, sha256 {sha}")
    if header.get("source") == "synthetic":
        for side in ("input", "output"):
            req, real = header["requested"][side], header["realized"][side]
            typer.echo(f"{side:7} requested mean {req['mean']:.1f} variance {req['variance']:.1f}   "
                       f"realized mean {real['mean']:.1f} variance {real['variance']:.1f}")
