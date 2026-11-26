"""`servecast compare`: a prediction against a measurement of the same manifest, and comparison.json.

Exit statuses: 0 scored (flags are printed, not failures); 2 the two directories are not a prediction and a
measurement of the same experiment, or cannot be read (what differs is named); 3 the comparison cannot be written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import typer

LABELS = {"mean_ttft_ms": "mean TTFT ms", "p95_ttft_ms": "P95 TTFT ms", "mean_tpot_ms": "mean TPOT ms",
          "p95_tpot_ms": "P95 TPOT ms", "request_throughput_rps": "requests/s",
          "output_throughput_tok_s": "output tokens/s"}


def fail(status: int, message: str) -> NoReturn:
    typer.echo(f"servecast compare: {message}", err=True)
    raise typer.Exit(status)


def _f(v: float | None, width: int, digits: int, sign: bool = False) -> str:
    if v is None:
        return "-".rjust(width)
    return format(v, f"{'+' if sign else ''}{width}.{digits}f")


def print_table(doc: dict[str, Any]) -> None:
    typer.echo(f"{'clients':>7} {'think ms':>8}  {'metric':<16} {'predicted':>10} {'measured':>10} "
               f"{'measured 95 % CI':>23} {'error %':>8}  in CI")
    for point in doc["points"]:
        load = point["load"]
        for name, label in LABELS.items():
            c = point["metrics"][name]
            ci = c.get("measured_ci95")
            interval = f"[{ci['low']:>9.2f}, {ci['high']:>9.2f}]" if ci else "-"
            within = {True: "yes", False: "no", None: "-"}[c["within_ci"]]
            typer.echo(f"{load['concurrency']:>7} {load['think_time_ms']:>8.0f}  {label:<16} "
                       f"{_f(c['predicted'], 10, 2)} {_f(c['measured'], 10, 2)} {interval:>23} "
                       f"{_f(c['signed_error_pct'], 8, 1, sign=True)}  {within}")
    for flag in doc["flags"]:
        typer.echo(f"flag: {flag}")


def _is_comparison(path: Path) -> bool:
    """An earlier comparison, which compare may replace: anything else at that path is someone's data."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and str(doc.get("schema", "")).startswith("servecast-comparison/")


def main(prediction_dir: Path, measurement_dir: Path, output: Path | None, output_format: str) -> None:
    from servecast.compare import score as S

    try:
        doc = S.score(prediction_dir, measurement_dir)
    except S.NotComparableError as error:
        fail(2, str(error))
    path = output if output is not None else prediction_dir / "comparison.json"
    replaced = path.exists()
    # a file the user names that is not an earlier comparison: --output prediction.json destroyed the prediction
    if output is not None and path.is_file() and not _is_comparison(path):
        fail(3, f"--output: {path} exists and is not a comparison; give a new path, or remove it first")
    try:
        path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    except OSError as error:
        fail(3, f"{'--output: ' if output is not None else ''}cannot write {path}: {error.strerror}")
    if output_format == "json":
        typer.echo(json.dumps(doc, indent=1))
    else:
        print_table(doc)
    typer.echo(f"wrote {path}{' (it replaces the earlier comparison)' if replaced else ''}", err=True)
