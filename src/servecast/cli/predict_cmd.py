"""`servecast predict`: the manifest in, prediction.json out, and a summary on standard output.

Exit statuses (the protocol of the command line): 0 ok; 1 the computation failed; 2 the manifest or the options
cannot be run (every problem named, with its fix); 3 a resource refused (an output directory in use); 124 a load
point reached --timeout before converging; 130 interrupted.
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import typer
import yaml

from servecast.core import manifest as mf
from servecast.core.paths import data_dir

KNOWN_OUTPUTS = ("prediction.json", "pool.jsonl", "requests.jsonl", "manifest.yaml", "run.json")
#: files that describe an earlier prediction in the same directory and would no longer match a new one
STALE = ("requests.jsonl", "comparison.json")


def fail(status: int, message: str, command: str = "predict") -> NoReturn:
    typer.echo(f"servecast {command}: {message}", err=True)
    raise typer.Exit(status)


def load_manifest(path: Path, sets: list[str], extra_sets: list[str], command: str = "predict") -> mf.Manifest:
    path = mf.find(path)
    unreadable = mf.config_problem(path)  # a directory or standard input used to be "no such file"
    if unreadable:
        fail(2, unreadable, command)
    try:
        raw = mf.load(path)
    except yaml.YAMLError as error:
        fail(2, f"{path}: not valid YAML: {error}", command)
    if not isinstance(raw, dict):
        fail(2, f"{path}: a manifest is a YAML mapping", command)
    raw, problems = mf.apply_sets(raw, [*sets, *extra_sets])
    manifest, invalid = mf.validate(raw)
    problems += invalid
    if problems or manifest is None:
        for problem in problems:
            typer.echo(f"{path}: {problem}", err=True)
        raise typer.Exit(2)
    return manifest


#: the manifest field `--to` leaves as it is, although vLLM sizes it per LLM
CAPACITY = "engine.workers.aggregated.kv_cache.capacity"


def carried_capacity(to: str | None, sets: list[str], original: str | None) -> str | None:
    """The warning for `--to` with the manifest's KV-cache block count: vLLM sizes the cache per LLM, so a count set
    for one LLM is not the engine's count for another."""
    if not to or to == original or any(s.split("=", 1)[0].strip().startswith(CAPACITY) for s in sets):
        return None
    return (f"--to {to}: the KV-cache block count comes from the manifest, where it was set for "
            f"{original or 'another LLM'}; vLLM sizes the cache per LLM, so give {to}'s own count with --set "
            f"{CAPACITY}.blocks=N (N = the 'GPU KV cache size' vLLM logs for it, divided by the block size)")


def find_calibration(name: str) -> tuple[dict[str, Any], str, bytes]:
    """A shipped calibration, one of the user's (saved by `servecast calibrate`), or a file. ManifestError when there
    is none, or the file is not a calibration: `manifest validate` reports it as predict does."""
    from servecast.core import calibration as calibrations
    from servecast.model import predict as P

    shipped = P.shipped_calibrations()
    user = data_dir() / "calibrations" / f"{name}.json"
    for path in (shipped.get(name), user if user.is_file() else None, Path(name)):
        if path is not None and Path(path).is_file():
            body = Path(path).read_bytes()
            try:
                doc = json.loads(body)
            except ValueError as error:  # not JSON, or not UTF-8
                raise mf.ManifestError(f"calibration.use: {path} is not valid JSON: {error}") from None
            try:
                doc = calibrations.parse(doc, path)  # servecast-calibration/2; a /1 file is converted
            except ValueError as error:
                raise mf.ManifestError(f"calibration.use: {error}") from None
            return doc, str(doc.get("name") or name), body
    # the shipped example names my-a100, which Get started saves first: say how to save a calibration under a name
    how = "" if name.endswith(".json") else (f"; `servecast calibrate --data <profiling data> --name {name}` saves one "
                                             "of yours under that name")
    raise mf.ManifestError(f"calibration.use: no calibration {name!r}; shipped: {', '.join(shipped)}{how} "
                           "(servecast list calibrations shows yours too)")


def fit_on_the_fly(manifest: mf.Manifest) -> tuple[dict[str, Any], str, bytes]:
    """The calibration of the manifest's profiling data: fitted now, or read from the cache."""
    from servecast.calibrate import fit
    from servecast.core import versions

    try:  # the profiling data name their cost law; versions, if set, must select the same
        law = None if manifest.versions is None else versions.resolve(manifest.versions)["cost_law"]
        typer.echo("servecast predict: fitting the calibration (cached afterwards)", err=True)
        doc, body = fit.cached(Path(str(manifest.calibration.data)), law)
    except ValueError as error:
        raise mf.ManifestError(f"calibration.data: {error}") from None
    sha = str(doc["fit"]["profiling_sha256"])
    typer.echo(f"servecast predict: calibration {doc['name']} (profiling data sha256 {sha[:12]}); "
               f"save it with `servecast calibrate` to reuse it with --with", err=True)
    return doc, f"{doc['name']} (fitted from profiling data)", body


def resolve_calibration(manifest: mf.Manifest) -> tuple[dict[str, Any], str, bytes]:
    """The manifest's calibration, with the name a prediction records and the file's bytes: fitted from its profiling
    data (cached), or found by name or path. ManifestError when there is none."""
    if manifest.calibration.data is not None:
        return fit_on_the_fly(manifest)
    if manifest.calibration.use is None:
        raise mf.ManifestError("calibration: give `use` (a calibration name or path, or --with) or `data` (profiling "
                               "data)")
    return find_calibration(str(manifest.calibration.use))


def parse_sweep(texts: list[str] | None, command: str = "predict") -> tuple[str, list[str]] | None:
    if not texts:
        return None
    if len(texts) > 1:  # one field is swept per run; Typer used to keep only the last --sweep
        fail(2, f"--sweep: one field per run, got {len(texts)} ({', '.join(t.partition('=')[0] for t in texts)}); "
                "sweep the other field in a second run, or fix it with --set", command)
    text = texts[0]
    field, sep, values = text.partition("=")
    items = [v.strip() for v in values.split(",") if v.strip()]
    if not sep or not field or not items:
        fail(2, f"--sweep: expected PATH=V1,V2,..., got {text!r}", command)
    return field, items


def print_table(doc: dict[str, Any]) -> None:
    head = (f"{'clients':>8} {'think ms':>9} {'TTFT ms':>9} {'P95 TTFT':>9} {'TPOT ms':>8} {'P95 TPOT':>9} "
            f"{'req/s':>8} {'tok/s':>9}  converged")
    typer.echo(f"calibration: {doc['provenance']['calibration']}")
    typer.echo(head)

    def f(v: float | None, fmt: str) -> str:
        return format(v, fmt) if v is not None else "-".rjust(int(fmt.split(".")[0]))

    for point in doc["points"]:
        s, load = point["summary"], point["load"]
        typer.echo(f"{load['concurrency']:>8} {load['think_time_ms']:>9.0f} {f(s['mean_ttft_ms'], '9.1f')} "
                   f"{f(s['p95_ttft_ms'], '9.1f')} {f(s['mean_tpot_ms'], '8.2f')} {f(s['p95_tpot_ms'], '9.2f')} "
                   f"{f(s['request_throughput_rps'], '8.3f')} {f(s['output_throughput_tok_s'], '9.1f')}  "
                   f"{'yes' if point['converged'] else 'NO (' + point['stop_reason'] + ')'}")


@contextlib.contextmanager
def _removed_if_left_empty(directory: Path, created: bool) -> Iterator[None]:
    """A prediction refused after it made its output folder leaves no empty folder behind (it used to)."""
    try:
        yield
    except BaseException:
        if created and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
        raise


def main(config: Path, *, with_: str | None, to: str | None, sets: list[str], sweep: list[str] | None,
         output_dir: Path, overwrite: bool, output_format: str, capture_per_request: bool, jobs: int,
         timeout: float | None) -> None:
    from servecast import __version__
    from servecast.core import runrecord
    from servecast.model import run as R

    started = runrecord.now()
    config = mf.find(config)  # a file, or a shipped manifest's name: run.json records the file read
    extra = ([f"calibration.use={with_}"] if with_ else [])
    extra += [f"engine.model={to}", "calibration.transfer=true"] if to else []  # another LLM: a transfer
    manifest = load_manifest(config, sets, extra)
    original = None
    if to:  # the LLM the manifest names before --to, for the warning on its block count
        raw, _ = mf.apply_sets(mf.load(config), sets)
        original = (raw.get("engine") or {}).get("model") if isinstance(raw, dict) else None
    if manifest.calibration.use is None and manifest.calibration.data is None:
        fail(2, "calibration: give `use` (a calibration name or path, or --with) or `data` (profiling data)")
    known = [p for p in output_dir.iterdir()] if output_dir.is_dir() else []
    if known and not overwrite:
        fail(3, f"--output-dir: {output_dir} is not empty; give --overwrite to replace {', '.join(KNOWN_OUTPUTS)} "
                f"(and remove an earlier {' and '.join(STALE)})")
    created = not output_dir.exists()  # a refused prediction removes the folder it made
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_dir):  # an existing read-only directory passes mkdir
            pass
    except OSError as error:
        fail(3, f"--output-dir: cannot write {output_dir}: {error.strerror}")
    with _removed_if_left_empty(output_dir, created):
        try:
            calibration, name, body = resolve_calibration(manifest)
        except mf.ManifestError as error:
            fail(2, str(error))

        def progress(msg: str) -> None:
            typer.echo(f"servecast predict: {msg}", err=True)

        try:
            pl = R.plan(manifest, calibration, name, body, sweep=parse_sweep(sweep), progress=progress,
                        fit_seconds=timeout)
            carried = carried_capacity(to, sets, None if original is None else str(original))
            if carried:
                pl.warnings.append(carried)
            for warning in pl.warnings:
                progress(f"warning: {warning}")
            progress(f"{len(pl.points)} load point(s)")
            rows = R.run(pl, jobs=jobs, seconds_budget=timeout, keep_requests=capture_per_request)
        except R.ManifestError as error:
            fail(2, str(error))
        except R.FitTimeoutError as error:
            fail(124, str(error))
        except KeyboardInterrupt:
            fail(130, "interrupted")
    doc = R.prediction_document(pl, rows, R.resolved_manifest(manifest, pl))
    try:
        for stale in STALE:  # not `name`, which is the calibration's and goes into run.json below
            if (output_dir / stale).is_file():
                (output_dir / stale).unlink()
                typer.echo(f"servecast predict: removed {output_dir / stale}, which described the earlier "
                           "prediction", err=True)
        written = R.write_outputs(output_dir, doc, pl, rows)
        # the manifest as it ran, every default filled in: `servecast predict -c manifest.yaml` runs it again
        (output_dir / "manifest.yaml").write_text(yaml.safe_dump(doc["resolved_manifest"], sort_keys=False),
                                                  encoding="utf-8")
        runrecord.write(output_dir, version=__version__, started=started, components=pl.components,
                        inputs={"manifest": {"path": str(config), "sha256": runrecord.sha256_of(config)},
                                "calibration": {"name": name, "sha256": pl.calibration_sha256},
                                "pool": {"sha256": pl.pool_sha256}},
                        outputs=[p.name for p in written] + ["manifest.yaml"])
    except OSError as error:
        fail(3, f"--output-dir: cannot write {error.filename or output_dir}: {error.strerror}")
    if output_format == "json":
        typer.echo(json.dumps(doc, indent=1))
    else:
        print_table(doc)
        typer.echo(f"wrote {output_dir / 'prediction.json'}", err=True)
    if any(not p["converged"] for p in doc["points"]):
        sys.stderr.flush()
        raise typer.Exit(124 if timeout else 1)
