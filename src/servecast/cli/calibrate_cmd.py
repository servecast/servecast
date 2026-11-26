"""`servecast calibrate`: profiling data in, a calibration saved under a name `--with` accepts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import typer

#: what calibrate writes in --output-dir: a folder holding only these is an earlier calibration, which it replaces
OUTPUTS = ("calibration.json", "run.json")


def _fail(status: int, message: str) -> NoReturn:
    typer.echo(f"servecast calibrate: {message}", err=True)
    raise typer.Exit(status)


def main(config: Path | None, data: Path | None, versions_date: str | None, name: str | None, output_format: str,
         overwrite: bool, output_dir: Path = Path("servecast-calibration")) -> None:
    from servecast import __version__
    from servecast.calibrate import fit
    from servecast.core import runrecord, versions
    from servecast.model import predict as P

    started = runrecord.now()

    if (config is None) == (data is None):
        _fail(2, "give either -c MANIFEST (whose calibration.data names the profiling data) or --data DIR")
    if config is not None:
        from servecast.cli.predict_cmd import load_manifest
        from servecast.core.manifest import find

        config = find(config)  # a file, or a shipped manifest's name: run.json records the file read

        m = load_manifest(config, [], [])
        if m.calibration.data is None:
            _fail(2, "calibration.data: the manifest names no profiling data")
        data = Path(str(m.calibration.data))
        when = m.versions
        hardware = m.engine.hardware
    else:
        when, hardware = versions_date, None
    assert data is not None
    shipped = set(P.shipped_calibrations())
    try:
        law = None if when is None else versions.resolve(when)["cost_law"]  # the data name their law; given, the same
        profiling, sha = fit.read_profiling(data)
        if hardware is not None and hardware != profiling["gpu"]:  # the manifest's GPU is the one calibrated
            raise ValueError(f"engine.hardware: {hardware}, but the profiling data in {data} were measured on "
                             f"{profiling['gpu']}")
        # before the fit: a taken name used to be refused only after it, with exit 2
        fit.check_free(fit.check_name(name) if name is not None else fit.default_name(profiling), shipped, overwrite)
        foreign = sorted(p.name for p in output_dir.iterdir() if p.name not in OUTPUTS) if output_dir.is_dir() else []
        if foreign and not overwrite:  # its calibration.json and run.json replaced someone's files without a word
            raise fit.NameTakenError(f"--output-dir: {output_dir} holds {', '.join(foreign)}, not an earlier "
                                     "calibration; give another --output-dir, or --overwrite")
        typer.echo(f"servecast calibrate: fitting cost law {profiling['cost_law']} to {data}", err=True)
        doc = fit.calibrate(profiling, sha, cost_law=law, name=name)
        path = fit.save(doc, shipped, overwrite=overwrite)
    except fit.NameTakenError as error:
        _fail(3, str(error))
    except ValueError as error:
        _fail(2, str(error))
    inputs = {"profiling data": {"path": str(data), "sha256": sha}}
    if config is not None:
        inputs["manifest"] = {"path": str(config), "sha256": runrecord.sha256_of(config)}
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "calibration.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        runrecord.write(output_dir, version=__version__, started=started, inputs=inputs,
                        components={"cost_law": doc["law"]["cost_law"],
                                    "calibration_procedure": doc["law"]["cost_law"]},  # part of the law's entry
                        outputs=["calibration.json"])
    except OSError as error:
        _fail(3, f"--output-dir: cannot write {output_dir}: {error.strerror}")
    if output_format == "json":
        typer.echo(json.dumps({"name": doc["name"], "path": str(path), "calibration": doc}, indent=1))
        return
    typer.echo(f"calibration {doc['name']}: cost law {doc['law']['cost_law']} on {doc['conditions']['gpu']}, "
               f"{doc['fit']['cells']} cells, "
               f"{doc['fit']['evaluations']} evaluations")
    fit_error = doc["fit"]["error_pct"]
    typer.echo(f"training MAPE {fit_error['mape']:.2f} % over the {doc['fit']['cells']} cells (weighted as the fit "
               f"weights them); largest cell error {fit_error['max']:.2f} %")
    for n, v in doc["law"]["parameters"].items():
        typer.echo(f"  {n:10} {v:.6g}")
    from servecast.core.paths import data_dir

    try:  # the same words on every machine: the data directory's place is in the docs (Files and folders)
        where = f"in your data directory as {Path(path).resolve().relative_to(data_dir().resolve()).as_posix()}"
    except ValueError:
        where = f"to {path}"
    typer.echo(f"saved {where}; use it with --with {doc['name']}")
