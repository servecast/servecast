"""`servecast replicate list | fetch | predict | run | verify`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, NoReturn

import typer

from servecast.paper import data
from servecast.paper import replicate as R
from servecast.paper import tables as T


def _fail(status: int, message: str) -> NoReturn:
    typer.echo(f"servecast replicate: {message}", err=True)
    raise typer.Exit(status)


def list_(output_format: str) -> None:
    rows = [{"experiment": e.key, "what it reproduces": e.title,
             "tables": ", ".join(f"{t} (Table {data.tables()[t]['number']})" for t in e.tables),
             "recomputed": {"diagonal": "324 load points (ServeCast); Vidur: stored",
                            "transfers": "1,944 load points (ServeCast); Vidur: stored",
                            "refits": "the 18 calibrations, fitted again from their shipped profiling data; the "
                                      "measured times: stored",
                            "workload": "from the shipped predictions of the ablated models; the workload laws "
                                        "fitted again from the shipped traces"}.get(
                 e.recompute or "", "printed table only")} for e in R.EXPERIMENTS.values()]
    if output_format == "json":
        typer.echo(json.dumps(rows, indent=2))
        return
    for r in rows:
        typer.echo(f"{r['experiment']:<9} {r['what it reproduces']}")
        typer.echo(f"{'':<9} tables: {r['tables']}; recomputed: {r['recomputed']}")


def fetch(output_format: str = "table") -> None:
    why = ("the paper's data record (raw probes and per-request measurements) is published with the first release; "
           "`replicate run --from-frozen` and the recomputation of rq1 and rq2 need no download")
    if output_format == "json":  # it printed the plain text even with --format json
        typer.echo(json.dumps({"fetched": False, "reason": why}, indent=1))
        raise typer.Exit(1)
    _fail(1, why)


def predict(with_: str, to: str, trace: str, clients: int, output_format: str, output_dir: Path) -> None:
    from servecast.core import versions

    law, parts = versions.PAPER["cost_law"], with_.split("/")
    if len(parts) != 3 or parts[1] != law or parts[0] not in data.GPUS:
        _fail(2, f"--with: {with_!r} is not one of the paper's calibrations: <gpu>/{law}/<llm>, e.g. "
                 f"a100_sxm_40gb/{law}/llama-3-8b")
    gpu = parts[0]
    try:
        source, target = data.llm_by_name(parts[2]), data.llm_by_name(to)
        camp = data.campaign(gpu, target, trace)
        got = R.predict_one(gpu, source, target, trace, clients)
    except ValueError as error:
        _fail(2, str(error))
    cell = next(c for c in camp["cells"] if c["concurrency"] == clients)
    shipped = (cell.get("predicted") or {}).get(source)
    measured = cell["measured"]
    doc = {"components": R.COMPONENTS, "gpu": gpu, "calibrated_on": source, "llm": target, "trace": trace,
           "concurrency": clients, "predicted": got, "paper_prediction": shipped, "measured": measured,
           "identical_to_paper": shipped is not None and all(got[k] == shipped[k] for k in data.METRICS.values())}
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "replication.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    except OSError as error:
        _fail(3, f"--output-dir: cannot write {output_dir}: {error.strerror}")
    if output_format == "json":
        typer.echo(json.dumps(doc, indent=1))
        return
    typer.echo(f"paper replication: {data.LLMS[target]} on {data.GPUS[gpu]}, {data.TRACES[trace]}, {clients} clients, "
               f"calibrated on {data.LLMS[source]}")
    typer.echo(f"{'':12}{'throughput req/s':>18}{'mean TTFT ms':>14}{'mean TPOT ms':>14}")
    for label, row in (("predicted", got), ("the paper's", shipped), ("measured", measured)):
        if row is None:
            typer.echo(f"{label:12}{'(none)':>18}")
            continue
        typer.echo(f"{label:12}{row['throughput_rps']:>18.3f}{row['mean_ttft_s'] * 1e3:>14.1f}"
                   f"{row['mean_tpot_s'] * 1e3:>14.2f}")
    if shipped is not None:
        typer.echo("identical to the paper's prediction: " + ("yes" if doc["identical_to_paper"] else "NO"))


def _print_table(name: str, rows: list[list[Any]]) -> None:
    pub = data.tables()[name]
    typer.echo(f"\n{name} (Table {pub['number']})")
    # a group's one-cell row (a GPU's name over its block) is shorter than the columns
    widths = [max(len(T.printed(r[i])) for r in [pub["columns"], *rows] if i < len(r))
              for i in range(len(pub["columns"]))]
    typer.echo("  ".join(str(c).ljust(w) for c, w in zip(pub["columns"], widths, strict=False)))
    for r in rows:
        typer.echo("  ".join(T.printed(c).ljust(w) for c, w in zip(r, widths, strict=False)))


def run(experiment: str, gpu: str | None, from_frozen: bool, output_dir: Path, output_format: str, jobs: int) -> None:
    exp = R.EXPERIMENTS[experiment]
    if gpu is not None and gpu not in data.GPUS:
        _fail(2, f"--gpu: one of {', '.join(data.GPUS)}")
    if gpu is not None and exp.recompute not in ("diagonal", "transfers"):  # it was ignored without a word
        _fail(2, f"--gpu: restricts rq1 and rq2; {experiment} prints every GPU's rows at once")
    recomputed = None
    refitted = None
    laws = None
    if exp.recompute is None and not from_frozen:
        _fail(2, f"{experiment}: this version prints its tables as published (--from-frozen)")
    if exp.recompute == "refits" and not from_frozen:
        typer.echo(f"servecast replicate: fitting the paper's calibrations again with {jobs} process(es)", err=True)
        refitted = R.refit.refits(jobs)
    elif exp.recompute == "workload" and not from_frozen:
        typer.echo(f"servecast replicate: fitting the workload law of each trace again with {jobs} process(es); up "
                   "to about 4 minutes per trace", err=True)
        laws = R.refit.law_refits(tuple(T.TRACE_SHORT), jobs)
    elif exp.recompute is not None and not from_frozen:
        todo = R.jobs_of(exp, gpu)
        typer.echo(f"servecast replicate: recomputing {len(todo)} load points with {jobs} process(es)", err=True)

        def report(i: int, n: int, key: tuple, row: dict[str, Any]) -> None:
            if i == n or i % 25 == 0:
                typer.echo(f"servecast replicate: {i}/{n}", err=True)

        recomputed = R.recompute(exp, gpu, jobs, report)
    computed = R.compute_tables(exp, recomputed, gpu, refitted, laws)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"experiment": experiment, "components": R.COMPONENTS, "gpu": gpu, "from_frozen": from_frozen,
           "tables": {k: [[T.printed(c) for c in r] for r in v] for k, v in computed.items()},
           "table_numbers": {k: data.tables()[k]["number"] for k in computed}}
    if refitted is not None:
        from servecast.model import predict as P

        fits = {k: v for k, v in refitted.items() if gpu is None or k[0] == gpu}
        same = sum(1 for (g, llm), fit in fits.items()
                   if P.load_calibration(R.refit.name(g, llm))["law"]["parameters"] == fit["parameters"])
        doc["refitted"] = {"calibrations": len(fits), "identical_to_shipped": same,
                           "computed": {k: list(T.REFITTED[k]) for k in computed if k in T.REFITTED},
                           "as_printed": [k for k in computed if k not in T.REFITTED]}
    if laws is not None:
        shipped = T.shipped_laws()
        same = sum(1 for t, fit in laws.items()
                   if json.dumps(fit["law"], sort_keys=True) == json.dumps(shipped[t]["law"], sort_keys=True))
        doc["refitted_laws"] = {"laws": len(laws), "identical_to_shipped": same}
    if recomputed is not None:
        same = sum(1 for (g, src, llm, tr, n), row in recomputed.items()
                   if (next(c for c in data.campaign(g, llm, tr)["cells"] if c["concurrency"] == n)
                       .get("predicted") or {}).get(src) == {k: row[k] for k in (*data.METRICS.values(), "converged")})
        doc["recomputed"] = {"load_points": len(recomputed), "identical_to_shipped": same}
    (output_dir / f"{experiment}.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    if output_format == "json":
        typer.echo(json.dumps(doc, indent=1))
        return
    typer.echo(f"paper replication, {experiment}: {exp.title}"
               + (f" (restricted to the {data.GPUS[gpu]})" if gpu is not None else ""))
    for name, rows in computed.items():
        _print_table(name, rows)
    if recomputed is not None:
        r = doc["recomputed"]
        typer.echo(f"\n{r['identical_to_shipped']} of {r['load_points']} load points identical to the shipped "
                   "predictions")
    if refitted is not None:
        r = doc["refitted"]
        typer.echo(f"\n{r['identical_to_shipped']} of {r['calibrations']} calibrations fitted again identical to the "
                   "shipped ones")
        if "rq3-cost" in computed:
            typer.echo("rq3-cost: the training MAPE is computed from the fits; the profiling minutes and the "
                       "simulation times are the paper's measurements, as printed")
        if "appendix-runtime" in computed:
            typer.echo("appendix-runtime: the paper's measured run times, as printed")
    if laws is not None:
        r = doc["refitted_laws"]
        typer.echo(f"\n{r['identical_to_shipped']} of {r['laws']} workload laws fitted again identical to the shipped "
                   "ones")
    if experiment == "ablation":
        typer.echo("residual-ablation, length-ablation and clock-ablation are computed from the predictions of the "
                   "ablated models that ship with ServeCast")


def verify(experiments: list[str], output_format: str, jobs: int | None = None) -> None:
    printed_only = [k for k in experiments if not R.EXPERIMENTS[k].recompute]
    checked = [k for k, e in R.EXPERIMENTS.items() if e.recompute]
    if printed_only:  # 0 cells compared used to pass, with nothing checked
        _fail(2, f"{', '.join(printed_only)}: prints the tables as published, so there is nothing to verify; "
                 f"verify checks {', '.join(checked)}")
    keys = list(dict.fromkeys(experiments)) or checked  # an experiment named twice is checked once
    compared, gaps = R.verify(keys, jobs=jobs or os.cpu_count() or 1)
    if output_format == "json":
        typer.echo(json.dumps({"compared": compared, "differ": gaps}, indent=1))
    else:
        typer.echo(f"{compared} printed cells compared at their printed precision, {len(gaps)} differ")
        for g in gaps:
            typer.echo(f"  {g}")
    if gaps:
        raise typer.Exit(1)
