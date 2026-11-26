"""The `servecast` command: every command, option and help text.

This is the user-facing contract. The help pages are snapshot-tested (tests/test_cli_help.py) and the CLI
reference (docs/reference/cli.md) is generated from them, so a change here is reviewed like code.
"""

from __future__ import annotations

import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import yaml
from typer._completion_classes import completion_init

from servecast import __version__

if TYPE_CHECKING:
    from servecast.core.manifest import Manifest, Problem

HELP_OPTIONS = {"help_option_names": ["-h", "--help"]}

# Typer registers its shell-completion handlers only for its own --install-completion option, which this CLI
# replaces with the `completion` group; register them here (Typer is pinned in constraints/ci.txt).
completion_init()


def _group(help_text: str) -> typer.Typer:
    return typer.Typer(help=help_text, rich_markup_mode="markdown", no_args_is_help=True, add_completion=False,
                       context_settings=HELP_OPTIONS)


def _examples(*lines: str, see_also: str) -> str:
    block = "\n".join(lines)
    return f"**Examples**\n\n```\n{block}\n```\n\nSee also: {see_also}."


def _print_rows(rows: list[dict[str, str]], output_format: Format) -> None:
    """A list command's output: aligned columns for people, JSON for scripts."""
    if output_format is Format.json:
        typer.echo(json.dumps(rows, indent=2))
        return
    headers = list(rows[0])
    widths = [max(len(h), *(len(r[h]) for r in rows)) for h in headers]
    typer.echo("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip())
    for row in rows:
        typer.echo("  ".join(row[h].ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip())


class Format(StrEnum):
    table = "table"
    json = "json"


class Template(StrEnum):
    predict = "predict"
    profile = "profile"


class SchemaKind(StrEnum):
    manifest = "manifest"
    pool = "pool"
    profiling_data = "profiling-data"
    probe = "probe"
    gemm = "gemm"
    calibration = "calibration"
    prediction = "prediction"
    measurement = "measurement"
    comparison = "comparison"


class Experiment(StrEnum):
    rq1 = "rq1"
    rq2 = "rq2"
    rq3 = "rq3"
    ablation = "ablation"
    appendix = "appendix"


class Trace(StrEnum):
    azure_conversation = "azure-conversation"
    burstgpt = "burstgpt"
    servegen = "servegen"


class Shell(StrEnum):
    bash = "bash"
    zsh = "zsh"
    fish = "fish"


# options shared by several commands ----------------------------------------------------------------------

Config = Annotated[Path, typer.Option("-c", "--config", metavar="MANIFEST", rich_help_panel="Inputs",
                                      help="The manifest: a YAML file, or the name of one that ships with "
                                           "ServeCast (servecast list manifests).")]
Sets = Annotated[list[str] | None, typer.Option(
    "--set", metavar="PATH=VALUE", rich_help_panel="Inputs",
    help="Override one manifest field; the value is read as YAML. Repeatable, applied left to right, then the "
         "whole manifest is validated.")]
Sweep = Annotated[list[str] | None, typer.Option(
    "--sweep", metavar="PATH=V1,V2,...", rich_help_panel="Workload",
    help="Repeat for each value of ONE field, e.g. traffic.load.concurrency=40,80,120.")]
OutputFormat = Annotated[Format, typer.Option("--format", rich_help_panel="Output",
                                              help="How the summary is printed. Files on disk do not change.")]
Overwrite = Annotated[bool, typer.Option("--overwrite", rich_help_panel="Output",
                                         help="Replace the known output files in a non-empty output directory.")]
Poweroff = Annotated[bool, typer.Option(
    "--poweroff-when-safe", rich_help_panel="Advanced",
    help="Power this machine off (sudo shutdown -h now) once the results are written and match their checksums; "
         "not after a failure. Many cloud providers bill a powered-off machine until it is terminated.")]


def _output_dir(
        help_text: str = "Where the results go. A non-empty directory is refused without --overwrite.") -> object:
    return typer.Option("--output-dir", metavar="DIR", rich_help_panel="Output", help=help_text)


# the application -------------------------------------------------------------------------------------------

app = _group(
    "Predict how an LLM serves on a GPU running vLLM: time to first token, time per output token and "
    "throughput, from a calibrated cost law and a workload.\n\n"
    "**Start here**\n\n"
    "```\nservecast manifest init --template predict > m.yaml\nservecast predict -c m.yaml\n"
    "servecast replicate list\n```"
)


#: set by --debug: an unexpected failure shows its traceback instead of one line
_DEBUG = False


def _version(value: bool) -> None:
    if value:
        typer.echo(f"servecast {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True,
                                          help="Print the version and exit.")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show the full traceback when a command fails.")] = False,
) -> None:
    """Predict how an LLM serves on a GPU running vLLM."""
    global _DEBUG
    _DEBUG = debug


@app.command(rich_help_panel="Predict and calibrate", epilog=_examples(
    "servecast predict -c m.yaml",
    "servecast predict -c m.yaml --with a100_sxm_40gb/c62ebe75 --to Qwen/Qwen2.5-7B",
    "servecast predict -c m.yaml --sweep traffic.load.concurrency=40,80,120 --format json",
    see_also="`servecast compare`, `servecast manifest explain`, `servecast list calibrations`"))
def predict(
    config: Config,
    with_: Annotated[str | None, typer.Option(
        "--with", metavar="CALIBRATION", rich_help_panel="Inputs",
        help="A calibration name or path. Same as --set calibration.use=CALIBRATION.")] = None,
    to: Annotated[str | None, typer.Option(
        "--to", metavar="LLM", rich_help_panel="Inputs",
        help="The LLM to predict with the calibration, a transfer: a Hugging Face id or a local directory holding "
             "config.json. Same as --set engine.model=LLM --set calibration.transfer=true.")] = None,
    set_: Sets = None,
    sweep: Sweep = None,
    output_dir: Annotated[Path, _output_dir()] = Path("servecast-output"),
    overwrite: Overwrite = False,
    output_format: OutputFormat = Format.table,
    capture_per_request: Annotated[bool, typer.Option(
        "--capture-per-request", rich_help_panel="Output",
        help="Also write requests.jsonl, one record per simulated request.")] = False,
    jobs: Annotated[int, typer.Option(
        "--jobs", min=1, rich_help_panel="Advanced",
        help="Load points of a sweep run in parallel processes; the numbers do not change.")] = 1,
    timeout: Annotated[float | None, typer.Option(
        "--timeout", metavar="SECONDS", min=1, rich_help_panel="Advanced",
        help="Stop the workload-law fit, or a load point that has not converged, after this long: a stopped fit "
             "ends the command, a stopped point's numbers are written, marked as not converged, and either way the "
             "command exits with status 124.")] = None,
) -> None:
    """Predict serving metrics for one deployment and workload.

    Reads the manifest, resolves the calibration and the component versions, simulates the closed workload to
    steady state, and writes prediction.json: means, P95s and 95 % confidence intervals, with the components that
    made them. When the manifest gives profiling data instead of a calibration, the calibration is fitted first.
    """
    from servecast.cli import predict_cmd

    predict_cmd.main(config, with_=with_, to=to, sets=set_ or [], sweep=sweep, output_dir=output_dir,
                     overwrite=overwrite, output_format=output_format.value, capture_per_request=capture_per_request,
                     jobs=jobs, timeout=timeout)


@app.command(rich_help_panel="Predict and calibrate", epilog=_examples(
    "servecast calibrate -c m.yaml",
    "servecast calibrate --data ./a100-profiling-data --versions 2026-11 --name my-a100",
    see_also="`servecast profile calibration`, `servecast list calibrations`"))
def calibrate(
    config: Annotated[Path | None, typer.Option(
        "-c", "--config", metavar="MANIFEST", rich_help_panel="Inputs",
        help="A manifest whose calibration.data names the profiling data.")] = None,
    data: Annotated[Path | None, typer.Option(
        "--data", metavar="PATH", rich_help_panel="Inputs",
        help="The profiling data, without a manifest: a file, a directory holding profiling.json, or the name of a "
             "shipped calibration whose profiling data ships (servecast list calibrations).")] = None,
    versions: Annotated[str | None, typer.Option(
        "--versions", metavar="DATE", rich_help_panel="Inputs",
        help="Optional: the profiling data names its cost law. A year-month (2026-11) given here must select "
             "the same law.")] = None,
    name: Annotated[str | None, typer.Option(
        "--name", rich_help_panel="Output",
        help="The calibration's name, to pass to --with (default local/GPU/COST_LAW, plus /LLM for a per-LLM "
             "calibration).")] = None,
    output_dir: Annotated[Path, _output_dir(
        "Where calibration.json and run.json go; the calibration is also saved under its name in your data "
        "directory.")] = Path("servecast-calibration"),
    output_format: OutputFormat = Format.table,
    overwrite: Annotated[bool, typer.Option("--overwrite", rich_help_panel="Output",
                                            help="Replace your calibration of the same name, and an --output-dir "
                                                 "that holds other files.")] = False,
) -> None:
    """Fit a cost law to profiling data and save the calibration.

    Fits the cost law selected by the versions to the profiling data, and saves the calibration in your data
    directory under a name that --with accepts. `servecast predict` runs the same fit on the fly when its
    manifest gives profiling data.
    """
    from servecast.cli import calibrate_cmd

    calibrate_cmd.main(config, data, versions, name, output_format.value, overwrite, output_dir)


# profile -------------------------------------------------------------------------------------------------

profile_app = _group(
    "Measure on a real GPU: Linux, an NVIDIA GPU of compute capability 8.0 or newer, and vLLM 0.25.1, installed "
    "with `pip install 'servecast[profile]'`; `servecast doctor` checks them. Every other command runs on any "
    "computer."
)
app.add_typer(profile_app, name="profile", rich_help_panel="Measure on a GPU",
              short_help="Measure on an NVIDIA GPU with vLLM.")


@profile_app.command("calibration", epilog=_examples(
    "servecast profile calibration -c m.yaml",
    see_also="`servecast calibrate`, `servecast doctor`"))
def profile_calibration(
    config: Config,
    set_: Sets = None,
    output_dir: Annotated[Path, _output_dir()] = Path("servecast-profiling-data"),
    overwrite: Overwrite = False,
    output_format: OutputFormat = Format.table,
    poweroff_when_safe: Poweroff = False,
) -> None:
    """Probe the cost law on this GPU and write its profiling data.

    Runs the calibration probe of the cost law the manifest's versions select: batch compositions of controlled
    sizes, each timed step by step. For the per-GPU law (c62ebe75), the default, it probes 8 synthetic models served
    with random weights, then a grid of matrix multiplications; for the paper's law (d9cbf1d3, once per LLM and GPU),
    the manifest's LLM. vLLM runs with the probe's own engine settings, those every shipped calibration was measured
    with (a context of 4,096 tokens, 512 sequences, 2,048 batched tokens, 9,965 KV blocks, or the manifest's block
    count if smaller for the paper's law); the manifest's other engine settings are for serving, so the same manifest
    serves both commands. Writes every step and the audit (probe.json, or one probe-MODEL.json per synthetic
    model and gemm.json) and, when every audit is clean, profiling.json, which `servecast calibrate` fits.
    """
    from servecast.cli import profile_cmd

    profile_cmd.calibration(config, sets=set_ or [], output_dir=output_dir, overwrite=overwrite,
                            output_format=output_format.value, poweroff_when_safe=poweroff_when_safe)


@profile_app.command("serving", epilog=_examples(
    "servecast profile serving -c m.yaml",
    "servecast profile serving -c m.yaml --sweep traffic.load.concurrency=40,80,120",
    see_also="`servecast compare`, `servecast doctor`, `servecast workload sample`"))
def profile_serving(
    config: Config,
    set_: Sets = None,
    sweep: Sweep = None,
    output_dir: Annotated[Path, _output_dir()] = Path("servecast-measurement"),
    overwrite: Overwrite = False,
    output_format: OutputFormat = Format.table,
    timeout: Annotated[float | None, typer.Option(
        "--timeout", metavar="SECONDS", min=1, rich_help_panel="Advanced",
        help="Stop a load point that has not converged after this long: its numbers are written, marked as not "
             "converged, and the command exits with status 124.")] = None,
    poweroff_when_safe: Poweroff = False,
) -> None:
    """Serve the workload with vLLM on this GPU and measure it.

    Starts vLLM 0.25.1 with the manifest's engine settings, drives the closed workload from the same request pool
    that `servecast predict` uses, and stops each load point by the manifest's rule, with the GPU clock sampled.
    Writes measurement.json, with the same summary fields as prediction.json, for `servecast compare`.
    """
    from servecast.cli import profile_cmd

    profile_cmd.serving(config, sets=set_ or [], sweep=sweep, output_dir=output_dir, overwrite=overwrite,
                        output_format=output_format.value, timeout=timeout, poweroff_when_safe=poweroff_when_safe)


@app.command(rich_help_panel="Measure on a GPU", epilog=_examples(
    "servecast compare servecast-output servecast-measurement",
    "servecast compare prediction/ measurement/ --format json",
    see_also="`servecast predict`, `servecast profile serving`"))
def compare(
    prediction_dir: Annotated[Path, typer.Argument(metavar="PREDICTION_DIR", help="What `predict` wrote.")],
    measurement_dir: Annotated[Path, typer.Argument(metavar="MEASUREMENT_DIR",
                                                    help="What `profile serving` wrote.")],
    output: Annotated[Path | None, typer.Option(
        "--output", metavar="FILE", rich_help_panel="Output",
        help="Where comparison.json goes (default inside PREDICTION_DIR).")] = None,
    output_format: OutputFormat = Format.table,
) -> None:
    """Score a prediction against a measurement of the same manifest.

    Refuses unless both come from the same manifest, request pool, sampler and estimator. For each load point,
    prints the signed error of every metric next to the measurement's confidence interval, and flags a clock or
    KV-capacity mismatch.
    """
    from servecast.cli import compare_cmd

    compare_cmd.main(prediction_dir, measurement_dir, output, output_format.value)


# replicate -----------------------------------------------------------------------------------------------

replicate_app = _group(
    "Reproduce the paper with its pinned component versions, calibrations, LLMs and traces. Every output is "
    "labelled *paper replication*. rq1 and rq2 are recomputed; rq3 and appendix fit the paper's calibrations again "
    "from their shipped profiling data; ablation computes its tables from the shipped predictions of the ablated "
    "models; Vidur is not run: its numbers are the ones the paper recorded."
)
app.add_typer(replicate_app, name="replicate", rich_help_panel="Reproduce the paper",
              short_help="Reproduce the paper with its pinned versions and data.")


@replicate_app.command("list", epilog=_examples("servecast replicate list",
                                                see_also="`servecast replicate run`"))
def replicate_list(output_format: OutputFormat = Format.table) -> None:
    """List the paper's experiments and what each one reproduces."""
    from servecast.cli import replicate_cmd

    replicate_cmd.list_(output_format.value)


@replicate_app.command("fetch", epilog=_examples("servecast replicate fetch",
                                                 see_also="`servecast replicate run`"))
def replicate_fetch(
    dest: Annotated[Path | None, typer.Option(
        "--dest", metavar="DIR", rich_help_panel="Output",
        help="Where the data go (default your cache directory).")] = None,
    output_format: OutputFormat = Format.table,
) -> None:
    """Download the paper's data record (not in this release).

    The record is published with the first release. `replicate run --from-frozen` and the recomputation of rq1 and
    rq2 need no download.
    """
    from servecast.cli import replicate_cmd

    replicate_cmd.fetch()


@replicate_app.command("predict", epilog=_examples(
    "servecast replicate predict --with a100_sxm_40gb/d9cbf1d3/llama-3-8b --to qwen2.5-7b \\",
    "    --trace burstgpt --clients 66",
    see_also="`servecast replicate run`, `servecast list calibrations`"))
def replicate_predict(
    with_: Annotated[str, typer.Option("--with", metavar="CALIBRATION", rich_help_panel="Inputs",
                                       help="One of the paper's calibrations: GPU/d9cbf1d3/LLM.")],
    to: Annotated[str, typer.Option("--to", metavar="LLM", rich_help_panel="Inputs",
                                    help="One of the paper's LLMs, on the same GPU.")],
    trace: Annotated[Trace, typer.Option("--trace", rich_help_panel="Workload", help="One of the paper's traces.")],
    clients: Annotated[int, typer.Option("--clients", metavar="N", rich_help_panel="Workload",
                                         help="One of the paper's client counts for that LLM and GPU.")],
    output_dir: Annotated[Path, _output_dir(
        "Where replication.json goes; an earlier one there is replaced.")] = Path("servecast-output"),
    output_format: OutputFormat = Format.table,
) -> None:
    """Predict one of the paper's configurations beside its measurement.

    The calibration, the LLM, the trace and the client count must all be the paper's; the output prints the
    paper's measurement next to the prediction, and replication.json in the output directory keeps both.
    """
    from servecast.cli import replicate_cmd

    replicate_cmd.predict(with_, to, trace.value, clients, output_format.value, output_dir)


@replicate_app.command("run", epilog=_examples(
    "servecast replicate run rq1 --from-frozen",
    "servecast replicate run rq2 --gpu h100_sxm",
    see_also="`servecast replicate verify`, `servecast replicate fetch`"))
def replicate_run(
    experiment: Annotated[Experiment, typer.Argument(help="Which experiment.")],
    gpu: Annotated[str | None, typer.Option("--gpu", rich_help_panel="Inputs",
                                            help="Restrict to one of the paper's GPUs.")] = None,
    from_frozen: Annotated[bool, typer.Option(
        "--from-frozen", rich_help_panel="Inputs",
        help="Print the tables from the shipped predictions, or as published, in seconds, instead of recomputing.")
    ] = False,
    output_dir: Annotated[Path, _output_dir(
        "Where EXPERIMENT.json goes; an earlier result of the same experiment is replaced.")] = Path(
        "servecast-replicate"),
    output_format: OutputFormat = Format.table,
    jobs: Annotated[int, typer.Option("--jobs", min=1, rich_help_panel="Advanced",
                                      help="Load points predicted, or calibrations and workload laws fitted, "
                                           "in parallel processes.")] = 1,
) -> None:
    """Recompute one of the paper's experiments and print its table.

    rq1 and rq2 predict every configuration with the paper's pinned pipeline and shipped calibrations, score it
    against the paper's measurements, and print the table in the paper's layout; with --from-frozen the table is
    computed from the shipped predictions instead, in seconds. rq3 and appendix fit the paper's 18 calibrations again
    from their shipped profiling data and print the training MAPE and the coefficients they give, beside the times the
    paper measured, as printed. ablation computes its tables from the shipped predictions of the ablated models
    and fits the workload laws again from the traces. Vidur is not run: its numbers are the ones the paper recorded.
    """
    from servecast.cli import replicate_cmd

    replicate_cmd.run(experiment.value, gpu, from_frozen, output_dir, output_format.value, jobs)


@replicate_app.command("verify", epilog=_examples("servecast replicate verify", "servecast replicate verify rq1",
                                                  see_also="`servecast replicate run`"))
def replicate_verify(
    experiments: Annotated[list[Experiment] | None, typer.Argument(
        help="Which experiments (default: all of them).")] = None,
    output_format: OutputFormat = Format.table,
) -> None:
    """Check the paper's printed cells against what the shipped data give.

    Rebuilds the accuracy table, its load bands and the transfer tables from the shipped predictions, the ablation
    tables from the shipped predictions of the ablated models and the workload-law table from the shipped laws, fits
    the paper's 18 calibrations again from their shipped profiling data for the coefficients and the training MAPE,
    and compares each cell with the printed one at its printed precision. It takes seconds. A damaged data file or a
    changed fit shows here. `replicate run rq1` and `rq2` recompute the predictions themselves.
    """
    from servecast.cli import replicate_cmd

    replicate_cmd.verify([e.value for e in experiments or []], output_format.value)


# workload ------------------------------------------------------------------------------------------------

workload_app = _group("Inspect a trace, or turn a mean-and-variance workload into concrete requests.")
app.add_typer(workload_app, name="workload", rich_help_panel="Helpers",
              short_help="Inspect a trace, or sample a workload into requests.")


@workload_app.command("describe", epilog=_examples("servecast workload describe azure_conv.jsonl",
                                                   "servecast workload describe burstgpt",
                                                   see_also="`servecast workload sample`"))
def workload_describe(
    trace_file: Annotated[Path, typer.Argument(
        metavar="TRACE", help="A Mooncake JSONL file, a lengths file, or a trace the package ships "
                              "(azure-conversation, "
                              "burstgpt, servegen).")],
    output_format: OutputFormat = Format.table,
) -> None:
    """Summarize the request lengths of a trace.

    Prints the number of requests, the mean, variance and quantiles of the input and output lengths, and their
    correlation.
    """
    from servecast.cli import workload_cmd

    workload_cmd.describe(trace_file, output_format.value)


@workload_app.command("sample", epilog=_examples("servecast workload sample -c m.yaml --out pool.jsonl",
                                                 see_also="`servecast predict`, `servecast profile serving`"))
def workload_sample(
    config: Config,
    set_: Sets = None,
    out: Annotated[Path, typer.Option("--out", metavar="FILE", rich_help_panel="Output",
                                      help="Where the pool goes. An existing FILE is refused without "
                                           "--overwrite.")] = Path("pool.jsonl"),
    overwrite: Annotated[bool, typer.Option("--overwrite", rich_help_panel="Output",
                                            help="Replace an existing FILE.")] = False,
    output_format: OutputFormat = Format.table,
) -> None:
    """Draw the request pool of a manifest's workload and write it.

    The same seeded pool that `predict` and `profile serving` draw from; prints the requested and the realized
    moments side by side.
    """
    from servecast.cli import workload_cmd

    workload_cmd.sample(config, set_ or [], out, output_format.value, overwrite=overwrite)


# manifest ------------------------------------------------------------------------------------------------

manifest_app = _group("Create, check and explain manifests.")
app.add_typer(manifest_app, name="manifest", rich_help_panel="Helpers",
              short_help="Create, check and explain manifests.")


@manifest_app.command("init", epilog=_examples("servecast manifest init > m.yaml",
                                               "servecast manifest init --template profile --out profile.yaml",
                                               see_also="`servecast manifest explain`"))
def manifest_init(
    template: Annotated[Template, typer.Option("--template", help="What the manifest is for.")] = Template.predict,
    out: Annotated[Path | None, typer.Option("--out", metavar="FILE",
                                             help="Write to FILE instead of standard output (- is standard "
                                                  "output). An existing FILE is refused without --overwrite.")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace an existing FILE.")] = False,
) -> None:
    """Write a commented manifest to start from.

    It holds the fields a prediction needs, each with an example value and a one-line note, and versions set to the
    month of the newest components. `servecast manifest explain` describes any field, those left out too.
    """
    from servecast.core import versions

    text = versions.template(template.value)
    if out is None or str(out) == "-":  # --out - wrote a file named "-"
        typer.echo(text, nl=False)
        return
    if out.exists() and not overwrite:  # a manifest the user had edited was replaced without a word
        typer.echo(f"servecast manifest init: --out: {out} exists; give --overwrite to replace it", err=True)
        raise typer.Exit(3)
    try:
        out.write_text(text, encoding="utf-8")
    except OSError as error:  # an internal error (FileNotFoundError) used to say "please report it"
        typer.echo(f"servecast manifest init: --out: cannot write {out}: {error.strerror}", err=True)
        raise typer.Exit(3) from None
    typer.echo(f"wrote {out}", err=True)


@manifest_app.command("show", epilog=_examples("servecast manifest show example-a100 > example-a100.yaml",
                                               see_also="`servecast list manifests`"))
def manifest_show(
    name: Annotated[str, typer.Argument(metavar="NAME", help="A manifest that ships with ServeCast, as servecast "
                                                             "list manifests names it.")],
) -> None:
    """Print a manifest that ships with ServeCast, to save as a file of yours.

    The manifests that ship with ServeCast are examples inside the installed package, which a reinstall replaces. Save
    one to a file, edit the file, and give its path to -c.
    """
    from servecast.core import manifest

    shipped = manifest.shipped()
    if name not in shipped:
        typer.echo(f"servecast manifest show: no manifest {name!r} ships with ServeCast; shipped: "
                   f"{', '.join(shipped)}", err=True)
        raise typer.Exit(2)
    typer.echo(shipped[name].read_text(encoding="utf-8"), nl=False)


def _what_predict_refuses(checked: Manifest) -> list[Problem]:
    """The first problem predict would refuse in a valid file, found by the same lookups predict runs (the
    calibration, then `model.run.resolve`), short of fitting the workload law: a calibration or a trace that does not
    exist, or a GPU other than the calibration's, used to be "valid"."""
    from servecast.cli import predict_cmd
    from servecast.core.manifest import ManifestError, Problem
    from servecast.model import run as R

    try:
        calibration, _, _ = predict_cmd.resolve_calibration(checked)
        R.resolve(checked, calibration)
    except ManifestError as error:
        path, sep, message = str(error).partition(": ")
        return [Problem(path, message) if sep else Problem("(manifest)", str(error))]
    return []


@manifest_app.command("validate", epilog=_examples("servecast manifest validate -c m.yaml",
                                                   see_also="`servecast manifest explain`"))
def manifest_validate(config: Config, set_: Sets = None, output_format: OutputFormat = Format.table) -> None:
    """Check a manifest the way predict does, without predicting.

    Reports every problem in the file at once, as `file: path: message` with the allowed values. A file without any is
    then resolved as predict resolves it (the calibration, the component versions, the LLM, the deployment and the
    request pool), and the first problem predict would refuse is reported. The workload law is not fitted, and
    predict can still refuse a small pool when it fits it.
    """
    from servecast.core import manifest

    config = manifest.find(config)
    unreadable = manifest.config_problem(config)  # a directory or standard input used to be "no such file"
    if unreadable:
        typer.echo(unreadable, err=True)
        raise typer.Exit(2)
    try:
        raw = manifest.load(config)
    except yaml.YAMLError as error:
        typer.echo(f"{config}: not valid YAML: {error}", err=True)
        raise typer.Exit(2) from error
    if not isinstance(raw, dict):  # as predict says it: an empty file was reported as "(manifest): unknown field"
        typer.echo(f"{config}: a manifest is a YAML mapping", err=True)
        raise typer.Exit(2)
    raw, problems = manifest.apply_sets(raw, set_ or [])
    checked, invalid = manifest.validate(raw)
    problems += invalid
    if not problems and checked is not None:  # a valid file: what predict refuses before it runs, refused here too
        problems += _what_predict_refuses(checked)
    if output_format is Format.json:
        typer.echo(json.dumps({"file": str(config), "valid": not problems,
                               "problems": [{"path": p.path, "message": p.message} for p in problems]}, indent=2))
    else:
        for problem in problems:
            typer.echo(f"{config}: {problem}", err=True)
        if not problems:
            typer.echo(f"{config}: valid")
    if problems:
        raise typer.Exit(2)


@manifest_app.command("explain", epilog=_examples("servecast manifest explain traffic.load.think_time_ms",
                                                  see_also="`servecast manifest init`"))
def manifest_explain(
    field: Annotated[str, typer.Argument(metavar="PATH", help="A dotted field path, e.g. traffic.load.concurrency.")],
) -> None:
    """Explain one manifest field: meaning, type, default and allowed values."""
    from servecast.core import manifest

    try:
        typer.echo(manifest.explain(field))
    except KeyError as error:
        typer.echo(str(error.args[0]), err=True)
        raise typer.Exit(2) from error


@manifest_app.command("schema", epilog=_examples("servecast manifest schema > servecast.schema.json",
                                                 "servecast manifest schema --kind prediction",
                                                 see_also="`servecast manifest validate`"))
def manifest_schema(
    kind: Annotated[SchemaKind, typer.Option("--kind", help="Which file's schema.")] = SchemaKind.manifest,
) -> None:
    """Print the JSON Schema of the manifest or of an output file.

    Point an editor at the manifest's schema to get completion and checks while writing it.
    """
    from servecast.core.formats import SCHEMAS

    typer.echo(json.dumps(SCHEMAS[kind.value].model_json_schema(by_alias=True), indent=2))


# list ----------------------------------------------------------------------------------------------------

list_app = _group("Show what is known and installed.")
app.add_typer(list_app, name="list", rich_help_panel="Helpers",
              short_help="Show what is known and installed.")


@list_app.command("gpus", epilog=_examples("servecast list gpus", see_also="`servecast list calibrations`"))
def list_gpus(output_format: OutputFormat = Format.table) -> None:
    """GPUs in the registry, and what each one's calibrations cover.

    The shipped GPUs, then those of gpus.json in your data directory.
    """
    from servecast.core import gpus, versions
    from servecast.model import predict as P

    laws: dict[str, set[str]] = {}
    for name in P.shipped_calibrations():  # what each covers, read from the file: a name is only a label
        calibration = P.load_calibration(name)
        laws.setdefault(calibration["conditions"]["gpu"], set()).add(calibration["law"]["cost_law"])
    order = list(versions.entries("cost_law"))
    try:
        registry = gpus.registry()
    except ValueError as error:  # your gpus.json: an internal error used to say "please report it"
        typer.echo(f"servecast list gpus: {error}", err=True)
        raise typer.Exit(2) from None
    rows = [{"gpu": g.id, "device": g.device_name, "memory GiB": str(g.memory_gib), "SMs": str(g.n_sm),
             "shipped cost laws": ", ".join(sorted(laws.get(g.id, ()), key=order.index)) or "none"}
            for g in registry.values()]
    _print_rows(rows, output_format)


@list_app.command("laws", epilog=_examples("servecast list laws", see_also="`servecast list components`"))
def list_laws(output_format: OutputFormat = Format.table) -> None:
    """Cost laws by name: what one calibration covers, and what each computes."""
    from servecast.core import versions

    per = {"llm_and_gpu": "per LLM and GPU", "gpu": "per GPU"}
    rows = [{"law": name, "released": law.released, "calibration": per[law.calibrated_per],
             "the paper's": "yes" if law.in_paper else "no", "summary": law.summary}
            for name, law in ((name, versions.law(name)) for name in versions.entries("cost_law"))]
    _print_rows(rows, output_format)


@list_app.command("components", epilog=_examples("servecast list components",
                                                 see_also="`servecast manifest explain versions`"))
def list_components(output_format: OutputFormat = Format.table) -> None:
    """Every component ServeCast can select, by name, with what it computes.

    A cost law's entry includes its calibration procedure.
    """
    from servecast.core import formats, versions

    rows = [{"component": versions.COMPONENTS[key].name, "key in versions": key, "name": name,
             "released": e.released, "the paper's": "yes" if e.in_paper else "no",
             "default": "yes" if e.default else "no", "summary": e.summary}
            for key in formats.COMPONENT_KINDS for name, e in versions.entries(key).items()]
    _print_rows(rows, output_format)


@list_app.command("calibrations", epilog=_examples("servecast list calibrations",
                                                   see_also="`servecast calibrate`"))
def list_calibrations(output_format: OutputFormat = Format.table) -> None:
    """Installed calibrations, shipped and your own, with what each covers."""
    from servecast.core import calibration as calibrations
    from servecast.core.paths import data_dir
    from servecast.model import predict as P

    found = [(name, path, "shipped") for name, path in P.shipped_calibrations().items()]
    profiled = P.shipped_profiling()
    user_root = data_dir() / "calibrations"
    found += [(p.relative_to(user_root).with_suffix("").as_posix(), p, "yours")
              for p in sorted(user_root.rglob("*.json"))] if user_root.is_dir() else []
    rows = []
    for name, path, origin in found:
        try:
            doc = calibrations.parse(json.loads(Path(path).read_text(encoding="utf-8")), path)
        except ValueError as error:
            rows.append({"name": name, "gpu": "-", "cost law": "-", "calibrated on": f"unreadable: {error}",
                         "origin": origin, "profiling data": "-"})
            continue
        on = doc["conditions"]["fitted_on"]
        rows.append({"name": name, "gpu": doc["conditions"]["gpu"], "cost law": doc["law"]["cost_law"],
                     "calibrated on": on if isinstance(on, str) else f"{len(on or [])} synthetic models",
                     "origin": origin, "profiling data": "shipped" if origin == "shipped" and name in profiled
                     else "-"})
    _print_rows(rows, output_format)


@list_app.command("manifests", epilog=_examples("servecast list manifests", see_also="`servecast manifest show`"))
def list_manifests(output_format: OutputFormat = Format.table) -> None:
    """Manifests that ship with ServeCast, examples to start from.

    They are inside the installed package. servecast manifest show NAME prints one, to save as a file of yours and
    give to -c by its path.
    """
    from servecast.core import manifest

    rows = []
    for name, path in manifest.shipped().items():
        m, _ = manifest.validate(manifest.load(path))
        assert m is not None and m.traffic is not None, name
        source = m.traffic.source
        rows.append({"manifest": name, "llm": str(m.engine.model), "gpu": str(m.engine.hardware),
                     "calibration": str(m.calibration.use),
                     "workload": ", ".join(str(t) for t in getattr(source, "paths", None) or ["synthetic lengths"]),
                     "load": "; ".join(f"{level.concurrency} clients, think {level.think_time_ms:g} ms"
                                       for level in m.traffic.levels())})
    _print_rows(rows, output_format)


@list_app.command("models", epilog=_examples("servecast list models", see_also="`servecast predict --to`"))
def list_models(output_format: OutputFormat = Format.table) -> None:
    """LLMs whose geometry is known without downloading their config.json.

    The shipped LLMs, then those of models.json in your data directory.
    """
    from servecast.model import geometry

    try:
        known = geometry.known()
    except ValueError as error:  # your models.json
        typer.echo(f"servecast list models: {error}", err=True)
        raise typer.Exit(2) from None
    rows = [{"llm": g.name, "layers": str(g.n_layers), "width": str(g.d_model), "query/KV heads":
             f"{g.n_heads}/{g.h_kv}", "vocabulary": str(g.vocab), "context": str(g.max_context)}
            for g in known.values()]
    _print_rows(rows, output_format)


# completion and doctor -----------------------------------------------------------------------------------

completion_app = _group("Shell completion for bash, zsh and fish.")
app.add_typer(completion_app, name="completion", rich_help_panel="Helpers",
              short_help="Shell completion for bash, zsh and fish.")


@completion_app.command("install", epilog=_examples("servecast completion install",
                                                    see_also="`servecast completion show`"))
def completion_install(
    shell: Annotated[Shell | None, typer.Argument(help="The shell (default the one running).")] = None,
) -> None:
    """Install completion for your shell, and print what was changed."""
    from servecast.cli import completion_cmd

    completion_cmd.main(shell.value if shell is not None else None)


@completion_app.command("show", epilog=_examples(
    "mkdir -p ~/.zfunc && servecast completion show zsh > ~/.zfunc/_servecast",
    see_also="`servecast completion install`"))
def completion_show(shell: Annotated[Shell, typer.Argument(help="The shell.")]) -> None:
    """Print the completion script for a shell."""
    from servecast.cli import completion_cmd

    typer.echo(completion_cmd.script(shell.value))


@app.command(rich_help_panel="Helpers",
             epilog=_examples("servecast doctor", see_also="`servecast profile calibration`"))
def doctor(output_format: OutputFormat = Format.table) -> None:
    """Check Python, the GPU, the driver and vLLM.

    Prints what was found and, for each `profile` command, whether it can run on this computer or why not. Every
    other command runs on any computer.
    """
    from servecast.cli import doctor_cmd

    doctor_cmd.main(output_format.value)


def main() -> None:
    """The `servecast` program. A failure the commands did not foresee is reported in one line with exit status 1;
    --debug shows its traceback instead."""
    try:
        app()
    except Exception as error:
        if _DEBUG:
            raise
        typer.echo(f"servecast: internal error: {type(error).__name__}: {error} (for the traceback, run it again as "
                   "`servecast --debug <command> ...`, and please report it)", err=True)
        sys.exit(1)
