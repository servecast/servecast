"""`servecast replicate`: the paper's tables from its shipped predictions, one configuration recomputed identically,
and the refusals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servecast.cli.app import app
from servecast.core import versions
from servecast.paper import replicate as R

RUN = CliRunner()


def test_list_names_the_five_experiments() -> None:
    rows = json.loads(RUN.invoke(app, ["replicate", "list", "--format", "json"]).output)
    assert [r["experiment"] for r in rows] == ["rq1", "rq2", "rq3", "ablation", "appendix"]
    assert rows[1]["tables"] == ("rq2-matrix-ttft (Table 4), rq2-matrix-throughput (Table 13), "
                                 "rq2-matrix-tpot (Table 14)")


def test_each_table_is_printed_under_its_name_and_its_number_in_the_arxiv_version(tmp_path: Path) -> None:
    """The names follow the text's research questions (the paper source's file names number RQ2 and RQ3 the other way
    round); the number points into the PDF."""
    r = RUN.invoke(app, ["replicate", "run", "rq2", "--from-frozen", "--output-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert [x for x in r.output.splitlines() if x.startswith("rq2-matrix-")] == [
        "rq2-matrix-ttft (Table 4)", "rq2-matrix-throughput (Table 13)", "rq2-matrix-tpot (Table 14)"]
    doc = json.loads((tmp_path / "rq2.json").read_text(encoding="utf-8"))
    assert doc["table_numbers"] == {"rq2-matrix-ttft": 4, "rq2-matrix-throughput": 13, "rq2-matrix-tpot": 14}


def test_every_shipped_table_belongs_to_one_experiment_in_the_papers_order() -> None:
    from servecast.paper import data

    numbers = [[data.tables()[name]["number"] for name in e.tables] for e in R.EXPERIMENTS.values()]
    assert all(n == sorted(n) for n in numbers), numbers
    assert sorted(x for n in numbers for x in n) == sorted(t["number"] for t in data.tables().values())


def test_verify_reproduces_every_computed_printed_cell() -> None:
    """rq1 and rq2's 600 cells from the shipped predictions (the accuracy table, its 18 load-band cells and the
    transfer matrices), 60 from the 18 calibrations fitted again (the appendix's 54 coefficients and the cost table's
    6 training-MAPE cells), and the ablation's 99: 87 from the ablated models' shipped predictions and 12 from the
    shipped workload laws."""
    r = RUN.invoke(app, ["replicate", "verify", "--format", "json"])
    assert r.exit_code == 0, r.output
    doc = json.loads(r.output)
    assert doc["compared"] == 759 and doc["differ"] == []
    r = RUN.invoke(app, ["replicate", "verify", "rq1", "rq2", "--format", "json"])
    assert json.loads(r.output) == {"compared": 600, "differ": []}


def test_the_load_bands_are_computed_from_the_accuracy_tables_predictions(tmp_path: Path) -> None:
    from servecast.paper import data, tables

    r = RUN.invoke(app, ["replicate", "run", "rq1", "--from-frozen", "--output-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    doc = json.loads((tmp_path / "rq1.json").read_text(encoding="utf-8"))
    assert doc["tables"]["load-bands"] == data.tables()["load-bands"]["rows"]

    def slower(campaign: dict, cell: dict, method: str) -> dict | None:  # ServeCast's TTFT 10 % higher everywhere
        pred = tables.frozen(campaign, cell, method)
        return pred if method == "vidur" or pred is None else {**pred, "mean_ttft_s": pred["mean_ttft_s"] * 1.1}

    published = data.tables()["load-bands"]
    rows = tables.compute("load-bands", data.campaigns(), slower, published)
    _, gaps = tables.compare("load-bands", rows, published, tables.COLUMNS["load-bands"])
    assert {g.split(" column ")[1].split(":")[0] for g in gaps} == {"ServeCast TTFT MAPE (%)"}


def test_the_coefficients_and_the_training_mape_are_fitted_again_from_the_shipped_profiling_data(
        tmp_path: Path) -> None:
    r = RUN.invoke(app, ["replicate", "run", "appendix", "--jobs", "2", "--output-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    doc = json.loads((tmp_path / "appendix.json").read_text(encoding="utf-8"))
    assert doc["refitted"]["calibrations"] == 18
    assert doc["tables"]["calibration-coefficients"][0] == ["A100", "Llama-3.2-3B", "65.8", "3.5", "6.5"]
    r = RUN.invoke(app, ["replicate", "run", "rq3", "--gpu", "h100_sxm", "--output-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    doc = json.loads((tmp_path / "rq3.json").read_text(encoding="utf-8"))
    assert doc["tables"]["rq3-cost"] == [["H100 SXM", "5.0", "4.7--9.0", "2.0", "13.3", "11.1", "27.5", "2.5×"]]
    assert "the profiling minutes and the simulation times are the paper's measurements" in r.output


@pytest.mark.exact
def test_every_calibration_fitted_again_is_the_shipped_one_to_the_last_bit() -> None:
    from servecast.model import predict as P
    from servecast.paper import refit

    for (gpu, llm), fit in refit.refits(1).items():
        assert fit["parameters"] == P.load_calibration(refit.name(gpu, llm))["law"]["parameters"], (gpu, llm)


@pytest.mark.parametrize(("table", "field", "where"), [
    ("calibration-coefficients", "eta_c", "A100/Llama-3.2-3B column compute efficiency eta_c (%)"),
    ("rq3-cost", "mape", "A100 column training MAPE, median (%)"),
])
def test_a_refitted_cell_off_by_its_printed_precision_is_caught(table: str, field: str, where: str) -> None:
    from servecast.paper import data, refit, tables

    fits = {k: {"parameters": dict(v["parameters"]), "mape": v["mape"]} for k, v in refit.refits(1).items()}
    # the A100's median MAPE is the mean of its two middle calibrations: moving both moves it by the same amount
    for llm in ([*data.LLMS][:1] if field == "eta_c" else data.LLMS):
        fit = fits[("a100_sxm_40gb", llm)]
        if field == "eta_c":
            fit["parameters"]["eta_c"] += 0.001
        else:
            fit["mape"] += 0.1
    rows = R.compute_tables(R.EXPERIMENTS["appendix" if table != "rq3-cost" else "rq3"], None, refitted=fits)[table]
    _, gaps = tables.compare(table, rows, data.tables()[table], tables.REFITTED[table])
    assert any(where in g for g in gaps), gaps


def test_the_lockfile_is_the_papers_date() -> None:
    resolved = versions.resolve("2026-10")
    assert {k: resolved[k] for k in R.COMPONENTS} == R.COMPONENTS


@pytest.mark.exact
def test_one_transfer_is_recomputed_identically(tmp_path: Path) -> None:
    r = RUN.invoke(app, ["replicate", "predict", "--with", "a100_sxm_40gb/d9cbf1d3/llama-3-8b", "--to", "qwen2.5-7b",
                         "--trace", "burstgpt", "--clients", "66", "--format", "json", "--output-dir",
                         str(tmp_path / "out")])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["identical_to_paper"] is True


@pytest.mark.parametrize("args, message", [
    (["--with", "a100_sxm_40gb/c62ebe75", "--to", "qwen2.5-7b", "--trace", "burstgpt", "--clients", "66"],
     "not one of the paper's calibrations"),
    (["--with", "a100_sxm_40gb/d9cbf1d3/llama-3-8b", "--to", "qwen2.5-7b", "--trace", "burstgpt", "--clients", "64"],
     "has no load point at 64 clients"),
    (["--with", "a100_sxm_40gb/d9cbf1d3/llama-3-8b", "--to", "mistral-7b", "--trace", "burstgpt", "--clients", "66"],
     "not one of the paper's LLMs"),
])
def test_a_configuration_outside_the_paper_is_refused(args: list[str], message: str) -> None:
    r = RUN.invoke(app, ["replicate", "predict", *args])
    assert r.exit_code == 2 and message in r.output, r.output


def test_run_from_frozen_prints_and_writes_the_table(tmp_path: Path) -> None:
    r = RUN.invoke(app, ["replicate", "run", "rq2", "--from-frozen", "--gpu", "h100_sxm", "--output-dir",
                         str(tmp_path)])
    assert r.exit_code == 0, r.output
    doc = json.loads((tmp_path / "rq2.json").read_text())
    assert set(doc["tables"]) == {"rq2-matrix-throughput", "rq2-matrix-ttft", "rq2-matrix-tpot"}
    assert all(len(rows) == 8 for rows in doc["tables"].values())


def test_an_experiment_that_cannot_be_recomputed_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every experiment of the paper is computed now; a printed-only one would say how to print it."""
    printed = R.Experiment("printed", "a table printed as published", ("appendix-runtime",), None)
    monkeypatch.setitem(R.EXPERIMENTS, "printed", printed)
    from servecast.cli import replicate_cmd

    with pytest.raises(Exception) as stopped:
        replicate_cmd.run("printed", None, False, tmp_path, "table", 1)
    assert getattr(stopped.value, "exit_code", None) == 2


def test_the_three_ablations_are_computed_from_the_shipped_predictions(tmp_path: Path) -> None:
    from servecast.paper import data

    r = RUN.invoke(app, ["replicate", "run", "ablation", "--from-frozen", "--output-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    doc = json.loads((tmp_path / "ablation.json").read_text(encoding="utf-8"))
    for name in ("workload-fit", "length-ablation", "residual-ablation", "clock-ablation"):
        assert doc["tables"][name] == data.tables()[name]["rows"], name
    assert "computed from the predictions of the ablated models that ship with ServeCast" in r.output


@pytest.mark.parametrize(("arm", "table", "column"), [
    ("constant_only", "residual-ablation", "Thr."),
    ("fixed_mean_lengths", "length-ablation", "TTFT"),
    ("no_clock_adjustment", "clock-ablation", "TPOT"),
])
def test_an_ablated_prediction_moved_is_caught(arm: str, table: str, column: str,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    import copy

    from servecast.paper import data, tables

    key = {"Thr.": "throughput_rps", "TTFT": "mean_ttft_s", "TPOT": "mean_tpot_s"}[column]
    moved = copy.deepcopy(data.ablations())
    part = {"residual-ablation": "residual", "length-ablation": "length", "clock-ablation": "clock"}[table]
    for c in moved[part]:  # every load point of the arm 20 % higher on one metric
        for x in c["cells"]:
            if x.get(arm):
                x[arm][key] *= 1.2
    monkeypatch.setattr(data, "ablations", lambda: moved)
    published = data.tables()[table]
    _, gaps = tables.compare(table, tables.compute(table, data.campaigns(), published=published), published,
                             tables.COLUMNS[table])
    assert gaps and all(f"column {column}:" in g for g in gaps), gaps


@pytest.mark.exact
def test_a_workload_law_fitted_again_is_the_shipped_one_to_the_last_bit() -> None:
    """Azure's, the quickest of the three (about 20 s); BurstGPT's and ServeGen's take minutes."""
    from servecast.paper import refit, tables

    fit = refit.refit_law("azure-conversation")
    assert json.dumps(fit["law"], sort_keys=True) == json.dumps(tables.shipped_laws()["azure-conversation"]["law"],
                                                                sort_keys=True)


def test_run_ablation_fits_the_workload_laws_again_and_says_how_many_are_the_shipped_ones(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from servecast.paper import refit, tables

    shipped = tables.shipped_laws()
    monkeypatch.setattr(refit, "law_refits", lambda traces, jobs=1: {t: dict(shipped[t]) for t in traces})
    r = RUN.invoke(app, ["replicate", "run", "ablation", "--output-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert json.loads((tmp_path / "ablation.json").read_text(encoding="utf-8"))["refitted_laws"] == {
        "laws": 3, "identical_to_shipped": 3}


def test_fetch_says_when_the_record_comes() -> None:
    r = RUN.invoke(app, ["replicate", "fetch"])
    assert r.exit_code == 1 and "first release" in r.output


@pytest.mark.parametrize("experiment", ["ablation", "appendix", "rq3"])
def test_every_printed_table_prints(tmp_path: Path, experiment: str) -> None:
    """The ablation tables have one-cell group rows (a GPU's name over its block): the printer took them for full
    rows and crashed."""
    r = RUN.invoke(app, ["replicate", "run", experiment, "--from-frozen", "--output-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output


@pytest.mark.parametrize("gpu", ["a100_sxm_40gb", "h100_sxm", "rtx_3090_ti"])
def test_the_accuracy_table_of_one_gpu_leaves_out_the_other_gpus(tmp_path: Path, gpu: str) -> None:
    r = RUN.invoke(app, ["replicate", "run", "rq1", "--from-frozen", "--gpu", gpu, "--format", "json",
                         "--output-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output


def test_verify_refuses_an_experiment_it_cannot_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """`verify rq3` printed '0 printed cells compared, 0 differ' and passed with nothing checked. Every experiment of
    the paper has computed tables now, so a printed-only one is made up here."""
    from servecast.cli import replicate_cmd

    monkeypatch.setitem(R.EXPERIMENTS, "printed", R.Experiment("printed", "as published", ("appendix-runtime",), None))
    with pytest.raises(Exception) as stopped:
        replicate_cmd.verify(["printed"], "table")
    assert getattr(stopped.value, "exit_code", None) == 2


def test_list_says_vidur_is_not_run() -> None:
    rows = {r["experiment"]: r for r in json.loads(RUN.invoke(app, ["replicate", "list", "--format",
                                                                     "json"]).output)}
    assert "Vidur: stored" in rows["rq1"]["recomputed"] and "Vidur: stored" in rows["rq2"]["recomputed"]


def test_predict_writes_its_document_to_the_output_directory(tmp_path: Path) -> None:
    """--output-dir was accepted and ignored."""
    r = RUN.invoke(app, ["replicate", "predict", "--with", "a100_sxm_40gb/d9cbf1d3/deepseek-coder-1.3b-base",
                         "--to", "deepseek-coder-1.3b-base", "--trace", "azure-conversation", "--clients", "40",
                         "--output-dir", str(tmp_path / "rp")])
    assert r.exit_code == 0, r.output
    assert json.loads((tmp_path / "rp" / "replication.json").read_text(encoding="utf-8"))["identical_to_paper"]


# the outside review's replicate findings ------------------------------------------------------------------------

def test_verify_takes_jobs_and_checks_an_experiment_named_twice_once() -> None:
    """The review's site-8 (no --jobs: one process per core, always) and F18 (rq1 rq1 counted 192 cells, not 96)."""
    once = json.loads(RUN.invoke(app, ["replicate", "verify", "rq1", "--jobs", "1", "--format", "json"]).output)
    twice = json.loads(RUN.invoke(app, ["replicate", "verify", "rq1", "rq1", "--jobs", "1", "--format", "json"]).output)
    assert once["compared"] == twice["compared"] > 0 and twice["differ"] == []


@pytest.mark.parametrize("experiment", ["rq3", "appendix", "ablation"])
def test_gpu_is_refused_where_it_cannot_restrict(tmp_path: Path, experiment: str) -> None:
    """The review's F09: --gpu was ignored for these, without a word."""
    r = RUN.invoke(app, ["replicate", "run", experiment, "--from-frozen", "--gpu", "h100_sxm", "--output-dir",
                         str(tmp_path)])
    assert r.exit_code == 2 and "--gpu: restricts rq1 and rq2" in r.output, r.output


def test_rq1_restricted_to_one_gpu_says_so_and_names_its_all_row(tmp_path: Path) -> None:
    """The review's F09: the restricted table still labelled its first row "All"."""
    r = RUN.invoke(app, ["replicate", "run", "rq1", "--from-frozen", "--gpu", "a100_sxm_40gb", "--output-dir",
                         str(tmp_path)])
    assert r.exit_code == 0 and "(restricted to the A100)" in r.output and "All on the A100" in r.output, r.output


def test_fetch_answers_in_json_when_asked() -> None:
    """The review's F20: --format json printed the plain text."""
    r = RUN.invoke(app, ["replicate", "fetch", "--format", "json"])
    assert r.exit_code == 1 and json.loads(r.stdout)["fetched"] is False


def test_recompute_with_one_job_starts_no_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """The review's Q4: --jobs 1 started a pool and sent every load point to a child process."""
    def no_pool(*args: object, **kwargs: object) -> None:
        raise AssertionError("a process pool for --jobs 1")

    monkeypatch.setattr(R, "ProcessPoolExecutor", no_pool)
    monkeypatch.setattr(R, "jobs_of", lambda experiment, gpu: [("g", "s", "l", "t", 1)])
    monkeypatch.setattr(R, "_job", lambda t: (t, {"x": 1}))
    assert R.recompute(R.EXPERIMENTS["rq1"], None, 1) == {("g", "s", "l", "t", 1): {"x": 1}}


def test_a_refit_reads_the_shipped_profiling_data_whatever_the_working_directory_holds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The review's Q2: the calibration's name, read as a path, found a folder of that name in the working directory
    first, and replicate refitted from it without a word."""
    from servecast.paper import refit

    decoy = tmp_path / refit.name("a100_sxm_40gb", "meta-llama/Meta-Llama-3-8B")
    decoy.mkdir(parents=True)
    (decoy / "profiling.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert refit.refit_one("a100_sxm_40gb", "meta-llama/Meta-Llama-3-8B")["parameters"]
