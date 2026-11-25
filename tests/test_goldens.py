"""The reference data in tests/goldens: present, complete, and in the command line's vocabulary.

Shape only. The model's tests (reproduce each prediction exactly) compare against these files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from servecast.cli.app import Trace

GOLDENS = Path(__file__).parent / "goldens"
PAPER_DATA = Path(__file__).resolve().parents[1] / "src" / "servecast" / "data" / "paper"
HOMES = {"paper.json": PAPER_DATA / "campaigns.json", "tables.json": PAPER_DATA / "tables.json",
         "newest-a100.json": GOLDENS / "newest-a100.json"}
GPUS = {"a100_sxm_40gb", "h100_sxm", "rtx_3090_ti"}
LLMS = {"meta-llama/Llama-3.2-3B", "meta-llama/Meta-Llama-3-8B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B",
        "deepseek-ai/deepseek-coder-1.3b-base", "deepseek-ai/deepseek-llm-7b-base"}
TABLES = {"appendix-runtime", "calibration-coefficients", "clock-ablation", "length-ablation", "load-bands",
          "residual-ablation", "rq1-accuracy", "rq2-matrix-throughput", "rq2-matrix-tpot", "rq2-matrix-ttft",
          "rq3-cost", "workload-fit"}


def load(name: str) -> dict:
    return json.loads(HOMES[name].read_text(encoding="utf-8"))


def floats(obj: object) -> list[float]:
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in floats(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in floats(v)]
    return [obj] if isinstance(obj, float) else []


def test_the_paper_has_every_campaign_and_the_diagonal_everywhere() -> None:
    paper = load("paper.json")
    keys = {(c["gpu"], c["llm"], c["trace"]) for c in paper["campaigns"]}
    assert len(paper["campaigns"]) == len(keys) == 54
    assert {g for g, _, _ in keys} == GPUS and {m for _, m, _ in keys} == LLMS
    assert {t for _, _, t in keys} == {t.value for t in Trace}
    for c in paper["campaigns"]:
        assert len(c["cells"]) == 6, (c["gpu"], c["llm"], c["trace"])  # the designed load points, measured or not
        for cell in c["cells"]:
            # the diagonal and the five transfers from the same GPU's other calibrations: 54 x 6 x 6 = 1,944
            assert set(cell["predicted"]) == LLMS, (c["gpu"], c["llm"], c["trace"], cell["concurrency"])


def test_the_paper_scores_only_cells_whose_prediction_converged() -> None:
    for c in load("paper.json")["campaigns"]:
        for cell in c["cells"]:
            if cell["primary"]:
                assert cell["predicted"][c["llm"]]["converged"] and cell["measured"]["certified"]


def test_the_newest_law_covers_the_a100_grid() -> None:
    newest = load("newest-a100.json")
    assert newest["gpu"] == "a100_sxm_40gb" and len(newest["campaigns"]) == 18
    cells = [cell for c in newest["campaigns"] for cell in c["cells"]]
    assert len(cells) == 108 and all(cell["predicted"]["converged"] for cell in cells)
    for cell in cells:
        if cell["scored_p95_ttft"] or cell["scored_p95_tpot"]:
            assert cell["scored_means"]


def test_the_twelve_printed_tables_are_there() -> None:
    assert set(load("tables.json")["tables"]) == TABLES


@pytest.mark.parametrize("name", ["paper.json", "newest-a100.json"])
def test_every_number_is_finite(name: str) -> None:
    values = floats(load(name))
    assert values and all(math.isfinite(v) for v in values)


def test_the_served_kv_capacity_is_an_override_and_vllms_own_count_is_kept() -> None:
    """The paper's campaigns set the KV-block count; the reference for estimating one is the count vLLM computed."""
    paper = load("paper.json")
    engines = [c["engine"] for c in paper["campaigns"] if c["engine"] is not None]
    assert engines and all(e["num_gpu_blocks"] == e["num_gpu_blocks_override"] for e in engines)
    assert all(isinstance(c["num_gpu_blocks_computed"], int) for c in paper["campaigns"])


def test_a_load_point_never_measured_is_explicit_and_never_scored() -> None:
    for c in load("paper.json")["campaigns"]:
        for cell in c["cells"]:
            if cell["measured"] is None:
                assert not c["complete"] and not cell["primary"] and not cell["evaluated"]


def test_every_null_run_fact_states_why() -> None:
    reasons = {"engine": "engine report not recorded", "clock_lock_sm_mhz": "clock lock not recorded"}
    for c in load("paper.json")["campaigns"]:
        assert sorted(c["missing"]) == sorted(why for key, why in reasons.items() if c[key] is None)


def test_every_printed_table_has_its_columns() -> None:
    for name, table in load("tables.json")["tables"].items():
        assert table["columns"], name
        assert all(len(row) <= len(table["columns"]) for row in table["rows"]), name


SETTINGS = {"seed", "kv_blocks", "block_size", "max_batched_tokens", "max_sequences", "checkpoint_ratio",
            "max_completions", "targets"}


@pytest.mark.parametrize("name", ["paper.json", "newest-a100.json"])
def test_every_campaign_carries_its_prediction_settings(name: str) -> None:
    doc = json.loads(HOMES[name].read_text(encoding="utf-8"))
    for camp in doc["campaigns"]:
        s = camp["prediction"]
        assert set(s) == SETTINGS
        assert {"throughput", "e2e", "ttft", "tpot"} <= set(s["targets"])
        assert all(0 < v < 1 for v in s["targets"].values())
        # every prediction in these files ran until its precision targets were met: no ceiling
        assert s["max_completions"] is None
        if name == "newest-a100.json":
            assert {"ttft_p95", "tpot_p95"} <= set(s["targets"])


def test_the_paper_campaigns_kv_blocks_are_the_engines_override() -> None:
    doc = json.loads(HOMES["paper.json"].read_text(encoding="utf-8"))
    for camp in doc["campaigns"]:
        if camp["engine"] is not None:
            assert camp["prediction"]["kv_blocks"] == camp["engine"]["num_gpu_blocks"]
            assert camp["prediction"]["max_sequences"] == camp["engine"]["max_num_seqs"]
            assert camp["prediction"]["max_batched_tokens"] == camp["engine"]["max_num_batched_tokens"]
