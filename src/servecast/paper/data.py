"""The paper's shipped data: its 54 measured campaigns (each with its settings, its six load points, the measurement,
Vidur's prediction and the paper's predictions), the per-load-point predictions of its three ablations, and the 12
tables printed in the arXiv version.

A table that answers a research question is named after it, as the arXiv version's text numbers them, and each
table carries the number it has in that version (`number`).
"""

from __future__ import annotations

import functools
import json
from importlib import resources
from pathlib import Path
from typing import Any

#: the paper's LLMs, in the order its tables print them, with their printed names
LLMS = {"meta-llama/Llama-3.2-3B": "Llama-3.2-3B", "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B", "Qwen/Qwen2.5-7B": "Qwen2.5-7B",
        "deepseek-ai/deepseek-coder-1.3b-base": "DS-Coder-1.3B", "deepseek-ai/deepseek-llm-7b-base": "DS-LLM-7B"}
GPUS = {"a100_sxm_40gb": "A100", "h100_sxm": "H100 SXM", "rtx_3090_ti": "RTX 3090 Ti"}
TRACES = {"burstgpt": "BurstGPT", "azure-conversation": "Azure conv.", "servegen": "ServeGen"}
#: the metrics the accuracy questions score, and their per-cell fields
METRICS = {"X": "throughput_rps", "TTFT": "mean_ttft_s", "TPOT": "mean_tpot_s"}


def _path(name: str) -> Path:
    return Path(str(resources.files("servecast").joinpath("data", "paper", name)))


@functools.cache
def campaigns() -> list[dict[str, Any]]:
    return json.loads(_path("campaigns.json").read_text(encoding="utf-8"))["campaigns"]


@functools.cache
def tables() -> dict[str, Any]:
    return json.loads(_path("tables.json").read_text(encoding="utf-8"))["tables"]


@functools.cache
def ablations() -> dict[str, Any]:
    """The predictions of the paper's three ablations per load point: the residual correction reduced, the length
    model reduced, and the clock treatments at 900 and 750 MHz with the measurements made at those clocks."""
    return json.loads(_path("ablations.json").read_text(encoding="utf-8"))


def slug(llm: str) -> str:
    """The LLM's part of a calibration name."""
    return llm.split("/")[-1].lower().removeprefix("meta-")


def llm_by_name(name: str) -> str:
    """One of the paper's LLMs, by its Hugging Face id, its printed name or its calibration name."""
    for hf, printed in LLMS.items():
        if name in (hf, printed, slug(hf)) or name.lower() == printed.lower():
            return hf
    raise ValueError(f"not one of the paper's LLMs: {name!r}; they are {', '.join(LLMS.values())}")


def campaign(gpu: str, llm: str, trace: str) -> dict[str, Any]:
    for c in campaigns():
        if (c["gpu"], c["llm"], c["trace"]) == (gpu, llm, trace):
            return c
    raise ValueError(f"the paper has no campaign of {llm} on {gpu} with {trace}")
