"""vLLM 0.25.1 as the engine `closed_loop.serve` drives. Runs only on a GPU host with vLLM 0.25.1 installed.

Ported unchanged from the measurements behind the paper:

* the engine is vLLM's in-process async engine, `AsyncLLMEngine.from_engine_args(AsyncEngineArgs(...))`, in bfloat16,
  with prefix caching off, CUDA graphs on, and the manifest's token budget, sequence cap, context length and KV-cache
  blocks (imposed with `num_gpu_blocks_override`, the KV capacity the model is given);
* the attention kernel is the GPU backend's (`gpu.Backend.attention`: FlashAttention 2 on NVIDIA), pinned through
  `attention_config` and READ BACK from the engine that was built, since vLLM picks the version by architecture (3 on
  H100-class GPUs) and an unattested kernel is not a pinned one; asynchronous scheduling is left at vLLM's default, as
  in every serving measurement of the paper;
* each request is a `TokensPrompt` of exactly p tokens (a prefix of one fixed sequence, tokenized once before the
  first request, with the LLM's chat template when it has one) and `SamplingParams(ignore_eos=True, max_tokens=g,
  temperature=0.0)`, so the output is exactly g tokens;
* the KV blocks the engine runs with are read back from its cache configuration.

UNVERIFIED in this repository (its CI has no GPU; the per-release A100 run checks them): the vLLM calls themselves,
and that vLLM's own block count reaches engine.log. The engine core prints it in its own process, on standard output,
as "Overriding num_gpu_blocks=X with num_gpu_blocks_override=Y": that text is the one vLLM 0.25.1 printed in the
paper's serving runs.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from servecast.core.deployment import Deployment
from servecast.profile.gpu import Backend

#: the text every prompt is cut from: the paper's measurements used the same
PROMPT_TEXT = "Summarize the following text in detail. " * 200
OVERRIDE_LINE = re.compile(r"Overriding num_gpu_blocks=(\d+) with num_gpu_blocks_override=(\d+)")


class EngineError(RuntimeError):
    """The engine cannot be built as the manifest says, or does not run what it was asked to."""


def engine_args(model: str, d: Deployment, attention: Mapping[str, Any]) -> dict[str, Any]:
    """The engine's arguments from the resolved deployment and the GPU backend's attention kernel. Pure, so it is
    tested without vLLM."""
    return {"model": model, "dtype": "bfloat16", "gpu_memory_utilization": d.memory_fraction,
            "max_model_len": d.context_length, "enable_prefix_caching": False, "enforce_eager": False,
            "max_num_seqs": d.max_sequences, "max_num_batched_tokens": d.max_batched_tokens,
            "num_gpu_blocks_override": d.kv_blocks, "block_size": d.block_size, "attention_config": dict(attention)}


def computed_blocks(log_text: str) -> int | None:
    """vLLM's own block count, from the line it prints when the override replaces it."""
    found = OVERRIDE_LINE.findall(log_text)
    return int(found[-1][0]) if found else None


def prompt_ids(model: str, p_max: int) -> list[int]:  # pragma: no cover - needs the LLM's tokenizer
    """p_max token ids; every request's prompt is a prefix of them, so none is tokenized while the clock runs."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    if getattr(tok, "chat_template", None):
        out = tok.apply_chat_template([{"role": "user", "content": PROMPT_TEXT}], add_generation_prompt=True,
                                      tokenize=True)
        if hasattr(out, "keys"):
            out = out["input_ids"]
        out = list(out)
        if out and isinstance(out[0], list | tuple):
            out = list(out[0])
        ids = [int(i) for i in out]
    else:
        ids = list(tok(PROMPT_TEXT)["input_ids"])
    if len(ids) < p_max:
        ids += list(tok("word " * (p_max - len(ids)))["input_ids"])
    ids = ids[:p_max]
    if len(ids) != p_max:
        raise EngineError(f"the prompt could not be built to exactly {p_max} tokens (got {len(ids)})")
    return ids


class VllmEngine:  # pragma: no cover - needs a GPU and vLLM
    """vLLM's async engine behind `closed_loop.Engine`."""

    def __init__(self, model: str, d: Deployment, backend: Backend) -> None:
        import torch
        import vllm
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        if not torch.cuda.is_available():
            raise EngineError("no CUDA device: profile serving runs on the GPU host")
        self.vllm_version = str(vllm.__version__)
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args(model, d, backend.attention)))
        config = getattr(self.engine, "vllm_config", None) or getattr(getattr(self.engine, "engine", None),
                                                                      "vllm_config", None)
        mismatch = backend.attention_mismatch(config)
        if mismatch is not None:
            self.shutdown()
            raise EngineError(mismatch)
        self.config = config

    def kv_blocks(self) -> int | None:
        """The KV-cache blocks the engine runs with, read back from its cache configuration."""
        for path in ("cache_config", "vllm_config.cache_config", "engine.cache_config"):
            obj: Any = self.engine
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
            except AttributeError:
                continue
            blocks = getattr(obj, "num_gpu_blocks", None) or getattr(obj, "num_gpu_blocks_override", None)
            if blocks:
                return int(blocks)
        return None

    async def generate(self, request_id: str, prompt: Sequence[int], max_tokens: int) -> AsyncIterator[int]:
        from vllm import SamplingParams, TokensPrompt

        params = SamplingParams(ignore_eos=True, max_tokens=int(max_tokens), temperature=0.0)
        async for out in self.engine.generate(TokensPrompt(prompt_token_ids=list(prompt)), params,
                                              request_id=request_id):
            if out.outputs:
                yield len(out.outputs[0].token_ids)

    async def abort(self, request_id: str) -> None:
        await self.engine.abort(request_id)

    def shutdown(self) -> None:
        stop = getattr(self.engine, "shutdown", None)
        if callable(stop):
            stop()
