# Limits

ServeCast predicts one serving setup well and refuses the others rather than guess. This page lists what is in scope
and why each exclusion exists.

## In scope

- Dense decoder-only LLMs with 16-bit weights (bf16 or fp16), described by a Hugging Face `config.json`.
- One GPU, with no tensor or pipeline parallelism.
- vLLM 0.25.1 with FlashAttention 2, continuous batching and chunked prefill, prefix caching off.
- Closed workloads, with N clients and an exponential think time.
- Python 3.11 to 3.14 for the commands that run on any computer.

The per-GPU cost law, `c62ebe75`, is calibrated for the A100-SXM4-40GB in this release. It was fitted on synthetic
models and checked on the six LLMs of the paper, which have 12 to 36 layers and widths from 1024 to 4096, and its grid
of synthetic layers covers feed-forward ratios from 2.67 to 5.5. The calibrations of the law of
[the paper](cite.md), `d9cbf1d3`, cover the A100-SXM4-40GB, the H100 SXM and the RTX 3090 Ti for the six LLMs of the
paper.

## Not modelled

| what | why |
|---|---|
| tensor and pipeline parallelism | the cost laws price one GPU; communication between GPUs is not in them. A manifest with `parallelism.tensor` above 1 is refused |
| mixture-of-experts models | the cost of a beat depends on which experts the tokens reach, which the laws do not represent. Refused when `config.json` names experts |
| quantized weights (FP8, FP4, INT8, INT4) | the laws assume 16-bit weights and the matching kernels |
| prefix caching and shared prefixes | the model computes every prompt in full. A trace whose prefix-block ids repeat is accepted with a warning |
| open-loop arrivals (Poisson or a trace's timestamps) | the population model and the estimator are built for closed workloads |
| speculative decoding | a beat would emit a variable number of tokens per request |
| disaggregated prefill and decode | the model has one engine serving both phases |
| other serving engines, or other vLLM versions | the scheduler rules and the kernels are vLLM 0.25.1's |
| multi-node serving | as for tensor parallelism |

## The KV-cache size is an input

ServeCast does not estimate how many KV-cache blocks the engine will get. The manifest gives the count, in blocks
(`engine.workers.aggregated.kv_cache.capacity.blocks`). vLLM prints its own count when it starts, as "GPU KV cache
size: T tokens", which is T / block_size blocks. `--num-gpu-blocks-override` imposes a count on the engine, and
`servecast profile serving` passes the manifest's count that way, so the model and the measurement serve the same
capacity.

## A Python API

There is none. The command line and the files it writes are the interface, and their formats are versioned
([file formats](reference/formats.md)). The Python modules are internal and may change in any release.
