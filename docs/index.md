---
title: ServeCast
hide:
  - navigation
  - toc
---

<div class="sc-hero" markdown>

<span class="sc-pill"><span class="sc-dot"></span>vLLM 0.25 · A100, H100, RTX 3090 Ti</span>

# Know how vLLM will serve your LLM before you run it

<p class="sc-lead">ServeCast predicts the serving performance of an LLM that vLLM serves on a GPU, so that you can check it against your service-level objectives (SLOs) before you deploy. For a given workload, it predicts the time to first token (TTFT) and the time per output token (TPOT), as means and 95th percentiles, and the throughput in completed requests per second. A prediction takes seconds on a laptop,
without a GPU.</p>

<p class="sc-lead" markdown>ServeCast was introduced in the paper *ServeCast: Predicting LLM Serving Throughput and Latency
Across Models from Microbenchmarks* (Incerto, Yang, Sun and Tribastone, 2026). Its experiments ship with ServeCast,
so they can be replicated, and the tool also predicts GPUs, workloads and scenarios the paper did not study.</p>

[Get started](get-started.md){ .md-button .md-button--primary }
[How it works](concepts.md){ .md-button }
[Cite](cite.md){ .md-button }

</div>

## What you can do with it

<div class="sc-bento" markdown>

<div class="sc-card sc-feature" markdown>

:lucide-chart-spline:{ .sc-ic }

[Predict](guides/predict.md){ .sc-title }

TTFT, TPOT and throughput, each with its 95&nbsp;% interval, at one load or along a load curve. No GPU needed.

`servecast predict -c m.yaml`{ .sc-cmd }

<div class="sc-stats"><span class="sc-stat"><span class="sc-k">TTFT</span><span class="sc-v">213 ms</span><span class="sc-s">P95 463 ms</span></span><span class="sc-stat"><span class="sc-k">TPOT</span><span class="sc-v">33.2 ms</span><span class="sc-s">P95 41.3 ms</span></span><span class="sc-stat"><span class="sc-k">Throughput</span><span class="sc-v">7.53</span><span class="sc-s">requests / s</span></span></div>

<div class="sc-curve"><svg viewBox="0 0 400 122" role="img" aria-label="Predicted requests per second at 16, 32, 64 and 128 clients: 3.3, 5.4, 7.5 and 8.3"><line x1="14" y1="100" x2="388" y2="100" class="ax"/><path d="M30,100 L30,68.5 L126,48.4 L222,28.5 L318,21.6 L318,100 Z" class="ar"/><polyline points="30,68.5 126,48.4 222,28.5 318,21.6" class="ln"/><circle cx="30" cy="68.5" r="3.6"/><circle cx="126" cy="48.4" r="3.6"/><circle cx="222" cy="28.5" r="3.6"/><circle cx="318" cy="21.6" r="3.6"/><text x="30" y="59.5" text-anchor="middle" class="v">3.3</text><text x="126" y="39.4" text-anchor="middle" class="v">5.4</text><text x="222" y="19.5" text-anchor="middle" class="v">7.5</text><text x="318" y="12.6" text-anchor="middle" class="v">8.3</text><text x="30" y="116" text-anchor="middle">16</text><text x="126" y="116" text-anchor="middle">32</text><text x="222" y="116" text-anchor="middle">64</text><text x="318" y="116" text-anchor="middle">128</text><text x="388" y="116" text-anchor="end" class="u">clients</text><text x="14" y="12" class="u">requests / s, from 0</text></svg></div>

Qwen2.5-7B on an A100-SXM4-40GB, predicted without a GPU. The tiles are at 64 clients, the curve from 16 to 128 (the
Get started example).
{ .sc-caption }

</div>

<div class="sc-card sc-compact" markdown>

:lucide-gauge:{ .sc-ic }

<div class="sc-body" markdown>

[Measure](guides/calibrate.md#4-measure-the-serving){ .sc-title }

Serve the same manifest with vLLM on the GPU, with the same request pool and the same stopping rule.

`servecast profile serving -c m.yaml`{ .sc-cmd }

</div>

</div>

<div class="sc-card sc-compact" markdown>

:lucide-git-compare-arrows:{ .sc-ic }

<div class="sc-body" markdown>

[Compare](guides/calibrate.md#5-compare){ .sc-title }

Score the prediction against the measurement, load point by load point, with signed errors.

`servecast compare servecast-output servecast-measurement`{ .sc-cmd }

</div>

</div>

<div class="sc-card sc-compact" markdown>

:lucide-sliders-horizontal:{ .sc-ic }

<div class="sc-body" markdown>

[Calibrate](guides/calibrate.md){ .sc-title }

Fit a GPU's cost law once from microbenchmarks, then predict any dense LLM on that GPU.

`servecast calibrate --data profiling --name my-gpu`{ .sc-cmd }

</div>

</div>

</div>

## A first prediction

Three commands make a first prediction with data that ship with ServeCast. The first fits the cost law to an
A100-SXM4-40GB. The second writes the example manifest to a file of yours, `example-a100.yaml`. The third predicts
from it Qwen2.5-7B served by vLLM on that GPU under a fixed load of 64 concurrent clients, each of which pauses one
second on average between a response and its next prompt.

```bash
pip install servecast
# 1. fit the cost law to the profiling data of an A100-SXM4-40GB, which ship with ServeCast
servecast calibrate --data a100_sxm_40gb/c62ebe75 --name my-a100
# 2. write the example manifest that ships with ServeCast to a file of yours, example-a100.yaml
servecast manifest show example-a100 > example-a100.yaml
# 3. predict Qwen2.5-7B on that GPU, from that file
servecast predict -c example-a100.yaml
```

```text
calibration: my-a100
 clients  think ms   TTFT ms  P95 TTFT  TPOT ms  P95 TPOT    req/s     tok/s  converged
      64      1000     213.3     463.4    33.24     41.32    7.528    1657.7  yes
```

`prediction.json` holds the same numbers with their 95 % intervals, the manifest with every default filled in, and
the components that produced them, by name. The calibration is saved in your data directory, so the first command
stops when it is run again; add `--overwrite` to fit `my-a100` anew. [Get started](get-started.md) explains each
step.

## A load curve in one command

`--sweep` repeats the prediction for each value of one field. Here it doubles the clients from 16 to 128, and
`--jobs` runs the four points in parallel without changing their numbers.

```bash
servecast predict -c example-a100.yaml --sweep traffic.load.concurrency=16,32,64,128 --jobs 4 --output-dir curve
```

```text
calibration: my-a100
 clients  think ms   TTFT ms  P95 TTFT  TPOT ms  P95 TPOT    req/s     tok/s  converged
      16      1000     106.3     238.0    16.62     18.86    3.311     751.2  yes
      32      1000     136.5     300.1    20.99     25.37    5.432    1213.5  yes
      64      1000     213.3     463.4    33.24     41.32    7.528    1657.7  yes
     128      1000     360.6     742.1    63.06     74.93    8.255    1855.0  yes
```

Doubling the clients from 64 to 128 adds little throughput while TTFT and TPOT keep growing. This deployment is near
its throughput limit for this workload.

## How it works

<div class="sc-flow">
<svg viewBox="-20 -18 1040 420" role="img" aria-label="The manifest and the GPU's calibration go to servecast predict, which writes prediction.json without a GPU. The same manifest goes to servecast profile serving, which writes measurement.json on the GPU, with the same request pool and the same stopping rule. servecast compare reads both and writes comparison.json.">
<!-- the icons are Lucide's (ISC License, https://lucide.dev) -->
<!-- arrow heads and the cards' shadow -->
  <defs id="sc-flow-defs">
  <marker id="sc-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ah"/></marker>
  <filter id="sc-sh" x="-10%" y="-10%" width="120%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f1720" flood-opacity="0.10"/></filter>
  </defs>
  <rect x="-20" y="-18" width="1040" height="420" rx="18" class="bg"/>
<!-- the inputs -->
  <rect x="0" y="0" width="236" height="384" rx="16" class="grp"/>
  <text x="18" y="28" class="pl">INPUTS</text>
  <rect x="14" y="44" width="208" height="120" rx="12" class="cd"/>
  <rect x="28" y="58" width="34" height="34" rx="9" class="it"/>
  <svg x="33" y="63" width="24" height="24" viewBox="0 0 24 24" class="ico"><path d="M10 5H3M12 19H3M14 3v4M16 17v4M21 12h-9M21 19h-5M21 5h-7M8 10v4M8 12H3"/></svg>
  <text x="74" y="81" class="t">Calibration</text>
  <text x="28" y="120" class="s">the GPU's cost law,</text>
  <text x="28" y="141" class="s">shipped or fitted on the GPU</text>
  <rect x="14" y="178" width="208" height="192" rx="12" class="cd"/>
  <rect x="28" y="192" width="34" height="34" rx="9" class="it"/>
  <svg x="33" y="197" width="24" height="24" viewBox="0 0 24 24" class="ico"><path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"/><path d="M14 2v5a1 1 0 0 0 1 1h5M10 9H8M16 13H8M16 17H8"/></svg>
  <text x="74" y="208" class="t">Manifest</text>
  <text x="74" y="226" class="m">m.yaml</text>
  <text x="28" y="262" class="s">the LLM and the GPU</text>
  <text x="28" y="284" class="s">vLLM's settings</text>
  <text x="28" y="306" class="s">the KV-cache size</text>
  <text x="28" y="328" class="s">the workload</text>
<!-- the prediction, without a GPU -->
  <rect x="300" y="24" width="290" height="150" rx="12" class="cd"/>
  <rect x="316" y="40" width="132" height="24" rx="12" class="cb"/>
  <text x="328" y="56.5" class="ct cbt">PREDICT · NO GPU</text>
  <svg x="316" y="80" width="20" height="20" viewBox="0 0 24 24" class="ico icb"><path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M7 16c.5-2 1.5-7 4-7 2 0 2 3 4 3 2.5 0 4.5-5 5-7"/></svg>
  <text x="344" y="96" class="mt">servecast predict</text>
  <text x="316" y="132" class="s">simulates vLLM's scheduler,</text>
  <text x="316" y="153" class="s">one engine step at a time</text>
<!-- what the two runs share -->
  <line x1="445" y1="174" x2="445" y2="210" class="dl"/>
  <rect x="318" y="181" width="254" height="22" rx="11" class="pill"/>
  <text x="445" y="196" class="pt" text-anchor="middle">same request pool, same stopping rule</text>
<!-- the measurement, on the GPU -->
  <rect x="300" y="210" width="290" height="150" rx="12" class="cd"/>
  <rect x="316" y="226" width="170" height="24" rx="12" class="cy"/>
  <text x="328" y="242.5" class="ct cyt">MEASURE · ON THE GPU</text>
  <svg x="316" y="266" width="20" height="20" viewBox="0 0 24 24" class="ico icy"><path d="m12 14 4-4M3.34 19a10 10 0 1 1 17.32 0"/></svg>
  <text x="344" y="282" class="mt">servecast profile serving</text>
  <text x="316" y="318" class="s">vLLM serves the same workload</text>
  <text x="316" y="339" class="s">and records every request</text>
<!-- the outputs and the comparison -->
  <rect x="618" y="84" width="156" height="30" rx="8" class="fl"/>
  <svg x="627" y="91" width="16" height="16" viewBox="0 0 24 24" class="icf"><path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"/><path d="M14 2v5a1 1 0 0 0 1 1h5M10 12a1 1 0 0 0-1 1v1a1 1 0 0 1-1 1 1 1 0 0 1 1 1v1a1 1 0 0 0 1 1M14 18a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1 1 1 0 0 1-1-1v-1a1 1 0 0 0-1-1"/></svg>
  <text x="649" y="103.5" class="fm">prediction.json</text>
  <rect x="618" y="270" width="156" height="30" rx="8" class="fl"/>
  <svg x="627" y="277" width="16" height="16" viewBox="0 0 24 24" class="icf"><path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"/><path d="M14 2v5a1 1 0 0 0 1 1h5M10 12a1 1 0 0 0-1 1v1a1 1 0 0 1-1 1 1 1 0 0 1 1 1v1a1 1 0 0 0 1 1M14 18a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1 1 1 0 0 1-1-1v-1a1 1 0 0 0-1-1"/></svg>
  <text x="649" y="289.5" class="fm">measurement.json</text>
  <rect x="806" y="96" width="194" height="192" rx="12" class="cd"/>
  <rect x="822" y="112" width="90" height="24" rx="12" class="cg"/>
  <text x="834" y="128.5" class="ct cgt">COMPARE</text>
  <svg x="822" y="152" width="20" height="20" viewBox="0 0 24 24" class="ico icg"><circle cx="5" cy="6" r="3"/><path d="M12 6h5a2 2 0 0 1 2 2v7"/><path d="m15 9-3-3 3-3"/><circle cx="19" cy="18" r="3"/><path d="M12 18H7a2 2 0 0 1-2-2V9"/><path d="m9 15 3 3-3 3"/></svg>
  <text x="848" y="168" class="mt">servecast compare</text>
  <text x="822" y="202" class="s">signed error per metric,</text>
  <text x="822" y="223" class="s">load point by load point</text>
  <rect x="822" y="240" width="162" height="30" rx="8" class="fl"/>
  <svg x="831" y="247" width="16" height="16" viewBox="0 0 24 24" class="icf"><path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"/><path d="M14 2v5a1 1 0 0 0 1 1h5M10 12a1 1 0 0 0-1 1v1a1 1 0 0 1-1 1 1 1 0 0 1 1 1v1a1 1 0 0 0 1 1M14 18a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1 1 1 0 0 1-1-1v-1a1 1 0 0 0-1-1"/></svg>
  <text x="853" y="259.5" class="fm">comparison.json</text>
<!-- the arrows: straight ones level, the others curved -->
  <path d="M222,104 L298,104" class="eg" marker-end="url(#sc-ah)"/>
  <path d="M222,250 C262,250 260,140 298,140" class="eg" marker-end="url(#sc-ah)"/>
  <path d="M222,300 L298,300" class="eg" marker-end="url(#sc-ah)"/>
  <path d="M590,99 L616,99" class="eg" marker-end="url(#sc-ah)"/>
  <path d="M590,285 L616,285" class="eg" marker-end="url(#sc-ah)"/>
  <path d="M774,99 C792,99 786,170 804,170" class="eg" marker-end="url(#sc-ah)"/>
  <path d="M774,285 C792,285 786,214 804,214" class="eg" marker-end="url(#sc-ah)"/>
</svg>
</div>

ServeCast follows every request through vLLM's scheduler one engine step at a time. Requests are admitted while the
KV cache has room, their prompts are computed in chunks within the step's token budget, and each running request
decodes one token per step. The KV cache is a finite pool of blocks. When it runs out, vLLM preempts the most recently
admitted request and recomputes it later, and ServeCast does the same.

The duration of each engine step comes from a cost law. It reads the LLM's architecture from its `config.json`, the
GPU's clock and memory bandwidth, and a few coefficients fitted to microbenchmarks of that GPU. The simulation stops
when the 95 % interval of every metric is narrow enough. A measurement on the GPU applies the same stopping rule to
the real engine, so the two can be compared number by number.

## What it covers

| | Supported |
|---|---|
| Engine | vLLM 0.25.1 on one GPU, FlashAttention 2, prefix caching off |
| GPUs | A100-SXM4-40GB, H100 SXM and RTX 3090 Ti with the shipped calibrations; others once calibrated |
| LLMs | Dense decoder-only transformers with 16-bit weights and attention over the whole context, from a Hugging Face id or a local `config.json`. The A100's calibration covers any of them; on the H100 and the RTX 3090 Ti the shipped calibrations are per LLM, for the paper's six. |
| Workload | N clients that wait an exponential think time between requests; request lengths from a trace or from a mean and a variance. The Azure conversation, BurstGPT and ServeGen traces ship with it. |
| Metrics | TTFT and TPOT, mean and 95th percentile; request and output-token throughput |

Mixture-of-experts, quantized and sliding-window models, tensor parallelism and open-loop arrivals are refused by
name. [Limits](limits.md) lists what is not modelled and why.

## Reproduce the paper

The paper's experiments ship with ServeCast, pinned to the paper's components and calibrations. `replicate verify`
rebuilds the accuracy, transfer and ablation tables from the shipped predictions, fits the paper's 18 calibrations
again from their shipped profiling data, and compares each cell with the printed one, in a few seconds.
`replicate run rq1` and `rq2` recompute the predictions themselves.

```bash
servecast replicate verify
```

```text
759 printed cells compared at their printed precision, 0 differ
```

[Reproduce the paper](guides/reproduce.md) shows how to recompute every prediction from the calibrations.
