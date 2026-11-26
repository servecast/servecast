# Reproduce the paper

`servecast replicate` reruns the experiments of [the ServeCast paper](../cite.md) with the paper's pinned components, calibrations,
LLMs, traces and settings, and every output names those components. There is no option to change the versions,
so the numbers cannot drift with a newer release.

## The experiments

```bash
servecast replicate list
```

| experiment | what it reproduces | recomputed |
|---|---|---|
| `rq1` | accuracy, with ServeCast and Vidur each calibrated on the LLM and GPU it predicts, overall and by load band | 324 load points |
| `rq2` | transfer, a calibration of one LLM predicting another LLM on the same GPU | 1,944 load points |
| `rq3` | cost, the profiling GPU time and the simulation time against Vidur's | the training MAPE, from the 18 calibrations fitted again |
| `ablation` | what the model's parts are worth: the residual correction, the length model, the clock, the workload law | from the ablated models' shipped predictions; the workload laws fitted again |
| `appendix` | the calibration coefficients and the simulation run times | the coefficients, from the 18 calibrations fitted again |

The paper covers three GPUs (A100-SXM4-40GB, H100 SXM, RTX 3090 Ti), six LLMs (Llama-3.2-3B, Llama-3-8B,
Qwen2.5-3B, Qwen2.5-7B, DeepSeek-Coder-1.3B, DeepSeek-LLM-7B) and three traces (Azure conversation, BurstGPT,
ServeGen), with six client levels for each and a mean think time of 1 s. Each table in the output shows the number
it has in the arXiv version beside its name, for example `rq2-matrix-ttft (Table 4)`, and a table that answers a
research question is named after it.

## Check the published tables

```bash
servecast replicate verify
```

`verify` recomputes the accuracy table, its load bands and the three transfer matrices from the shipped per-load-point
predictions. It fits the paper's 18 calibrations again from the profiling data that ship with them, for the appendix's
coefficients and the cost table's training MAPE. It computes the residual, length-model and clock ablations from the
shipped predictions of the ablated models, and the workload-law table from the shipped laws. It then compares every
cell with the printed one at its printed precision, 759 cells in a few seconds with one process per core (`--jobs`
sets how many).

## Recompute an experiment

```bash
servecast replicate run rq1 --from-frozen
servecast replicate run rq1 --jobs 8
servecast replicate run rq2 --gpu h100_sxm --jobs 8
```

With `--from-frozen` the table comes from the shipped predictions. Without it, ServeCast predicts every load point of
the experiment again with the shipped calibrations, scores the predictions against the paper's measurements and
prints the table in the paper's layout. `rq1` recomputes 324 load points and `rq2` 1,944, which takes hours on one
core. `--gpu` restricts the run to one GPU. On the reference platform (macOS on Apple silicon, Python 3.12, numpy
2.5.0, scipy 1.18.0) every recomputed prediction equals the shipped one to the last bit, and the output says how many
did (`recomputed.identical_to_shipped`). Other platforms may differ in the last digits.

```bash
servecast replicate run appendix --jobs 8
```

`rq3` and `appendix` fit the paper's 18 calibrations again from the profiling data that ship with them, in seconds.
They print the training MAPE and the coefficients these fits give, and the output says how many fits equal the
shipped calibration to the last bit (`refitted.identical_to_shipped`), all 18 on the reference platform. The
profiling minutes and the run times beside them are measurements of the paper's machines, printed as published.

`ablation` computes its three ablation tables from the predictions of the ablated models that ship with ServeCast,
the reduced residual corrections, the reduced length models and the clock treatments at 900 and 750 MHz, beside the
measurements they were scored against. It does not predict them again. It also fits the workload law of each trace
again from the shipped trace, in up to about 4 minutes per trace, and says how many laws equal the shipped ones
(`refitted_laws.identical_to_shipped`). With `--from-frozen` it reads the shipped laws instead.

## Predict one configuration

```bash
servecast replicate predict --with a100_sxm_40gb/d9cbf1d3/llama-3-8b --to qwen2.5-7b --trace burstgpt --clients 66
```

This predicts one load point of the paper with the calibration of one LLM (`--with`) for another LLM (`--to`) and
prints the paper's measurement beside it. The client levels differ by configuration, and the command lists them when
asked for one that does not exist.

## Vidur

Vidur is not run. Its predictions ship as the paper recorded them, made with FlashAttention-2 profiles measured on
each GPU for the paper.

The shipped data cannot reproduce the KV-cache occupancy of Fig. 3 and its medians, which need the per-request
measurements of the data record. The profiling minutes of Table 5 and the run times of Table 12 are measurements of
the paper's machines, printed as published.

## The data record

The paper's raw data (the calibration probes and the per-request measurements) will be published as a separate
record. `servecast replicate fetch` downloads it once it is published. Nothing on this page needs it.
