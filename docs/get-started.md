# Get started

ServeCast predicts the serving performance of an LLM that vLLM serves on a GPU, so that you can check it against your
service-level objectives (SLOs) before you deploy. For a given workload, it predicts the time to first token (TTFT)
and the time per output token (TPOT), as means and 95th percentiles, and the throughput in completed requests per
second.

This page shows how to use ServeCast on an example that ships with it: Qwen2.5-7B served by vLLM on an A100-SXM4-40GB
under a fixed load of 64 concurrent clients. Each client sends a prompt, waits for the whole response, then pauses for
a random time, one second on average, before sending the next prompt.

## Install

Install ServeCast as [Install](install.md) describes. Predicting needs no GPU.

## 1. Fit ServeCast to the GPU

ServeCast prices each engine step of vLLM with a cost law, whose coefficients it fits to measurements of the GPU, called
its profiling data. This release has two versions of the cost law, the one of [the paper](cite.md) and a research preview that is
more accurate and transfers better to LLMs not used to calibrate the GPU. The example uses the research preview. Its
profiling data for an A100-SXM4-40GB ship with ServeCast under the name `a100_sxm_40gb/c62ebe75`, the GPU followed by
the name of the law.

This command fits the cost law to those data and saves the result, a calibration, under the name given with
`--name`.

```bash
servecast calibrate --data a100_sxm_40gb/c62ebe75 --name my-a100
```

It takes a few seconds and saves the calibration as `my-a100`, the name the manifest of step 2 uses. The
calibration goes to your data directory, which `servecast doctor` names, and not to the directory you run in, so the
name stays taken. Run this step again with `--overwrite` to replace it.

## 2. Predict

A manifest is a file of inputs, which the commands that read one take by its path. ServeCast ships an example,
`example-a100`, inside the installed package. The first command writes it to `example-a100.yaml`, a file of yours.
The second predicts from that file, whose calibration is the one of step 1, prints the prediction and saves it in
`servecast-output/`.

```bash
servecast manifest show example-a100 > example-a100.yaml
servecast predict -c example-a100.yaml
```

```text title="Output"
calibration: my-a100
 clients  think ms   TTFT ms  P95 TTFT  TPOT ms  P95 TPOT    req/s     tok/s  converged
      64      1000     213.3     463.4    33.24     41.32    7.528    1657.7  yes
```

### The manifest

A manifest is a YAML file that holds every input of a prediction. This is `example-a100.yaml`, as the first command
of step 2 wrote it. To change the example, edit this file. To make another manifest, copy it.

```yaml title="example-a100.yaml"
engine:
  model: Qwen/Qwen2.5-7B
  hardware: a100_sxm_40gb
  workers:
    aggregated:
      kv_cache:
        capacity: {blocks: 23488}
calibration:
  use: my-a100
  transfer: true
traffic:
  source: {type: trace, paths: [azure-conversation]}
  load: {type: concurrency, concurrency: 64, think_time_ms: 1000}
  stop: {steady_state: {}}
```

| field | meaning |
|---|---|
| `engine.model` | The LLM, as a Hugging Face id. `servecast list models` lists the LLMs ServeCast knows without downloading anything. For any other dense LLM, ServeCast reads its `config.json`. |
| `engine.hardware` | The GPU, as its id in ServeCast's registry (`servecast list gpus`). It must be the calibration's GPU. |
| `kv_cache.capacity.blocks` | The size of vLLM's KV cache, in blocks of 16 tokens. vLLM logs it at start-up as `GPU KV cache size: T tokens`, which is T / 16 blocks. ServeCast does not estimate it ([why](limits.md#the-kv-cache-size-is-an-input)). |
| `calibration.use` | The calibration, here `my-a100`, the one step 1 saved. It can also be one that ships with ServeCast (`servecast list calibrations`). |
| `calibration.transfer` | The prediction is a transfer, for an LLM the calibration was not fitted on. The research-preview law is fitted on synthetic models, so every LLM it predicts is a transfer. Without this field, `predict` refuses such an LLM ([transfer](guides/predict.md#choose-the-calibration-and-the-llm)). |
| `traffic.source` | Where each request's number of input and output tokens comes from. `azure-conversation` is a trace that ships with ServeCast, as are `burstgpt` and `servegen`. [A file of your own](guides/predict.md#input-and-output-tokens) also works. |
| `traffic.load` | 64 clients. Each one sends a request, waits for its last token, and then waits again for an exponential time with a mean of 1,000 ms, its think time, before sending the next. |
| `traffic.stop` | The prediction stops once the 95 % confidence interval of every metric is narrow enough. By default the interval's half-width must fall below 10 % of the predicted value for TTFT, the end-to-end latency and the 95th percentiles, and below 5 % for TPOT and the throughput. `traffic.stop.steady_state` sets other targets. |

`engine.model` and `engine.hardware` are names that ServeCast looks up in two files that ship with it. `models.json`
holds each LLM's architecture, such as its layers, widths and attention heads
([fields](reference/formats.md#models-modelentry)), and `gpus.json` each GPU's datasheet numbers, such as its peak
compute and memory bandwidth ([fields](reference/formats.md#gpus-gpuentry)). The cost law prices each engine step of vLLM
from these numbers, and an LLM that `models.json` does not list is read from its `config.json`.

The manifest sets only what this example needs. Every other vLLM setting, such as the maximum number of tokens in a
batch, takes the value vLLM 0.25.1 uses by default on this GPU. `servecast manifest explain <field>` prints what any
field means, for example `servecast manifest explain traffic.load.think_time_ms`.

### The output

The table has one row per load point. This manifest has one, the 64 clients with a mean think time of 1,000 ms that
`traffic.load` sets. You can also predict your own workload, from the number of input and output tokens of your
requests and a list of load levels, and the table then has one row per level
([Bring your own workload](guides/predict.md#bring-your-own-workload)).

| column | meaning |
|---|---|
| `clients`, `think ms` | The load point, from the manifest. |
| `TTFT ms`, `P95 TTFT` | The mean and the 95th percentile of the time to first token, in milliseconds. |
| `TPOT ms`, `P95 TPOT` | The mean and the 95th percentile of the time per output token, in milliseconds. |
| `req/s` | The throughput, in completed requests per second. |
| `tok/s` | The output tokens per second of the completed requests. Unlike the other metrics, it has no confidence interval. |
| `converged` | Whether the 95 % confidence interval of every metric became narrow enough before the prediction stopped. |

The `calibration` line names the calibration. The files in `servecast-output/` hold the same prediction with the 95 %
interval of each metric ([Files and folders](reference/files.md)).

## Next

- [Predict](guides/predict.md) covers predicting another LLM, a load curve and a trace of your own.
- [On your GPU](guides/calibrate.md) runs this example on your own GPU and compares the measurement with the
  prediction.
- [How ServeCast models vLLM](concepts.md) explains the model, and [Limits](limits.md) what it leaves out.
- `servecast list manifests` lists the manifests that ship with ServeCast, `servecast manifest show` prints one to
  save as a file of yours, and `servecast manifest init` prints a commented template for writing your own.
