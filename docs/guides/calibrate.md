# On your GPU

This page runs on your GPU what [Get started](../get-started.md) predicted. It measures the GPU and fits its
calibration, predicts the same manifest with it, then serves the manifest with vLLM and compares the measurement with
the prediction. The example is the one in Get started: Qwen2.5-7B on an A100-SXM4-40GB under 64 clients.

Steps 1 and 4 measure on the GPU, so they need Linux and an NVIDIA GPU. Steps 2, 3 and 5 need no GPU and run on any
operating system.

Install ServeCast with the `profile` extra, and check with `servecast doctor` that this computer can profile, as
[Install](../install.md) describes.

Steps 1, 3 and 4 read the manifest from a file of yours, `example-a100.yaml`, the example that ships with ServeCast.
This command writes it in the current directory, as Get started does.

```bash
servecast manifest show example-a100 > example-a100.yaml
```

## 1. Measure the GPU

```bash
servecast profile calibration -c example-a100.yaml --output-dir profiling
```

This command measures the GPU for the manifest's cost law, the research preview. It serves 8 small synthetic models
with random weights, timing each step of batches of controlled sizes. Then it times a grid of matrix multiplications
of different shapes. From the manifest it reads the GPU and the cost law. The probes run with their own vLLM settings,
which are the same on every GPU.

Before it measures, it locks the GPU's clock at the highest level the GPU holds under a dense load, trying the levels
from the top down, and it releases the lock at the end. On the A100-SXM4-40GB this page was run on, it locked at 1200
MHz, because at 1350 and 1275 MHz the power cap held the clock down. Locking needs root, or `sudo` without a password.
Where the clock cannot be locked, it measures the clock instead. Without the lock, the profiling data are written only
if the 8 synthetic models and the grid ran at the same clock.

It writes to `profiling/` the profiling data (`profiling.json`), every probe step with its audit (one
`probe-<model>.json` per synthetic model, and `gemm.json` for the grid), the GPU's description with the clock levels it
tried (`environment.json`), vLLM's output (`engine.log`) and the files' checksums (`SHA256SUMS`). A probe that fails its
audit stops the run, and no profiling data are written.

## 2. Fit the calibration

```bash
servecast calibrate --data profiling --name my-a100 --overwrite
```

This command fits the cost law to the profiling data and saves the calibration as `my-a100`, the name the manifest
uses. `--overwrite` replaces the `my-a100` that Get started fitted from the profiling data that ship with ServeCast. It
prints the fit's error on the probe's measurements, as the paper reports it.

`--data` takes the folder step 1 wrote, by a relative or an absolute path, the `profiling.json` file itself under any
name, or the name of profiling data that ship with ServeCast, as in Get started. The fit reads only `profiling.json`,
so it can run on another computer.

## 3. Predict

```bash
servecast predict -c example-a100.yaml --with my-a100 --output-dir prediction
```

This is the Get started prediction, now made with your GPU's calibration at the clock it was measured at. `--with`
names the calibration step 2 saved, which is also the one the manifest uses. It prints the prediction and saves it in
`prediction/`.

## 4. Measure the serving

```bash
servecast profile serving -c example-a100.yaml --output-dir measurement
```

This command serves the manifest with vLLM on the GPU, with the manifest's settings, and measures it the way the
prediction was computed. The 64 clients draw their requests from the same pool as the prediction, and the run stops
once every metric's 95 % confidence interval is narrow enough. Until then it prints a line at each check of that rule,
with the requests served so far and the widest interval, the one the run is waiting for.

Where it can, it locks the clock at the level the calibration was made at, which it reads from the calibration the
manifest names. A manifest whose calibration it cannot find is refused before the GPU is touched, since the
measurement would otherwise run at another clock than the prediction.

It writes to `measurement/` the measured metrics (`measurement.json`), every request (`requests.jsonl`), the pool
(`pool.jsonl`), the GPU's description with the clock (`environment.json`), vLLM's output (`engine.log`) and the
checksums (`SHA256SUMS`).

## 5. Compare

```bash
servecast compare prediction measurement
```

This command scores the prediction against the measurement, metric by metric, and saves the comparison in
`prediction/comparison.json`. It prints one row per metric of each load point.

| column | meaning |
|---|---|
| `metric` | The mean and 95th percentile of TTFT and TPOT in milliseconds, requests per second, and output tokens per second. |
| `predicted`, `measured` | The two values. |
| `measured 95 % CI` | The 95 % confidence interval of the measured value. |
| `error %` | The prediction's error relative to the measurement, positive when the prediction is higher. |
| `in CI` | Whether the predicted value falls within the measured interval. |

A `flag:` line names anything that differs between the two runs, such as the clock or vLLM's version.

On the A100-SXM4-40GB this page was run on, with the clock locked at 1200 MHz in steps 1 and 4, step 5 printed the
following.

```text
clients think ms  metric            predicted   measured        measured 95 % CI  error %  in CI
     64     1000  mean TTFT ms         212.38     197.83  [   178.35,    217.14]     +7.4  yes
     64     1000  P95 TTFT ms          456.38     399.75  [   379.91,    425.22]    +14.2  no
     64     1000  mean TPOT ms          32.97      33.59  [    33.35,     33.83]     -1.8  no
     64     1000  P95 TPOT ms           40.04      40.05  [    39.55,     40.70]     -0.0  yes
     64     1000  requests/s             7.57       7.33  [     7.25,      7.42]     +3.2  no
     64     1000  output tokens/s     1678.09    1650.51                       -     +1.7  -
```

Read `error %` first. Here the means and the throughput are within 10 % of the measurement. `in CI` depends on how
narrow the measured interval is as much as on the prediction. The run stops only when every interval meets its target,
10 % for TTFT and 5 % for TPOT and the throughput by default. TTFT varies the most from one request to the next, so it
gets there last, and by then the TPOT and throughput intervals are much narrower, here ±0.7 % and ±1.2 %. A prediction
2 or 3 % off falls outside them.

## A GPU of your own

The five steps run on an NVIDIA GPU of compute capability 8.0 or newer, which `servecast doctor` reports, and one that
ServeCast knows. For a GPU it does not know, add the GPU to ServeCast's registry, write a manifest that names it, and
run the five steps with that manifest.

### Add the GPU to the registry

`servecast list gpus` lists the GPUs ServeCast knows. ServeCast recognizes the GPU of the computer it runs on by the
name the driver reports, and its cost law prices each engine step of vLLM from the GPU's datasheet numbers. A GPU of your
own goes in `gpus.json` in your data directory, `~/.local/share/servecast` on Linux
([Files and folders](../reference/files.md#your-files)). On a computer with a GPU it does not know, `servecast doctor`
names that file. For an A100 with 80 GB of memory, the file reads as follows.

```json title="gpus.json"
{
  "schema": "servecast-gpus/1",
  "gpus": {
    "a100_sxm_80gb": {
      "device_name": "NVIDIA A100-SXM4-80GB",
      "memory_gib": 80,
      "sm_count": 108,
      "peak_flop_per_s": 312e12,
      "memory_bandwidth_bytes_per_s": 2039e9,
      "boost_clock_mhz": 1410
    }
  }
}
```

The key, `a100_sxm_80gb`, is the GPU's id, the name manifests and calibrations use for it. The name comes from the
driver, and every number from the GPU's datasheet.

| field | value |
|---|---|
| `device_name` | The name exactly as `nvidia-smi --query-gpu=name --format=csv,noheader` prints it. |
| `memory_gib` | The memory, in GiB. |
| `sm_count` | The number of streaming multiprocessors. |
| `peak_flop_per_s` | The dense BF16 tensor-core peak, in FLOP/s. A datasheet that quotes the peak with sparsity gives twice the dense value. |
| `memory_bandwidth_bytes_per_s` | The memory bandwidth, in bytes per second. |
| `boost_clock_mhz` | The boost clock the peak is quoted at, in MHz. |

Each computer that runs a step for this GPU needs the same entry, the GPU host for steps 1 and 4 and the computer that
calibrates and predicts. A calibration records the numbers it was fitted with, and `predict` refuses it if the
registry's numbers differ.

### Write the manifest

Copy `example-a100.yaml` into a file of your own, `my-gpu.yaml`, and change the GPU, the calibration and the size of the
KV cache.

```yaml title="my-gpu.yaml"
engine:
  model: Qwen/Qwen2.5-7B
  hardware: a100_sxm_80gb
  workers:
    aggregated:
      kv_cache:
        capacity: {blocks: 23488}
calibration:
  use: my-gpu
  transfer: true
traffic:
  source: {type: trace, paths: [azure-conversation]}
  load: {type: concurrency, concurrency: 64, think_time_ms: 1000}
  stop: {steady_state: {}}
```

`engine.hardware` is the GPU's id, and `calibration.use` the name step 2 gives the calibration.

`kv_cache.capacity.blocks` is the size of the KV cache, in blocks of 16 tokens, and ServeCast does not estimate it.
Step 3 simulates it and step 4 makes vLLM run with it, so the prediction and the measurement serve with the same cache.
For a deployment that leaves vLLM to size its own cache, it is the size vLLM logs when it starts, `GPU KV cache size: T
tokens`, which is T / 16 blocks. The manifest above keeps the size of Get started, smaller than the one vLLM gives
itself on 80 GB. ServeCast refuses a size that cannot fit in the GPU's memory next to the LLM's weights, with the
number of blocks that fit at most, and `compare` flags a size larger than the one vLLM computed.

### Run the five steps

Run steps 1 to 5 with this manifest, and give the calibration the name the manifest uses.

```bash
servecast profile calibration -c my-gpu.yaml --output-dir profiling
servecast calibrate --data profiling --name my-gpu
servecast predict -c my-gpu.yaml --output-dir prediction
servecast profile serving -c my-gpu.yaml --output-dir measurement
servecast compare prediction measurement
```

Step 1 probes the 8 synthetic models with 9,965 KV-cache blocks each, as the paper did, which takes a GPU with 40 GB
of memory or more. On a smaller GPU, calibrate the paper's law instead, by writing `versions: 2026-10` in the
manifest. Step 1 then probes the manifest's LLM itself, and the calibration is fitted on that LLM.

## An LLM of your own

A calibration of the research preview belongs to the GPU. It was fitted on synthetic models, so it predicts any LLM on
that GPU, and a new LLM needs no new calibration. Copy `example-a100.yaml` into `my-llm.yaml`, change the LLM, and
give the size of its KV cache, since vLLM sizes the cache per LLM. For Qwen2.5-3B on the A100-SXM4-40GB of this page,
the file reads as follows.

```yaml title="my-llm.yaml"
engine:
  model: Qwen/Qwen2.5-3B
  hardware: a100_sxm_40gb
  workers:
    aggregated:
      kv_cache:
        capacity: {blocks: 52875}
calibration:
  use: my-a100
  transfer: true
traffic:
  source: {type: trace, paths: [azure-conversation]}
  load: {type: concurrency, concurrency: 64, think_time_ms: 1000}
  stop: {steady_state: {}}
```

`engine.model` is a Hugging Face id, or a local directory holding the LLM's `config.json`. ServeCast reads only that
file, never the weights, and `servecast list models` lists the LLMs it knows without a network. It predicts dense LLMs
with 16-bit weights, and the research preview was checked on LLMs of 24 to 36 layers
([Limits](../limits.md#in-scope)). The 52,875 blocks are the size vLLM gave Qwen2.5-3B on this A100 in a run of the
paper.

Predict the LLM, which needs no GPU, then serve it on the GPU and compare, as steps 3 to 5 do.

```bash
servecast predict -c my-llm.yaml --output-dir prediction-3b
servecast profile serving -c my-llm.yaml --output-dir measurement-3b
servecast compare prediction-3b measurement-3b
```

`profile serving` has vLLM download the LLM's weights, so a gated LLM needs `HF_TOKEN` in the environment. On a GPU of
your own, start from `my-gpu.yaml`, with the size vLLM logs for the new LLM on that GPU.

A calibration of the paper's law (`versions: 2026-10`) is fitted on one LLM. It predicts another one by transfer
([Predict](predict.md#choose-the-calibration-and-the-llm)), or step 1, run with the new LLM in the manifest, probes
that LLM and step 2 fits a calibration on it.

## A workload of your own

The calibration belongs to the GPU, so a workload of your own needs no new calibration either. The workload is the
manifest's `traffic`. Its `source` gives each request's numbers of input and output tokens, and its `load` the load
levels. The file below, `my-workload.yaml`, is a copy of `example-a100.yaml` at two load levels, 16 and 64 clients.

```yaml title="my-workload.yaml"
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
  load:
    - {type: concurrency, concurrency: 16, think_time_ms: 1000}
    - {type: concurrency, concurrency: 64, think_time_ms: 1000}
  stop: {steady_state: {}}
```

For requests of your own, `source` takes your trace, or a mean and a variance of the input and output tokens
([Bring your own workload](predict.md#bring-your-own-workload)). Then run steps 3 to 5 with this manifest. `predict`
predicts each level, `profile serving` measures each one in turn, and `compare` pairs them.

```bash
servecast predict -c my-workload.yaml --output-dir prediction-workload
servecast profile serving -c my-workload.yaml --output-dir measurement-workload
servecast compare prediction-workload measurement-workload
```

On a GPU of your own, start from `my-gpu.yaml`.

## On a Slurm cluster

[`examples/slurm/servecast.sbatch`](https://github.com/servecast/servecast/blob/main/examples/slurm/servecast.sbatch)
runs steps 1 to 5 as one job on a GPU node. Activate the environment ServeCast is installed in, set your cluster's
partition, account and time limit in the script's `#SBATCH` lines, and submit it from the directory that should receive
the outputs:

```bash
sbatch servecast.sbatch
```

A compute node without internet needs the LLM, and the configuration and tokenizer of
`deepseek-ai/deepseek-coder-1.3b-base`, from which the synthetic models are built, in a shared Hugging Face cache, with
`HF_HUB_OFFLINE=1` set.
