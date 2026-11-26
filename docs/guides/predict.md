# Predict

`servecast predict` reads a manifest and writes the predicted serving metrics. It runs on any computer, needs no GPU,
and gives the same numbers for the same manifest and seed.

## Write a manifest

```bash
servecast manifest init --template predict > m.yaml
```

The template is commented field by field. To start from the example of Get started instead, `servecast manifest show
example-a100 > m.yaml` writes it. Either way the manifest is a file of yours, given to every command by its path.
`servecast manifest explain <path>` prints any field's meaning, type, default and allowed values, and `servecast
manifest validate -c m.yaml` lists every problem in the file at once. It then refuses what predict would refuse before
running, such as a calibration or a trace that does not exist.

```bash
servecast manifest explain engine.workers.aggregated.scheduler.max_sequences
```

The manifest has four parts.

- `engine` is the deployment: the LLM (`model`), the GPU (`hardware`), and vLLM's settings under
  `workers.aggregated`. Scheduler fields left unset take vLLM 0.25.1's defaults for that GPU. The KV-cache size,
  `kv_cache.capacity.blocks`, is required ([why](../limits.md#the-kv-cache-size-is-an-input)).
- `calibration` names a calibration (`use`) or the profiling data to fit one from (`data`).
- `versions` selects the components: unset, each one's default; a year and month (`2026-11`), each one's newest
  entry up to it; or one by one, by name (`servecast list components`).
- `traffic` is the workload: where each request's number of input and output tokens comes from (`source`), the load
  levels (`load`), when a run is long enough (`stop`), the seed and, for drawn token counts, the pool size.

`--set PATH=VALUE` changes one field from the command line, and `--with` and `--to` are short forms for the
calibration and the LLM.

## Choose the calibration and the LLM

`servecast list calibrations` lists the calibrations installed with ServeCast and the ones you fitted. The newest law
needs one calibration per GPU. The law of [the paper](../cite.md) needs one per LLM and GPU: to run the paper's model, write
`versions: 2026-10` and a calibration `<gpu>/d9cbf1d3/<llm>`, such as `a100_sxm_40gb/d9cbf1d3/qwen2.5-7b`.

```bash
servecast predict -c m.yaml --with a100_sxm_40gb/c62ebe75 --to Qwen/Qwen2.5-3B
```

`--to` takes a Hugging Face id or a local directory holding the LLM's `config.json`. ServeCast reads only that file,
never the weights. A gated model needs `HF_TOKEN` in the environment, and the six LLMs of the paper resolve without a
network. The calibration's GPU must be the manifest's `engine.hardware`.

A prediction for an LLM other than the one the calibration was fitted on is a transfer, and the manifest says so with
`calibration.transfer: true`. Without it, `predict` stops and names the field. A calibration of the research-preview
law is fitted on synthetic models, so every LLM it predicts is a transfer. A calibration of the paper's law is fitted
on one LLM and predicts any other one by transfer. `--to` sets `calibration.transfer: true` itself. ServeCast compares
LLMs by their architecture, so a fine-tune or a local copy of the calibrated LLM is not a transfer.

`--to` keeps the manifest's KV-cache block count, but vLLM sizes the cache per LLM: give the new LLM's own count with
`--set engine.workers.aggregated.kv_cache.capacity.blocks=N`, or the prediction carries a warning. A cache that, with
the weights, needs more than `memory_fraction` of the GPU's memory (0.9 unless set) is refused, with the number of
blocks that would fit.

## Bring your own workload

Your workload enters the manifest in two places. `traffic.source` gives each request's number of input tokens (its
prompt) and output tokens (its response), and `traffic.load` the load levels to run it at.

### Input and output tokens

A trace gives each request's number of input and output tokens. ServeCast reads Mooncake JSONL (`input_length` and
`output_length` on each line) and files in the `lengths` format (`input_tokens` and `output_tokens`, as JSONL or
CSV). The traces of the paper ship with the package under the names `azure-conversation`, `burstgpt` and `servegen`. A
request whose input and output tokens together exceed `engine.context_length` is left out, as vLLM would refuse it,
and the prediction says how many. At least 100 requests must remain, as in a pool of synthetic lengths.

```yaml
traffic:
  source: {type: trace, paths: [my-requests.jsonl], format: mooncake}
```

Without a trace, give a mean and an optional variance for the input tokens and for the output tokens, and ServeCast
draws `pool_size` requests.

```yaml
traffic:
  source: {type: synthetic, input_tokens: {mean: 1200}, output_tokens: {mean: 250, variance: 40000}}
```

A request whose input and output tokens together exceed the context length is drawn again, so the pool can differ
from what the manifest asks for when those numbers come close to the context length; the prediction says so when a
mean or a variance ends more than 5 % off.

`servecast workload describe <trace>` prints a trace's moments and `servecast workload sample -c m.yaml` writes the
pool a prediction would use. A new workload law is fitted on its first use and cached. The fit takes longer for a
larger pool, and the default 10,000-request synthetic pool above took about 6 minutes on an Apple M1 Pro. `--timeout`
bounds the fit as well as the load points.

### Load levels

`traffic.load` holds one level, the number of clients and the mean think time in milliseconds, or a list of levels.
Each level is predicted on its own, at steady state, and reported in the order listed. `profile serving` measures the
same levels, so `compare` pairs them.

```yaml
traffic:
  load:
    - {type: concurrency, concurrency: 16, think_time_ms: 1000}
    - {type: concurrency, concurrency: 64, think_time_ms: 500}
```

`--sweep` repeats the prediction for each value of one field of a single level, and `--jobs` runs the load points in
parallel without changing the numbers. A new `--output-dir` keeps the earlier prediction; `--overwrite` replaces it.

```bash
servecast predict -c m.yaml --sweep traffic.load.concurrency=16,32,64,128 --jobs 4 --output-dir sweep
```

## Read the results

The command prints a table and writes `servecast-output/` (`--output-dir` changes it).

- `prediction.json` holds, for each load point, the mean and 95th percentile of TTFT and TPOT, the request and token
  throughputs, their 95 % confidence intervals, and whether the run converged. It records the resolved manifest and
  the components that ran, by name.
- `pool.jsonl` is the request pool the prediction drew from, with its digest.
- `requests.jsonl`, with `--capture-per-request`, holds every simulated request of the steady-state window.
- `manifest.yaml` is the manifest as it ran, with every default filled in: `servecast predict -c manifest.yaml`
  predicts the same numbers again.
- `run.json` records the command, the version, and each input with its sha256 ([Files and
  folders](../reference/files.md)).

A run stops when every metric's interval is narrow enough (`traffic.stop.steady_state`). A heavily loaded point can
take minutes. `--timeout SECONDS` stops a point that has not converged by then, writes its numbers marked as not
converged, and exits with status 124.

## Exit statuses

| status | meaning |
|---|---|
| 0 | every load point converged |
| 1 | the computation failed, or a point did not converge |
| 2 | the manifest or an option cannot run; every problem is named with its fix |
| 3 | a resource was refused, such as a non-empty output directory without `--overwrite` |
| 124 | a load point reached `--timeout` |
| 130 | interrupted |
