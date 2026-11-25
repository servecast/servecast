# File formats

ServeCast's files are its interface. Each carries `schema: servecast-<kind>/<major>`. A reader refuses a major version
it does not know and ignores fields it does not know inside a known one, so a minor release can add fields without
breaking a reader. Units are in the field names. `servecast manifest schema --kind <kind>` prints the JSON Schema of
any of them, for an editor or a validator.

This page is generated from the formats' definitions (`python tools/gen_formats_reference.py`).

## manifest

You write it; every command reads it.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `engine` | Engine |  |  | The deployment: LLM, GPU and engine settings. |
| `calibration` | Calibration |  |  | Which calibration, or what to fit. |
| `versions` | string or VersionsSpec or null |  |  | The components: a year-month (each component's newest entry up to it), or {as_of: ..., &lt;component&gt;: &lt;name&gt;}. Unset: each component's default (servecast list components). |
| `traffic` | Traffic or null |  |  | The workload. Required by predict, profile serving and workload sample. |
| `profile` | Profile |  |  | Read by the profile commands only. |

### manifest: Architecture

| field | type | required | default | meaning |
|---|---|---|---|---|
| `layers` | integer | yes |  | Transformer layers (num_hidden_layers). |
| `hidden` | integer | yes |  | Hidden width (hidden_size). |
| `ffn` | integer | yes |  | Feed-forward width (intermediate_size). |
| `heads` | integer | yes |  | Query heads (num_attention_heads). |
| `kv_heads` | integer | yes |  | Key-value heads (num_key_value_heads). |
| `kv_width` | integer | yes |  | Key-value heads times the head size. |
| `vocab` | integer | yes |  | Vocabulary size (vocab_size). |
| `max_context` | integer | yes |  | The LLM's own context limit (max_position_embeddings); 0 when config.json gives none. |

### manifest: Calibration

| field | type | required | default | meaning |
|---|---|---|---|---|
| `use` | string or null |  |  | A calibration name or path. `--with` sets it. |
| `data` | string or null |  |  | A profiling-data directory to fit a calibration from. |
| `transfer` | boolean |  | `False` | Predict an LLM other than the one the calibration was fitted on: a transfer. A per-GPU calibration is fitted on synthetic models, so every LLM it predicts is a transfer. `--to` sets it. |

### manifest: Capacity

| field | type | required | default | meaning |
|---|---|---|---|---|
| `type` | `fixed` |  | `fixed` | fixed: the engine gets exactly `blocks` KV-cache blocks. The KV capacity is an input, as in the paper; ServeCast does not estimate it. |
| `blocks` | integer | yes |  | KV-cache blocks the engine gets (vLLM --num-gpu-blocks-override). vLLM logs its own count at start-up as 'GPU KV cache size: T tokens' (blocks = T / block_size). |
| `memory_fraction` | number or null |  |  | Fraction of GPU memory vLLM may use (--gpu-memory-utilization), passed to the engine by profile serving. Unset: 0.9, the value the paper's measurements used. predict and profile serving refuse a KV cache that, with the weights, needs more than this fraction of the GPU's memory. |

### manifest: ClockLock

| field | type | required | default | meaning |
|---|---|---|---|---|
| `lock` | integer | yes |  | Lock the GPU clock at this many MHz, if the machine allows it and the clock holds it under load. |

### manifest: Engine

| field | type | required | default | meaning |
|---|---|---|---|---|
| `mode` | `aggregated` |  | `aggregated` | One engine serving prefill and decode. |
| `model` | string or null |  |  | The LLM to predict: a Hugging Face id or a local directory holding config.json. `--to` sets it. |
| `hardware` | string or null |  |  | The GPU, as an id that servecast list gpus lists, e.g. a100_sxm_40gb (shipped, or in gpus.json in your data directory). Required: the GPU predict simulates, with a calibration of that GPU; the GPU profile calibration and profile serving measure, which refuse another; the GPU whose profiling data calibrate fits. |
| `backend` | `vllm` |  | `vllm` | The serving engine. |
| `backend_version` | string or null |  |  | The vLLM version. ServeCast models vLLM 0.25.1, the version every calibration and measurement uses; another is refused. Unset: 0.25.1. |
| `context_length` | integer or `max` |  | `max` | Longest prompt plus output, in tokens; max: the LLM's own limit (its config.json's max_position_embeddings). A trace request longer than this is left out, as vLLM would refuse it. |
| `gpu_clock_mhz` | number or null |  |  | The GPU clock to predict at, in MHz. This version predicts at the calibration's clock only: predict refuses another value, and the profile commands refuse the field (profile.clock locks the clock). Unset: the clock the calibration was made at. |
| `architecture` | Architecture or null |  |  | Written by ServeCast in a resolved manifest: the numbers that define the LLM, read from its config.json. When given, they must be the LLM's. |
| `workers` | Workers |  |  | The engine's settings. |

### manifest: KvCache

| field | type | required | default | meaning |
|---|---|---|---|---|
| `block_size` | integer or null |  |  | Tokens per KV-cache block, a multiple of 16: vLLM 0.25.1's FlashAttention backend takes no other. Unset: 16, vLLM 0.25.1's default on these GPUs. |
| `prefix_caching` | boolean or null |  |  | vLLM's prefix caching. ServeCast computes every prompt in full, so true is refused, and profile serving runs vLLM with it off (its prompts all start with the same tokens). Unset: off. |
| `capacity` | Capacity or null |  |  | How much KV cache the engine gets. Required: ServeCast does not estimate it, and every command refuses a manifest without it. |

### manifest: LengthSpec

| field | type | required | default | meaning |
|---|---|---|---|---|
| `mean` | number | yes |  | Mean length, in tokens. |
| `variance` | number or null |  |  | Variance, in tokens squared. Unset: the length sampler's default; sampler 2026-11 uses mean (mean - 1), the variance of a geometric distribution. |

### manifest: Load

| field | type | required | default | meaning |
|---|---|---|---|---|
| `type` | `concurrency` | yes |  | A closed workload: a fixed number of clients. |
| `concurrency` | integer | yes |  | N, the number of clients. |
| `think_time_ms` | number | yes |  | Mean think time between a response and the client's next request, in milliseconds; exponential. |

### manifest: Parallelism

| field | type | required | default | meaning |
|---|---|---|---|---|
| `tensor` | integer |  | `1` | Tensor-parallel degree. Only 1 is supported in this version. |

### manifest: Profile

| field | type | required | default | meaning |
|---|---|---|---|---|
| `clock` | `auto` or `measured` or ClockLock |  | `auto` | auto locks the GPU clock where this machine allows it (NVML, or sudo -n nvidia-smi): profile calibration at the highest level the clock holds under load, profile serving at the level its calibration was made at. Where the clock cannot be locked, or does not hold, it is measured. measured never locks; {lock: MHz} locks at that level. The clock is always measured and recorded. |
| `poweroff_when_safe` | boolean |  | `False` | Power the machine off once the results are written and match their checksums, as --poweroff-when-safe does. |

### manifest: Scheduler

| field | type | required | default | meaning |
|---|---|---|---|---|
| `max_batched_tokens` | integer or null |  |  | Most tokens in one engine step (vLLM --max-num-batched-tokens). Unset: vLLM 0.25.1's default for `vllm serve`: 8192 on a GPU with at least 70 GiB other than the A100, else 2048. |
| `max_sequences` | integer or null |  |  | Most requests running at once (vLLM --max-num-seqs). Unset: vLLM 0.25.1's default for `vllm serve`: 1024 on a GPU with at least 70 GiB other than the A100, else 256. |

### manifest: SteadyState

| field | type | required | default | meaning |
|---|---|---|---|---|
| `ttft` | number |  | `0.1` | Target relative half-width of TTFT's 95 % interval. |
| `tpot` | number |  | `0.05` | Target relative half-width of TPOT's 95 % interval. |
| `throughput` | number |  | `0.05` | Target relative half-width of throughput's interval. |
| `e2e` | number |  | `0.1` | Target relative half-width of the end-to-end latency's interval. |
| `ttft_p95` | number or null |  |  | Target relative half-width of the 95th percentile of TTFT. Unset: 0.10 with the steady-state estimator 2026-11; percentiles are not judged by 2026-10. |
| `tpot_p95` | number or null |  |  | Target relative half-width of the 95th percentile of TPOT. Unset: 0.10 with the steady-state estimator 2026-11; percentiles are not judged by 2026-10. |

### manifest: Stop

| field | type | required | default | meaning |
|---|---|---|---|---|
| `steady_state` | SteadyState | yes |  | Run until every metric's interval is narrow enough. |

### manifest: SyntheticSource

| field | type | required | default | meaning |
|---|---|---|---|---|
| `type` | `synthetic` | yes |  | Lengths drawn from a mean and a variance. |
| `input_tokens` | LengthSpec | yes |  | The prompt length. |
| `output_tokens` | LengthSpec | yes |  | The output length. |

### manifest: TraceSource

| field | type | required | default | meaning |
|---|---|---|---|---|
| `type` | `trace` | yes |  | Lengths taken from a trace. |
| `paths` | list of string | yes |  | Trace files, or the name of a trace the package ships: azure-conversation, burstgpt, servegen. |
| `format` | `mooncake` or `lengths` |  | `mooncake` | mooncake: JSONL with input_length and output_length; lengths: JSONL or CSV with input_tokens and output_tokens. |

### manifest: Traffic

| field | type | required | default | meaning |
|---|---|---|---|---|
| `source` | TraceSource or SyntheticSource | yes |  | Where request lengths come from. |
| `load` | Load or list of Load | yes |  | How requests arrive: one load level, or a list of levels. Each level is predicted and measured on its own, at steady state, and reported in the order listed. |
| `stop` | Stop | yes |  | When a run is long enough. |
| `seed` | integer |  | `0` | Seed of the request pool and of the simulation. |
| `pool_size` | integer |  | `10000` | Requests in the pool both arms draw from, for synthetic lengths; a trace's pool is its requests that fit the context length, at least 100. |

### manifest: VersionsSpec

| field | type | required | default | meaning |
|---|---|---|---|---|
| `as_of` | string or null |  |  | Each component at its newest entry up to this month. |
| `cost_law` | string or null |  |  | Override: the cost law, by name (or release month). |
| `calibration_procedure` | string or null |  |  | Override: the calibration procedure, part of its cost law's entry: the cost law's name, if set. |
| `population_model` | string or null |  |  | Override: the population model, by name. |
| `workload_fitter` | string or null |  |  | Override: the workload fitter, by name. |
| `length_sampler` | string or null |  |  | Override: the length sampler, by name. |
| `estimator` | string or null |  |  | Override: the steady-state estimator, by name. |

### manifest: Worker

| field | type | required | default | meaning |
|---|---|---|---|---|
| `parallelism` | Parallelism |  |  | How the LLM is split across GPUs. |
| `scheduler` | Scheduler |  |  | The engine's batching limits. |
| `kv_cache` | KvCache |  |  | The engine's KV cache. |

### manifest: Workers

| field | type | required | default | meaning |
|---|---|---|---|---|
| `aggregated` | Worker |  |  | The single engine that serves every phase. |

## pool

`predict` and `workload sample` write `pool.jsonl`: this header line, then one request per line.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `requests` | integer | yes |  | The requests that follow. |
| `sha256` | string | yes |  | Their digest, checked when the pool is read. |
| `source` | `trace` or `synthetic` | yes |  |  |
| `files` | list of string or null |  |  | A trace source: the traces the requests come from. |
| `sampler` | string or null |  |  | A synthetic source: the length sampler, by name. |
| `seed` | integer or null |  |  | A synthetic source: the seed of the draw. |
| `requested` | LengthMoments or null |  |  | A synthetic source: the moments the manifest asked for (a mean alone sets the variance of its law). |
| `realized` | LengthMoments or null |  |  | A synthetic source: the moments of the requests drawn. |
| `laws` | map to object or null |  |  | A synthetic source: the law each length was drawn from. |

### pool: LengthMoments

| field | type | required | default | meaning |
|---|---|---|---|---|
| `input` | Moments | yes |  |  |
| `output` | Moments | yes |  |  |

### pool: Moments

| field | type | required | default | meaning |
|---|---|---|---|---|
| `mean` | number | yes |  |  |
| `variance` | number | yes |  |  |

## profiling-data

`profile calibration` writes it; `calibrate` and `predict` read it.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `gpu` | string | yes |  | The GPU's id (servecast list gpus). |
| `cost_law` | string | yes |  | The cost law the probes were run for, by name (in a file written before names, the month it was released). |
| `calibration_procedure` | string | yes |  | The same entry: the procedure is part of its cost law's. |
| `reference_mhz` | number | yes |  | The SM clock the probes ran at. |
| `llm` | string or null |  |  | A per-LLM law: the LLM probed. |
| `cells` | list of ProbeCell or null |  |  | A per-LLM law: its probe cells. |
| `models` | list of ProbedModel or null |  |  | A per-GPU law: the synthetic models probed. |
| `gemm_grid` | GemmGrid or null |  |  | A per-GPU law: the grid of synthetic layers timed. |

### profiling-data: GemmGrid

| field | type | required | default | meaning |
|---|---|---|---|---|
| `clock_mhz` | number | yes |  | The SM clock the grid ran at; the probes' clock. |
| `runs` | list of GemmRun | yes |  |  |

### profiling-data: GemmPoint

| field | type | required | default | meaning |
|---|---|---|---|---|
| `n` | integer | yes |  | Rows of the matrix multiplications. |
| `s_per_iter` | number | yes |  | Seconds per layer iteration. |
| `flops` | number | yes |  | Floating-point operations per layer iteration. |

### profiling-data: GemmRun

| field | type | required | default | meaning |
|---|---|---|---|---|
| `model` | string | yes |  | The LLM whose layer shapes were timed, or a synthetic layer (synthetic-d&lt;d&gt;-f&lt;d_ff&gt;-k&lt;d_kv&gt;). |
| `reps` | list of list of GemmPoint | yes |  | Repetitions, each a sweep over the row counts. |

### profiling-data: ProbeCell

| field | type | required | default | meaning |
|---|---|---|---|---|
| `q` | number | yes |  | Prefill tokens computed in the iteration. |
| `nd` | number | yes |  | Decoding requests in the iteration. |
| `A` | list of number | yes |  | The prefill attention work, the sum of q (w + q/2) over the prefill chunks (w: tokens already computed). |
| `K` | list of number | yes |  | The decode context: the decoding requests' contexts, summed, in tokens. |
| `n_pf` | list of number | yes |  | Prefill requests in the iteration. |
| `T` | number | yes |  | The measured iteration duration, in ms. |
| `w` | number | yes |  | The cell's weight in the fit, before the procedure's class shares. |
| `cls` | `decode` or `prefill` or `mixed` | yes |  | The probe class: decoding only, prefill only, or both. |

### profiling-data: ProbedModel

| field | type | required | default | meaning |
|---|---|---|---|---|
| `llm` | string | yes |  | The synthetic model probed (synthetic/&lt;family&gt;-l&lt;L&gt;-d&lt;D&gt;-f&lt;F&gt;-h&lt;H&gt;-kv&lt;KV&gt;). |
| `cells` | list of ProbeCell | yes |  |  |

## probe

`profile calibration` writes `probe.json` (for a per-GPU law, one `probe-<model>.json` per synthetic model): every engine step it timed, and its audit.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `gpu` | string | yes |  | The GPU's id. |
| `llm` | string | yes |  | The LLM probed. |
| `cost_law` | string | yes |  | The cost law the probe was designed for, by name. |
| `design` | object | yes |  | The compositions and their order. |
| `spans` | list of ProbeSpan | yes |  | One per composition measured, in the order they ran. |
| `beats` | list of ProbeBeat | yes |  | Every engine step timed. |
| `clock` | list of ClockSample | yes |  | The SM clock, sampled every few beats. |
| `audit` | ProbeAudit | yes |  |  |

### probe: ClockSample

| field | type | required | default | meaning |
|---|---|---|---|---|
| `beat_index` | integer | yes |  | The beat after which the clock was read. |
| `sm_clock_mhz` | number | yes |  | The SM clock, MHz. |

### probe: ProbeAudit

| field | type | required | default | meaning |
|---|---|---|---|---|
| `clean` | boolean | yes |  | No flag: the probe makes profiling data. |
| `flags` | list of string | yes |  | Why the probe makes no profiling data, each a sentence. |
| `compositions_designed` | map to integer | yes |  |  |
| `compositions` | integer | yes |  | Compositions measured. |
| `beats_kept` | integer | yes |  | Beats of the composition their cell was made for. |
| `reference_drift` | number or null | yes |  | max / min - 1 of the reference revisits' mean beat. |
| `clock_held` | boolean | yes |  |  |
| `cells` | list of object | yes |  | Per composition: beats run and kept, and whether enough. |

### probe: ProbeBeat

| field | type | required | default | meaning |
|---|---|---|---|---|
| `prefill` | list of list of integer | yes |  | The prefill chunks, as [tokens, tokens already computed]. |
| `decode` | list of list of integer | yes |  | The decoding requests, as [context, requests]. |
| `dur_ms` | number | yes |  | The engine step's duration on the GPU, ms. |
| `t_ms` | number | yes |  | When it ended, ms from the first beat. |

### probe: ProbeSpan

| field | type | required | default | meaning |
|---|---|---|---|---|
| `kind` | `decode` or `prefill` or `mixed` or `reference` | yes |  |  |
| `task` | string | yes |  | The design's task: &lt;kind&gt;:&lt;index&gt;. |
| `beat_from` | integer | yes |  | The span's first beat, an index into beats. |
| `beat_to` | integer | yes |  | One past its last beat. |

## gemm

`profile calibration` writes `gemm.json` for a per-GPU law: the GEMM grid, point by point.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `gpu` | string | yes |  | The GPU's id. |
| `clock_mhz` | number or null | yes |  | The grid's SM clock: the lock, else its points' mean, snapped to 25 MHz; null when they disagree. |
| `seconds` | number | yes |  | Wall-clock seconds each point was timed for. |
| `runs` | list of GemmTimedRun | yes |  |  |
| `clean` | boolean | yes |  | No flag: the grid enters the profiling data. |
| `flags` | list of string | yes |  | Why the grid cannot calibrate the law, each a sentence. |

### gemm: GemmTimed

| field | type | required | default | meaning |
|---|---|---|---|---|
| `n` | integer | yes |  | Rows of the matrix multiplications. |
| `s_per_iter` | number | yes |  | Seconds per layer iteration. |
| `iters` | integer | yes |  | Iterations timed. |
| `flops` | number | yes |  | Floating-point operations per layer iteration. |
| `sm_mhz` | number or null | yes |  | The mean SM clock while the point ran, MHz. |

### gemm: GemmTimedRun

| field | type | required | default | meaning |
|---|---|---|---|---|
| `model` | string | yes |  | The layer timed: an LLM's, or a synthetic one (synthetic-d&lt;d&gt;-f&lt;d_ff&gt;-k&lt;d_kv&gt;). |
| `reps` | list of list of GemmTimed | yes |  | Repetitions, each a sweep over the row counts. |

## calibration

`calibrate` writes it; `predict` reads it.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `name` | string | yes |  | Its name, &lt;gpu&gt;/&lt;cost law&gt;[/&lt;llm&gt;] for a shipped one: a label, which nothing parses. |
| `law` | CalibratedLaw | yes |  |  |
| `conditions` | CalibrationConditions | yes |  |  |
| `fit` | FitRecord or null |  |  | How it was fitted: provenance, which predict never reads. The shipped calibrations, fitted before ServeCast, have none. |

### calibration: CalibratedLaw

| field | type | required | default | meaning |
|---|---|---|---|---|
| `cost_law` | string | yes |  | The cost law, by name (servecast list laws); its calibration procedure is part of its entry. |
| `parameters` | map to number | yes |  | Each of the law's parameters by name, in the law's order; components.json gives their units. |
| `eta_c_grid` | EtaCGrid or null |  |  | A per-GPU law reads each LLM's eta_c off this grid. |

### calibration: CalibrationConditions

| field | type | required | default | meaning |
|---|---|---|---|---|
| `gpu` | string | yes |  | The GPU's id (servecast list gpus). |
| `gpu_spec` | GpuSpec | yes |  | The datasheet numbers the fit assumed; predict refuses a calibration whose numbers are not the registry's. |
| `clock_mhz` | number | yes |  | The SM clock the probes ran at. The law's latencies are in ms at this clock, and it ran at clock_mhz / boost_clock_mhz of the peak. |
| `tensor_parallel` | integer |  | `1` | GPUs per engine; only 1 in this version. |
| `fitted_on` | string or list of string | yes |  | The LLM a per-LLM calibration was fitted on, or the synthetic models of a per-GPU one. |

### calibration: EtaCCurve

| field | type | required | default | meaning |
|---|---|---|---|---|
| `a` | number | yes |  | 1/eta_c = a + b / (the LLM's weights per layer). |
| `b` | number | yes |  |  |

### calibration: EtaCGrid

| field | type | required | default | meaning |
|---|---|---|---|---|
| `rule` | `interpolation` or `curve` | yes |  | How an LLM's eta_c is read: bilinear interpolation on the points, clamped to the grid, or the curve when every point lies near it. |
| `points` | list of EtaCPoint | yes |  | The GPU's grid of synthetic layers. |
| `curve` | EtaCCurve or null |  |  |  |

### calibration: EtaCPoint

| field | type | required | default | meaning |
|---|---|---|---|---|
| `d_model` | integer | yes |  | The synthetic layer's width. |
| `ffn_ratio` | number | yes |  | Its feed-forward width over its width. |
| `eta_c` | number | yes |  | The fraction of the peak arithmetic rate its matrix multiplications reached. |

### calibration: FitRecord

| field | type | required | default | meaning |
|---|---|---|---|---|
| `cells` | integer or null |  |  | The probe cells fitted. |
| `evaluations` | integer or null |  |  | The cost evaluations the optimizer made. |
| `error_pct` | TrainingError or null |  |  |  |
| `profiling_sha256` | string or null |  |  | The sha256 of the profiling data it was fitted on. |

### calibration: GpuSpec

| field | type | required | default | meaning |
|---|---|---|---|---|
| `peak_flop_per_s` | number | yes |  | The registry's peak the fit assumed (gpus.json). |
| `memory_bandwidth_bytes_per_s` | number | yes |  | The registry's bandwidth the fit assumed. |
| `boost_clock_mhz` | integer | yes |  | The registry's boost clock, the peak's clock. |

### calibration: TrainingError

| field | type | required | default | meaning |
|---|---|---|---|---|
| `mape` | number | yes |  | The paper's training MAPE: the cells' absolute percentage errors averaged with the fit's weights, in %. |
| `max` | number | yes |  | The largest cell error, in %. |

## prediction

`predict` writes `prediction.json`; `compare` reads it.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `points` | list of Point | yes |  | One per load point; several with --sweep. |
| `estimand` | Estimand | yes |  |  |
| `validity` | Validity | yes |  | Internal to ServeCast; not part of its interface. |
| `provenance` | Provenance | yes |  |  |
| `resolved_manifest` | object | yes |  | The manifest with every default filled in. |
| `warnings` | list of string |  |  | What the run noticed but did not refuse. |

### prediction: Estimand

| field | type | required | default | meaning |
|---|---|---|---|---|
| `rule` | `steady_state` | yes |  |  |
| `ttft` | number | yes |  | Target relative half-width for TTFT. |
| `tpot` | number | yes |  | Target relative half-width for TPOT. |
| `throughput` | number | yes |  | Target relative half-width for throughput. |
| `e2e` | number or null |  |  | Target relative half-width for the end-to-end latency. |
| `ttft_p95` | number or null |  |  | Target relative half-width for TTFT's 95th percentile. |
| `tpot_p95` | number or null |  |  | Target relative half-width for TPOT's 95th percentile. |

### prediction: Interval

| field | type | required | default | meaning |
|---|---|---|---|---|
| `low` | number | yes |  |  |
| `high` | number | yes |  |  |

### prediction: LoadPoint

| field | type | required | default | meaning |
|---|---|---|---|---|
| `concurrency` | integer | yes |  | Clients. |
| `think_time_ms` | number | yes |  | Mean think time, ms. |

### prediction: Point

| field | type | required | default | meaning |
|---|---|---|---|---|
| `load` | LoadPoint | yes |  |  |
| `summary` | Summary | yes |  |  |
| `converged` | boolean |  | `True` | Whether every judged metric met its target; false only when a time budget stopped the run first. |
| `stop_reason` | string or null |  |  | converged, or what stopped the run. |
| `diagnostics` | object |  |  | Model internals, for reading only. |

### prediction: Provenance

| field | type | required | default | meaning |
|---|---|---|---|---|
| `components` | map to string | yes |  | Each component that ran, by name (servecast list components). |
| `servecast_version` | string | yes |  |  |
| `seed` | integer or null |  |  |  |
| `pool_sha256` | string or null |  |  |  |
| `calibration` | string or null |  |  | The calibration's name. |
| `calibration_sha256` | string or null |  |  |  |
| `vllm_version` | string or null |  |  | Measurements only. |

### prediction: Summary

| field | type | required | default | meaning |
|---|---|---|---|---|
| `mean_ttft_ms` | number or null | yes |  | Mean time to first token, ms; null when unavailable. |
| `p95_ttft_ms` | number or null | yes |  | 95th percentile of time to first token, ms. |
| `mean_tpot_ms` | number or null | yes |  | Mean time per output token, ms: per request, (last token - first token) / (output tokens - 1). |
| `p95_tpot_ms` | number or null | yes |  | 95th percentile of time per output token, ms. |
| `output_throughput_tok_s` | number or null | yes |  | Output tokens per second. |
| `request_throughput_rps` | number or null | yes |  | Completed requests per second. |
| `num_ttft_samples` | integer | yes |  | Requests behind the TTFT statistics. |
| `num_tpot_samples` | integer | yes |  | Requests behind the TPOT statistics. |
| `duration_ms` | number | yes |  | The steady-state window the statistics cover, ms. |
| `wall_time_ms` | number or null |  |  | Time the computation took on this computer, ms. |
| `ci95` | map to Interval |  |  | 95 % interval of each metric above, keyed by the metric's field name. |

### prediction: Validity

| field | type | required | default | meaning |
|---|---|---|---|---|
| `status` | `inside` or `outside` | yes |  |  |
| `reasons` | list of string |  |  |  |

## measurement

`profile serving` writes `measurement.json`; `compare` reads it.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `points` | list of MeasuredPoint | yes |  |  |
| `estimand` | Estimand | yes |  |  |
| `provenance` | Provenance | yes |  |  |
| `resolved_manifest` | object | yes |  | The manifest with every default filled in, written by the same resolver as a prediction's, so compare can pair them. |

### measurement: Estimand

| field | type | required | default | meaning |
|---|---|---|---|---|
| `rule` | `steady_state` | yes |  |  |
| `ttft` | number | yes |  | Target relative half-width for TTFT. |
| `tpot` | number | yes |  | Target relative half-width for TPOT. |
| `throughput` | number | yes |  | Target relative half-width for throughput. |
| `e2e` | number or null |  |  | Target relative half-width for the end-to-end latency. |
| `ttft_p95` | number or null |  |  | Target relative half-width for TTFT's 95th percentile. |
| `tpot_p95` | number or null |  |  | Target relative half-width for TPOT's 95th percentile. |

### measurement: Interval

| field | type | required | default | meaning |
|---|---|---|---|---|
| `low` | number | yes |  |  |
| `high` | number | yes |  |  |

### measurement: LoadPoint

| field | type | required | default | meaning |
|---|---|---|---|---|
| `concurrency` | integer | yes |  | Clients. |
| `think_time_ms` | number | yes |  | Mean think time, ms. |

### measurement: MeasuredPoint

| field | type | required | default | meaning |
|---|---|---|---|---|
| `load` | LoadPoint | yes |  |  |
| `summary` | Summary | yes |  |  |
| `converged` | boolean | yes |  | Whether every judged metric met its target before the run was stopped. |
| `stop_reason` | string or null |  |  | converged, or what stopped the run. |
| `diagnostics` | object |  |  | Model internals, for reading only. |
| `measured_clock_mhz` | number or null | yes |  | Mean GPU clock over the steady-state window, MHz. |
| `kv_blocks` | integer or null | yes |  | The KV-cache blocks the engine ran with, read back from it. |
| `kv_blocks_computed` | integer or null |  |  | The blocks vLLM computed from the GPU's free memory, before the manifest's capacity overrode them. |

### measurement: Provenance

| field | type | required | default | meaning |
|---|---|---|---|---|
| `components` | map to string | yes |  | Each component that ran, by name (servecast list components). |
| `servecast_version` | string | yes |  |  |
| `seed` | integer or null |  |  |  |
| `pool_sha256` | string or null |  |  |  |
| `calibration` | string or null |  |  | The calibration's name. |
| `calibration_sha256` | string or null |  |  |  |
| `vllm_version` | string or null |  |  | Measurements only. |

### measurement: Summary

| field | type | required | default | meaning |
|---|---|---|---|---|
| `mean_ttft_ms` | number or null | yes |  | Mean time to first token, ms; null when unavailable. |
| `p95_ttft_ms` | number or null | yes |  | 95th percentile of time to first token, ms. |
| `mean_tpot_ms` | number or null | yes |  | Mean time per output token, ms: per request, (last token - first token) / (output tokens - 1). |
| `p95_tpot_ms` | number or null | yes |  | 95th percentile of time per output token, ms. |
| `output_throughput_tok_s` | number or null | yes |  | Output tokens per second. |
| `request_throughput_rps` | number or null | yes |  | Completed requests per second. |
| `num_ttft_samples` | integer | yes |  | Requests behind the TTFT statistics. |
| `num_tpot_samples` | integer | yes |  | Requests behind the TPOT statistics. |
| `duration_ms` | number | yes |  | The steady-state window the statistics cover, ms. |
| `wall_time_ms` | number or null |  |  | Time the computation took on this computer, ms. |
| `ci95` | map to Interval |  |  | 95 % interval of each metric above, keyed by the metric's field name. |

## comparison

`compare` writes `comparison.json`.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `prediction_sha256` | string | yes |  |  |
| `measurement_sha256` | string | yes |  |  |
| `checks` | Checks | yes |  |  |
| `points` | list of ComparedPoint | yes |  |  |
| `flags` | list of string |  |  | e.g. a clock or KV-capacity mismatch. |

### comparison: Checks

| field | type | required | default | meaning |
|---|---|---|---|---|
| `manifest` | boolean | yes |  | Same manifest, compared on its meaning. |
| `estimand` | boolean |  | `True` | Same stop rule, each the one its resolved manifest records. |
| `pool` | boolean | yes |  | Same request pool, by the digest each provenance records. |
| `sampler` | boolean or null | yes |  | Same length-sampler id; null when the lengths come from a trace. |
| `estimator` | boolean | yes |  | Same steady-state estimator id. |

### comparison: Compared

| field | type | required | default | meaning |
|---|---|---|---|---|
| `predicted` | number or null | yes |  |  |
| `measured` | number or null | yes |  |  |
| `measured_ci95` | Interval or null |  |  |  |
| `signed_error_pct` | number or null | yes |  | (predicted - measured) / measured, in percent. |
| `within_ci` | boolean or null |  |  | Whether the prediction falls inside the measured interval. |

### comparison: ComparedPoint

| field | type | required | default | meaning |
|---|---|---|---|---|
| `load` | LoadPoint | yes |  |  |
| `metrics` | map to Compared | yes |  |  |
| `measured_certified` | boolean | yes |  | Whether the measurement met every target of its stop rule. |
| `predicted_converged` | boolean |  | `True` | Whether the prediction met every target of its stop rule. |

### comparison: Interval

| field | type | required | default | meaning |
|---|---|---|---|---|
| `low` | number | yes |  |  |
| `high` | number | yes |  |  |

### comparison: LoadPoint

| field | type | required | default | meaning |
|---|---|---|---|---|
| `concurrency` | integer | yes |  | Clients. |
| `think_time_ms` | number | yes |  | Mean think time, ms. |

## run

`predict` and `calibrate` write `run.json` beside their other outputs.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `command` | string | yes |  | The command line, as typed. |
| `servecast` | string | yes |  | The version of ServeCast that ran it. |
| `started` | string | yes |  | When it started: local time with its UTC offset (ISO 8601). |
| `finished` | string | yes |  | When it wrote this record. |
| `inputs` | map to object | yes |  | Each input by its role (manifest, calibration, pool, profiling data): its path or name, and its sha256. |
| `components` | map to string |  |  | The components that ran, by name. |
| `outputs` | list of string | yes |  | The other files the command wrote in this directory. |

## gpus

ServeCast ships `gpus.json`; a `gpus.json` in your data directory adds GPUs.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `gpus` | map to GpuEntry | yes |  | Each GPU by its id (a100_sxm_40gb), as manifests and calibrations name it. |

### gpus: GpuEntry

| field | type | required | default | meaning |
|---|---|---|---|---|
| `device_name` | string | yes |  | The name the NVIDIA driver reports, as `nvidia-smi --query-gpu=name` prints it. |
| `memory_gib` | integer | yes |  | The GPU's memory, in GiB. |
| `sm_count` | integer | yes |  | Streaming multiprocessors. |
| `peak_flop_per_s` | number | yes |  | The dense bf16 tensor-core peak at the boost clock, in FLOP/s, from the datasheet. |
| `memory_bandwidth_bytes_per_s` | number | yes |  | The datasheet memory bandwidth, in bytes/s. |
| `boost_clock_mhz` | integer | yes |  | The boost clock the peak is quoted at, in MHz. |

## models

ServeCast ships `models.json`; a `models.json` in your data directory adds LLMs.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `models` | map to ModelEntry | yes |  | Each LLM by the name engine.model gives it (a Hugging Face id). |

### models: ModelEntry

| field | type | required | default | meaning |
|---|---|---|---|---|
| `num_hidden_layers` | integer | yes |  | Decoder layers. |
| `hidden_size` | integer | yes |  | The hidden width. |
| `intermediate_size` | integer | yes |  | The width of the gated feed-forward block. |
| `num_attention_heads` | integer | yes |  | Query heads. |
| `num_key_value_heads` | integer or null |  |  | KV heads. Unset: the query heads. |
| `head_dim` | integer or null |  |  | The head width. Unset: hidden_size / num_attention_heads. |
| `vocab_size` | integer | yes |  | The vocabulary. |
| `max_position_embeddings` | integer | yes |  | The longest context, in tokens. |
| `torch_dtype` | string |  | `bfloat16` | The weights' type: bfloat16 or float16. |

## components

ServeCast ships `components.json`, the one list of the components it can select.

| field | type | required | default | meaning |
|---|---|---|---|---|
| `schema` | string | yes |  | servecast-&lt;kind&gt;/&lt;major&gt;. |
| `about` | list of string |  |  | The rules of the file, in words. |
| `cost_law` | map to CostLawEntry | yes |  | The cost laws, each with its calibration procedure. |
| `population_model` | map to ComponentEntry | yes |  | The population models. |
| `workload_fitter` | map to ComponentEntry | yes |  | The workload fitters. |
| `length_sampler` | map to ComponentEntry | yes |  | The length samplers. |
| `estimator` | map to ComponentEntry | yes |  | The steady-state estimators. |

### components: ComponentEntry

| field | type | required | default | meaning |
|---|---|---|---|---|
| `summary` | string | yes |  | One sentence, printed beside the name. |
| `released` | string | yes |  | The month it was released (2026-11; 2026-11b for a second entry that month). `versions: &lt;month&gt;` selects each component's newest entry up to it, and `versions` accepts it in place of the name. |
| `in_paper` | boolean | yes |  | Whether the paper used it; one entry of each kind at most. |
| `default` | boolean | yes |  | Whether it is used when `versions` does not choose; one entry of each kind. |
| `description` | string | yes |  | What it computes, in full. |
| `paper` | string or null |  |  | Where the paper describes it. |
| `paper_symbols` | map to string |  |  | Its parameters' names in the paper. |
| `code` | list of string | yes |  | The modules that implement it. |

### components: CostLawEntry

| field | type | required | default | meaning |
|---|---|---|---|---|
| `summary` | string | yes |  | One sentence, printed beside the name. |
| `released` | string | yes |  | The month it was released (2026-11; 2026-11b for a second entry that month). `versions: &lt;month&gt;` selects each component's newest entry up to it, and `versions` accepts it in place of the name. |
| `in_paper` | boolean | yes |  | Whether the paper used it; one entry of each kind at most. |
| `default` | boolean | yes |  | Whether it is used when `versions` does not choose; one entry of each kind. |
| `description` | string | yes |  | What it computes, in full. |
| `paper` | string or null |  |  | Where the paper describes it. |
| `paper_symbols` | map to string |  |  | Its parameters' names in the paper. |
| `code` | list of string | yes |  | The modules that implement it. |
| `calibrated_per` | `llm_and_gpu` or `gpu` | yes |  | What one calibration covers: one LLM on one GPU, or one GPU for any LLM. |
| `calibration` | LawCalibration | yes |  | Its calibration procedure, part of the same entry. |
| `parameters` | map to string | yes |  | Each parameter, with its unit. |

### components: LawCalibration

| field | type | required | default | meaning |
|---|---|---|---|---|
| `measures` | string | yes |  | What a calibration measures on the GPU. |
| `fit` | string | yes |  | How the parameters are fitted to it. |
| `fitted` | list of string | yes |  | The parameters the fit determines. |
| `pinned` | map to string |  |  | The parameters fixed before the fit, and to what. |
