# The paper's data

`servecast replicate` reads these files. They were extracted from the runs behind the arXiv version and re-emitted
field by field, through a writer that accepts only the fields and names below.

| file | content |
|---|---|
| `campaigns.json` | the 54 measured serving campaigns of the paper: 3 GPUs, 6 LLMs, 3 workloads |
| `ablations.json` | the predictions of the paper's three ablations per load point, and the measurements of the clock ablation |
| `tables.json` | the 12 tables printed in the arXiv version, cell by cell |

`ablations.json` holds, for each ablation, the predictions of the reduced models at the load points the paper scored
them on, with the same fields as a campaign's predictions (`throughput_rps`, `mean_ttft_s`, `mean_tpot_s`,
`converged`). Under `residual`, every campaign's scored load points with the residual correction reduced to its
constant (`constant_only`) and removed (`no_correction`). Under `length`, the three campaigns the length-model
ablation selected, with independent lengths (`independent_lengths`) and fixed mean lengths (`fixed_mean_lengths`).
The full model and the joint length law are each campaign's own prediction in `campaigns.json`, so they are not
repeated. Under `clock` is the A100 serving of Llama-3-8B on the Azure conversation trace with the clock locked at
900 and at 750 MHz. For each load point it gives the measured means, whether the measurement is certified, and the
prediction of each clock treatment (`adjust_f_and_residual`, `adjust_f_only`, `no_clock_adjustment`,
`refit_at_target_clock`).

The newest cost law's A100 predictions, which the model's tests also compare against, are test data and live in
`tests/goldens/newest-a100.json`; they make no claim about accuracy. The published accuracy figures are the ones in
the arXiv version.

## Units and statistics

Times are in seconds and throughput in requests per second, the units the model computes in. Every float
is written as Python writes it, so it reads back bit for bit. A null means that the value does not exist. When
a field of a campaign is null, the campaign's `missing` list says why.


## Campaigns

A campaign is identified by `gpu`, `llm` (the Hugging Face id) and `trace`. It lists its six designed load
points under `cells`. A load point is a number of clients, `concurrency`, each thinking for `think_time_s`
on average between requests. Two campaigns stopped early (`complete: false`). Their unmeasured load points
have `measured: null` but keep their predictions.

`engine` holds two kinds of settings:

- What the engine reported when it started: `num_gpu_blocks`, `max_num_seqs`, `max_num_batched_tokens`,
  `gpu_memory_utilization`, `block_size`, `dtype`, the CUDA-graph capture sizes and the attention backend.
- What the campaign requested, which the engine does not report back: `max_model_len`,
  `enable_prefix_caching` and `enforce_eager`.

`engine` is null for three campaigns whose run kept no engine report.

The paper's campaigns set the KV-cache capacity themselves with `num_gpu_blocks_override`, so the served
`num_gpu_blocks` is a design choice. On the campaign, `num_gpu_blocks_computed` is the count that vLLM's
memory profile allowed on that machine, logged before the override replaced it. This count is the
reference for estimating a capacity. Under identical settings it differs between runs of the same GPU and
LLM:

| GPU | difference between runs |
|---|---|
| H100 SXM | none |
| A100 | up to 3.2 % |
| RTX 3090 Ti | up to 11.9 % |

On two RTX 3090 Ti campaigns, the override was larger than the count vLLM computed for that run.

`clock_lock_sm_mhz` is the SM clock the run locked. `clock_lock_verified` says whether the lock was verified
under load at both the start and the end of the run. For each load point, `measured.sm_clock_mean_mhz` is
the mean SM clock sampled during the steady-state window. It is null when the run did not sample it.

## Prediction settings

Each campaign's `prediction` holds what the model was given, read from the campaign's own manifest:

- `seed`: the seed of the simulation (0 everywhere);
- `kv_blocks` and `block_size`: the KV capacity in blocks, the same count the engine was given;
- `max_batched_tokens` and `max_sequences`: the engine's batching limits;
- `checkpoint_ratio`: the ratio of the stop rule's geometric checkpoint schedule;
- `max_completions`: a ceiling on the run, or null for none. It is null everywhere: every prediction ran until its
  precision targets were met;
- `targets`: the stop rule's relative half-widths, per judged metric (`throughput`, `e2e`, `ttft`, `tpot`).

Two A100 BurstGPT campaigns of the paper (Llama-3-8B and Qwen2.5-7B) predate the final stop rule: their schedule
doubles at each checkpoint and every metric's target is 5 %. The Llama-3-8B one also capped sequences at 256.

## Load points

In `campaigns.json`, each load point holds:

- `measured`: the steady-state means, their half-widths and the GPU clock. `certified` is the measurement's
  own verdict: the window reached the target precision and was stationary.
- `predicted`: the paper's prediction, one entry per LLM whose calibration on the same GPU was used. The
  entry for the served LLM itself is the paper's accuracy result, and the other entries are its transfer
  results.
- `vidur`: Vidur's prediction, with its own `certified` verdict.
- `primary`: whether the paper's accuracy question scores this load point. It does exactly when the
  measurement is certified, the served LLM's own prediction converged and Vidur's run is certified.
- `evaluated`: the same rule, for the transfer question.

## Tables

Each table in `tables.json` has `number`, its number in the arXiv version, and `columns` and `rows`. For eight
tables, the printed header is not part of the generated file. Their `columns` repeat the header of the arXiv
version and add the unit stated in its caption: MAPE in percent; times in seconds, milliseconds or microseconds;
GPU-minutes. The other four tables print their header as their first row, which became their `columns`. A row
with a single cell is a group label.

A table that answers a research question is named after it, as the arXiv version's text numbers them, so
`rq2-matrix-*` are the transfer tables of RQ2 (Tables 4, 13 and 14) and `rq3-cost` is the cost table of RQ3
(Table 5).
