# Concepts

This page describes what ServeCast computes and the parts it is made of. The [guides](guides/predict.md) show how to
use them.

## The question

An LLM served by vLLM processes many requests at once. At every engine step, which this documentation calls a beat,
the engine admits waiting requests, computes part of one prompt (prefill) and produces one token for every request
that is generating (decode). Requests join and leave between beats, so the cost of a beat changes with the batch, and
the batch changes with the cost. ServeCast predicts the result of that loop for a closed workload, in which each of N
clients sends a request, waits for the whole response, thinks for an exponentially distributed time with mean Z and
then sends its next request. It reports the request throughput, the time to first token (TTFT), the time per output
token (TPOT) and the output tokens per second.

TPOT is measured per request as (time of the last token - time of the first token) / (output tokens - 1), so a
request with one output token has none. TTFT runs from the moment the client sends the request to its first token.

## The population model with eviction

The model follows the engine beat by beat. Between two beats its state is a census: the clients thinking, the
requests waiting in arrival order, the one request whose prompt is partly computed, and the requests decoding, each
with its exact context length. A beat then does what vLLM's scheduler does. It admits requests from the head of the
queue while the token budget (`max_batched_tokens`) and the free KV-cache blocks allow, computes the next chunk of the
partly computed prompt, and lets every decoding request emit a token.

The KV cache is a pool of blocks of `block_size` tokens. When a beat's growth does not fit, the engine evicts a
request to free its blocks, and so does the model. The partly computed prompt goes first, otherwise a decoding request
from the most recently admitted group. The evicted request goes back to the head of the queue and must compute its
whole context again before it emits another token. Eviction is how a full KV cache shows up in TTFT and throughput, so
the KV capacity is an input in blocks, the same number the engine is given.

The model is a simulation. It draws think times, request lengths and eviction victims from seeded random streams, so
the same manifest and seed give the same numbers on the reference platform.

## Cost laws

A cost law gives the duration of a beat from its composition: how many prompt tokens it computes, how many requests
decode, and their contexts. Each law has a name and the month it was released (`servecast list laws`).

The law of [the paper](cite.md), `d9cbf1d3` (released 2026-10), works as follows. A beat runs the layer matrix multiplications, the vocabulary projection and the
attention kernels one after another, and each kind costs the larger of its memory time and its arithmetic time. The
memory time reads the weight and KV-cache bytes at the GPU's datasheet bandwidth, and the arithmetic time divides the
floating-point work by the GPU's peak at the clock of the calibration. A fixed latency per beat and a cost per
decoding request are added. The law has three fitted coefficients: the fraction of the peak the matrix
multiplications reach, the fixed latency and the per-request cost. They are fitted for each LLM on each GPU.

The law `c62ebe75` (released 2026-11) keeps that form and changes six things. The fixed latency is per layer, and a per-token elementwise
cost is added. The fraction of the peak the matrix multiplications reach comes from a grid of synthetic layers
measured on the GPU, read off at the LLM's width and feed-forward ratio. The weight read runs at an efficiency measured
on that grid instead of the datasheet bandwidth, the KV-cache read has its own fitted efficiency, and the per-request
cost is dropped. It is calibrated once per GPU, on synthetic models, and predicts an LLM that was never profiled, from
its `config.json`.

## Calibration

A calibration file has three sections. `law` names the cost law and gives its fitted coefficients. `conditions` says
what they hold for: the GPU, the datasheet numbers the fit assumed, the clock the probes ran at, and the LLM or the
synthetic models it was fitted on. `fit` says how it was obtained: the cells, the training error and the sha256 of the
profiling data; predict never reads it. A shipped calibration is named after its GPU and its cost law, plus the LLM
for the paper's law: `a100_sxm_40gb/c62ebe75`, `h100_sxm/d9cbf1d3/llama-3-8b`. `servecast profile calibration`
measures what a law needs on a GPU, and `servecast calibrate` fits it ([On your GPU](guides/calibrate.md)). The fit is
a weighted least-squares fit on the relative error of the predicted beat durations. Every calibration that ships with ServeCast is refitted from its profiling data in the test
suite, with identical coefficients on the reference platform.

## Workloads

A workload is a pool of requests, each a prompt length and an output length. A pool comes either from a trace (all its
requests, in order) or from the length sampler, which draws `pool_size` pairs from a mean and an optional variance
for each length. The sampler matches the two moments with sums of geometric stages, the family the model represents.

The model does not replay the pool request by request. The workload fitter turns the pool into a workload law, a set
of prompt-length classes and a chain of decode phases whose exit probabilities fit the output lengths, and the
model draws each arriving request's type from that law. The fit of a large trace can take minutes. ServeCast ships the
laws of its three traces and caches every other law by the pool's digest.

## The steady-state estimator

A closed workload settles into a steady state after a warm-up, and ServeCast reports that state. At each checkpoint
of a growing schedule the estimator removes the warm-up with the MSER-5 rule (at most 40 % of the completions), builds
the window of the remaining completions, and computes a 95 % confidence interval for each metric from 32 batch means.
The run stops when every interval's half-width is below its target, relative to the mean, and Little's law holds on
the window within 1 %. The targets are in the manifest (`traffic.stop.steady_state`). With the estimator `c3ad4006`
(released 2026-11) the 95th percentiles of TTFT and TPOT can be targets too.

## Components and versions

ServeCast's predictor has five components: the cost law, whose entry includes its calibration procedure, the
population model, the workload fitter, the length sampler and the steady-state estimator. Each version of a
component is an entry with a name, eight hexadecimal characters such as `c62ebe75`, and the month it was
released. A name never changes meaning. The same name always gives the same numbers, and a change that moves them
is a new entry with a new name. `servecast list components` shows every entry with a one-sentence summary, and
`components.json`, installed with the package ([Files and folders](reference/files.md)), explains each one in
full: what it computes, its parameters, its calibration and where the paper describes it.

A manifest chooses them in `versions`. Unset, each component takes its default entry. A month, such as
`versions: 2026-11`, selects each component's newest entry up to that month. The components can also be chosen
one by one, by name (`versions: {estimator: cabeb968}`), and there the month an entry was released stands for
its name, so manifests and calibrations written before names keep working. Every output records the components
that ran, by name.
