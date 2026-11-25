# AGENTS.md

Instructions for AI agents that use ServeCast to answer a question, or that change this repository. People should
start with the [README](README.md) and the [documentation](https://servecast.github.io/servecast/).

## What ServeCast answers

ServeCast predicts how vLLM 0.25.1 serves a dense LLM on one NVIDIA GPU under a closed workload, in which each of N
clients sends a prompt, waits for the whole response and pauses before the next one. For each load it predicts the time
to first token (TTFT) and the time per output token (TPOT), as means and 95th percentiles, and the throughput in
completed requests per second, each with a 95 % confidence interval. A prediction needs no GPU and no model weights,
only the LLM's `config.json`.

ServeCast refuses what it does not model instead of guessing. That covers tensor or pipeline parallelism,
mixture-of-experts models, quantized weights, open-loop arrivals, speculative decoding, and any engine or vLLM version
other than vLLM 0.25.1 ([Limits](docs/limits.md)). When a question needs one of these, tell the user, and do not work
around the refusal.

## Using ServeCast

Install it with `pip install servecast`. Only `servecast profile ...` needs a GPU, with Linux, an NVIDIA GPU of compute
capability 8.0 or newer and `pip install 'servecast[profile]'`; `servecast doctor` says whether this computer has them.

| to get | run |
|---|---|
| a prediction | `servecast manifest show example-a100 > m.yaml`, edit `m.yaml`, then `servecast predict -c m.yaml --with a100_sxm_40gb/c62ebe75 --format json` |
| the same for another LLM | add `--to <Hugging Face id>` and that LLM's own KV-cache size, `--set engine.workers.aggregated.kv_cache.capacity.blocks=N` |
| a load curve | add `--sweep traffic.load.concurrency=16,32,64` |
| what a manifest field means | `servecast manifest explain <dotted.path>` |
| every problem in a manifest | `servecast manifest validate -c m.yaml` |
| what is installed | `servecast list calibrations`, `servecast list models`, `servecast list gpus` |
| a calibration for another GPU | the [On your GPU](docs/guides/calibrate.md) guide, which ends with `servecast compare` |
| the paper's results | `servecast replicate verify`, then `servecast replicate run <experiment> --from-frozen` |

The shipped calibration `a100_sxm_40gb/c62ebe75` covers any dense LLM on the A100-SXM4-40GB. The other GPUs have the
paper's calibrations, one per LLM, and the [Predict](docs/guides/predict.md) guide says how to select them.

- Read results from `--format json` or from the files a command writes, not from the printed table. Each entry of
  `points` holds the load, a `summary` with the means, the P95s and their `ci95`, and `converged`.
- Report every number with what produced it, meaning the calibration, the components under `provenance`, and whether
  the point converged. Pass on what `warnings` says. A prediction for an LLM other than the one a calibration was
  fitted on is a transfer, and the answer should say so.
- A predicted number is not a measurement. Do not extrapolate it to loads, LLMs or GPUs the run did not cover.
- Run `servecast manifest validate -c m.yaml` before a long run.
- Exit status 2 means that the manifest or an option cannot run. The message names each problem and its fix, so fix
  the input rather than retry it unchanged. Status 3 is a refused resource, such as a non-empty output directory, and
  `--overwrite` is for when the user agrees. Status 1 is a failed computation or a load point that did not converge,
  and 124 a load point stopped by `--timeout`.
- Ask the user before running `servecast profile ...`. It holds a GPU for a long run, and what it writes is a
  measurement.
- A manifest is a file of the user's, given to `-c` by its path. The shipped manifests are examples, so save one with
  `servecast manifest show <name> > file.yaml` and edit the copy. Fitted calibrations live in the data directory
  (`SERVECAST_DATA_DIR`), and downloaded `config.json` files and fitted workload laws in the cache
  (`SERVECAST_CACHE_DIR`).

## Working on this repository

Install a clone with the tools CI uses:

```bash
pip install -c constraints/ci.txt -e . pytest import-linter ruff mypy types-PyYAML
```

A change is done when what CI runs passes:

```bash
ruff check src tools tests
mypy
python -m pytest -q
lint-imports
python tools/gen_cli_reference.py --check
python tools/privacy_scan.py
```

- A changed help text changes the help snapshots and the command-line reference. Regenerate them with
  `SERVECAST_UPDATE_SNAPSHOTS=1 python -m pytest tests/test_cli_help.py` and `python tools/gen_cli_reference.py`, then
  review both diffs like code.
- `docs/reference/cli.md` and `docs/reference/formats.md` are generated, the second by `tools/gen_formats_reference.py`.
  Do not edit them by hand.
- `tests/test_docs_examples.py` runs every command in README.md and in docs/ as a reader would, and a `text` block after
  a command must equal its output on the reference environment. When an output changes, change the page with it.
- Tests marked `exact` compare bit for bit and run on the reference environment (macOS on Apple silicon, Python 3.12,
  numpy 2.5.0, scipy 1.18.0). Elsewhere they are skipped, and `SERVECAST_EXACT=1` runs them anyway.
- `lint-imports` enforces the layers of the package, for example that the measurement code never imports the model.
- The command line and the files it writes are the interface. The Python modules are internal and may change in any
  release.
