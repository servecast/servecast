<p align="center">
  <img src="docs/assets/logo.svg" width="64" alt="">
</p>

<h1 align="center">ServeCast</h1>

<p align="center">Know how vLLM will serve your LLM before you run it.</p>

<p align="center"><a href="https://arxiv.org/abs/XXXX.XXXXX"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg"></a></p>

<!-- pypi-links: the PyPI page is built from this README (pyproject.toml, tool.hatch.metadata.hooks.fancy-pypi-readme) -->

ServeCast predicts how fast vLLM will serve an LLM on a given GPU, so you can check the numbers against your
service-level objectives (SLOs) before you deploy. For a given workload, it predicts the time to first token (TTFT)
and the time per output token (TPOT), both as means and 95th percentiles, and the throughput in completed requests per
second.

ServeCast comes from the paper *ServeCast: Predicting LLM Serving Throughput and Latency Across Models from
Microbenchmarks* (Incerto, Yang, Sun and Tribastone, 2026). This repository holds the tool. You can use it to replicate
the paper's experiments, or to predict GPUs, LLMs and workloads that the paper did not study.

## Quick start

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

The manifest `example-a100.yaml` describes Qwen2.5-7B served by vLLM on an A100-SXM4-40GB under a fixed load of 64
concurrent clients. Each client sends a prompt, waits for the whole response, then pauses for a random time, one second
on average, before sending the next prompt. The file is now yours to edit or copy, and every command takes it by its
path. [Get started](docs/get-started.md) explains each step.

## On your GPU

These commands run the whole flow on an A100-SXM4-40GB under Linux, the GPU the example manifest names. They measure
the GPU, fit its calibration, predict, serve the same workload with vLLM, and compare the measurement with the
prediction.

```bash
# installs ServeCast with what profiling needs (vLLM 0.25.1, NVIDIA's NVML library)
pip install 'servecast[profile]'
# checks that this computer can profile: the GPU, its driver and vLLM
servecast doctor
# writes the example manifest that ships with ServeCast to example-a100.yaml
servecast manifest show example-a100 > example-a100.yaml
# measures the GPU (8 synthetic models, then a grid of matrix multiplications), with the clock locked where
# possible; reads the GPU and the cost law from example-a100.yaml, writes the profiling data to profiling/
servecast profile calibration -c example-a100.yaml --output-dir profiling
# fits the cost law to the profiling data; reads profiling/, saves the calibration as my-a100, the name the
# manifest uses (--overwrite replaces the Quick start's)
servecast calibrate --data profiling --name my-a100 --overwrite
# predicts TTFT, TPOT and throughput; reads example-a100.yaml (Qwen2.5-7B, 64 clients) and my-a100, writes
# prediction/
servecast predict -c example-a100.yaml --output-dir prediction
# serves the same workload with vLLM on this GPU at my-a100's clock and measures it; writes measurement/
servecast profile serving -c example-a100.yaml --output-dir measurement
# scores the prediction against the measurement, metric by metric; reads both folders, writes
# prediction/comparison.json
servecast compare prediction measurement
```

To install from a clone of this repository instead, see [Install](docs/install.md).

## What it does

- `servecast predict` turns a manifest (the LLM, the GPU, vLLM's settings and KV-cache size, the workload) into the
  serving metrics with their 95 % intervals, at one load or along a load curve.
- `servecast profile serving` measures the same manifest with vLLM on an NVIDIA GPU.
- `servecast compare` scores a prediction against a measurement of the same manifest, load point by load point.
- `servecast profile calibration` measures a GPU with microbenchmarks, and `servecast calibrate` fits the cost law to
  those measurements, so that any dense LLM can be predicted on that GPU. Each shipped calibration comes with the
  profiling data it was fitted on, so you can refit it.
- Every command writes its outputs to one folder, with a `run.json` that records the command, the version and
  each input with its sha256.
- Every component of the predictor (the cost law, the population model, the workload fitter, the length
  sampler, the steady-state estimator) has a name that never changes meaning, such as `c62ebe75`, and every
  output says which ran. `servecast list components` explains each one.
- `servecast replicate` reruns the experiments of the paper with its pinned components.

## Documentation

[Install](docs/install.md) · [Get started](docs/get-started.md) · [Predict](docs/guides/predict.md) ·
[On your GPU](docs/guides/calibrate.md) · [How it works](docs/concepts.md) · [Limits](docs/limits.md) ·
[Command line](docs/reference/cli.md) · [Files and folders](docs/reference/files.md) ·
[File formats](docs/reference/formats.md)

## Citation

If you use ServeCast, please cite the paper that introduced it:

```bibtex
@misc{incerto2026servecast,
  title         = {{ServeCast}: Predicting {LLM} Serving Throughput and Latency Across Models from Microbenchmarks},
  author        = {Incerto, Emilio and Yang, Lishan and Sun, Yu and Tribastone, Mirco},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv}
}
```

## License

Apache License 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The request lengths of the three shipped traces
keep their own licences, listed in [DATA_LICENSES.md](DATA_LICENSES.md).
