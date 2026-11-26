# Changelog

Each release of ServeCast has a section here, the newest first. The PyPI page and the GitHub release show the newest
section. A component's name, such as `c62ebe75`, never changes meaning from one release to the next, so a manifest
that sets `versions:` selects the same components in a later release. `servecast list components` explains each
component.

## 0.1.0 (unreleased)

The first release.

- `servecast predict` predicts how vLLM 0.25.1 serves a dense LLM on one GPU under a closed workload: the mean and the
  95th percentile of TTFT and TPOT, and the throughput, each with its 95 % interval.
- `servecast profile calibration` measures a GPU of yours and `servecast calibrate` fits a cost law to it.
  `servecast profile serving` measures a manifest with vLLM, and `servecast compare` scores a prediction against
  that measurement.
- 19 calibrations ship with the package, each with the profiling data it was fitted on. One is for the
  A100-SXM4-40GB with the per-GPU cost law `c62ebe75`. The other 18 are the paper's, with the cost law `d9cbf1d3`,
  for six LLMs on each of the A100-SXM4-40GB, the H100 SXM and the RTX 3090 Ti.
- `servecast replicate` reruns the paper's experiments and rebuilds its tables from the shipped data.
- ServeCast runs on Python 3.11 to 3.14. Profiling needs Linux, an NVIDIA GPU and the `profile` extra, which installs
  vLLM 0.25.1.
