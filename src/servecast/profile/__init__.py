"""Measurement on a real GPU: the calibration probes and the serving runs. It never imports the model, so no
measurement can lean on the model's assumptions.

May import: servecast.core. Third-party: torch, vLLM and NVML (nvidia-ml-py), installed with the `profile`
extra.
"""
