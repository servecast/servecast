"""What `profile serving` writes. Each load point is summarised exactly as a prediction is (`core.summary`, over the
window the same estimator kept), and the files are written so that an interrupted run keeps what it measured: the pool
and the environment first, then, after every load point, its requests and an atomic rewrite of measurement.json and of
SHA256SUMS.

Files: `measurement.json` (`servecast-measurement/1`), `pool.jsonl` (the requests served from), `requests.jsonl`
(every request served, warm-up included, each marked whether it is in the window), `environment.json` (the GPU, the
driver, CUDA, vLLM and the clock policy; never a serial number, UUID, bus id or host name) and `SHA256SUMS`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from servecast import __version__
from servecast.core import formats
from servecast.core import pool as pools
from servecast.core import summary as summaries

SCHEMA = "servecast-measurement/1"
FILES = ("measurement.json", "pool.jsonl", "requests.jsonl", "environment.json", "SHA256SUMS")
#: identifiers of one machine, never written
FORBIDDEN_KEYS = ("uuid", "serial", "bus_id", "pci_bus_id", "hostname", "host")


def point(result: Mapping[str, Any], concurrency: int, think_time_ms: float, estimator: ModuleType, *,
          clock_mhz: float | None, kv_blocks: int | None, kv_blocks_computed: int | None) -> dict[str, Any]:
    """A measured load point (`MeasuredPoint`), from `closed_loop.serve`'s result."""
    last = result["assessment"]
    q = summaries.window_quantities(last["kept"], last["window"])
    records = result["records"]
    return {"load": {"concurrency": int(concurrency), "think_time_ms": float(think_time_ms)},
            "summary": summaries.point_summary(q, last["decision"]["per_metric"], estimator, result["wall_s"]),
            "converged": bool(result["converged"]), "stop_reason": result["stopped_because"],
            "diagnostics": {"warmup_discarded": int(last["trunc"]), "requests_served": len(records),
                            "requests_aborted": sum(1 for r in records if not math.isfinite(r["completion_t"])),
                            "little_residual": last["window"]["little_res"]},
            "measured_clock_mhz": clock_mhz, "kv_blocks": kv_blocks, "kv_blocks_computed": kv_blocks_computed}


def document(points: Sequence[Mapping[str, Any]], targets: Mapping[str, float], *, components: Mapping[str, str],
             seed: int, pool_sha256: str, vllm_version: str, resolved_manifest: Mapping[str, Any]) -> dict[str, Any]:
    doc = {"schema": SCHEMA, "points": list(points), "estimand": {"rule": "steady_state", **targets},
           "provenance": {"components": dict(components), "servecast_version": __version__, "seed": int(seed),
                          "pool_sha256": pool_sha256, "vllm_version": vllm_version},
           "resolved_manifest": dict(resolved_manifest)}
    formats.Measurement.model_validate(doc)
    return doc


def _clean(environment: Mapping[str, Any]) -> dict[str, Any]:
    """The environment without any key that identifies one machine."""
    def walk(node: Any) -> Any:
        if isinstance(node, Mapping):
            return {k: walk(v) for k, v in node.items() if str(k).lower() not in FORBIDDEN_KEYS}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(environment)


def _atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _finite_or_none(v: float) -> float | None:
    return v if math.isfinite(v) else None


class Writer:
    """The output directory of one `profile serving` run."""

    def __init__(self, out_dir: Path, pool_p: Sequence[int], pool_g: Sequence[int], pool_header: Mapping[str, Any],
                 environment: Mapping[str, Any]) -> None:
        self.dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.pool_sha256 = pools.write(out_dir / "pool.jsonl", pool_p, pool_g, dict(pool_header))
        _atomic(out_dir / "environment.json", json.dumps(_clean(environment), indent=1) + "\n")
        (out_dir / "requests.jsonl").write_text("", encoding="utf-8")
        self._sums()

    def add(self, doc: Mapping[str, Any], records: Sequence[Mapping[str, Any]], kept_rids: set[int],
            concurrency: int, think_time_ms: float) -> None:
        """One more load point: its requests appended, measurement.json and SHA256SUMS rewritten."""
        with (self.dir / "requests.jsonl").open("a", encoding="utf-8") as handle:
            for r in records:
                handle.write(json.dumps({
                    "concurrency": int(concurrency), "think_time_ms": float(think_time_ms), "request": r["rid"],
                    "client": r["slot"], "input_tokens": r["p"], "output_tokens": r["g"],
                    "output_tokens_received": r["n_out"], "think_s": r["think_t"], "submit_s": r["arrival_t"],
                    "first_token_s": _finite_or_none(r["first_t"]), "completion_s": _finite_or_none(r["completion_t"]),
                    "ttft_ms": _finite_or_none(r["ttft"] * 1e3), "tpot_ms": _finite_or_none(r["tpot"] * 1e3),
                    "e2e_ms": _finite_or_none(r["e2e"] * 1e3), "in_window": r["rid"] in kept_rids,
                    "aborted": not math.isfinite(r["completion_t"])}) + "\n")
        _atomic(self.dir / "measurement.json", json.dumps(doc, indent=1) + "\n")
        self._sums()

    def _sums(self) -> None:
        lines = []
        for name in FILES[:-1]:
            path = self.dir / name
            if path.is_file():
                lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
        _atomic(self.dir / "SHA256SUMS", "\n".join(lines) + "\n")
