"""One synthetic model's calibration probe, in a process of its own (`profile calibration` for the per-GPU law).

    python -m servecast.profile.probe_child JOB.json RECORD.json

Each synthetic model is probed by a fresh process, as the probes behind the shipped per-GPU calibration were run one
per model, so that the engine's GPU memory is freed when the process ends before the next model's engine starts.
The job names the model's directory, the engine settings, the design and the lock the parent holds; the record is
the probe's steps, clock samples and audit (`probe.audit`), written whatever happened, before a failure ends the
process with status 1.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any


def run(job: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - needs a GPU and vLLM
    from servecast.core.deployment import Deployment
    from servecast.profile import gpu as G
    from servecast.profile import probe as PR
    from servecast.profile.probe_vllm import VllmProbeEngine

    backend = G.detect()
    design = job["design"]
    eng = VllmProbeEngine(job["dir"], Deployment(**job["deployment"]), backend, kv_tokens=design["kv_tokens"],
                          sample_clock=backend.sm_clock_mhz, load_format="dummy")
    spans: list[dict[str, Any]] = []
    aborted = None
    try:
        PR.drive(eng, design, lambda: len(eng.beats), spans)
    except (PR.ProbeError, RuntimeError, ValueError, OSError) as error:
        aborted = f"{type(error).__name__}: {error}"
    finally:
        eng.shutdown()
    beats = list(eng.beats)
    samples = [dict(t) for t in eng.clock]
    return {"spans": spans, "beats": [dataclasses.asdict(b) for b in beats], "clock": samples,
            "audit": PR.audit(beats, spans, design, samples, lock_mhz=job.get("lock_mhz"), aborted=aborted,
                              degraded=eng.degraded)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a GPU and vLLM
    job_path, out_path = (argv if argv is not None else sys.argv[1:])
    record = run(json.loads(Path(job_path).read_text(encoding="utf-8")))
    Path(out_path).write_text(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if record["audit"]["clean"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
