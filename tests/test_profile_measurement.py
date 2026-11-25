"""What profile serving writes: a load point summarised as a prediction is, the files, their order and checksums."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from servecast.core import estimator as estimators
from servecast.core import formats
from servecast.core import summary as summaries
from servecast.profile import closed_loop as C
from servecast.profile import measurement as M
from tests.test_profile_closed_loop import TARGETS, FakeEngine

EST = estimators.get("2026-10")


def result() -> dict:
    run = C.serve(FakeEngine(), list(range(20)), C.lengths([5, 9, 13, 2], [3, 1, 4, 6], 0), 6, 0.001, EST, TARGETS,
                  seed=0, n_ceiling=700)
    return asyncio.run(asyncio.wait_for(run, timeout=60))


def test_a_measured_point_is_summarised_as_a_prediction_is() -> None:
    r = result()
    p = M.point(r, 6, 1.0, EST, clock_mhz=1210.0, kv_blocks=9965, kv_blocks_computed=10400)
    formats.MeasuredPoint.model_validate(p)
    last = r["assessment"]
    want = summaries.point_summary(summaries.window_quantities(last["kept"], last["window"]),
                                   last["decision"]["per_metric"], EST, r["wall_s"])
    assert p["summary"] == want and p["converged"] == r["converged"]
    assert p["diagnostics"]["requests_served"] == len(r["records"])


def test_the_files_their_checksums_and_no_machine_identifier(tmp_path: Path) -> None:
    env = {"gpu": {"name": "NVIDIA A100-SXM4-40GB", "uuid": "GPU-x", "serial": "1", "pci_bus_id": "0"},
           "hostname": "box", "driver": "580.0", "nested": [{"Host": "h", "cuda": "12.9"}]}
    w = M.Writer(tmp_path / "m", [5, 9], [3, 1], {"source": "trace", "files": ["test"]}, env)
    written = json.loads((tmp_path / "m" / "environment.json").read_text(encoding="utf-8"))
    assert written == {"gpu": {"name": "NVIDIA A100-SXM4-40GB"}, "driver": "580.0", "nested": [{"cuda": "12.9"}]}
    r = result()
    p = M.point(r, 6, 1.0, EST, clock_mhz=1210.0, kv_blocks=9965, kv_blocks_computed=None)
    doc = M.document([p], {"throughput": 0.05, "e2e": 0.1, "ttft": 0.1, "tpot": 0.05},
                     components={"estimator": "2026-10"}, seed=0, pool_sha256=w.pool_sha256, vllm_version="0.25.1",
                     resolved_manifest={"engine": {}, "traffic": {}})
    kept = {q["rid"] for q in r["assessment"]["kept"]}
    w.add(doc, r["records"], kept, 6, 1000.0)
    lines = [json.loads(x) for x in (tmp_path / "m" / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(r["records"]) and sum(x["in_window"] for x in lines) == len(kept)
    assert all(x["aborted"] == (x["completion_s"] is None) for x in lines)
    sums = dict(reversed(line.split("  ")) for line in
                (tmp_path / "m" / "SHA256SUMS").read_text(encoding="utf-8").splitlines())
    assert set(sums) == {"measurement.json", "pool.jsonl", "requests.jsonl", "environment.json"}
    for name, digest in sums.items():
        assert hashlib.sha256((tmp_path / "m" / name).read_bytes()).hexdigest() == digest
    formats.Measurement.model_validate(json.loads((tmp_path / "m" / "measurement.json").read_text(encoding="utf-8")))
    assert not list((tmp_path / "m").glob("*.tmp"))
