"""Every file ServeCast ships or writes follows the format docs/reference/formats.md documents for it (audit AREA-4,
COST-LAW-C6: the documented calibration, profiling-data and pool formats described no real file)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from servecast.calibrate import fit
from servecast.core import formats
from servecast.core import pool as pools
from servecast.model import predict as P


def test_every_shipped_profiling_file_follows_its_format() -> None:
    shipped = P.shipped_profiling()
    assert len(shipped) == 19
    for name, path in shipped.items():
        doc = formats.ProfilingData.model_validate(json.loads(path.read_text(encoding="utf-8")))
        assert (doc.llm is None) == name.endswith("c62ebe75"), name  # the per-GPU law probes synthetic models


@pytest.mark.parametrize("header", [{"source": "trace", "files": ["azure-conversation"]},
                                    {"source": "synthetic", "sampler": "fe3db391", "seed": 0,
                                     "requested": {"input": {"mean": 3.0, "variance": 6.0},
                                                   "output": {"mean": 2.0, "variance": 2.0}},
                                     "realized": {"input": {"mean": 3.0, "variance": 1.0},
                                                  "output": {"mean": 2.0, "variance": 1.0}},
                                     "laws": {}}])
def test_a_written_pool_follows_its_format_and_reads_back(tmp_path: Path, header: dict) -> None:
    path = tmp_path / "pool.jsonl"
    pools.write(path, [2, 4], [1, 3], header)
    lines = path.read_text(encoding="utf-8").splitlines()
    formats.PoolHeader.model_validate(json.loads(lines[0]))
    for line in lines[1:]:
        formats.PoolRecord.model_validate(json.loads(line))
    assert pools.read(path)[:2] == ([2, 4], [1, 3])


def test_a_pool_header_that_is_not_one_is_refused_on_write_and_on_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="names its files"):
        pools.write(tmp_path / "w.jsonl", [2], [1], {"source": "trace"})
    path = tmp_path / "r.jsonl"
    pools.write(path, [2], [1], {"source": "trace", "files": ["x"]})
    lines = path.read_text(encoding="utf-8").splitlines()
    head = {k: v for k, v in json.loads(lines[0]).items() if k != "files"}
    path.write_text("\n".join([json.dumps(head), *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"r\.jsonl: not a servecast-pool/1 file"):
        pools.read(path)


def test_profiling_data_that_is_not_is_refused_naming_the_file(tmp_path: Path) -> None:
    doc = json.loads(P.shipped_profiling()["h100_sxm/d9cbf1d3/llama-3-8b"].read_text(encoding="utf-8"))
    del doc["cells"]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=r"(?s)broken\.json: not profiling data .*either llm and cells"):
        fit.read_profiling(path)
