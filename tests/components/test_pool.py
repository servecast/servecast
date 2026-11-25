"""Request pools: trace formats, the shipped traces, and the pool file's digest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from servecast.core import pool


def test_mooncake_and_lengths_files(tmp_path: Path) -> None:
    moon = tmp_path / "t.jsonl"
    moon.write_text('{"input_length": 10, "output_length": 3, "hash_ids": [1, 2]}\n'
                    '{"input_length": 20, "output_length": 5, "hash_ids": [1, 3]}\n')
    p, g, warnings = pool.read_trace(moon, "mooncake")
    assert (p, g) == ([10, 20], [3, 5]) and warnings and "prefix reuse is not modelled" in warnings[0]
    lengths = tmp_path / "l.csv"
    lengths.write_text("input_tokens,output_tokens\n7,1\n8,2\n")
    assert pool.read_trace(lengths, "lengths")[:2] == ([7, 8], [1, 2])


@pytest.mark.parametrize("body, message", [('{"input_length": 10}\n', "needs integer"),
                                           ('{"input_length": 0, "output_length": 3}\n', "below 1 token"),
                                           ("", "no request")])
def test_a_bad_trace_is_refused_with_the_line(tmp_path: Path, body: str, message: str) -> None:
    f = tmp_path / "t.jsonl"
    f.write_text(body)
    with pytest.raises(ValueError, match=message):
        pool.read_trace(f, "mooncake")


def test_a_pool_file_round_trips_and_its_digest_is_checked(tmp_path: Path) -> None:
    f = tmp_path / "pool.jsonl"
    sha = pool.write(f, [5, 6, 7], [1, 2, 3], {"source": "trace", "files": ["test"]})
    p, g, head = pool.read(f)
    assert (p, g, head["sha256"]) == ([5, 6, 7], [1, 2, 3], sha)
    lines = f.read_text().splitlines()
    f.write_text("\n".join([lines[0], lines[1], json.dumps({"input_tokens": 9, "output_tokens": 2}), lines[3]]) + "\n")
    with pytest.raises(ValueError, match="digest"):
        pool.read(f)


@pytest.mark.parametrize("name, n", [("azure-conversation", 17754), ("burstgpt", 50000), ("servegen", 44990)])
def test_the_shipped_traces(name: str, n: int) -> None:
    doc = pool.shipped_trace(name)
    assert doc["requests"] == len(doc["input_tokens"]) == len(doc["output_tokens"]) == n
    assert min(doc["input_tokens"]) >= 1 and min(doc["output_tokens"]) >= 1
    assert max(a + b for a, b in zip(doc["input_tokens"], doc["output_tokens"], strict=False)) <= 4096
    assert doc["license"] in ("CC-BY-4.0", "Apache-2.0") and doc["changes"]
