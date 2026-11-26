"""Request pools: the concrete (input, output) lengths that the prediction fits its workload law to and that the
measurement serves. A pool comes from a trace (all of its requests, in order) or from the length sampler.

Trace formats: Mooncake JSONL (`input_length`, `output_length` per line; `hash_ids` is accepted, with a warning when
ids repeat, since prefix reuse is not modelled), or a lengths file (JSONL with `input_tokens` and `output_tokens`, or
CSV with those two columns). The paper's three traces ship with the package.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from servecast.core.formats import PoolHeader

SCHEMA = "servecast-pool/1"
TRACE_SCHEMA = "servecast-trace/1"
SHIPPED_TRACES = ("azure-conversation", "burstgpt", "servegen")


def shipped_trace(name: str) -> dict[str, Any]:
    """One of the paper's three traces: lengths, source, licence and our changes."""
    if name not in SHIPPED_TRACES:
        raise ValueError(f"no shipped trace {name!r}; shipped: {', '.join(SHIPPED_TRACES)}")
    path = Path(str(resources.files("servecast").joinpath("data", "traces", f"{name}.json")))
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"{path}: not a trace ({TRACE_SCHEMA})")
    return doc


def _length(value: Any) -> int:
    """A token count as a trace writes it: an integer, or a whole number written as one (512.0, "512"); a fraction, a
    boolean or a value too large to be finite is refused (2.5 used to be read as 2, true as 1, 1e400 crashed)."""
    if isinstance(value, bool):
        raise ValueError("a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"{value!r}")


def _json_line(path: Path, n: int, line: str) -> Any:
    try:
        return json.loads(line)
    except ValueError as error:  # the file and the line, which json leaves out
        hint = "; a CSV trace needs format: lengths" if path.suffix.lower() == ".csv" else ""
        raise ValueError(f"{path}:{n}: not a JSON line ({error.args[0].split(':')[0]}){hint}") from None


def read_trace(path: Path, fmt: str) -> tuple[list[int], list[int], list[str]]:
    """(input lengths, output lengths, warnings) of a trace file, every request in file order. ValueError names the
    file and the line of anything that is not one."""
    try:
        return _read_trace(path, fmt)
    except UnicodeDecodeError:
        raise ValueError(f"{path}: not UTF-8 text; a trace is JSON lines or CSV") from None


def _read_trace(path: Path, fmt: str) -> tuple[list[int], list[int], list[str]]:
    p: list[int] = []
    g: list[int] = []
    warnings: list[str] = []
    if fmt == "mooncake":
        seen: set[Any] = set()
        repeated = 0
        with path.open(encoding="utf-8") as handle:
            for n, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = _json_line(path, n, line)
                try:
                    p.append(_length(row["input_length"]))
                    g.append(_length(row["output_length"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{n}: needs integer input_length and output_length") from exc
                for h in row.get("hash_ids") or ():
                    repeated += h in seen
                    seen.add(h)
        if repeated:
            warnings.append(f"{path}: {repeated} prefix-block ids repeat; prefix reuse is not modelled, every prompt "
                            "is computed in full")
    elif fmt == "lengths":
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                for n, row in enumerate(csv.DictReader(handle), 2):
                    try:
                        p.append(_length(row["input_tokens"]))
                        g.append(_length(row["output_tokens"]))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"{path}:{n}: needs integer input_tokens and output_tokens") from exc
        else:
            with path.open(encoding="utf-8") as handle:
                for n, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = _json_line(path, n, line)
                    try:
                        p.append(_length(row["input_tokens"]))
                        g.append(_length(row["output_tokens"]))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"{path}:{n}: needs integer input_tokens and output_tokens") from exc
    else:
        raise ValueError(f"unknown trace format {fmt!r}; known: mooncake, lengths")
    if not p:
        raise ValueError(f"{path}: no request")
    bad = next((i for i, (a, b) in enumerate(zip(p, g, strict=False)) if a < 1 or b < 1), None)
    if bad is not None:
        raise ValueError(f"{path}: request {bad + 1} has a length below 1 token")
    return p, g, warnings


def digest(p: Sequence[int], g: Sequence[int]) -> str:
    """The pool's identity: sha256 over its lengths, in order."""
    h = hashlib.sha256()
    for a, b in zip(p, g, strict=False):
        h.update(f"{int(a)},{int(b)}\n".encode())
    return h.hexdigest()


def write(path: Path, p: Sequence[int], g: Sequence[int], header: dict[str, Any]) -> str:
    """A pool file: a header line, then one request per line. Returns its digest."""
    sha = digest(p, g)
    head = {"schema": SCHEMA, "requests": len(p), "sha256": sha, **header}
    PoolHeader.model_validate(head)  # what formats.md documents, or a defect here
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(head) + "\n")
        for a, b in zip(p, g, strict=False):
            handle.write(f'{{"input_tokens": {int(a)}, "output_tokens": {int(b)}}}\n')
    return sha


def read(path: Path) -> tuple[list[int], list[int], dict[str, Any]]:
    """A pool file written by `write`; its digest is checked."""
    with path.open(encoding="utf-8") as handle:
        head = json.loads(handle.readline())
        if str(head.get("schema", "")).split("/")[0] != SCHEMA.split("/")[0] or head["schema"] != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} file")
        try:
            PoolHeader.model_validate(head)
        except ValidationError as error:
            raise ValueError(f"{path}: not a {SCHEMA} file: {error}") from None
        p, g = [], []
        for line in handle:
            row = json.loads(line)
            p.append(int(row["input_tokens"]))
            g.append(int(row["output_tokens"]))
    if digest(p, g) != head["sha256"] or len(p) != head["requests"]:
        raise ValueError(f"{path}: the requests do not match the header's digest")
    return p, g, head
