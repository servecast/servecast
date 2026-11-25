"""run.json, the record a command leaves beside its outputs (servecast-run/1).

It says what ran (the command line), with which version of ServeCast, when, from which inputs (each with its sha256),
with which components, and which files it wrote. With it, a directory of outputs explains itself, whatever the command
that made it: `predict` and `calibrate` write one, and `profile` will.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "servecast-run/1"
NAME = "run.json"


def now() -> str:
    """The local time with its UTC offset, to the second."""
    return dt.datetime.now(dt.UTC).astimezone().isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(out_dir: Path, *, version: str, started: str, inputs: Mapping[str, Mapping[str, Any]],
          outputs: Sequence[str], components: Mapping[str, str] | None = None,
          argv: Sequence[str] | None = None) -> Path:
    """Write `out_dir/run.json`; `outputs` are the names of the other files the command wrote there."""
    args = list(sys.argv if argv is None else argv)
    doc = {"schema": SCHEMA, "command": shlex.join(["servecast", *args[1:]]), "servecast": version,
           "started": started, "finished": now(), "inputs": {k: dict(v) for k, v in inputs.items()},
           "components": dict(components or {}), "outputs": sorted(set(outputs))}
    path = Path(out_dir) / NAME
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
