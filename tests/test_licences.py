"""The licences the package declares are those of what it ships: the code under Apache-2.0, with a NOTICE that every
redistribution keeps, and the request lengths of the shipped traces under their own licences."""

from __future__ import annotations

import importlib.metadata as md
import json
from importlib import resources
from pathlib import Path


def test_the_declared_licence_covers_every_shipped_trace() -> None:
    """The metadata said Apache-2.0 alone, while two shipped traces are CC-BY-4.0."""
    traces = resources.files("servecast").joinpath("data", "traces")
    licences = {json.loads(t.read_text(encoding="utf-8"))["license"] for t in traces.iterdir()
                if t.name.endswith(".json")}
    assert licences == {"Apache-2.0", "CC-BY-4.0"}
    assert set(md.metadata("servecast")["License-Expression"].split(" AND ")) == licences


def test_the_licence_files_ship_with_the_package() -> None:
    """The wheel carried LICENSE alone: no NOTICE, and no list of the traces' licences."""
    shipped = {Path(str(f)).name for f in md.files("servecast") or [] if "licenses" in Path(str(f)).parts}
    assert {"LICENSE", "NOTICE", "DATA_LICENSES.md"} <= shipped
