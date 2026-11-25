"""The beat durations of every shipped calibration, for each of ServeCast's six known LLMs, on the 120 compositions of
tests/components/goldens/law.json: the reference tests/components/test_law.py compares the shipped calibrations with,
exactly. Run on the commit whose numbers are the reference, before the change it guards:

    PYTHONPATH=<checkout>/src python tools/shipped_beats.py > tests/components/goldens/shipped-beats.json

Each entry is keyed by what the calibration is, not by its name: its GPU, its cost law's name and what it was fitted
on, so a renamed or reformatted calibration finds its entry. It reads either calibration format.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def key(doc: dict[str, Any]) -> str:
    from servecast.core import versions

    if doc.get("schema") == "servecast-calibration/1":
        gpu, law, on = doc["gpu"], doc["cost_law"], doc["calibrated_on"]
    else:
        gpu, law, on = doc["conditions"]["gpu"], doc["law"]["cost_law"], doc["conditions"]["fitted_on"]
    on = on if isinstance(on, str) else f"{len(on)} synthetic models"
    return f"{gpu} | {versions.canonical('cost_law', str(law))} | {on}"


def main() -> int:
    import numpy

    import servecast
    from servecast.model import geometry
    from servecast.model import predict as P
    from servecast.model.population.v2026_10 import Beat
    from tests.components.inputs import load

    beats = [Beat(prefill=tuple(map(tuple, c["prefill"])), decode=tuple(map(tuple, c["decode"])))
             for c in load("law")["compositions"]]
    cases: dict[str, dict[str, list[float]]] = {}
    for name, path in P.shipped_calibrations().items():
        raw = json.loads(path.read_text(encoding="utf-8"))
        calibration = P.load_calibration(name)
        cases[key(raw)] = {llm: [P.clock(calibration, g)(b) for b in beats] for llm, g in geometry.KNOWN.items()}
    src = Path(servecast.__file__).resolve().parents[1]
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(src), "status", "--porcelain"], capture_output=True,
                                text=True).stdout.strip())
    doc = {"about": "Beat durations in ms of every shipped calibration x LLM on the 120 compositions of law.json "
                    "(tools/shipped_beats.py)",
           "made_by": {"commit": commit, "dirty": dirty, "python": platform.python_version(),
                       "numpy": numpy.__version__},
           "cases": dict(sorted(cases.items()))}
    print(json.dumps(doc, indent=0, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
