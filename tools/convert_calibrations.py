"""Convert calibration files to servecast-calibration/2, in place: the same numbers, moved into law, conditions and
fit (servecast.core.calibration.upgrade, the function predict uses on a /1 file it reads). A /2 file is left as it is,
so running it twice changes nothing.

    python tools/convert_calibrations.py                 # the shipped calibrations
    python tools/convert_calibrations.py FILE [FILE ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from servecast.core import calibration as calibrations  # noqa: E402

SHIPPED = ROOT / "src" / "servecast" / "data" / "calibrations"


def convert(path: Path) -> bool:
    """Whether the file was converted; it is first read back as predict reads it, and compared number for number."""
    old = json.loads(path.read_text(encoding="utf-8"))
    if old.get("schema") == calibrations.SCHEMA:
        return False
    new = calibrations.parse(json.loads(path.read_text(encoding="utf-8")), path)
    names, values = old["parameters"]["names"], old["parameters"]["values"]
    moved = [new["law"]["parameters"][n] for n in names] != values
    if moved or calibrations.peak_fraction(new) != old["clock"]["peak_fraction"]:
        raise SystemExit(f"{path}: the conversion moved a number")
    path.write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", type=Path, help="calibration files (default: the shipped ones)")
    paths = ap.parse_args(argv).files or sorted(SHIPPED.rglob("*.json"))
    converted = [p for p in paths if convert(p)]
    print(f"{len(converted)} converted, {len(paths) - len(converted)} already servecast-calibration/2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
