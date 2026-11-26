"""The released component modules, held to their sha256 in versions.lock, so that an edit to one cannot pass unseen.

A released module (src/servecast/**/v20*.py) computes what its version name promises, for good: a change to what it
computes is a new version, in a new module. Its tests compare it with reference outputs, which a change that leaves
those outputs unchanged would pass; the lock does not.

    python tools/versions_lock.py            # exit 1 if a released module differs from the lock, or is not in it
    python tools/versions_lock.py --update   # write the lock: only when a released module is added
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "versions.lock"
HEADER = ("# The released component modules and their sha256: a change to one is a new version in a new module.\n"
          "# Written by `python tools/versions_lock.py --update`, only when a released module is added.\n")


def released() -> dict[str, str]:
    """Each released module, by its path in the repository, with its sha256."""
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src" / "servecast").rglob("v20*.py"))}


def locked() -> dict[str, str]:
    out = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, path = line.split(None, 1)
            out[path.strip()] = digest
    return out


def problems() -> list[str]:
    """What differs between the released modules and the lock, one line each."""
    now, lock = released(), locked()
    out = [f"{path}: changed (sha256 {now[path][:12]}, locked {lock[path][:12]})"
           for path in sorted(now.keys() & lock.keys()) if now[path] != lock[path]]
    out += [f"{path}: released but not in versions.lock" for path in sorted(now.keys() - lock.keys())]
    out += [f"{path}: in versions.lock but gone" for path in sorted(lock.keys() - now.keys())]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hold the released component modules to versions.lock.")
    parser.add_argument("--update", action="store_true", help="write the lock (only when a released module is added)")
    args = parser.parse_args(argv)
    if args.update:
        LOCK.write_text(HEADER + "".join(f"{digest}  {path}\n" for path, digest in released().items()),
                        encoding="utf-8")
        print(f"wrote {LOCK.name}: {len(released())} released modules")
        return 0
    found = problems()
    for line in found:
        print(line, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
