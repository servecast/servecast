"""Where ServeCast keeps files on the user's computer: a cache (fitted workload laws, downloaded config.json files)
and a data directory (the user's calibrations). Both can be moved with an environment variable."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def cache_dir() -> Path:
    if os.environ.get("SERVECAST_CACHE_DIR"):
        return Path(os.environ["SERVECAST_CACHE_DIR"])
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "servecast"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "servecast"


def data_dir() -> Path:
    if os.environ.get("SERVECAST_DATA_DIR"):
        return Path(os.environ["SERVECAST_DATA_DIR"])
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "servecast"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "servecast"
