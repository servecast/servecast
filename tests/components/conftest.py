"""Bit-exact checks run on the reference environment (macOS arm64, Python 3.12, numpy 2.5.0, scipy 1.18.0).

Elsewhere a different math library may round one operation differently, and a simulation then follows another random
path: the tests marked `exact` are skipped there, unless SERVECAST_EXACT=1 asks for them (CI's informational Linux
job does). SERVECAST_EXACT=0 skips them anywhere.
"""

from __future__ import annotations

import os
import platform

import pytest

REFERENCE = platform.system() == "Darwin" and platform.machine() == "arm64"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    wanted = os.environ.get("SERVECAST_EXACT")
    if wanted == "1" or (REFERENCE and wanted != "0"):
        return
    skip = pytest.mark.skip(reason="bit-exact against the reference data: runs on macOS arm64, or with "
                                   "SERVECAST_EXACT=1")
    for item in items:
        if item.get_closest_marker("exact"):
            item.add_marker(skip)
