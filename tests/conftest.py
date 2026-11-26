"""Every test session gets its own ServeCast cache and data directories, so no test reads or writes the user's: a
workload law another run cached, or a calibration the user saved, would otherwise make a test pass for a reason it
does not state.

Bit-exact checks run on the reference environment (macOS arm64, Python 3.12, numpy 2.5.0, scipy 1.18.0). Elsewhere a
different math library may round one operation differently, and a simulation then follows another random path: the
tests marked `exact` are skipped there, unless SERVECAST_EXACT=1 asks for them (CI's informational Linux job does).
SERVECAST_EXACT=0 skips them anywhere. The hook lives here, so that it applies to any test file run on its own (it
lived under tests/components, and tests/test_docs_examples.py alone ran its exact tests under SERVECAST_EXACT=0).
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import sys
from collections.abc import Iterator

import pytest


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


#: what `servecast doctor` calls the reference environment, where the reference outputs were made
REFERENCE = (platform.system() == "Darwin" and platform.machine() == "arm64" and sys.version_info[:2] == (3, 12)
             and _version("numpy") == "2.5.0" and _version("scipy") == "1.18.0")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    wanted = os.environ.get("SERVECAST_EXACT")
    if wanted == "1" or (REFERENCE and wanted != "0"):
        return
    skip = pytest.mark.skip(reason="bit-exact against the reference data: runs on the reference environment (macOS "
                                   "arm64, Python 3.12, numpy 2.5.0, scipy 1.18.0), or with SERVECAST_EXACT=1")
    for item in items:
        if item.get_closest_marker("exact"):
            item.add_marker(skip)


@pytest.fixture(autouse=True, scope="session")
def _private_directories(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SERVECAST_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
        mp.setenv("SERVECAST_DATA_DIR", str(tmp_path_factory.mktemp("data")))
        yield
