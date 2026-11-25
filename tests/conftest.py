"""Every test session gets its own ServeCast cache and data directories, so no test reads or writes the user's: a
workload law another run cached, or a calibration the user saved, would otherwise make a test pass for a reason it
does not state."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True, scope="session")
def _private_directories(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SERVECAST_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
        mp.setenv("SERVECAST_DATA_DIR", str(tmp_path_factory.mktemp("data")))
        yield
