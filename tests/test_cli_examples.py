"""The executed examples: every example in every help page runs as a user would type it and ends with the status its
page implies.

Each example line runs on its own, through bash with pipefail, in a fresh scratch directory that holds what the example
presumes (SETUP: a manifest and its trace for `predict`, profiling data for `calibrate`, a prediction and a measurement
for `compare`), with HOME and ServeCast's cache and data directories inside it. An example that cannot succeed on this
machine says why in EXPECTED; one that runs for hours is in SLOW and runs only with SERVECAST_SLOW=1.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from servecast.core import pool
from servecast.model.predict import shipped_profiling
from tests.test_cli_help import PATHS, _command

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.executable).parent
SLOW_ON = os.environ.get("SERVECAST_SLOW") == "1"


#: pages whose example block is one sequence (each line needs the ones before it); every other block lists
#: alternatives, each run on its own
SEQUENCES = {()}


def blocks() -> list[tuple[tuple[str, ...], str, tuple[str, ...]]]:
    """(command path, example line, the lines that run before it) for every line of every code block in the help,
    continuations joined."""
    out = []
    for path in PATHS:
        command = _command(path)
        text = f"{command.help or ''}\n{command.epilog or ''}"
        for block in re.findall(r"```\n(.*?)```", text, flags=re.S):
            lines = [re.sub(r"\s+", " ", x).strip()
                     for x in block.replace("\\\n", " ").splitlines() if "servecast" in x]
            for i, line in enumerate(lines):
                out.append((path, line, tuple(lines[:i]) if path in SEQUENCES else ()))
    return out


def _manifest(d: Path) -> None:
    run(d, "servecast manifest init --template predict > m.yaml")


def _trace_file(d: Path) -> None:
    trace = pool.shipped_trace("azure-conversation")
    with (d / "azure_conv.jsonl").open("w", encoding="utf-8") as handle:
        for p, g in zip(trace["input_tokens"], trace["output_tokens"], strict=True):
            handle.write(json.dumps({"input_length": p, "output_length": g}) + "\n")


def _calibrate_files(d: Path) -> None:
    (d / "a100-profiling-data").mkdir()
    shutil.copyfile(shipped_profiling()["a100_sxm_40gb/c62ebe75"],
                    d / "a100-profiling-data" / "profiling.json")
    _manifest(d)
    doc = yaml.safe_load((d / "m.yaml").read_text(encoding="utf-8"))
    doc["calibration"] = {"data": "./a100-profiling-data"}
    (d / "m.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")


def _compare_files(d: Path) -> None:
    from tests.test_compare import measurement, prediction, write

    for pred, meas in (("servecast-output", "servecast-measurement"), ("prediction", "measurement")):
        write(d / pred, "prediction.json", prediction())
        write(d / meas, "measurement.json", measurement())


#: what an example presumes beyond what an earlier line of its own page makes: the root page's "Start here" needs
#: nothing (its first line writes m.yaml), the other pages' examples presume the manifest a user would have
SETUP: dict[tuple[str, ...], Callable[[Path], None]] = {
    ("predict",): _manifest, ("calibrate",): _calibrate_files, ("compare",): _compare_files,
    ("manifest", "validate"): _manifest, ("workload", "sample"): _manifest, ("workload", "describe"): _trace_file,
    ("profile", "serving"): _manifest, ("profile", "calibration"): _manifest,
}
#: examples that cannot succeed here: the status they end with and a phrase their error must contain (a crash, which
#: also exits 1, does not pass)
EXPECTED: dict[str, tuple[int, str]] = {
    "servecast replicate fetch": (1, "is published with the first release"),
    "servecast profile calibration -c m.yaml": (3, "this computer cannot profile"),
    # on a computer without an NVIDIA GPU, as CI's are (tests/test_cli_profile.py runs them on a fake GPU host)
    "servecast profile serving -c m.yaml": (3, "this computer cannot profile"),
    "servecast profile serving -c m.yaml --sweep traffic.load.concurrency=40,80,120":
        (3, "this computer cannot profile"),
}
#: examples that run for hours: the full recomputation of a GPU's transfer matrices. Its predictions are the paper
#: parity's (tools/parity.py paper, in the parity-full workflow), so only the command around them waits for
#: SERVECAST_SLOW=1.
SLOW = {"servecast replicate run rq2 --gpu h100_sxm"}
EXAMPLES = list({line: (path, line, before) for path, line, before in reversed(blocks())}.values())[::-1]


def run(d: Path, line: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(d), "SERVECAST_CACHE_DIR": str(d / "cache"),
           "SERVECAST_DATA_DIR": str(d / "data"), "PATH": f"{BIN}{os.pathsep}{os.environ.get('PATH', '')}",
           "NO_COLOR": "1", "ZDOTDIR": str(d), "XDG_CONFIG_HOME": str(d / ".config")}
    # bash with pipefail, so a failing command piped into another (`servecast ... | cat`) still fails
    return subprocess.run(["bash", "-o", "pipefail", "-c", line], cwd=d, env=env, capture_output=True, text=True,
                          timeout=1800)


def test_the_examples_are_found() -> None:
    assert len(EXAMPLES) >= 30 and len(blocks()) > len(EXAMPLES)       # some pages repeat another's example
    start = [line for path, line, _ in blocks() if path == ()]
    assert start[0].startswith("servecast manifest init") and "servecast predict -c m.yaml" in start
    assert set(EXPECTED) | SLOW <= {line for _, line, _ in EXAMPLES}


@pytest.mark.parametrize(("path", "line", "before"), EXAMPLES, ids=[line for _, line, _ in EXAMPLES])
def test_every_example_runs(tmp_path: Path, path: tuple[str, ...], line: str, before: tuple[str, ...]) -> None:
    if line in SLOW and not SLOW_ON:
        pytest.skip("runs for hours; SERVECAST_SLOW=1 runs it")
    setup = SETUP.get(path)
    if setup is not None:
        setup(tmp_path)
    for earlier in before:
        assert run(tmp_path, earlier).returncode == 0, earlier
    result = run(tmp_path, line)
    status, phrase = EXPECTED.get(line, (0, ""))
    assert result.returncode == status, f"{line}: status {result.returncode}, expected {status}\n" \
                                        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    assert "Traceback" not in result.stderr, result.stderr[-2000:]
    assert phrase in result.stderr, f"{line}: the error does not say {phrase!r}: {result.stderr[-500:]}"
