"""examples/slurm/servecast.sbatch, the On your GPU guide's five steps as one Slurm job, run here by bash with a
stand-in `servecast` that records each call: the job runs the guide's steps in the guide's order, takes a manifest and
a calibration name of your own, and stops at the first command that fails. Slurm itself is not run."""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

from tests.test_docs_examples import SHELL, commands, fences_in

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "slurm" / "servecast.sbatch"
GUIDE = "docs/guides/calibrate.md"


def steps(page: str) -> list[str]:
    """The servecast commands of a guide's numbered steps, its sections headed `## 1.`, `## 2.` and so on."""
    sections = re.split(r"(?m)^(?=## )", (ROOT / page).read_text(encoding="utf-8"))
    return [line.strip() for section in sections if re.match(r"## \d+\. ", section)
            for language, body in fences_in(section) if language.split(" ")[0] in SHELL
            for line in body.splitlines() if line.strip().startswith("servecast ")]


def run_job(tmp_path: Path, *args: str, refuse: str | None = None) -> tuple[subprocess.CompletedProcess[str],
                                                                             list[str]]:
    """The job's script under bash, with a `servecast` that records its arguments (and exits 3 on `refuse`)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    stand_in = bin_dir / "servecast"
    stand_in.write_text("#!/bin/bash\n"
                        f'echo "servecast $*" >> "{calls}"\n'
                        + (f'[[ "$*" == "{refuse}"* ]] && exit 3\n' if refuse else "")
                        + "exit 0\n", encoding="utf-8")
    stand_in.chmod(stand_in.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    done = subprocess.run(["bash", str(SCRIPT), *args], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=60)
    return done, calls.read_text(encoding="utf-8").splitlines() if calls.is_file() else []


def test_the_guides_steps_are_found() -> None:
    """Five steps, after the command that writes the example manifest; the sections after them (a GPU, an LLM or a
    workload of your own) are not the job's."""
    assert len(steps(GUIDE)) == 5
    assert commands(GUIDE)[0] == "servecast manifest show example-a100 > example-a100.yaml"
    assert steps(GUIDE) == commands(GUIDE)[1:6]


def test_the_job_runs_the_guides_steps_in_order(tmp_path: Path) -> None:
    """Without a manifest of yours, the job writes the example first, as the guide does before step 1."""
    done, calls = run_job(tmp_path)
    assert done.returncode == 0, done.stderr
    assert calls == ["servecast doctor", "servecast manifest show example-a100", *steps(GUIDE)]
    assert "servecast manifest show example-a100 > example-a100.yaml" in commands(GUIDE)


def test_the_job_keeps_an_example_manifest_already_there(tmp_path: Path) -> None:
    (tmp_path / "example-a100.yaml").write_text("# edited\n", encoding="utf-8")
    done, calls = run_job(tmp_path)
    assert done.returncode == 0, done.stderr
    assert calls == ["servecast doctor", *steps(GUIDE)]
    assert (tmp_path / "example-a100.yaml").read_text(encoding="utf-8") == "# edited\n"


def test_the_job_takes_a_manifest_and_a_calibration_name_of_your_own(tmp_path: Path) -> None:
    done, calls = run_job(tmp_path, "my.yaml", "my-h100")
    assert done.returncode == 0, done.stderr
    assert not (tmp_path / "my.yaml").exists()  # a manifest of yours is never written by the job
    assert calls[1:4] == ["servecast profile calibration -c my.yaml --output-dir profiling",
                          "servecast calibrate --data profiling --name my-h100 --overwrite",
                          # the review's MF-15: the prediction read the manifest's calibration, not the one just fitted
                          "servecast predict -c my.yaml --with my-h100 --output-dir prediction"]
    assert calls[-2] == "servecast profile serving -c my.yaml --output-dir measurement"


def test_the_job_refuses_a_manifest_without_a_calibration_name(tmp_path: Path) -> None:
    """The review's MF-15: one argument fitted a calibration under the default name, replacing one of yours, and
    predicted with whatever calibration the manifest names."""
    done, calls = run_job(tmp_path, "my.yaml")
    assert done.returncode == 2 and "usage:" in done.stderr and calls == []


def test_the_job_stops_at_the_first_command_that_fails(tmp_path: Path) -> None:
    done, calls = run_job(tmp_path, refuse="profile calibration")
    assert done.returncode == 3
    assert calls == ["servecast doctor", "servecast manifest show example-a100",
                     "servecast profile calibration -c example-a100.yaml --output-dir profiling"]


def test_the_job_asks_slurm_for_one_gpu_and_a_time_limit() -> None:
    directives = [line for line in SCRIPT.read_text(encoding="utf-8").splitlines() if line.startswith("#SBATCH")]
    assert "#SBATCH --gres=gpu:1" in directives
    assert any(line.startswith("#SBATCH --time=") for line in directives)
