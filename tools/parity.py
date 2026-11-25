"""Parity of the model with the reference data: recompute every predicted load point of
tests/goldens/ with this package and compare, bit for bit.

    python tools/parity.py paper  [--jobs 8] [--limit N] [--match TEXT] [--out FILE]
    python tools/parity.py newest [--jobs 8] [--limit N] [--match TEXT] [--out FILE]

`paper` recomputes the arXiv paper's predictions (cost law d9cbf1d3, each GPU's six calibrations on each campaign: the
diagonal and the transfers); `newest` those of the per-GPU law c62ebe75 on the A100-SXM4-40GB. Exit status 0 only when
every value compared is identical. The comparison is exact on the reference environment (Python 3.12, numpy 2.5.0,
scipy 1.18.0).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = ROOT / "tests" / "goldens"
PAPER_COMPONENTS = {"cost_law": "d9cbf1d3", "population_model": "f2f5d726", "workload_fitter": "cd88bc61",
                    "estimator": "cabeb968"}
NEWEST_COMPONENTS = {"cost_law": "c62ebe75", "population_model": "f2f5d726", "workload_fitter": "cd88bc61",
                     "estimator": "c3ad4006"}
PAPER_FIELDS = ("throughput_rps", "mean_ttft_s", "mean_tpot_s", "converged")
NEWEST_FIELDS = ("throughput_rps", "mean_ttft_s", "mean_tpot_s", "p95_ttft_s", "p95_tpot_s", "ttft_samples",
                 "tpot_samples", "converged")


def slug(hf: str) -> str:
    return hf.split("/")[-1].lower().removeprefix("meta-")


def tasks(pipeline: str) -> list[dict[str, Any]]:
    path = (ROOT / "src" / "servecast" / "data" / "paper" / "campaigns.json" if pipeline == "paper"
            else GOLDENS / "newest-a100.json")
    doc = json.loads(path.read_text("utf-8"))
    out = []
    for camp in doc["campaigns"]:
        gpu = camp.get("gpu", doc.get("gpu"))
        for cell in camp["cells"]:
            if pipeline == "paper":
                for cal_llm, want in (cell.get("predicted") or {}).items():
                    out.append({"gpu": gpu, "llm": camp["llm"], "trace": camp["trace"], "settings": camp["prediction"],
                                "n": cell["concurrency"], "z": cell["think_time_s"],
                                "calibration": f"{gpu}/d9cbf1d3/{slug(cal_llm)}", "want": want,
                                "components": PAPER_COMPONENTS, "fields": PAPER_FIELDS})
            elif cell.get("predicted"):
                out.append({"gpu": gpu, "llm": camp["llm"], "trace": camp["trace"], "settings": camp["prediction"],
                            "n": cell["concurrency"], "z": cell["think_time_s"],
                            "calibration": f"{gpu}/c62ebe75", "want": cell["predicted"],
                            "components": NEWEST_COMPONENTS, "fields": NEWEST_FIELDS})
    return out


def run(task: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "src"))
    from servecast.model import geometry
    from servecast.model import predict as P

    s = task["settings"]
    t0 = time.time()
    row = P.predict_point(P.load_calibration(task["calibration"]), geometry.resolve(task["llm"]),
                          P.load_workload_law(task["trace"]),
                          P.Engine(s["kv_blocks"], s["block_size"], s["max_batched_tokens"], s["max_sequences"]),
                          task["n"], task["z"], components=task["components"], targets=s["targets"], seed=s["seed"],
                          checkpoint_ratio=s["checkpoint_ratio"], max_completions=s["max_completions"])
    got = {"throughput_rps": row["throughput_rps"], "mean_ttft_s": row["ttft_mean"], "mean_tpot_s": row["tpot_mean"],
           "p95_ttft_s": row["p95_ttft_s"], "p95_tpot_s": row["p95_tpot_s"], "ttft_samples": len(row["ttft"]),
           "tpot_samples": len(row["tpot"]), "converged": row["converged"]}
    diffs = {}
    for f in task["fields"]:
        a, b = got[f], task["want"].get(f)
        equal = (a == b) or (isinstance(a, float) and b is None and math.isnan(a))
        if not equal:
            diffs[f] = {"got": a, "want": b}
    return {"key": f"{task['gpu']} {task['llm']} {task['trace']} N{task['n']} [{task['calibration']}]",
            "diffs": diffs, "wall_s": time.time() - t0}


#: every predicted load point of the reference data: a run that is not filtered must compare all of them
EXPECTED = {"paper": 1944, "newest": 108}


def provenance() -> dict[str, Any]:
    """What the run is evidence for: the commit, whether the tree was clean, the environment."""
    import platform
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()

    import numpy
    import scipy
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}", "numpy": numpy.__version__,
            "scipy": scipy.__version__}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute the reference predictions and compare them bit for bit.")
    ap.add_argument("pipeline", choices=("paper", "newest"))
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run on uncommitted changes; the result is then not evidence for any commit")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="the first N load points only")
    ap.add_argument("--match", action="append", default=None,
                    help="only load points whose key contains this text (repeatable: any of them)")
    ap.add_argument("--out", type=Path, default=None, help="write every result here, as JSON")
    a = ap.parse_args(argv)
    prov = provenance()
    if prov["dirty"] and not a.allow_dirty:
        print("the working tree has uncommitted changes: commit them, or run from a clean checkout, or pass "
              "--allow-dirty for a result that is evidence for no commit", file=sys.stderr)
        return 2
    todo = tasks(a.pipeline)
    if a.match is None and a.limit is None and len(todo) != EXPECTED[a.pipeline]:
        print(f"{len(todo)} load points in the reference data, {EXPECTED[a.pipeline]} expected", file=sys.stderr)
        return 2
    for t in todo:
        t["key"] = f"{t['gpu']} {t['llm']} {t['trace']} N{t['n']} [{t['calibration']}]"
    if a.match:
        todo = [t for t in todo if any(m in t["key"] for m in a.match)]
    if a.limit is not None:
        todo = todo[: a.limit]
    results: list[dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futures = [ex.submit(run, t) for t in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            mark = "identical" if not r["diffs"] else "DIFFERS " + ", ".join(sorted(r["diffs"]))
            print(f"[{i:>5}/{len(todo)}] {r['wall_s']:7.1f} s  {mark:<20} {r['key']}", flush=True)
    bad = [r for r in results if r["diffs"]]
    print(f"\n{len(results)} load points, {len(results) - len(bad)} identical, {len(bad)} differ "
          f"({time.time() - t0:.0f} s)")
    print(f"evidence for commit {prov['commit']}{' (with uncommitted changes)' if prov['dirty'] else ''} on "
          f"{prov['platform']}, Python {prov['python']}, numpy {prov['numpy']}, scipy {prov['scipy']}")
    if a.out:
        a.out.write_text(json.dumps({"pipeline": a.pipeline, "provenance": prov, "filtered": bool(a.match or a.limit),
                                     "results": results}, indent=1) + "\n", encoding="utf-8")
    return 0 if not bad and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
