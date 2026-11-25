"""Pin the length sampler's draws. The sampler has no reference implementation to match: this golden
records what the released 2026-11 module draws for fixed seeds, so any later change to those draws is caught.

    python tools/sampler_golden.py   # writes tests/components/goldens/sampler.json; only when a NEW id is released
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from servecast.core.sampler import v2026_11 as S

ROOT = Path(__file__).resolve().parents[1]
SPECS = [(250.0, None), (250.0, 250000.0), (250.0, 2500.0), (1200.0, None), (3.0, 1.0)]


def build() -> dict:
    draws = []
    for i, (m, v) in enumerate(SPECS):
        spec = S.match(m, v)
        draws.append({"mean": m, "variance": v, "law": spec.as_dict(),
                      "first": S.draw(spec, np.random.default_rng(100 + i), 200).tolist()})
    pool = S.pool(S.match(1200.0), S.match(250.0, 40000.0), size=500, seed=7, context_length=2048)
    return {"draws": draws, "pool": {"input_tokens": pool["input_tokens"], "output_tokens": pool["output_tokens"]}}


if __name__ == "__main__":
    out = ROOT / "tests" / "components" / "goldens" / "sampler.json"
    out.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")
