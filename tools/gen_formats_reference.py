"""Generate docs/reference/formats.md from the file formats' own models, so the two cannot drift.

    python tools/gen_formats_reference.py            # write the file
    python tools/gen_formats_reference.py --check    # exit 1 if the file is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from servecast.core import formats

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "reference" / "formats.md"

INTRO = """\
# File formats

ServeCast's files are its interface. Each carries `schema: servecast-<kind>/<major>`. A reader refuses a major version
it does not know and ignores fields it does not know inside a known one, so a minor release can add fields without
breaking a reader. Units are in the field names. `servecast manifest schema --kind <kind>` prints the JSON Schema of
any of them, for an editor or a validator.

This page is generated from the formats' definitions (`python tools/gen_formats_reference.py`).
"""

WHO = {
    "manifest": "you write it; every command reads it",
    "pool": "`predict` and `workload sample` write `pool.jsonl`: this header line, then one request per line",
    "profiling-data": "`profile calibration` writes it; `calibrate` and `predict` read it",
    "probe": "`profile calibration` writes `probe.json` (for a per-GPU law, one `probe-<model>.json` per synthetic "
             "model): every engine step it timed, and its audit",
    "gemm": "`profile calibration` writes `gemm.json` for a per-GPU law: the GEMM grid, point by point",
    "calibration": "`calibrate` writes it; `predict` reads it",
    "prediction": "`predict` writes `prediction.json`; `compare` reads it",
    "measurement": "`profile serving` writes `measurement.json`; `compare` reads it",
    "comparison": "`compare` writes `comparison.json`",
    "run": "`predict` and `calibrate` write `run.json` beside their other outputs",
    "gpus": "ServeCast ships `gpus.json`; a `gpus.json` in your data directory adds GPUs",
    "models": "ServeCast ships `models.json`; a `models.json` in your data directory adds LLMs",
    "components": "ServeCast ships `components.json`, the one list of the components it can select",
}


def _type(node: dict[str, Any], defs: dict[str, Any]) -> str:
    if "$ref" in node:
        return node["$ref"].rsplit("/", 1)[-1]
    if "anyOf" in node or "oneOf" in node:  # oneOf: a discriminated union, such as traffic.source
        return " or ".join(_type(n, defs) for n in node.get("anyOf") or node["oneOf"])
    if "enum" in node:
        return " or ".join(f"`{v}`" for v in node["enum"])
    if "const" in node:
        return f"`{node['const']}`"
    kind = node.get("type", "any")
    if kind == "array":
        return f"list of {_type(node.get('items', {}), defs)}"
    if kind == "object" and isinstance(node.get("additionalProperties"), dict):
        return f"map to {_type(node['additionalProperties'], defs)}"
    return str(kind)


def _table(schema: dict[str, Any], defs: dict[str, Any]) -> list[str]:
    required = set(schema.get("required", []))
    rows = ["| field | type | required | default | meaning |", "|---|---|---|---|---|"]
    for name, node in (schema.get("properties") or {}).items():
        default = f"`{node['default']}`" if node.get("default") not in (None, {}, []) else ""
        text = (str(node.get("description", "")).replace("|", "\\|").replace("\n", " ")
                .replace("<", "&lt;").replace(">", "&gt;"))
        rows.append(f"| `{name}` | {_type(node, defs)} | {'yes' if name in required else ''} | {default} | {text} |")
    return rows


def render() -> str:
    parts = [INTRO]
    for kind, model in formats.SCHEMAS.items():
        schema = model.model_json_schema(by_alias=True)
        defs = schema.get("$defs", {})
        who = WHO[kind]
        parts += [f"## {kind}", "", f"{who[0].upper()}{who[1:]}.", ""]
        parts += _table(schema, defs)
        for name, sub in sorted(defs.items()):
            if sub.get("properties"):
                parts += ["", f"### {kind}: {name}", ""]
                parts += _table(sub, defs)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the file-format reference from the formats' models.")
    parser.add_argument("--check", action="store_true", help="exit 1 if the file on disk is out of date")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        if not TARGET.is_file() or TARGET.read_text(encoding="utf-8") != text:
            print(f"{TARGET.relative_to(ROOT)} is out of date: run python tools/gen_formats_reference.py",
                  file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
