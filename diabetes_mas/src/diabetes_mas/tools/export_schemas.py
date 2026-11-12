"""Xuất JSON schema của mọi tool → docs/assets/tool_schemas.json (dùng bind cho LLM và tra cứu).

Chạy:  python -m diabetes_mas.tools.export_schemas
"""
from __future__ import annotations

import json
import sys

from diabetes_mas.config import PROJECT_ROOT
from diabetes_mas.tools import TOOLS

OUT = PROJECT_ROOT.parent / "docs" / "assets" / "tool_schemas.json"


def build() -> dict:
    return {tid: {"name": spec.name, "description": (spec.fn.__doc__ or "").strip(),
                  "timeout_s": spec.timeout_s,
                  "input": spec.input_model.model_json_schema(),
                  "output": spec.output_model.model_json_schema()}
            for tid, spec in sorted(TOOLS.items(), key=lambda kv: int(kv[0][2:]))}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[export] {len(TOOLS)} tool → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
