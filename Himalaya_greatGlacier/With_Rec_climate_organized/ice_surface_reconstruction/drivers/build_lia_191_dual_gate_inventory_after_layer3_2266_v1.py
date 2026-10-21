"""Extend the formal inventory with the completed system-2266 Layer-3 block."""

from __future__ import annotations

import json

from . import build_lia_191_dual_gate_inventory_after_batch2_v1 as _builder
from . import build_lia_191_iterative_loop as _base


def main() -> None:
    original = (
        _builder.INPUT_INVENTORY,
        _builder.EXPERIMENT,
        _builder.OUTPUT_NAME,
        _builder._batch2_rows,
    )
    original_rows = _builder._batch2_rows

    def rows():
        frame = original_rows()
        frame["score_source"] = "layer3_glen_2266_v1"
        frame["score_layer"] = "layer3_single_parameter"
        frame["score_priority"] = 190
        return frame

    try:
        _builder.INPUT_INVENTORY = "dual_gate_inventory_after_layer2_2266_v1"
        _builder.EXPERIMENT = "lia_191_layer3_glen_2266_v1"
        _builder.OUTPUT_NAME = "dual_gate_inventory_after_layer3_2266_v1"
        _builder._batch2_rows = rows
        _builder.main()
    finally:
        (
            _builder.INPUT_INVENTORY,
            _builder.EXPERIMENT,
            _builder.OUTPUT_NAME,
            _builder._batch2_rows,
        ) = original

    summary_path = (
        _base.controller_root()
        / "diagnostics"
        / "dual_gate_inventory_after_layer3_2266_v1"
        / "formal_inventory_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["new_layer3_glen_2266_row_count"] = int(summary.pop("new_batch2_row_count"))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
