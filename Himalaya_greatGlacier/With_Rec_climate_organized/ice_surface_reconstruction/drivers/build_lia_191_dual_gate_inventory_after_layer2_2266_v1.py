"""Extend the formal inventory with both completed system-2266 Layer-2 blocks."""

from __future__ import annotations

import json

from . import build_lia_191_dual_gate_inventory_after_batch2_v1 as _builder
from . import build_lia_191_iterative_loop as _base


def _run_one(input_inventory: str, experiment: str, output_name: str, label: str, priority: int) -> None:
    original_rows = _builder._batch2_rows

    def rows():
        frame = original_rows()
        frame["score_source"] = label
        frame["score_layer"] = "layer2_melt_delta"
        frame["score_priority"] = priority
        return frame

    _builder.INPUT_INVENTORY = input_inventory
    _builder.EXPERIMENT = experiment
    _builder.OUTPUT_NAME = output_name
    _builder._batch2_rows = rows
    try:
        _builder.main()
    finally:
        _builder._batch2_rows = original_rows


def main() -> None:
    original = (
        _builder.INPUT_INVENTORY,
        _builder.EXPERIMENT,
        _builder.OUTPUT_NAME,
        _builder._batch2_rows,
    )
    try:
        _run_one(
            "dual_gate_inventory_after_batch3_v1",
            "lia_191_layer2_melt_delta_2266_v1",
            "dual_gate_inventory_after_layer2_2266_stage1_v1",
            "layer2_melt_delta_2266_v1",
            180,
        )
        _run_one(
            "dual_gate_inventory_after_layer2_2266_stage1_v1",
            "lia_191_layer2_melt_delta_2266_refine_v1",
            "dual_gate_inventory_after_layer2_2266_v1",
            "layer2_melt_delta_2266_refine_v1",
            181,
        )
    finally:
        (
            _builder.INPUT_INVENTORY,
            _builder.EXPERIMENT,
            _builder.OUTPUT_NAME,
            _builder._batch2_rows,
        ) = original

    final_path = (
        _base.controller_root()
        / "diagnostics"
        / "dual_gate_inventory_after_layer2_2266_v1"
        / "formal_inventory_summary.json"
    )
    summary = json.loads(final_path.read_text(encoding="utf-8"))
    summary["new_layer2_2266_initial_row_count"] = 6
    summary["new_layer2_2266_refine_row_count"] = int(summary.pop("new_batch2_row_count"))
    final_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
