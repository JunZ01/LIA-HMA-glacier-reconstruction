"""Extend the formal inventory with batch 3 and the system-2266 side point."""

from __future__ import annotations

import json

from . import build_lia_191_dual_gate_inventory_after_batch2_v1 as _builder
from . import build_lia_191_iterative_loop as _base


def _run_one(input_inventory: str, experiment: str, output_name: str, label: str, priority: int) -> None:
    original_rows = _builder._batch2_rows

    def rows():
        frame = original_rows()
        frame["score_source"] = label
        frame["score_layer"] = "formal_delta_edge_screening"
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
            "dual_gate_inventory_after_batch2_v1",
            "lia_191_formal_delta_bisection_batch3_v1",
            "dual_gate_inventory_after_batch3_midpoints_v1",
            "formal_delta_bisection_batch3_v1",
            170,
        )
        _run_one(
            "dual_gate_inventory_after_batch3_midpoints_v1",
            "lia_191_formal_delta_bisection_batch3b_2266_v1",
            "dual_gate_inventory_after_batch3_v1",
            "formal_delta_bisection_batch3b_2266_v1",
            171,
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
        / "dual_gate_inventory_after_batch3_v1"
        / "formal_inventory_summary.json"
    )
    summary = json.loads(final_path.read_text(encoding="utf-8"))
    summary["new_batch3_midpoint_row_count"] = 3
    summary["new_batch3b_side_row_count"] = int(summary.pop("new_batch2_row_count"))
    final_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
