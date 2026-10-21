"""Merge second-round class pilots into the formal dual-gate inventory."""

from __future__ import annotations

import json

from . import build_lia_191_dual_gate_inventory_after_class_batches_v1 as _v1
from . import build_lia_191_iterative_loop as _base


INPUT_INVENTORY = "dual_gate_inventory_after_class_batches_v1"
OUTPUT_NAME = "dual_gate_inventory_after_class_round2_v1"
EXPERIMENTS = [
    (
        "lia_191_class_C1_melt_delta_spatial_pilot_v1",
        "class_C1_melt_delta_spatial_pilot_v1",
        "class_round2_preflight_stopped",
        220,
        "spatial-downstream-v2",
        "melt",
    ),
    (
        "lia_191_class_C1_melt_delta_spatial_pilot_v2",
        "class_C1_melt_delta_spatial_pilot_v2",
        "class_round2_representative",
        221,
        "spatial-downstream-v2",
        "melt",
    ),
    (
        "lia_191_class_C2_temp_melt_pilot_v1",
        "class_C2_temp_melt_pilot_v1",
        "class_round2_representative",
        222,
        "core191-length-corrected",
        "temp",
    ),
    (
        "lia_191_class_C4_temp_melt_spatial_pilot_v1",
        "class_C4_temp_melt_spatial_pilot_v1",
        "class_round2_preflight_stopped",
        223,
        "spatial-downstream-v2",
        "temp",
    ),
]


def main() -> None:
    original = (_v1.INPUT_INVENTORY, _v1.OUTPUT_NAME, _v1.EXPERIMENTS)
    try:
        _v1.INPUT_INVENTORY = INPUT_INVENTORY
        _v1.OUTPUT_NAME = OUTPUT_NAME
        _v1.EXPERIMENTS = EXPERIMENTS
        _v1.main()
    finally:
        _v1.INPUT_INVENTORY, _v1.OUTPUT_NAME, _v1.EXPERIMENTS = original

    summary_path = (
        _base.controller_root()
        / "diagnostics"
        / OUTPUT_NAME
        / "formal_inventory_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["round2_experiments"] = [item[0] for item in EXPERIMENTS]
    summary["round2_planned_or_completed_chain_count"] = 14
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
