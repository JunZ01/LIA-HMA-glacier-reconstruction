"""Merge the bounded N1 glen interpolation into the formal inventory."""

from __future__ import annotations

import json

from . import build_lia_191_dual_gate_inventory_after_class_batches_v1 as _v1
from . import build_lia_191_iterative_loop as _base


INPUT_INVENTORY = "dual_gate_inventory_after_no_formal_N2_v1"
OUTPUT_NAME = "dual_gate_inventory_after_no_formal_N1_v1"
EXPERIMENTS = [
    (
        "lia_191_no_formal_N1_glen_interpolation_v1",
        "no_formal_N1_glen_interpolation_v1",
        "no_formal_N1_interpolation",
        232,
        "spatial-downstream-v2",
        "glen",
    )
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
    path = (
        _base.controller_root()
        / "diagnostics"
        / OUTPUT_NAME
        / "formal_inventory_summary.json"
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["N1_experiment"] = EXPERIMENTS[0][0]
    summary["N1_completed_chain_count"] = 12
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
