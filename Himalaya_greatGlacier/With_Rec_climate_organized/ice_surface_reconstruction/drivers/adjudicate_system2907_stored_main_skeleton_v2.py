"""Corrected spatial-only adjudication of the system-2907 stored-main skeleton."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system2907_stored_main_skeleton_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_system2907_stored_main_skeleton_v2"


def main() -> None:
    v1 = json.loads(SOURCE.read_text(encoding="utf-8"))
    coverage_gate = 1.5 * float(v1["paired_cell_diagonal_m"])
    passed = bool(
        float(v1["registered_length_error_fraction"]) <= 0.005
        and bool(v1["head_to_outlet_direction_pass"])
        and int(v1["positive_mask_component_count"]) == 1
        and float(v1["main_node_to_positive_cell_max_distance_m"]) <= coverage_gate
        and int(v1["main_node_count"]) >= 2
        and int(v1["branch_count"]) >= 1
    )
    result = {
        "schema": "lia-191-system2907-stored-main-skeleton-v2",
        "system_id": 2907,
        "status": "stored-main-spatial-skeleton-pass-forced-root-builder-authorized" if passed else "stored-main-spatial-skeleton-fail-mechanism-closed",
        "v1_status_superseded": v1["status"],
        "v1_contract_error": "raw stored-network reference_surface reversals were incorrectly treated as a spatial-skeleton gate; the unchanged synthetic geometry builder owns the formal surface-slope gate",
        "registered_length_error_fraction": v1["registered_length_error_fraction"],
        "registered_length_gate": 0.005,
        "head_to_outlet_direction_pass": v1["head_to_outlet_direction_pass"],
        "positive_mask_component_count": v1["positive_mask_component_count"],
        "main_node_to_positive_cell_max_distance_m": v1["main_node_to_positive_cell_max_distance_m"],
        "spatial_coverage_gate_m": coverage_gate,
        "raw_reference_surface_reverse_count_diagnostic_only": v1["reference_surface_reverse_count"],
        "unchanged_geometry_surface_filter_degree": 1.5,
        "formal_surface_reversal_gate_deferred_to_materialized_geometry": True,
        "forced_root_builder_authorized": passed,
        "geometry_materialized": False, "inversion_run": False, "zero_mb_run": False, "historical_run": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
