"""Consolidate the completed continuous class loop into one review package."""

from __future__ import annotations

import json

import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
RESULTS = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
DIAG = RESULTS / "lia_191_controller_v1" / "diagnostics"
OUTPUT = DIAG / "continuous_class_loop_outcome_v1"


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    inventory = _read_json(
        DIAG
        / "dual_gate_inventory_after_no_formal_N3_v1"
        / "formal_inventory_summary.json"
    )
    round2 = _read_json(DIAG / "class_batch_round2_outcome_v1" / "class_round2_summary.json")
    n2 = _read_json(DIAG / "no_formal_N2_glen_outcome_v1" / "summary.json")
    n1 = _read_json(DIAG / "no_formal_N1_glen_outcome_v1" / "summary.json")
    n3 = _read_json(DIAG / "no_formal_N3_outcome_v1" / "summary.json")
    n4 = _read_json(DIAG / "no_formal_N4_branch_diagnosis_v1" / "summary.json")
    total_router = _read_json(
        DIAG
        / "total_length_only_batch_router_after_no_formal_v1"
        / "batch_router_summary.json"
    )
    frozen = _read_json(
        RESULTS
        / "lia_191_frozen_curve_registry_v1"
        / "registry_validation_v1"
        / "summary.json"
    )
    rows = [
        {
            "block": "class_round2",
            "new_oggm_chains": 14,
            "technical_failures": int(round2["technical_failure_count"]),
            "formal_passes": int(round2["formal_pass_count"]),
            "decision": "closed",
        },
        {
            "block": "N2_glen_representative_plus_promotion",
            "new_oggm_chains": 33,
            "technical_failures": int(n2["promotion_technical_failure_count"]),
            "formal_passes": int(n2["promotion_formal_pass_count"]),
            "decision": n2["decision"],
        },
        {
            "block": "N1_glen_interpolation",
            "new_oggm_chains": 12,
            "technical_failures": int(n1["technical_failure_count"]),
            "formal_passes": int(n1["formal_pass_count"]),
            "decision": n1["decision"],
        },
        {
            "block": "N3_melt_delta_representatives",
            "new_oggm_chains": 8,
            "technical_failures": int(n3["technical_failure_count"]),
            "formal_passes": int(n3["formal_pass_count"]),
            "decision": n3["decision"],
        },
    ]
    blocks = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(OUTPUT / "executed_block_summary.csv", index=False)
    summary = {
        "schema_version": 1,
        "executed_new_oggm_chain_count": int(blocks["new_oggm_chains"].sum()),
        "technical_failure_count": int(blocks["technical_failures"].sum()),
        "new_formal_pass_count": int(blocks["formal_passes"].sum()),
        "final_inventory_chain_count": int(inventory["score_row_count"]),
        "formal_system_count": int(inventory["systems_with_formal_dual_pass"]),
        "formal_systems": inventory["formal_dual_pass_systems"],
        "route_counts": inventory["route_counts"],
        "frozen_registry_validation": frozen,
        "N4_read_only_class_counts": n4["class_counts"],
        "updated_total_length_only_router": {
            "system_count": total_router["total_length_only_system_count"],
            "class_counts": total_router["class_counts"],
            "nominal_individual_eligible_count": total_router[
                "eligible_class_pilot_count"
            ],
            "class_level_decision": (
                "No release: C1/C3/C4 proposed families were already closed "
                "at class level in rounds 1-2; do not reopen by system."
            ),
        },
        "continuous_loop_state": "stopped_at_declared_parameter_family_boundary",
        "next_required_decision": (
            "Declare a new scientific family outside the closed C1-C5 and "
            "N1-N4 mechanisms, or move to consolidated curve review."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Continuous class-loop outcome V1",
        "",
        f"- New OGGM chains: {summary['executed_new_oggm_chain_count']}",
        f"- Technical failures: {summary['technical_failure_count']}",
        f"- New formal passes: {summary['new_formal_pass_count']}",
        f"- Final inventory: {summary['final_inventory_chain_count']} chains",
        f"- Frozen formal systems: {summary['formal_system_count']}",
        f"- Registry hashes pass: {frozen['all_pass']}",
        "",
        "The current declared class families are exhausted. No further OGGM "
        "batch is released automatically from the nominal per-system eligible "
        "flags because their C1/C3/C4 mechanisms were already closed at class level.",
    ]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
