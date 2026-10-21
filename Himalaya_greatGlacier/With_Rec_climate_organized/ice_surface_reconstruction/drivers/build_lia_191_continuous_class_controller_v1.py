"""Materialize persistent class queues for continuous unresolved-system routing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_controller_v1" / "diagnostics" / "unresolved_reachability_audit_v1" / "unresolved_reachability.csv"
OUTPUT = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_continuous_class_controller_v1"
ROUTES = {
    "neither_endpoint": "Q1_neither_endpoint",
    "rgi_only": "Q2_rgi_only",
    "lia_only": "Q3_lia_only",
    "no_stable_valid_candidate": "Q4_no_stable_valid_candidate",
    "both_endpoints_separately": "Q5_both_endpoints_separately",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE)
    if len(frame) != 170 or frame["lia_unique_id"].nunique() != 170:
        raise RuntimeError("continuous controller requires exactly 170 unresolved systems")

    queue_counts: dict[str, int] = {}
    for route, queue in ROUTES.items():
        subset = frame[frame["reachability_route"].eq(route)].copy()
        subset.insert(1, "queue", queue)
        subset["controller_state"] = "awaiting_declared_structural_mechanism"
        subset.to_csv(OUTPUT / f"{queue}.csv", index=False)
        queue_counts[queue] = int(len(subset))

    contract = {
        "schema_version": 1,
        "created_at": "2026-08-16",
        "facts_contract": "LIA_PROJECT_FACTS.md",
        "plan": "LIA_191_CONTINUOUS_EXECUTION_PLAN_V1.md",
        "frozen_system_count": 21,
        "unresolved_system_count": 170,
        "queue_counts": queue_counts,
        "formal_acceptance": {
            "technical_clean": True,
            "spinup_stability_max_fraction": 0.02,
            "observation_domain_valid": True,
            "lia_total_length_error_max": 0.10,
            "rgi_total_length_error_max": 0.10,
            "lia_separation_error_max": 0.20,
            "rgi_separation_error_max": 0.20,
        },
        "execution": {
            "maximum_total_oggm_workers": 12,
            "automatic_freeze": True,
            "per_glacier_user_approval": False,
            "pause_boundaries": [
                "climate change",
                "LIA boundary change",
                "formal score change",
                "new model family",
            ],
        },
        "class_rules": {
            "representative_count": "3-5",
            "preflight": "one complete technical/stable/domain-valid chain",
            "promotion": "at least two valid representative objective improvements or one formal pass",
            "stop": "invalid fraction above 0.30, zero representative improvement, or reproduction of a closed response",
        },
    }
    (OUTPUT / "controller_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    state = {
        "schema_version": 1,
        "phase": "structural_mechanism_design",
        "active_queues": list(queue_counts),
        "queue_counts": queue_counts,
        "launched_oggm": False,
        "next_transition": "freeze one declared structural mechanism per eligible queue",
    }
    (OUTPUT / "controller_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
