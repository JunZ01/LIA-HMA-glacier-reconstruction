"""Write the reproducible Cycle-1/early-Cycle-2 controller outcome."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
CTRL = BASE / "lia_191_continuous_class_controller_v1"
OUT = CTRL / "cycle1_outcome_v1"


def _truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def main() -> None:
    registry = pd.read_csv(BASE / "lia_191_frozen_curve_registry_v1/frozen_candidates.csv")
    validation = json.loads(
        (BASE / "lia_191_frozen_curve_registry_v1/registry_validation_v2_dynamic/summary.json").read_text(encoding="utf-8")
    )
    q2 = pd.read_csv(CTRL / "cycle1/Q2_25m_promotion35_v1/q2_25m_results.csv")
    q3_short = pd.read_csv(CTRL / "cycle1/Q3_rgi_short_25m_historical_v1/q3_rgi_short_25m_results.csv")
    q3_long = pd.read_csv(CTRL / "cycle2/Q3_rgi_long_25m_historical_v1/q3_rgi_long_25m_results.csv")
    q1 = json.loads((CTRL / "cycle1/Q1_monotonic_regularized/zero_mb_3850_v1/zero_mb_summary.json").read_text(encoding="utf-8"))
    q4 = json.loads((CTRL / "q4_preflight_audit_v1/summary.json").read_text(encoding="utf-8"))

    frozen_ids = sorted(registry["lia_unique_id"].astype(int).tolist())
    formal_q2 = sorted(q2.loc[_truth(q2["formal_dual_pass"]), "lia_unique_id"].astype(int).tolist())
    remaining = pd.DataFrame(
        [
            {"queue": "Q1_neither_endpoint", "system_count": 108, "cycle_status": "q1_mechanism_closed"},
            {"queue": "Q2_rgi_only", "system_count": 34, "cycle_status": "23_standard_closed;11_spatial_contract_pending"},
            {"queue": "Q3_lia_only", "system_count": 19, "cycle_status": "standard_25m_both_signed_directions_closed;spatial_contract_review_pending"},
            {"queue": "Q4_no_stable_valid_candidate", "system_count": 7, "cycle_status": "no_release;separate_geometry_state_routes"},
            {"queue": "Q5_both_endpoints_separately", "system_count": 1, "cycle_status": "hold_2697"},
        ]
    )
    if int(remaining["system_count"].sum()) != 169:
        raise RuntimeError("remaining-queue-count-not-169")
    OUT.mkdir(parents=True, exist_ok=True)
    remaining.to_csv(OUT / "remaining_queue_counts.csv", index=False)
    summary = {
        "schema_version": 1,
        "status": "cycle1-complete-cycle2-spatial-preflight-active",
        "total_system_count": 191,
        "frozen_system_count": len(frozen_ids),
        "frozen_systems": frozen_ids,
        "unresolved_system_count": 191 - len(frozen_ids),
        "newly_frozen_systems": formal_q2,
        "registry_validation_all_pass": bool(validation["all_pass"]),
        "registry_recorded_file_count": int(validation["recorded_file_count"]),
        "q1": {
            "decision": "closed-at-zero-mb",
            "representative": 3850,
            "drift_0p1y_fraction": 0.006854,
            "drift_5y_fraction": 0.122645,
            "solver_change": bool(q1["solver_change"]),
        },
        "q2": {
            "geometry_preflight_pass": 35,
            "standard_parent_count": 24,
            "spatial_downstream_contract_count": 11,
            "standard_base_gate_pass": int((q2["forward_status"].astype(str) == "ok").sum()),
            "standard_base_invalid": 4,
            "formal_pass_count": len(formal_q2),
            "formal_pass_ids": formal_q2,
        },
        "q3": {
            "rgi_short_representatives": len(q3_short),
            "rgi_short_formal_pass": int(_truth(q3_short["formal_dual_pass"]).sum()),
            "rgi_long_representatives": len(q3_long),
            "rgi_long_formal_pass": int(_truth(q3_long["formal_dual_pass"]).sum()),
            "decision": "standard-25m-closed-both-directions",
        },
        "q4": {
            "released_count": 0,
            "source_summary": str(CTRL / "q4_preflight_audit_v1/summary.json"),
            "summary_status": q4.get("status"),
        },
        "next_active_route": "25m-spatial-downstream-contract-preserving-static-and-zero-mb-preflight",
        "launched_oggm": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
