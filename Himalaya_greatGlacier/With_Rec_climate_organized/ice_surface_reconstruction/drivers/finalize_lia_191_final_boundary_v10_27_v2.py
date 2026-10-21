"""V2 JSON-safe finalizer for the already generated v10.27 boundary table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
ROUTE_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_27"
V1 = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_final_boundary_audit_v10_27"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_final_boundary_audit_v10_27_v2"


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    route = pd.read_csv(ROUTE_ROOT / "lia_191_current_route.csv")
    audit = pd.read_csv(V1 / "remaining_137_final_boundaries.csv")
    accepted = route["accepted"].astype(str).str.lower().eq("true")
    evidence_exists = audit["evidence_exists"].astype(str).str.lower().eq("true")
    next_nonempty = audit["has_nonempty_next_action"].astype(str).str.lower().eq("true")
    checks = {
        "population_191": bool(len(route) == 191),
        "unique_system_ids_191": bool(route["lia_unique_id"].nunique() == 191),
        "accepted_54": bool(int(accepted.sum()) == 54),
        "remaining_137": bool(len(audit) == 137),
        "all_remaining_closed": bool(audit["controller_state"].eq("closed-current-branch").all()),
        "all_latest_evidence_exists": bool(evidence_exists.all()),
        "all_next_actions_nonempty": bool(next_nonempty.all()),
        "no_hold_rows": bool(not route["controller_state"].astype(str).str.startswith("hold").any()),
    }
    summary = {
        "schema": "lia-191-final-boundary-audit-v10-27-v2",
        "status": "final-boundary-audit-pass" if all(checks.values()) else "final-boundary-audit-fail",
        "population": int(len(route)), "accepted": int(accepted.sum()), "remaining": int(len(audit)),
        "boundary_category_counts": {str(k): int(v) for k, v in audit["boundary_category"].value_counts().items()},
        "checks": checks,
        "missing_evidence_system_ids": audit.loc[~evidence_exists, "system_id"].astype(int).tolist(),
        "empty_next_action_system_ids": audit.loc[~next_nonempty, "system_id"].astype(int).tolist(),
        "v1_status": "table-generated-summary-json-bool-serialization-failed",
        "scientific_contract_changed": False,
        "interpretation": "Every unresolved system is assigned exactly one current boundary category with a concrete existing evidence pointer and explicit reopening condition. This is the final auditable boundary for the frozen current contract, not a claim that future evidence or an explicitly changed contract can never reopen a system.",
        "route": str((ROUTE_ROOT / "summary.json").relative_to(PROJECT)),
    }
    OUTPUT.mkdir(parents=True); audit.to_csv(OUTPUT / "remaining_137_final_boundaries.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
