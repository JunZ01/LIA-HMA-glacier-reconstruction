"""Final all-191 evidence and boundary audit for route v10.29.

The audit deliberately distinguishes accepted reconstructions from scientific
boundaries.  It performs no OGGM computation and changes no scientific gate.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE = RESULTS / "lia_191_current_route_v10_29"
ACCEPTED = RESULTS / "lia_191_accepted54_evidence_index_v3"
BOUNDARY = RESULTS / "lia_191_final_boundary_audit_v10_28"
OUT = RESULTS / "lia_191_full_completion_audit_v10_29"


def truth(v):
    return str(v).strip().lower() in {"true", "1", "yes"}


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def evidence_exists(value):
    p = Path(value)
    return p.exists() if p.is_absolute() else (ROOT / p).exists()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROUTE / "lia_191_current_route.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with (ACCEPTED / "accepted54_evidence_index.csv").open("r", encoding="utf-8-sig", newline="") as f:
        accepted_index = list(csv.DictReader(f))
    route_summary = read_json(ROUTE / "summary.json")
    accepted_summary = read_json(ACCEPTED / "summary.json")
    boundary_summary = read_json(BOUNDARY / "summary.json")

    accepted = [x for x in rows if truth(x["accepted"])]
    remaining = [x for x in rows if not truth(x["accepted"])]
    accepted_route_ids = {int(x["lia_unique_id"]) for x in accepted}
    accepted_index_ids = {int(x["system_id"]) for x in accepted_index}
    class_counts = Counter(x["mechanism_class"] for x in remaining)

    checks = {
        "route_summary_pass": route_summary.get("status") == "full-population-evidence-complete-current-contract-route",
        "population_191": len(rows) == 191,
        "unique_systems_191": len({int(x["lia_unique_id"]) for x in rows}) == 191,
        "accepted_54": len(accepted) == 54,
        "remaining_137": len(remaining) == 137,
        "accepted_index_status_pass": accepted_summary.get("status") == "accepted54-evidence-index-pass",
        "accepted_index_membership_exact": accepted_route_ids == accepted_index_ids,
        "all_accepted_index_rows_pass": all(truth(x["accepted_evidence_pass"]) for x in accepted_index),
        "all_accepted_evidence_exists": all(evidence_exists(x["latest_evidence"]) for x in accepted),
        "prior_remaining_boundary_audit_pass": boundary_summary.get("status") == "final-boundary-audit-pass",
        "all_remaining_closed_current_branch": all(x["controller_state"] == "closed-current-branch" for x in remaining),
        "all_remaining_evidence_exists": all(evidence_exists(x["latest_evidence"]) for x in remaining),
        "all_remaining_actions_explicit": all(bool(x["next_legal_action"].strip()) for x in remaining),
        "all_remaining_classes_explicit": all(bool(x["mechanism_class"].strip()) for x in remaining),
        "all_route_process_counts_zero": all(str(x["active_process_count"]) in {"", "0"} for x in rows),
        "accepted_and_remaining_disjoint": not bool(accepted_route_ids & {int(x["lia_unique_id"]) for x in remaining}),
    }
    status = "full191-current-contract-completion-audit-pass" if all(checks.values()) else "full191-audit-fail"

    summary = {
        "status": status,
        "read_only_audit": True,
        "scientific_contract_changed": False,
        "population": 191,
        "accepted_reconstructions": 54,
        "evidence_complete_current_contract_boundaries": 137,
        "successful_reconstructions_are_not_191": True,
        "checks": checks,
        "remaining_mechanism_class_counts": dict(sorted(class_counts.items())),
        "authoritative_route": str((ROUTE / "lia_191_current_route.csv").resolve()),
        "accepted_evidence_index": str((ACCEPTED / "accepted54_evidence_index.csv").resolve()),
        "remaining_boundary_audit": str((BOUNDARY / "summary.json").resolve()),
        "completion_interpretation": (
            "The registered 191-system program is fully auditable under the frozen current contract: "
            "54 systems are accepted, and each of the other 137 has an evidence pointer, an explicit "
            "closed mechanism class, and a stated reopening condition.  This is the scientific-boundary "
            "completion alternative, not a claim that all 191 reconstructions succeeded."
        ),
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (OUT / "remaining_mechanism_class_counts.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mechanism_class", "count"])
        w.writeheader()
        for key, count in sorted(class_counts.items()):
            w.writerow({"mechanism_class": key, "count": count})
    print(json.dumps(summary, indent=2))
    if status != "full191-current-contract-completion-audit-pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
