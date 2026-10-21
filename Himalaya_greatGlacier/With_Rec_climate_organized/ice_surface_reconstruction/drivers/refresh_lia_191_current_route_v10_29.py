"""Bind accepted54 evidence into the evidence-complete v10.28 route.

This is a read-only route refresh.  It copies all remaining137 scientific
decisions unchanged and adds authoritative per-system evidence pointers to all
54 accepted rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = RESULTS / "lia_191_current_route_v10_28"
ACCEPTED = RESULTS / "lia_191_accepted54_evidence_index_v3"
OUT = RESULTS / "lia_191_current_route_v10_29"


def truth(v):
    return str(v).strip().lower() in {"true", "1", "yes"}


def evidence_exists(value):
    p = Path(value)
    return p.exists() if p.is_absolute() else (ROOT / p).exists()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (SOURCE / "lia_191_current_route.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with (ACCEPTED / "accepted54_evidence_index.csv").open("r", encoding="utf-8-sig", newline="") as f:
        accepted_rows = list(csv.DictReader(f))
    evidence = {int(x["system_id"]): x for x in accepted_rows}
    if len(evidence) != 54 or not all(truth(x["accepted_evidence_pass"]) for x in accepted_rows):
        raise RuntimeError("accepted54 evidence index is incomplete")

    for row in rows:
        sid = int(row["lia_unique_id"])
        if truth(row["accepted"]):
            ev = evidence.get(sid)
            if ev is None:
                raise RuntimeError(f"accepted system {sid} missing evidence index row")
            row["latest_evidence_stage"] = "accepted-evidence-index-v3"
            row["latest_evidence_status"] = "accepted-evidence-pass"
            row["latest_flags"] = ""
            row["latest_evidence"] = ev["primary_evidence"]
            row["controller_state"] = "accepted"
            row["mechanism_class"] = f"accepted:{ev['route']}"
            row["next_legal_action"] = "frozen accepted result; read-only unless the scientific contract changes"
            row["update_source"] = str((ACCEPTED / "summary.json").resolve())
            row["active_process_count"] = "0"
            row["active_process_ids"] = ""
            row["mutually_exclusive_class"] = "accepted"

    ids = [int(x["lia_unique_id"]) for x in rows]
    accepted_count = sum(truth(x["accepted"]) for x in rows)
    remaining = [x for x in rows if not truth(x["accepted"])]
    checks = {
        "population_191": len(rows) == 191,
        "unique_191": len(set(ids)) == 191,
        "accepted_54": accepted_count == 54,
        "remaining_137": len(remaining) == 137,
        "accepted_evidence_index_54": len(evidence) == 54,
        "all_accepted_evidence_paths_exist": all(
            evidence_exists(x["latest_evidence"]) for x in rows if truth(x["accepted"])
        ),
        "all_remaining_evidence_paths_exist": all(evidence_exists(x["latest_evidence"]) for x in remaining),
        "all_remaining_closed": all(x["controller_state"] == "closed-current-branch" for x in remaining),
        "no_active_route_rows": all(str(x["active_process_count"]) in {"", "0"} for x in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v10.29 route checks failed: {checks}")

    fields = list(rows[0].keys())
    for name, selected in (
        ("lia_191_current_route.csv", rows),
        ("remaining_137_current_route.csv", remaining),
        ("accepted_54_current_route.csv", [x for x in rows if truth(x["accepted"])]),
    ):
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(selected)

    summary = {
        "schema": "lia-191-current-route-v10-29",
        "status": "full-population-evidence-complete-current-contract-route",
        "population": len(rows),
        "accepted": accepted_count,
        "remaining": len(remaining),
        "closed_current_branch": sum(x["controller_state"] == "closed-current-branch" for x in remaining),
        "active": 0,
        "queue": 0,
        "checks": checks,
        "supersedes": "lia_191_current_route_v10_28",
        "accepted_evidence_index": str((ACCEPTED / "summary.json").resolve()),
        "scientific_contract_changed": False,
        "interpretation": (
            "All 191 systems now have an existing evidence pointer.  This is 54 accepted "
            "reconstructions plus 137 evidence-complete current-contract boundaries, not 191 successes."
        ),
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
