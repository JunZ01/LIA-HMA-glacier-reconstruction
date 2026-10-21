"""Register policy-V4 disconnection rescore and the remaining length gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = RESULTS / "lia_191_current_route_v10_30"
EVIDENCE = RESULTS / "lia_191_system2005_50y_disconnection_policy_v4_rescore_v1/summary.json"
OUT = RESULTS / "lia_191_current_route_v10_31"


def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def evidence_exists(value):
    path = Path(value)
    return path.exists() if path.is_absolute() else (ROOT / path).exists()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUT}")
    with (SOURCE / "lia_191_current_route.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if evidence.get("status") != "policy-rescore-complete":
        raise RuntimeError("system-2005 policy-v4 rescore did not complete")
    matched = [row for row in rows if int(row["lia_unique_id"]) == 2005]
    if len(matched) != 1 or truth(matched[0]["accepted"]):
        raise RuntimeError("system-2005 route row is missing, duplicated, or accepted")
    evidence_rel = str(EVIDENCE.relative_to(ROOT)).replace("\\", "/")
    matched[0].update({
        "current_route": "disconnection-diagnostic-policy-v4-length-adjudication",
        "current_gate": "lia-separation-normalized-length-gate",
        "latest_evidence_stage": "system2005-50y-disconnection-policy-v4-rescore-v1",
        "latest_evidence_status": "policy-rescore-complete-length-fail",
        "latest_flags": (
            "numerical pass; disconnections diagnostic-only; LIA total length passes; "
            "LIA separation-normalized error 0.629635 exceeds 0.2"
        ),
        "latest_evidence": evidence_rel,
        "controller_state": "closed-current-branch",
        "mechanism_class": "topology-adaptive-checkpointed-solver-policy-v4-length-conflict",
        "next_legal_action": (
            "adjudicate separation-normalized length gate before any 530y launch"
        ),
        "update_source": evidence_rel,
        "active_process_count": "0",
        "active_process_ids": "",
        "mutually_exclusive_class": "closed-current-branch",
    })
    ids = [int(row["lia_unique_id"]) for row in rows]
    accepted = [row for row in rows if truth(row["accepted"])]
    remaining = [row for row in rows if not truth(row["accepted"])]
    checks = {
        "population_191": len(rows) == 191,
        "unique_191": len(set(ids)) == 191,
        "accepted_54": len(accepted) == 54,
        "remaining_137": len(remaining) == 137,
        "system2005_policy_rescore_complete": evidence["status"] == "policy-rescore-complete",
        "disconnection_no_longer_hard_gate": evidence["disconnection_diagnostics"]["hard_failure"] is False,
        "system2005_full_historical_not_authorized": evidence["full_historical_authorized"] is False,
        "all_evidence_paths_exist": all(evidence_exists(row["latest_evidence"]) for row in rows),
        "all_remaining_closed": all(row["controller_state"] == "closed-current-branch" for row in remaining),
        "no_active_route_rows": all(str(row["active_process_count"]) in {"", "0"} for row in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v10.31 route checks failed:{checks}")
    OUT.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    write_csv(OUT / "lia_191_current_route.csv", rows, fields)
    write_csv(OUT / "remaining_137_current_route.csv", remaining, fields)
    write_csv(OUT / "accepted_54_current_route.csv", accepted, fields)
    summary = {
        "schema": "lia-191-current-route-v10-31",
        "status": "system2005-policy-v4-registered-length-gate-remains",
        "population": len(rows),
        "accepted": len(accepted),
        "remaining": len(remaining),
        "closed_current_branch": len(remaining),
        "active": 0,
        "queue": 0,
        "checks": checks,
        "supersedes": "lia_191_current_route_v10_30",
        "system2005_evidence": str(EVIDENCE.resolve()),
        "system2005_remaining_blocker": evidence["remaining_blocker"],
        "scientific_contract_changed": True,
        "contract_change": "disconnections-are-diagnostic-not-hard-failure",
        "interpretation": (
            "Policy V4 removes root and tributary disconnections from hard acceptance. "
            "System 2005 remains unaccepted only because the unchanged LIA separation-normalized "
            "length gate fails."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
