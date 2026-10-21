"""Register the two authorized active system-2005 full historical endpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = RESULTS / "lia_191_current_route_v10_32"
EVIDENCE = RESULTS / "lia_191_system2005_formalsteady25250_policyv5_50y_gate_v1/summary.json"
OUT = RESULTS / "lia_191_current_route_v10_33"
PIDS = "13596;62252"


def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


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
    matched = [row for row in rows if int(row["lia_unique_id"]) == 2005]
    if evidence.get("full_historical_authorized") is not True or len(matched) != 1:
        raise RuntimeError("system-2005 full historical launch is not authorized")
    evidence_rel = str(EVIDENCE.relative_to(ROOT)).replace("\\", "/")
    matched[0].update({
        "current_route": "policy-v5-full-historical-endpoints-active",
        "current_gate": "complete-and-adjudicate-endpoints-0p375-and-0p875",
        "latest_evidence_stage": "system2005-formalsteady25250-policyv5-50y-gate-v1",
        "latest_evidence_status": "full-historical-authorized-and-active",
        "latest_flags": "two independent versioned endpoints active with annual checkpoints",
        "latest_evidence": evidence_rel,
        "controller_state": "active-historical",
        "mechanism_class": "topology-adaptive-v9-policy-v5-full-historical",
        "next_legal_action": "wait for both endpoints; adjudicate completed summaries without changing contract",
        "update_source": evidence_rel,
        "active_process_count": "2",
        "active_process_ids": PIDS,
        "mutually_exclusive_class": "active-historical",
    })
    accepted = [row for row in rows if truth(row["accepted"])]
    remaining = [row for row in rows if not truth(row["accepted"])]
    checks = {
        "population_191": len(rows) == 191,
        "accepted_54": len(accepted) == 54,
        "remaining_137": len(remaining) == 137,
        "full_historical_authorized": evidence["full_historical_authorized"] is True,
        "exactly_one_active_route_row": sum(row["controller_state"] == "active-historical" for row in rows) == 1,
        "two_active_processes_registered": matched[0]["active_process_count"] == "2",
    }
    if not all(checks.values()):
        raise RuntimeError(f"v10.33 route checks failed:{checks}")
    OUT.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    write_csv(OUT / "lia_191_current_route.csv", rows, fields)
    write_csv(OUT / "remaining_137_current_route.csv", remaining, fields)
    write_csv(OUT / "accepted_54_current_route.csv", accepted, fields)
    summary = {
        "schema": "lia-191-current-route-v10-33",
        "status": "system2005-full-historical-endpoints-active",
        "population": len(rows),
        "accepted": len(accepted),
        "remaining": len(remaining),
        "closed_current_branch": 136,
        "active_systems": 1,
        "active_processes": 2,
        "active_process_ids": [13596, 62252],
        "queue": 0,
        "checks": checks,
        "supersedes": "lia_191_current_route_v10_32",
        "authorization_evidence": str(EVIDENCE.resolve()),
        "outputs": [
            str((RESULTS / "s2005fs25250p5h0375v1").resolve()),
            str((RESULTS / "s2005fs25250p5h0875v1").resolve()),
        ],
        "scientific_contract_changed_since_authorization": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
