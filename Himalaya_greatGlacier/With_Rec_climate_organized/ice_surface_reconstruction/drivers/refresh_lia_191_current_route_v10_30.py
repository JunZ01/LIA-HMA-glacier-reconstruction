"""Register the system-2005 50-year solver/checkpoint gate in current route."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = RESULTS / "lia_191_current_route_v10_29"
EVIDENCE = RESULTS / "lia_191_system2005_v8_v9_checkpoint_50y_equivalence_v1/summary.json"
OUT = RESULTS / "lia_191_current_route_v10_30"


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
    if evidence.get("status") != "equivalence-pass":
        raise RuntimeError("system-2005 50-year equivalence evidence did not pass")

    matching = [row for row in rows if int(row["lia_unique_id"]) == 2005]
    if len(matching) != 1 or truth(matching[0]["accepted"]):
        raise RuntimeError("system-2005 route row is missing, duplicated, or accepted")
    row = matching[0]
    row.update(
        {
            "current_route": "topology-adaptive-checkpointed-historical-policy-adjudication",
            "current_gate": "root-continuity-and-lia-separation-conflict",
            "latest_evidence_stage": "system2005-v8-v9-checkpoint-50y-equivalence-v1",
            "latest_evidence_status": "equivalence-pass-historical-policy-fail",
            "latest_flags": (
                "50y/36600-step numerical pass; V8/V9 final arrays bitwise equal; "
                "50 annual checkpoints complete; root-gap policy and LIA-separation gates fail"
            ),
            "latest_evidence": str(EVIDENCE.relative_to(ROOT)).replace("\\", "/"),
            "controller_state": "closed-current-branch",
            "mechanism_class": (
                "topology-adaptive-checkpointed-solver-numerically-qualified-policy-conflict"
            ),
            "next_legal_action": (
                "adjudicate root-gap policy conflict before any 530y or population launch"
            ),
            "update_source": str(EVIDENCE.relative_to(ROOT)).replace("\\", "/"),
            "active_process_count": "0",
            "active_process_ids": "",
            "mutually_exclusive_class": "closed-current-branch",
        }
    )

    ids = [int(value["lia_unique_id"]) for value in rows]
    accepted = [value for value in rows if truth(value["accepted"])]
    remaining = [value for value in rows if not truth(value["accepted"])]
    checks = {
        "population_191": len(rows) == 191,
        "unique_191": len(set(ids)) == 191,
        "accepted_54": len(accepted) == 54,
        "remaining_137": len(remaining) == 137,
        "system2005_evidence_pass": evidence.get("status") == "equivalence-pass",
        "system2005_full_historical_not_authorized": (
            evidence.get("full_historical_authorized") is False
        ),
        "all_evidence_paths_exist": all(
            evidence_exists(value["latest_evidence"]) for value in rows
        ),
        "all_remaining_closed": all(
            value["controller_state"] == "closed-current-branch" for value in remaining
        ),
        "no_active_route_rows": all(
            str(value["active_process_count"]) in {"", "0"} for value in rows
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v10.30 route checks failed:{checks}")

    OUT.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0].keys())
    write_csv(OUT / "lia_191_current_route.csv", rows, fields)
    write_csv(OUT / "remaining_137_current_route.csv", remaining, fields)
    write_csv(OUT / "accepted_54_current_route.csv", accepted, fields)
    summary = {
        "schema": "lia-191-current-route-v10-30",
        "status": "system2005-optimized-checkpoint-gate-registered",
        "population": len(rows),
        "accepted": len(accepted),
        "remaining": len(remaining),
        "closed_current_branch": len(remaining),
        "active": 0,
        "queue": 0,
        "checks": checks,
        "supersedes": "lia_191_current_route_v10_29",
        "system2005_evidence": str(EVIDENCE.resolve()),
        "system2005_remaining_blocker": evidence.get("remaining_blocker"),
        "scientific_contract_changed": False,
        "interpretation": (
            "The system-2005 optimized solver and annual checkpoint mechanism pass the "
            "50-year equivalence gate. Accepted and remaining counts are unchanged; full "
            "historical execution remains unauthorized until the root-gap policy conflict "
            "is adjudicated."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
