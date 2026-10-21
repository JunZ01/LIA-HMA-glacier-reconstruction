"""Corrected v10.2 route refresh: system 1642 was already in the closed class."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
VALIDATION = PROJECT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = VALIDATION / "lia_191_current_route_v10_1/lia_191_current_route.csv"
OUTPUT = VALIDATION / "lia_191_current_route_v10_2"
EVIDENCE = "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux_monotone_variable_thinfilm_1642_decision_v1/summary.json"


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not (PROJECT / EVIDENCE).exists():
        raise FileNotFoundError(EVIDENCE)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    row = next(item for item in rows if item["lia_unique_id"] == "1642")
    if row["controller_state"] != "closed-current-branch":
        raise RuntimeError("1642-source-class-drift")
    row.update({
        "current_route": "negative-flux-initial-state-mechanisms-closed",
        "current_gate": "monotone-variable-thinfilm-zero-mb-fail",
        "latest_evidence_stage": "zero_mb",
        "latest_evidence_status": "zero-mb-fail",
        "latest_flags": "cfl-error;checkpoint-incomplete;negative-section-clip-mass-creation",
        "latest_evidence": EVIDENCE,
        "mechanism_class": "negative-flux-monotone-variable-thinfilm-closed",
        "next_legal_action": "requires a genuinely new conservative initial-state/dynamics representation; do not run 30s continuation, steady, history, or promotion from this candidate",
        "update_source": "lia_191_negative_flux_monotone_variable_thinfilm_1642_decision_v1",
    })
    accepted = [item for item in rows if item["accepted"].lower() == "true"]
    remaining = [item for item in rows if item["accepted"].lower() != "true"]
    states = Counter(item["controller_state"] for item in rows)
    classes = Counter(item["mutually_exclusive_class"] for item in rows)
    expected = Counter({"closed-current-branch": 109, "accepted": 54, "hold": 28})
    if len(rows) != 191 or len(accepted) != 54 or len(remaining) != 137 or states != expected:
        raise RuntimeError(f"route-invariant-failed:{len(rows)}:{len(accepted)}:{states}")
    OUTPUT.mkdir(parents=True)
    write_csv(OUTPUT / "lia_191_current_route.csv", rows, fields)
    write_csv(OUTPUT / "remaining_137_current_route.csv", remaining, fields)
    write_csv(OUTPUT / "mutually_exclusive_class_counts.csv", [{"class": k, "count": v} for k, v in sorted(classes.items())], ["class", "count"])
    write_csv(OUTPUT / "next_legal_queue.csv", [], ["lia_unique_id", "next_legal_action"])
    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "controller_state_counts": dict(states),
        "mutually_exclusive_class_counts": dict(classes),
        "active_systems": [],
        "active_process_count": 0,
        "queue_rows": 0,
        "actionable_now": 0,
        "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_1",
        "new_decision": {"1642": "a second negative-flux initial-state mechanism passed static but failed zero-MB at both CFL and section-clipping gates"},
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
