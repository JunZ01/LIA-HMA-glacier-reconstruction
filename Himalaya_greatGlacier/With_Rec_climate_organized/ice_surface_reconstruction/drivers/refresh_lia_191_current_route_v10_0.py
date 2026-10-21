"""Read-only route refresh after system-14 source-native 2-D CFL closure."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v9_9/lia_191_current_route.csv"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_0"


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    by_id = {row["lia_unique_id"]: row for row in rows}
    by_id["14"].update({
        "current_route": "geometry-dynamics-mechanism-closed",
        "current_gate": "source-native-planform2d-first-step-cfl-floor-fail",
        "latest_evidence_stage": "upstream2d-zero-mb-0p1y",
        "latest_evidence_status": "zero-mb-fail",
        "latest_flags": "required-timestep-1.188882s-below-30s;source-internal-76.014m-surface-step",
        "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_planform2d_source_native_zero_mb_v1/system_14/summary.json",
        "controller_state": "closed-current-branch",
        "mechanism_class": "source-native-24m-planform-upstream2d-closed",
        "next_legal_action": "requires a new implicit 2-D/state representation; do not run 5y, confirmation, or invalid-outline class promotion",
        "update_source": "lia_191_invalid_outline_planform2d_source_native_cfl_diagnostic_v2",
        "mutually_exclusive_class": "closed-current-branch",
    })
    accepted = [row for row in rows if row["accepted"].lower() == "true"]
    remaining = [row for row in rows if row["accepted"].lower() != "true"]
    states = Counter(row["controller_state"] for row in rows)
    classes = Counter(row["mutually_exclusive_class"] for row in rows)
    expected = Counter({"closed-current-branch": 109, "accepted": 54, "hold": 28})
    if len(rows) != 191 or len(accepted) != 54 or len(remaining) != 137 or states != expected:
        raise RuntimeError(
            f"route invariant failed: rows={len(rows)}, accepted={len(accepted)}, states={states}"
        )
    write_csv(OUTPUT / "lia_191_current_route.csv", rows, fields)
    write_csv(OUTPUT / "remaining_137_current_route.csv", remaining, fields)
    write_csv(
        OUTPUT / "mutually_exclusive_class_counts.csv",
        [{"class": key, "count": value} for key, value in sorted(classes.items())],
        ["class", "count"],
    )
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
        "supersedes": "lia_191_current_route_v9_9",
        "new_decision": {
            "14": "native 24 m planform static passed; explicit Upstream2D closed at 1.188882 s first-step CFL floor"
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
