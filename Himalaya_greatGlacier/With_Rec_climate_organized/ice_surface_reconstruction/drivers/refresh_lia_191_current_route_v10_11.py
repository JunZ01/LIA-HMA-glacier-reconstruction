"""Route refresh after system-1856 shared-confluence surface closure."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_10/lia_191_current_route.csv"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_11"


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open("r", encoding="utf-8-sig")))
    for row in rows:
        if int(row["lia_unique_id"]) != 1856:
            continue
        row.update({
            "current_route": "geometry-mechanism-closed",
            "current_gate": "shared-confluence-surface-local-interval-infeasible",
            "latest_evidence_stage": "geometry-static-mathematical-audit",
            "latest_evidence_status": "mathematical-fail",
            "latest_flags": "two-empty-confluence-intervals;new-within-reverse2;max-adjustment181.75m;cdf0.163pct",
            "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_system1856_shared_confluence_surface_v1/summary.json",
            "controller_state": "closed-current-branch",
            "mechanism_class": "shared-confluence-surface-control-volume-closed",
            "next_legal_action": "requires a geometry representation that changes neighboring control volumes or junction topology; do not retry shared-node elevation or generic smoothing",
            "update_source": "lia_191_system1856_shared_confluence_surface_v1",
            "mutually_exclusive_class": "closed-current-branch",
        })
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    for name, data in (("lia_191_current_route.csv", rows), ("remaining_137_current_route.csv", [r for r in rows if r["accepted"].lower() != "true"])):
        with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    controller_counts = Counter(r["controller_state"] for r in rows)
    class_counts = Counter(r["mutually_exclusive_class"] for r in rows)
    summary = {
        "status": "current-route-snapshot-complete", "population": 191, "accepted": 54, "remaining": 137,
        "controller_state_counts": dict(controller_counts), "mutually_exclusive_class_counts": dict(class_counts),
        "active_systems": [], "active_process_count": 0, "queue_rows": 0, "actionable_now": 0, "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_10",
        "new_decisions": {"1856": "shared confluence surface DOF has two empty local feasible intervals and creates within-branch reversals; no materialization or class confirmation"},
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
