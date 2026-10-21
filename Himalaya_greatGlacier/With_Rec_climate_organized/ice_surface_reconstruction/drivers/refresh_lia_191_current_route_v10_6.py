"""Route refresh after system-801 active-area consolidation closure."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_5/lia_191_current_route.csv"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_6"


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open("r", encoding="utf-8-sig")))
    for row in rows:
        if int(row["lia_unique_id"]) != 801:
            continue
        row.update({
            "current_route": "inversion-mechanism-closed",
            "current_gate": "active-area-consolidation-hypsometry-width-fail",
            "latest_evidence_stage": "initial-state-static-feasibility",
            "latest_evidence_status": "mathematical-fail",
            "latest_flags": "area-error0.080pct;volume-exact;hypsometry-cdf3.780pct;max-width-multiplier40.620;max-bed-step1809m",
            "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_system801_active_area_consolidation_v1/summary.json",
            "controller_state": "closed-current-branch",
            "mechanism_class": "active-area-consolidation-rectangular-representation-closed",
            "next_legal_action": "requires a different conservative initial-state representation; do not concentrate zero-thickness catchment area into active nodes",
            "update_source": "lia_191_system801_active_area_consolidation_v1",
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
        "supersedes": "lia_191_current_route_v10_5",
        "new_decisions": {"801": "active-area consolidation exactly restores inversion-width area and volume but fails hypsometry and width/bed plausibility gates; no materialization or promotion"},
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
