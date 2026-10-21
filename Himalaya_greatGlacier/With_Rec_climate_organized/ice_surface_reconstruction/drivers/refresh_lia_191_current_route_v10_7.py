"""Route refresh after system-2907 stored-main overlap geometry closure."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_6/lia_191_current_route.csv"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_7"


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open("r", encoding="utf-8-sig")))
    for row in rows:
        if int(row["lia_unique_id"]) != 2907:
            continue
        row.update({
            "current_route": "geometry-mechanism-closed",
            "current_gate": "stored-main-overlap-selection-static-fail",
            "latest_evidence_stage": "geometry",
            "latest_evidence_status": "geometry-fail",
            "latest_flags": "length-error1.052pct;area-error0.379pct;free5.150km;within0;junction6;root-stored-main-p951026.75m-over250m",
            "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v2/stored_main_overlap_2907_v1/geometry/geometry_summary_stored_main_overlap.json",
            "controller_state": "closed-current-branch",
            "mechanism_class": "stored-main-whole-route-candidate-selection-closed",
            "next_legal_action": "requires an explicitly constrained centerline generator; do not retry head-first or route-ranking variants",
            "update_source": "lia_191_geometry_repair_v2_stored_main_overlap_2907_v1",
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
        "supersedes": "lia_191_current_route_v10_6",
        "new_decisions": {"2907": "stored-main whole-route candidate selection passes area/length/domain but fails six junction slopes and the 250 m spatial-overlap gate; no inversion"},
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
