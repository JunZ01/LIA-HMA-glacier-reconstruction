"""Route refresh after final system-2907 explicit stored-network closure."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_8/lia_191_current_route.csv"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_9"


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open("r", encoding="utf-8-sig")))
    for row in rows:
        if int(row["lia_unique_id"]) != 2907:
            continue
        row.update({
            "current_route": "geometry-mechanism-closed",
            "current_gate": "explicit-stored-network-catchment-structure-fail",
            "latest_evidence_stage": "geometry-catchment-construction",
            "latest_evidence_status": "structural-fail",
            "latest_flags": "root-only-p95-34.1m-but-catchment-missing1688-and-junction8;full-network-catchment-cell-nonunique",
            "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v2/stored_full_network_2907_v1/decision.json",
            "controller_state": "closed-current-branch",
            "mechanism_class": "explicit-stored-root-and-full-network-injection-closed",
            "next_legal_action": "requires a new catchment-compatible geometry representation; do not jitter/delete stored branches or retry ranking",
            "update_source": "lia_191_geometry_repair_v2_stored_full_network_2907_v1",
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
        "supersedes": "lia_191_current_route_v10_8",
        "new_decisions": {"2907": "explicit stored root solves spatial path but fails catchment/junction gates; explicit full network fails catchment-cell uniqueness; structural family closed before inversion"},
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
