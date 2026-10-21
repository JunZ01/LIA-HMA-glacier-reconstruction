"""Route refresh after system-1014 raster/shared-confluence closure."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_11/lia_191_current_route.csv"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_12"


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open("r", encoding="utf-8-sig")))
    for row in rows:
        if int(row["lia_unique_id"]) != 1014:
            continue
        row.update({
            "current_route": "input-geometry-mechanism-closed",
            "current_gate": "raster-shared-confluence-local-interval-infeasible",
            "latest_evidence_stage": "geometry-static-mathematical-audit",
            "latest_evidence_status": "mathematical-fail",
            "latest_flags": "five-groups-three-feasible-two-empty;new-within-reverse2;max-adjustment144.77m;cdf0.044pct",
            "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_system1014_raster_shared_confluence_v1/summary.json",
            "controller_state": "closed-current-branch",
            "mechanism_class": "raster-support-outline-shared-confluence-closed",
            "next_legal_action": "requires a confluence-capable representation that changes neighboring control volumes or topology; do not combine raster outline with shared-node elevation again",
            "update_source": "lia_191_system1014_raster_shared_confluence_v1",
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
        "supersedes": "lia_191_current_route_v10_11",
        "new_decisions": {"1014": "raster-support outline plus shared confluence surface has two empty local intervals and creates within-branch reversals; no materialization or promotion"},
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
