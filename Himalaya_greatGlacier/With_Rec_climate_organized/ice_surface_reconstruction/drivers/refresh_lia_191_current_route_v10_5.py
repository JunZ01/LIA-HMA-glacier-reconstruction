"""Route refresh after system-1931 continuous-surface adjudication."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_4/lia_191_current_route.csv"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_5"


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open("r", encoding="utf-8-sig")))
    for row in rows:
        if int(row["lia_unique_id"]) != 1931:
            continue
        row.update({
            "current_route": "geometry-mechanism-closed",
            "current_gate": "continuous-surface-target-hypsometry-cdf-fail",
            "latest_evidence_stage": "continuous-surface-static-adjudication",
            "latest_evidence_status": "mathematical-fail",
            "latest_flags": "zero-reversals;max-adjustment77.776m;mean-adjustment2.755m;hypsometry-cdf4.458pct-over-2pct",
            "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_system1931_continuous_surface_target_adjudication_v1/summary.json",
            "controller_state": "closed-current-branch",
            "mechanism_class": "surface-moment-conserving-continuous-network-geometry-closed",
            "next_legal_action": "requires a new distribution-preserving discrete geometry representation; do not materialize this continuous target",
            "update_source": "lia_191_system1931_continuous_surface_target_adjudication_v1",
            "mutually_exclusive_class": "closed-current-branch",
        })
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    for name, data in (
        ("lia_191_current_route.csv", rows),
        ("remaining_137_current_route.csv", [row for row in rows if row["accepted"].lower() != "true"]),
    ):
        with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    controller_counts = Counter(row["controller_state"] for row in rows)
    class_counts = Counter(row["mutually_exclusive_class"] for row in rows)
    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191, "accepted": 54, "remaining": 137,
        "controller_state_counts": dict(controller_counts),
        "mutually_exclusive_class_counts": dict(class_counts),
        "active_systems": [], "active_process_count": 0, "queue_rows": 0,
        "actionable_now": 0, "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_4",
        "new_decisions": {
            "1931": "continuous monotone surface-moment target passes reversal and adjustment gates but fails the frozen 100 m hypsometry CDF gate; no materialization or OGGM run"
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
