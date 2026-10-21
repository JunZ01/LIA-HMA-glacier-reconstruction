"""Route refresh after system-84 exact integer min-cost transport closure."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_9/lia_191_current_route.csv"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_10"


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open("r", encoding="utf-8-sig")))
    for row in rows:
        if int(row["lia_unique_id"]) != 84:
            continue
        row.update({
            "current_route": "geometry-optimizer-backend-closed",
            "current_gate": "exact-integer-mincost-static-network-fail",
            "latest_evidence_stage": "integer-cell-transport-static-adjudication",
            "latest_evidence_status": "mathematical-candidate-fail",
            "latest_flags": "quota-exact;cdf0.637pct;within119;junction5;max-adjustment177.29m;mean-adjustment26.48m",
            "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_system84_network_simplex_integer_transport_v1/summary.json",
            "controller_state": "closed-current-branch",
            "mechanism_class": "network-simplex-exact-integer-mincost-backend-closed",
            "next_legal_action": "requires an independent exact MILP/CP backend that jointly encodes integer assignment and network mean inequalities; do not tune min-cost weights or locality",
            "update_source": "lia_191_system84_network_simplex_integer_transport_v1",
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
        "supersedes": "lia_191_current_route_v10_9",
        "new_decisions": {"84": "NetworkX exact integer quota transport solves the min-cost flow but fails network reversal and 150/20 m adjustment gates; backend closed without materialization"},
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
