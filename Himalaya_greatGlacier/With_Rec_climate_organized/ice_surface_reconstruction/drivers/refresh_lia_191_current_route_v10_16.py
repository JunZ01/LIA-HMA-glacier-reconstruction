"""Route refresh after system-334 terminal-condensation connector closure."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_current_route_v10_15/lia_191_current_route.csv"
)
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_current_route_v10_16"
)
HOLD_IDS = {1120, 1133, 1499, 1639, 1653, 2535, 2692, 3075, 5014, 5231, 5508, 5598}
EVIDENCE = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_invalid_outline334_terminal_control_volume_condensation_v1/"
    "zero_mb_connectors_mindt30_v2/summary.json"
)


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open("r", encoding="utf-8-sig")))
    for row in rows:
        system_id = int(row["lia_unique_id"])
        if system_id == 334:
            row.update(
                {
                    "current_route": "geometry-dynamics-mechanism-closed",
                    "current_gate": "connector-state-0p1y-observation-domain-overflow",
                    "latest_evidence_stage": "zero-mb-0p1y",
                    "latest_evidence_status": "zero-mb-fail",
                    "latest_flags": (
                        "60s pure-CFL 56.2s; unique30s continuation numerically clean "
                        "but root overflow one node with 0.004086m ice"
                    ),
                    "latest_evidence": EVIDENCE,
                    "mechanism_class": (
                        "terminal-control-volume-condensation-zero-ice-connectors-closed"
                    ),
                    "next_legal_action": (
                        "requires a genuinely different observation-domain-preserving "
                        "confluence/control-volume representation; do not relax overflow "
                        "or repeat 30s/connector-tail mechanism"
                    ),
                    "update_source": "system334-terminal-condensation-connector-v1-closed",
                }
            )
        elif system_id in HOLD_IDS:
            row.update(
                {
                    "current_gate": (
                        "representative-334-connector-state-zero-mb-failed-no-promotion"
                    ),
                    "latest_evidence_stage": "zero-mb-0p1y-representative-gate",
                    "latest_evidence_status": (
                        "not-run-after-representative-observation-overflow"
                    ),
                    "latest_flags": (
                        "334 static A/V/hypsometry/topology passed; 30s zero-MB "
                        "overflowed root observation domain by one node"
                    ),
                    "latest_evidence": EVIDENCE,
                    "next_legal_action": (
                        "do not promote terminal control-volume condensation or zero-ice "
                        "connectors; await a genuinely different observation-domain-"
                        "preserving geometry/state representation"
                    ),
                    "update_source": "system334-terminal-condensation-no-promotion-v1",
                }
            )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    remaining = [row for row in rows if row["accepted"].lower() != "true"]
    for name, data in (
        ("lia_191_current_route.csv", rows),
        ("remaining_137_current_route.csv", remaining),
    ):
        with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    summary = {
        "status": "current-route-snapshot-complete",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "controller_state_counts": dict(
            Counter(row["controller_state"] for row in rows)
        ),
        "mutually_exclusive_class_counts": dict(
            Counter(row["mutually_exclusive_class"] for row in rows)
        ),
        "active_systems": [],
        "active_process_count": 0,
        "queue_rows": 0,
        "actionable_now": 0,
        "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_15",
        "new_decisions": {
            "invalid-outline-terminal-condensation": (
                "334 pre/post-inversion static passed exact A/V and no reverse slopes; "
                "unique30s 0.1y zero-MB failed any-positive observation overflow "
                "(one root node, 0.004086m); no promotion to12 holds"
            )
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pending_fact = """## Pending: system 334 terminal-control-volume condensation closure

- Representative 334 passed mathematical and materialized pre-inversion geometry
  gates after condensing 18 terminal control volumes across six branches.  Area
  was exact, 100 m hypsometry CDF error was 0.1015%, and all six junction
  reversals were removed.
- Standard inversion completed but reintroduced four small junction reversals
  and positive tributary terminal ice.  A post-inversion exact-A/V connector
  state condensed at most one additional active 50 m node on four branches and
  appended zero-ice tails.  It passed static with area exact, volume relative
  error 1.61e-16, CDF error at machine precision, zero within/junction reversals,
  and zero positive terminal thickness.
- The 60 s zero-MB run stopped only at required 56.2 s.  The unique authorized
  30 s continuation was mass-clean through 0.1 y but placed 0.004086 m of ice in
  the first root node outside the initial observation domain.  Under the frozen
  any-positive overflow gate this closes the mechanism.  No 5 y, steady,
  historical, confirmation, or promotion run is authorized.
"""
    (OUTPUT / "pending_fact_section.md").write_text(pending_fact, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
