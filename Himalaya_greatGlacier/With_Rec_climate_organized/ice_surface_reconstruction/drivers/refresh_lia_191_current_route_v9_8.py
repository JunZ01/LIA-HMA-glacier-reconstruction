"""Read-only route refresh after system-84 and system-2005 mechanism closures."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_current_route_v9_7/lia_191_current_route.csv"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_current_route_v9_8"
)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 191 or len({row["lia_unique_id"] for row in rows}) != 191:
        raise RuntimeError("source route is not a unique 191-system table")

    by_id = {row["lia_unique_id"]: row for row in rows}
    by_id["84"].update({
        "current_route": "geometry-mechanism-closed",
        "current_gate": "deterministic-owner-map-not-junction-only",
        "latest_evidence_stage": "deterministic-cell-owner-audit",
        "latest_evidence_status": "audit-fail",
        "latest_flags": "182-within-branch-reversals;10-junction-reversals",
        "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_confluence_exact_cell_reassignment_v1/system_84/deterministic_owner_audit_v1/summary.json",
        "controller_state": "closed-current-branch",
        "mechanism_class": "confluence-local-exact-cell-reassignment-closed",
        "next_legal_action": "requires a genuinely new geometry representation; do not run MILP, static materialization, 1931 confirmation, or class promotion",
        "update_source": "lia_191_confluence_exact_cell_reassignment_v1",
        "mutually_exclusive_class": "closed-current-branch",
    })
    by_id["2005"].update({
        "current_route": "zero-mb-mechanism-closed",
        "current_gate": "network-semiimplicit-0p1y-floor-fail",
        "latest_evidence_stage": "zero-mb-0p1y",
        "latest_evidence_status": "zero-mb-fail",
        "latest_flags": "within-branch-reverse-flux-at-30s;target-incomplete",
        "latest_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_network_coupled_semiimplicit_system2005_zero_mb_0p1y_v3/summary.json",
        "controller_state": "closed-current-branch",
        "mechanism_class": "network-coupled-semiimplicit-finite-volume-closed",
        "next_legal_action": "requires a new state/geometry mechanism; do not run 5y zero-MB, steady, historical, confirmation, or promotion",
        "update_source": "lia_191_network_coupled_semiimplicit_system2005_zero_mb_0p1y_v3",
        "mutually_exclusive_class": "closed-current-branch",
    })

    accepted = [row for row in rows if row["accepted"].lower() == "true"]
    remaining = [row for row in rows if row["accepted"].lower() != "true"]
    states = Counter(row["controller_state"] for row in rows)
    classes = Counter(row["mutually_exclusive_class"] for row in rows)
    if len(accepted) != 54 or len(remaining) != 137:
        raise RuntimeError("accepted/remaining invariant changed unexpectedly")
    if states != Counter({"closed-current-branch": 107, "accepted": 54, "hold": 30}):
        raise RuntimeError(f"unexpected controller counts: {states}")

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
        "supersedes": "lia_191_current_route_v9_7",
        "no_additional_oggm_run": True,
        "new_decisions": {
            "84": "confluence-only exact-cell reassignment closed at deterministic owner-map audit",
            "2005": "network-coupled semiimplicit passed real 30s step but closed at 0.1y 30s-floor reverse-flux gate",
            "3660": "historical closure inherited unchanged from v9.7",
        },
        "facts_update": "pending sections 303-305 because LIA_PROJECT_FACTS.md update is blocked by deny-read ACL",
        "facts_patch_artifact": "validation_data/results/paleo_surface_validation_v0/lia_191_pending_facts_sections_303_305_v1.md",
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
