"""Refresh v10.22 with the 5598 global and 1499 datum adjudications."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_22"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_23"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    for path in SOURCE.glob("*.csv"):
        shutil.copy2(path, OUTPUT / path.name)
    route_path = OUTPUT / "lia_191_current_route.csv"
    route = pd.read_csv(route_path)

    mask = route["lia_unique_id"].astype(int).eq(5598)
    route.loc[mask, "current_gate"] = "global-topological-cell-assignment-locality-and-adjustment-fail"
    route.loc[mask, "latest_evidence_stage"] = "joint-spatial-network-monotonic-static-feasibility"
    route.loc[mask, "latest_evidence_status"] = "static-feasibility-fail"
    route.loc[mask, "latest_flags"] = (
        "A/V exact, CDF0.120%, reversals0; node adjustment187.38m, movement p954425.55m/max7503.76m"
    )
    route.loc[mask, "latest_evidence"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_invalid_outline5598_global_topological_cell_assignment_v2/summary.json"
    )
    route.loc[mask, "next_legal_action"] = (
        "requires a different joint representation that preserves network monotonicity without nonlocal cell transport"
    )
    route.loc[mask, "update_source"] = "invalid-outline5598-global-topological-v2"

    mask = route["lia_unique_id"].astype(int).eq(1499)
    route.loc[mask, "current_gate"] = "branch-rigid-vertical-datum-hypsometry-fail"
    route.loc[mask, "latest_evidence_stage"] = "branch-rigid-vertical-datum-lexicographic-feasibility"
    route.loc[mask, "latest_evidence_status"] = "static-mathematical-feasibility-fail"
    route.loc[mask, "latest_flags"] = (
        "junction reversals2->0 with max offset29.66m, but paired CDF rises from1.041% to3.384%"
    )
    route.loc[mask, "latest_evidence"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_invalid_outline1499_branch_rigid_vertical_datum_v2/summary.json"
    )
    route.loc[mask, "next_legal_action"] = (
        "closed for rigid branch datum; only a genuinely different joint geometry/state representation may reopen"
    )
    route.loc[mask, "update_source"] = "invalid-outline1499-branch-rigid-datum-v2"

    route.to_csv(route_path, index=False)
    accepted = route["accepted"].astype(str).str.lower().eq("true")
    route.loc[~accepted].to_csv(OUTPUT / "remaining_137_current_route.csv", index=False)
    summary = {
        "status": "current-route-snapshot-complete",
        "population": int(len(route)),
        "accepted": int(accepted.sum()),
        "remaining": int((~accepted).sum()),
        "controller_state_counts": route["controller_state"].value_counts().to_dict(),
        "mutually_exclusive_class_counts": route["mutually_exclusive_class"].value_counts().to_dict(),
        "active_systems": [],
        "active_process_count": 0,
        "queue_rows": 0,
        "actionable_now": 0,
        "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_22",
        "new_decisions": {
            "system5598": "global topological exact-cell assignment is monotonic and conservative but fails spatial locality and adjustment gates",
            "system1499": "branch-rigid datum clears junction reversals but fails the unchanged 2% paired-hypsometry gate after lexicographic refinement",
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
