"""Route refresh after system-1499 low-risk invalid-outline static closures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_20"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_21"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    for path in SOURCE.glob("*.csv"):
        shutil.copy2(path, OUTPUT / path.name)
    route_path = OUTPUT / "lia_191_current_route.csv"
    route = pd.read_csv(route_path)

    mask = route["lia_unique_id"].astype(int).eq(1499)
    route.loc[mask, "current_route"] = "input-geometry-mechanism-closed"
    route.loc[mask, "current_gate"] = "shared-confluence-observed-surface-static-fail"
    route.loc[mask, "latest_evidence_stage"] = "repaired-outline-geometry-static-feasibility"
    route.loc[mask, "latest_evidence_status"] = "static-feasibility-fail"
    route.loc[mask, "latest_flags"] = (
        "topology repair pass; unchanged geometry has2 junction reversals; one-node reindex needs36 nodes/1.833km; shared paired-confluence surfaces clear junctions but create2 within reversals"
    )
    route.loc[mask, "latest_evidence"] = (
        "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline1499_shared_confluence_observed_surface_v1/summary.json"
    )
    route.loc[mask, "controller_state"] = "closed-current-branch"
    route.loc[mask, "mechanism_class"] = "invalid-outline1499-lowrisk-network-static-families-closed"
    route.loc[mask, "next_legal_action"] = (
        "requires a genuinely joint network surface/topology representation; do not retry receiver shifts, shared-node sampling, terminal condensation, or local regularization"
    )
    route.loc[mask, "update_source"] = "invalid-outline1499-lowrisk-static-series-v1"
    route.loc[mask, "mutually_exclusive_class"] = "closed-current-branch"

    mask_2005 = route["lia_unique_id"].astype(int).eq(2005)
    route.loc[mask_2005, "latest_evidence_stage"] = "connected-front-node-depletion-readonly-diagnostic"
    route.loc[mask_2005, "latest_evidence_status"] = "diagnostic-complete-current-family-still-closed"
    route.loc[mask_2005, "latest_flags"] = (
        "branch10 local-MB depletion is terminus-first;0 interior-risk nodes; reported 50y gap arises from coupled transport/state evolution"
    )
    route.loc[mask_2005, "latest_evidence"] = (
        "validation_data/results/paleo_surface_validation_v0/lia_191_system2005_connected_front_node_depletion_timescale_v2/summary.json"
    )
    route.loc[mask_2005, "next_legal_action"] = (
        "branch-integrated MB redistribution is not justified; requires a new coupled transport/state representation with persisted pre-gap diagnostics"
    )
    route.loc[mask_2005, "update_source"] = "system2005-node-depletion-timescale-v2"

    route.to_csv(route_path, index=False)
    accepted_mask = route["accepted"].astype(str).str.lower().eq("true")
    route.loc[~accepted_mask].to_csv(OUTPUT / "remaining_137_current_route.csv", index=False)
    summary = {
        "status": "current-route-snapshot-complete",
        "population": len(route),
        "accepted": int(accepted_mask.sum()),
        "remaining": int((~accepted_mask).sum()),
        "controller_state_counts": route["controller_state"].value_counts().to_dict(),
        "mutually_exclusive_class_counts": route["mutually_exclusive_class"].value_counts().to_dict(),
        "active_systems": [],
        "active_process_count": 0,
        "queue_rows": 0,
        "actionable_now": 0,
        "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_20",
        "new_decisions": {
            "1499": "low-risk invalid-outline representative closed at static network gate; no promotion to other seven risk-selected holds",
            "2005": "local-MB depletion audit shows terminus-first retreat; branch-integrated MB hypothesis closed",
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
