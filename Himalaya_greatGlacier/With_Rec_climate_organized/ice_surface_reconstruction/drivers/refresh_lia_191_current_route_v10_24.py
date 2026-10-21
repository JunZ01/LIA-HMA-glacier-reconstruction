"""Refresh v10.23 with system 1120 and system 2005 terminal adjudications."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_23"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_24"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    for path in SOURCE.glob("*.csv"):
        shutil.copy2(path, OUTPUT / path.name)
    route_path = OUTPUT / "lia_191_current_route.csv"
    route = pd.read_csv(route_path)

    mask = route["lia_unique_id"].astype(int).eq(1120)
    route.loc[mask, "synthetic_route_attempted"] = True
    route.loc[mask, "current_route"] = "synthetic-standard-chain-steady-closed"
    route.loc[mask, "current_gate"] = "steady-50y-no-volume-sign-bracket"
    route.loc[mask, "latest_evidence_stage"] = "steady-50y-cold-bracket"
    route.loc[mask, "latest_evidence_status"] = "steady-50y-fail"
    route.loc[mask, "latest_flags"] = (
        "geometry/inversion/5y-zeroMB pass; parent dV50=-15.668%, cold -0.25K dV50=-2.584%, same sign and still outside 2%"
    )
    route.loc[mask, "latest_evidence"] = (
        "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline1120_single_outlier_v1/"
        "steady_cold_dtm0p25_v1/steady_50y_summary.json"
    )
    route.loc[mask, "controller_state"] = "closed-current-branch"
    route.loc[mask, "mutually_exclusive_class"] = "closed-current-branch"
    route.loc[mask, "mechanism_class"] = "invalid-polygon-plus1-topology-repair-standard-inversion-zeroMB-pass-steady-no-bracket"
    route.loc[mask, "next_legal_action"] = (
        "requires a genuinely new non-scalar initial-state or climate-state mechanism; do not add more delta-T points"
    )
    route.loc[mask, "update_source"] = "invalid-outline1120-single-outlier-full-gated-chain-v1"

    mask = route["lia_unique_id"].astype(int).eq(2005)
    route.loc[mask, "current_gate"] = "network-monotone-positive-solver-tributary-disconnection-at-30s"
    route.loc[mask, "latest_evidence_stage"] = "zero-mb-0p1y-monotone-positive-conservative-solver"
    route.loc[mask, "latest_evidence_status"] = "zero-mb-fail"
    route.loc[mask, "latest_flags"] = (
        "KKT failure-state step passes, but 0.1y at 30s disconnects branch8 active node11 from terminal node27"
    )
    route.loc[mask, "latest_evidence"] = (
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_network_monotone_positive_conservative_system2005_zero_mb_0p1y_v1/summary.json"
    )
    route.loc[mask, "next_legal_action"] = (
        "closed for network variational solver family; preserving the junction would require artificial ice storage"
    )
    route.loc[mask, "update_source"] = "system2005-network-monotone-positive-v1"

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
        "supersedes": "lia_191_current_route_v10_23",
        "new_decisions": {
            "system1120": "topology repair, static geometry, formal inversion and 5y zero-MB pass; finite parent/cold steady points share negative dV sign and current scalar steady branch closes",
            "system2005": "monotone-positive mass-conserving solver passes the targeted KKT step but disconnects a tributary at the 30s floor during 0.1y",
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
