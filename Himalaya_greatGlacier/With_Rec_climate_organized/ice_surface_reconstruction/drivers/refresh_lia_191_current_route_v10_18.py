"""Close current structural families for 2342 after spatial-control audit."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_17"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_18"
EVIDENCE = "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2342_spatial_station_rank_controls_v1/summary.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    for path in SOURCE.glob("*.csv"):
        shutil.copy2(path, OUTPUT / path.name)
    path = OUTPUT / "lia_191_current_route.csv"
    route = pd.read_csv(path)
    mask = route["lia_unique_id"].astype(int).eq(2342)
    route.loc[mask, "current_route"] = "negative-flux-current-structural-families-closed"
    route.loc[mask, "current_gate"] = "spatial-station-static-fail-after-reservoir-and-decoupled-families"
    route.loc[mask, "latest_evidence_stage"] = "static-spatial-control-feasibility"
    route.loc[mask, "latest_evidence_status"] = "static-feasibility-fail"
    route.loc[mask, "latest_flags"] = "exact A/V and CDF0.743pct but 327 raw spatial reversals and p95 movement1.060km>1km; no sorting/regularization authorized"
    route.loc[mask, "latest_evidence"] = EVIDENCE
    route.loc[mask, "controller_state"] = "closed-current-branch"
    route.loc[mask, "mutually_exclusive_class"] = "closed-current-branch"
    route.loc[mask, "next_legal_action"] = "only a genuinely new higher-order or independently validated state/dynamics family may reopen 2342; no current-family retry"
    route.loc[mask, "update_source"] = "negative-flux2342-current-structural-boundary-v1"
    route.to_csv(path, index=False)
    remaining = route[~route["accepted"].astype(str).str.lower().eq("true")].copy()
    remaining.to_csv(OUTPUT / "remaining_137_current_route.csv", index=False)
    accepted = int(route["accepted"].astype(str).str.lower().eq("true").sum())
    summary = {
        "status": "current-route-snapshot-complete",
        "population": len(route),
        "accepted": accepted,
        "remaining": len(route)-accepted,
        "controller_state_counts": route["controller_state"].value_counts().to_dict(),
        "mutually_exclusive_class_counts": route["mutually_exclusive_class"].value_counts().to_dict(),
        "active_systems": [], "active_process_count": 0, "queue_rows": 0,
        "actionable_now": 0, "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_17",
        "new_decisions": {
            "2342": "current standard-core, residual/subnetwork/root reservoirs, decoupled branch EB, and raw spatial-control families closed with versioned evidence"
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
