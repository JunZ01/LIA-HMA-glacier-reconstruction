"""Record stored-LIA-network surface infeasibility for low-risk invalid holds."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_21"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_22"
SYSTEM_IDS = (1499, 5598, 5014, 2535, 3075, 5231, 1653, 1133)
EVIDENCE = "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_lowrisk8_stored_lia_network_surface_v1/summary.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    for path in SOURCE.glob("*.csv"):
        shutil.copy2(path, OUTPUT / path.name)
    route_path = OUTPUT / "lia_191_current_route.csv"
    route = pd.read_csv(route_path)
    mask = route["lia_unique_id"].astype(int).isin(SYSTEM_IDS)
    still_hold = mask & route["controller_state"].eq("hold")
    route.loc[still_hold, "current_gate"] = "stored-lia-network-reference-surface-no-feasible-subclass"
    route.loc[still_hold, "latest_evidence_stage"] = "stored-lia-network-surface-topology-feasibility"
    route.loc[still_hold, "latest_evidence_status"] = "static-feasibility-fail"
    route.loc[still_hold, "latest_flags"] = (
        "stored networks retain 11-50 within-branch reversals; some also retain 1-8 junction reversals; registered main lengths close exactly"
    )
    route.loc[still_hold, "latest_evidence"] = EVIDENCE
    route.loc[still_hold, "next_legal_action"] = (
        "requires a joint spatial-network/current-paired/dynamic-monotonic state representation; do not materialize unchanged stored reference surfaces"
    )
    route.loc[still_hold, "update_source"] = "invalid-outline-lowrisk8-stored-network-surface-v1"
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
        "supersedes": "lia_191_current_route_v10_21",
        "new_decisions": {
            "invalid_outline_lowrisk8": "unchanged stored LIA network reference surfaces have no static-feasible subclass; seven holds remain pending a joint representation",
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
