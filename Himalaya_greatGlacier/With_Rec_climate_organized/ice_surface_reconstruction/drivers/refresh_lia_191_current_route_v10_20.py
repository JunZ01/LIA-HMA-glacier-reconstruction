"""Record remaining-six spatial-control audits and system-890 junction closure."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_19"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_20"
SYSTEM_IDS = (890, 988, 1026, 1743, 2689, 2793)
LOCAL = "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux_remaining6_local_pava_cell_reassignment_v1/summary.json"
SYSTEM_890 = "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux890_junction_coupled_cell_reassignment_v1/summary.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    for path in SOURCE.glob("*.csv"):
        shutil.copy2(path, OUTPUT / path.name)
    route_path = OUTPUT / "lia_191_current_route.csv"
    route = pd.read_csv(route_path)
    mask = route["lia_unique_id"].astype(int).isin(SYSTEM_IDS)
    route.loc[mask, "current_gate"] = "raw-spatial-and-local-pava-cell-controls-no-feasible-subclass"
    route.loc[mask, "latest_evidence_stage"] = "current-paired-spatial-control-static-feasibility"
    route.loc[mask, "latest_evidence_status"] = "static-feasibility-fail"
    route.loc[mask, "latest_flags"] = (
        "raw controls retain 178-618 within reversals; local PAVA clears within reversals but all six retain junction reversals and/or exceed 1km-p95/2.5km-max displacement"
    )
    route.loc[mask, "latest_evidence"] = LOCAL
    route.loc[mask, "next_legal_action"] = (
        "raw spatial controls and branch-local exact-cell PAVA are closed; require a genuinely rebuilt current-paired representation with explicit network-junction coupling"
    )
    route.loc[mask, "update_source"] = "negative-flux-remaining6-spatial-control-audits-v1"

    mask_890 = route["lia_unique_id"].astype(int).eq(890)
    route.loc[mask_890, "current_gate"] = "junction-coupled-exact-cell-static-feasibility-fail"
    route.loc[mask_890, "latest_evidence_stage"] = "junction-coupled-exact-cell-static-feasibility"
    route.loc[mask_890, "latest_flags"] = (
        "3/4 junction reversals repaired; one remains; solver stalled; Euclidean displacement p95=1010.289m>1km"
    )
    route.loc[mask_890, "latest_evidence"] = SYSTEM_890
    route.loc[mask_890, "next_legal_action"] = (
        "close local junction-block exact-cell reassignment; require a new global network-coupled state representation, not wider pools or relaxed displacement gates"
    )
    route.loc[mask_890, "update_source"] = "negative-flux890-junction-coupled-cell-v1"
    route.to_csv(route_path, index=False)

    accepted_mask = route["accepted"].astype(str).str.lower().eq("true")
    remaining = route[~accepted_mask]
    remaining.to_csv(OUTPUT / "remaining_137_current_route.csv", index=False)
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
        "supersedes": "lia_191_current_route_v10_19",
        "new_decisions": {
            "negative_flux_remaining6": "raw station controls and local branch-PAVA exact-cell reassignment have no feasible subclass",
            "890": "local junction-coupled exact-cell reassignment closed after 3/4 repairs; one junction and p95 displacement gate remain",
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
