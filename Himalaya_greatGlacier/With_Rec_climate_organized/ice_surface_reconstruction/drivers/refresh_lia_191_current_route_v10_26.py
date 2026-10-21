"""Refresh v10.25 with high-risk outline, 5598, and negative-flux evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_25"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_26"


def close_highrisk(route: pd.DataFrame, system_id: int, flags: str) -> None:
    mask = route["lia_unique_id"].astype(int).eq(system_id)
    route.loc[mask, "synthetic_route_attempted"] = True
    route.loc[mask, "current_route"] = "invalid-outline-highrisk-topology-repair-static-closed"
    route.loc[mask, "current_gate"] = "unchanged-standard-geometry-static-fail"
    route.loc[mask, "latest_evidence_stage"] = "highrisk-topology-repair-and-standard-geometry"
    route.loc[mask, "latest_evidence_status"] = "geometry-fail"
    route.loc[mask, "latest_flags"] = flags
    route.loc[mask, "latest_evidence"] = f"wcr10invpoly_highrisk3v1/system_{system_id}/geometry/geometry_summary.json"
    route.loc[mask, "controller_state"] = "closed-current-branch"
    route.loc[mask, "mutually_exclusive_class"] = "closed-current-branch"
    route.loc[mask, "mechanism_class"] = "invalid-polygon-highrisk-topology-only-repair-unchanged-standard-geometry-closed"
    route.loc[mask, "next_legal_action"] = "requires a new junction-aware spatial/state representation; do not repeat topology-only repair"
    route.loc[mask, "update_source"] = "invalid-outline-highrisk3-static-batch-v1"


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    for path in SOURCE.glob("*.csv"): shutil.copy2(path, OUTPUT / path.name)
    route_path = OUTPUT / "lia_191_current_route.csv"; route = pd.read_csv(route_path)
    close_highrisk(route, 1639, "junction-reverse-slope:8")
    close_highrisk(route, 2692, "junction-reverse-slope:12")
    close_highrisk(route, 5508, "within-branch-reverse-slope:1;junction-reverse-slope:13;registered-root-length-error:53.297%")

    mask = route["lia_unique_id"].astype(int).eq(5598)
    route.loc[mask, "synthetic_route_attempted"] = True
    route.loc[mask, "current_route"] = "branch-rigid-vertical-datum-zero-mb-closed"
    route.loc[mask, "current_gate"] = "negative-tributary-mass-creation-at-0p1y"
    route.loc[mask, "latest_evidence_stage"] = "zero-mb-0p1y"
    route.loc[mask, "latest_evidence_status"] = "zero-mb-fail"
    route.loc[mask, "latest_flags"] = "static/inversion pass; negative tributary creation 2066.930m3 > 0.002270m3"
    route.loc[mask, "latest_evidence"] = "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline5598_branch_rigid_v1/zero_mb_60s_v1/zero_mb_summary.json"
    route.loc[mask, "controller_state"] = "closed-current-branch"
    route.loc[mask, "mutually_exclusive_class"] = "closed-current-branch"
    route.loc[mask, "mechanism_class"] = "branch-rigid-vertical-datum-reconciliation-zeroMB-negative-mass-closed"
    route.loc[mask, "next_legal_action"] = "requires a solver/state mechanism eliminating negative tributary mass creation without clipping; do not repeat branch offsets"
    route.loc[mask, "update_source"] = "invalid-outline5598-branch-rigid-full-static-inversion-zero-v1"

    six = [890, 988, 1026, 1743, 2689, 2793]
    mask = route["lia_unique_id"].astype(int).isin(six)
    route.loc[mask, "current_gate"] = "paired-cell-initial-B-window-no-near-neutral-subclass"
    route.loc[mask, "latest_evidence_stage"] = "paired-cell-initial-mb-readonly-screen"
    route.loc[mask, "latest_evidence_status"] = "no-near-neutral-subclass"
    route.loc[mask, "latest_flags"] = "projected absolute 50y initial-MB volume fraction exceeds 10% for all six"
    route.loc[mask, "latest_evidence"] = "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux_remaining6_paired_cell_initial_mb_v2/summary.json"
    route.loc[mask, "mechanism_class"] = "negative-flux-zero-thickness-requires-spatially-local-network-coupled-state"
    route.loc[mask, "next_legal_action"] = "requires a new spatially local network-coupled paired-state representation; no scalar steady search or local PAVA retry"
    route.loc[mask, "update_source"] = "negative-flux-remaining6-paired-cell-initial-mb-v2"

    mask890 = route["lia_unique_id"].astype(int).eq(890)
    route.loc[mask890, "current_gate"] = "global-topological-exact-cell-spatial-locality-fail"
    route.loc[mask890, "latest_evidence_stage"] = "global-network-exact-cell-static-feasibility"
    route.loc[mask890, "latest_evidence_status"] = "static-feasibility-fail"
    route.loc[mask890, "latest_flags"] = "reverse slopes zero; p95 movement 8.844km; max 11.622km; branch volume change 131.7%"
    route.loc[mask890, "latest_evidence"] = "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux890_global_topological_exact_cell_v1/summary.json"
    route.loc[mask890, "next_legal_action"] = "global elevation ordering is closed; require a spatially local junction-coupled representation, not wider pools or global sorting"
    route.loc[mask890, "update_source"] = "negative-flux890-global-topological-exact-cell-v1"

    route.to_csv(route_path, index=False)
    accepted = route["accepted"].astype(str).str.lower().eq("true")
    route.loc[~accepted].to_csv(OUTPUT / "remaining_137_current_route.csv", index=False)
    summary = {
        "status": "current-route-snapshot-complete", "population": int(len(route)),
        "accepted": int(accepted.sum()), "remaining": int((~accepted).sum()),
        "controller_state_counts": route["controller_state"].value_counts().to_dict(),
        "mutually_exclusive_class_counts": route["mutually_exclusive_class"].value_counts().to_dict(),
        "active_systems": [], "active_process_count": 0, "queue_rows": 0, "actionable_now": 0, "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_25",
        "new_decisions": {
            "highrisk_invalid_outline": "1639/2692/5508 topology repairs pass but unchanged standard geometry fails; no inversion released",
            "system5598": "branch-rigid static and inversion pass but 0.1y zero-MB creates 2066.930m3 negative tributary mass",
            "negative_flux_remaining6": "paired-cell initial-MB screen has no near-neutral subclass",
            "system890_global_exact_cell": "zero reverse slopes but decisive spatial-locality and branch-volume distortion failure",
            "system2005_support_reserve": "diagnostic invalidated because positive support reserve conflicts with canonical prohibition on artificial ice storage; route unchanged",
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
