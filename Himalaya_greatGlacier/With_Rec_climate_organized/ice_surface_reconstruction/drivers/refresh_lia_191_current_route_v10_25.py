"""Refresh v10.24 with the low-risk invalid-outline six-case batch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_24"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_25"


def _close_static(route: pd.DataFrame, system_id: int, flags: str) -> None:
    mask = route["lia_unique_id"].astype(int).eq(system_id)
    evidence = f"wcr10invpoly_lowrisk6v1/system_{system_id}/geometry/geometry_summary.json"
    route.loc[mask, "synthetic_route_attempted"] = True
    route.loc[mask, "current_route"] = "invalid-outline-topology-repair-static-closed"
    route.loc[mask, "current_gate"] = "unchanged-standard-geometry-static-fail"
    route.loc[mask, "latest_evidence_stage"] = "topology-repair-and-standard-geometry"
    route.loc[mask, "latest_evidence_status"] = "geometry-fail"
    route.loc[mask, "latest_flags"] = flags
    route.loc[mask, "latest_evidence"] = evidence
    route.loc[mask, "controller_state"] = "closed-current-branch"
    route.loc[mask, "mutually_exclusive_class"] = "closed-current-branch"
    route.loc[mask, "mechanism_class"] = "invalid-polygon-plus1-topology-repair-unchanged-standard-geometry"
    route.loc[mask, "next_legal_action"] = (
        "requires a genuinely new junction-aware spatial/state representation; do not rerun topology-only repair"
    )
    route.loc[mask, "update_source"] = "invalid-outline-lowrisk6-static-batch-v1"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    for path in SOURCE.glob("*.csv"):
        shutil.copy2(path, OUTPUT / path.name)
    route_path = OUTPUT / "lia_191_current_route.csv"
    route = pd.read_csv(route_path)

    _close_static(route, 5014, "junction-reverse-slope:7")
    _close_static(route, 2535, "junction-reverse-slope:3;downstream-domain-too-short:4999.649m")
    _close_static(route, 5231, "within-branch-reverse-slope:1;junction-reverse-slope:5")
    _close_static(route, 1653, "junction-reverse-slope:11")
    _close_static(route, 1133, "within-branch-reverse-slope;junction-reverse-slope")

    mask = route["lia_unique_id"].astype(int).eq(3075)
    route.loc[mask, "synthetic_route_attempted"] = True
    route.loc[mask, "current_route"] = "synthetic-standard-chain-steady-and-endpoint-order-closed"
    route.loc[mask, "current_gate"] = "steady-growth-domain-fail-and-endpoint-order-impossible"
    route.loc[mask, "latest_evidence_stage"] = "steady-50y-parent-mindt30"
    route.loc[mask, "latest_evidence_status"] = "steady-50y-fail"
    route.loc[mask, "latest_flags"] = (
        "geometry/formal inversion/5y zeroMB pass; dV50=+21.795%; LIA length error=27.482%; "
        "free tail=4100m; L_RGI>L_LIA makes peak/final separation intervals disjoint"
    )
    route.loc[mask, "latest_evidence"] = (
        "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline3075_lowrisk_v1/"
        "steady_parent_dt0p25_v1/steady_50y_summary.json"
    )
    route.loc[mask, "controller_state"] = "closed-current-branch"
    route.loc[mask, "mutually_exclusive_class"] = "closed-current-branch"
    route.loc[mask, "mechanism_class"] = (
        "invalid-polygon-topology-repair-standard-inversion-zeroMB-pass-steady-growth-endpoint-order-impossible"
    )
    route.loc[mask, "next_legal_action"] = (
        "requires a new observation operator/target contract or a genuinely new non-scalar state mechanism; "
        "do not add delta-T points under the current peak/final operator"
    )
    route.loc[mask, "update_source"] = "invalid-outline3075-lowrisk-full-gated-chain-v1"

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
        "supersedes": "lia_191_current_route_v10_24",
        "new_decisions": {
            "lowrisk6_static": "all six topology repairs pass; only system3075 passes unchanged standard geometry",
            "system3075": "formal inversion and 30s 5y zero-MB pass, but 50y parent strongly grows and the frozen peak/final endpoint order is mathematically impossible",
            "static_closed": [5014, 2535, 5231, 1653, 1133],
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
