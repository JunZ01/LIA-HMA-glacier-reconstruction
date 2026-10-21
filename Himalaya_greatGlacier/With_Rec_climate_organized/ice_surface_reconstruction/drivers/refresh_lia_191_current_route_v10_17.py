"""Refresh route after negative-flux representative 2342 structural pilots."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_16"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_17"
NEGATIVE = {890, 988, 1026, 1743, 2342, 2689, 2793}
LATEST = "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2342_decoupled_branch_zero_mb_v1/summary.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    for path in SOURCE.glob("*.csv"):
        shutil.copy2(path, OUTPUT / path.name)
    route_path = OUTPUT / "lia_191_current_route.csv"
    route = pd.read_csv(route_path)
    mask = route["lia_unique_id"].astype(int).isin(NEGATIVE)
    route.loc[mask, "current_gate"] = "representative-2342-decoupled-branch-zero-mb-domain-fail-no-promotion"
    route.loc[mask, "latest_evidence_stage"] = "zero-mb-representative-gate"
    route.loc[mask, "latest_evidence_status"] = "not-run-after-representative-zero-mb-domain-failure"
    route.loc[mask, "latest_flags"] = (
        "2342 branch-ensemble static passed exact A/V, CDF0.781pct and length0.116pct; "
        "5y zero-MB mass drift2.11e-16 but main overflow307.006m, minimum free tail400m, area+4.449pct"
    )
    route.loc[mask, "latest_evidence"] = LATEST
    route.loc[mask, "next_legal_action"] = (
        "do not promote unchanged-core reservoirs or decoupled elevation-band ensemble; "
        "next new family must use current paired F19 spatial-station conservative branch controls"
    )
    route.loc[mask, "update_source"] = "negative-flux2342-structural-pilots-v1"
    route.to_csv(route_path, index=False)
    remaining = route[~route["accepted"].astype(str).str.lower().eq("true")].copy()
    remaining.to_csv(OUTPUT / "remaining_137_current_route.csv", index=False)
    accepted = int(route["accepted"].astype(str).str.lower().eq("true").sum())
    classes = route["mutually_exclusive_class"].value_counts().to_dict()
    states = route["controller_state"].value_counts().to_dict()
    summary = {
        "status": "current-route-snapshot-complete",
        "population": len(route),
        "accepted": accepted,
        "remaining": len(route) - accepted,
        "controller_state_counts": states,
        "mutually_exclusive_class_counts": classes,
        "active_systems": [],
        "active_process_count": 0,
        "queue_rows": 0,
        "actionable_now": 0,
        "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_16",
        "new_decisions": {
            "1026-eb50-diagnostic": "old elevation-band source differs from current paired F19; diagnostic only, not route evidence",
            "2342-unchanged-core-reservoir": "LP optimum CDF12.087pct, static mathematical fail",
            "2342-connected-subnetwork-reservoir": "326 connected subnetworks evaluated; none satisfy exact A/V and CDF<=2pct",
            "2342-decoupled-branch-ensemble": "static pass; 5y zero-MB domain/free-tail fail despite exact mass conservation",
            "2342-dynamic-root-reservoir": "LP optimum CDF17.899pct, static fail",
        },
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
