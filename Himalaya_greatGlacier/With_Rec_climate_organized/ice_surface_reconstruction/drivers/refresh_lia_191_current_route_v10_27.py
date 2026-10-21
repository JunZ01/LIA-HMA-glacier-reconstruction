"""Close the final six holds by the frozen-MB/hypsometry reachability bound."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_26"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_27"
BOUND = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux6_hypsometry_mb_reachability_v1"


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    bound = json.loads((BOUND / "summary.json").read_text(encoding="utf-8"))
    if bound["status"] != "all-six-initial-exact-steady-excluded": raise RuntimeError("reachability-bound-not-pass")
    detail = pd.read_csv(BOUND / "systems.csv").set_index("system_id")
    OUTPUT.mkdir(parents=True)
    for path in SOURCE.glob("*.csv"): shutil.copy2(path, OUTPUT / path.name)
    route_path = OUTPUT / "lia_191_current_route.csv"; route = pd.read_csv(route_path)
    for system_id in bound["zero_excluded_system_ids"]:
        row = detail.loc[int(system_id)]; mask = route["lia_unique_id"].astype(int).eq(int(system_id))
        interval = f"[{row.reachable_mean_mb_lower_m_ice:+.6f},{row.reachable_mean_mb_upper_m_ice:+.6f}]m ice yr-1"
        route.loc[mask, "current_route"] = "negative-flux-exact-steady-hypsometry-bound-closed"
        route.loc[mask, "current_gate"] = "zero-net-MB-excluded-within-frozen-2pct-hypsometry-gate"
        route.loc[mask, "latest_evidence_stage"] = "bounded-variation-hypsometry-mb-reachability"
        route.loc[mask, "latest_evidence_status"] = "initial-exact-steady-mathematically-excluded"
        route.loc[mask, "latest_flags"] = f"reachable mean annual MB interval {interval} excludes zero"
        route.loc[mask, "latest_evidence"] = "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux6_hypsometry_mb_reachability_v1/summary.json"
        route.loc[mask, "controller_state"] = "closed-current-branch"
        route.loc[mask, "mutually_exclusive_class"] = "closed-current-branch"
        route.loc[mask, "mechanism_class"] = "negative-flux-paired-state-exact-steady-impossible-under-frozen-MB-and-2pct-CDF"
        route.loc[mask, "next_legal_action"] = "reopening requires an explicit changed climate/MB contract, relaxed hypsometry gate, or non-steady LIA assumption; representation-only retries are closed"
        route.loc[mask, "update_source"] = "negative-flux6-hypsometry-mb-reachability-v1"
    route.to_csv(route_path, index=False)
    accepted = route["accepted"].astype(str).str.lower().eq("true")
    route.loc[~accepted].to_csv(OUTPUT / "remaining_137_current_route.csv", index=False)
    summary = {
        "status": "current-route-snapshot-complete", "population": int(len(route)), "accepted": int(accepted.sum()), "remaining": int((~accepted).sum()),
        "controller_state_counts": route["controller_state"].value_counts().to_dict(),
        "mutually_exclusive_class_counts": route["mutually_exclusive_class"].value_counts().to_dict(),
        "active_systems": [], "active_process_count": 0, "queue_rows": 0, "actionable_now": 0, "launchable_now": 0,
        "supersedes": "lia_191_current_route_v10_26",
        "new_decisions": {"negative_flux_final_six": "all six exact initial zero-net-MB states are excluded within the frozen 2% elevation-CDF gate; representation-only retries closed"},
        "driver": str(Path(__file__).resolve()),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
