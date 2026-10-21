"""Classify the 170 unresolved systems by saved formal endpoint reachability."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MASTER = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_controller_v1" / "diagnostics" / "unresolved_master_router_v1"
OUTPUT = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_controller_v1" / "diagnostics" / "unresolved_reachability_audit_v1"


def _passes_endpoint(frame: pd.DataFrame, endpoint: str) -> pd.Series:
    return (
        pd.to_numeric(frame[f"e_{endpoint}_sep"], errors="coerce").le(0.2)
        & pd.to_numeric(frame[f"e_{endpoint}_total"], errors="coerce").le(0.1)
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    queue = pd.read_csv(MASTER / "unresolved_master_queue.csv")
    candidates = pd.read_csv(MASTER / "all_current_valid_candidates.csv")
    master_summary = json.loads((MASTER / "summary.json").read_text(encoding="utf-8"))
    missing_ids = {int(value) for value in master_summary["unresolved_missing_valid_candidate_ids"]}
    references = pd.read_csv(
        ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
        / "lia_191_controller_v1" / "core191_length_references.csv"
    )
    stubs: list[dict] = []
    for system_id in sorted(missing_ids):
        reference = references[references["lia_unique_id"].astype(int).eq(system_id)]
        if len(reference) != 1:
            raise RuntimeError(f"missing unique reference for {system_id}")
        row = reference.iloc[0]
        separation = abs(float(row["lia_registered_length_m"]) - float(row["rgi_registered_length_m"]))
        stubs.append({
            "lia_unique_id": system_id,
            "separation_m": separation,
            "endpoint_class": "no_current_stable_valid_candidate",
        })
    queue = pd.concat([queue, pd.DataFrame(stubs)], ignore_index=True, sort=False)

    # Include the single stable/observation-valid result from the declared
    # state-transition midpoint pilot. The other four midpoint branches are
    # observation-domain overflows and therefore never enter reachability.
    for path in (
        ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_state_transition_midpoint_pilot_v2"
    ).glob("*/scores.csv"):
        extra = pd.read_csv(path)
        keep = (
            extra["summary_status"].eq("ok")
            & extra["spinup_converged"].eq(True)
            & pd.to_numeric(extra["spinup_stability_fraction"], errors="coerce").le(0.02)
            & extra["observation_domain_overflow"].eq(False)
        )
        if keep.any():
            candidates = pd.concat([candidates, extra.loc[keep]], ignore_index=True, sort=False)

    rows: list[dict] = []
    for item in queue.itertuples(index=False):
        system_id = int(item.lia_unique_id)
        group = candidates[pd.to_numeric(candidates["lia_unique_id"], errors="coerce").eq(system_id)].copy()
        if group.empty:
            route = "no_stable_valid_candidate"
            lia_reachable = rgi_reachable = joint = False
            min_lia = min_rgi = min_objective = float("nan")
        else:
            lia_mask = _passes_endpoint(group, "lia")
            rgi_mask = _passes_endpoint(group, "rgi")
            lia_reachable = bool(lia_mask.any())
            rgi_reachable = bool(rgi_mask.any())
            joint = bool((lia_mask & rgi_mask).any())
            if joint:
                route = "joint"
            elif lia_reachable and rgi_reachable:
                route = "both_endpoints_separately"
            elif lia_reachable:
                route = "lia_only"
            elif rgi_reachable:
                route = "rgi_only"
            else:
                route = "neither_endpoint"
            min_lia = pd.to_numeric(group["e_lia_sep"], errors="coerce").min()
            min_rgi = pd.to_numeric(group["e_rgi_sep"], errors="coerce").min()
            min_objective = pd.to_numeric(group["objective_sep"], errors="coerce").min()
        separation = pd.to_numeric(pd.Series([getattr(item, "separation_m", None)]), errors="coerce").iloc[0]
        tolerance_m = 0.2 * separation if pd.notna(separation) else float("nan")
        rows.append({
            "lia_unique_id": system_id,
            "candidate_count": int(len(group)),
            "reachability_route": route,
            "lia_reachable": lia_reachable,
            "rgi_reachable": rgi_reachable,
            "joint_reachable": joint,
            "min_e_lia_sep": min_lia,
            "min_e_rgi_sep": min_rgi,
            "min_objective_sep": min_objective,
            "separation_m": separation,
            "formal_tolerance_m": tolerance_m,
            "tolerance_below_50m_grid": bool(pd.notna(tolerance_m) and tolerance_m < 50.0),
            "best_endpoint_class": getattr(item, "endpoint_class", ""),
        })

    result = pd.DataFrame(rows).sort_values("lia_unique_id")
    if len(result) != 170 or result["lia_unique_id"].nunique() != 170:
        raise RuntimeError("expected exactly 170 unresolved systems")
    result.to_csv(OUTPUT / "unresolved_reachability.csv", index=False)
    counts = result["reachability_route"].value_counts().to_dict()
    summary = {
        "unresolved_systems": 170,
        "reachability_counts": {str(key): int(value) for key, value in counts.items()},
        "tolerance_below_50m_grid_count": int(result["tolerance_below_50m_grid"].sum()),
        "state_transition_midpoint_pilot": {
            "chains": 5,
            "stable_observation_valid": 1,
            "formal_passes": 0,
            "promotion_rule_met": False,
        },
        "decision": (
            "No additional scalar parameter family is population-releasable. "
            "The next experiment must test a declared structural model/geometry mechanism."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
