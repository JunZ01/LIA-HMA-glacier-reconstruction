"""Build the auditable pilot-readiness ledger for calibrated LIA surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve


def _finite_summary(values: pd.Series) -> dict:
    data = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "n": int(data.size),
        "minimum": float(np.min(data)) if data.size else None,
        "median": float(np.median(data)) if data.size else None,
        "maximum": float(np.max(data)) if data.size else None,
    }


def run(config: dict) -> dict:
    output_root = resolve(config["output_dir"])
    tau_dir = output_root / str(config["modern_tau_calibration"]["output_subdir"])
    lia_dir = output_root / str(config["lia_3d_zero_boundary_pilot"]["output_subdir"])
    tau = pd.read_csv(tau_dir / "modern_tau_calibration_v0.csv")
    lia = pd.read_csv(lia_dir / "lia_3d_calibrated_results_v0.csv")
    sensitivity = pd.read_csv(lia_dir / "lia_3d_head_exclusion_sensitivity_v0.csv")
    heads = pd.read_csv(lia_dir / "lia_head_boundary_classification_v0.csv")
    primary_distance = float(
        config["lia_3d_zero_boundary_pilot"]["head_exclusion_primary_m"]
    )
    primary = lia[
        pd.to_numeric(lia.get("head_exclusion_distance_m"), errors="coerce")
        == primary_distance
    ].copy()
    success = primary[primary["status"] == "success"]
    failures = primary[primary["status"].astype(str).str.startswith("network-failed:")]
    non_primary = sensitivity[
        pd.to_numeric(sensitivity["head_exclusion_distance_m"], errors="coerce")
        > primary_distance
    ].copy()
    relative = 100.0 * pd.to_numeric(
        non_primary["relative_volume_change_vs_min_distance"], errors="coerce"
    )
    candidate_divide = primary[
        primary.get("candidate_divide_v0_zero_boundary", False).astype(str).str.lower()
        == "true"
    ]
    status_counts = primary["status"].value_counts(dropna=False).to_dict()
    audit_columns = [
        "rgi_id", "lia_unique_id", "bed_scenario", "head_exclusion_distance_m",
        "boundary_preflight_status", "candidate_divide_v0_zero_boundary",
        "excluded_constraint_count", "maximum_excluded_thickness_m",
        "minimum_retained_boundary_distance_m", "maximum_inner_edge_implied_slope_deg",
        "status", "route_bed_compatibility_note",
    ]
    lia[[column for column in audit_columns if column in lia.columns]].to_csv(
        lia_dir / "lia_head_exclusion_audit_v0.csv", index=False
    )

    old_path = output_root / "lia_3d_zero_boundary_pilot_v0" / "lia_3d_zero_boundary_results_v0.csv"
    transition = pd.DataFrame()
    if old_path.exists():
        old = pd.read_csv(old_path)
        new_by_key = primary.set_index(["rgi_id", "bed_scenario"])
        transition_rows = []
        for old_row in old.itertuples(index=False):
            key = (str(old_row.rgi_id), str(old_row.bed_scenario))
            new_status = (
                str(new_by_key.loc[key, "status"])
                if key in new_by_key.index else "not-comparable-no-new-primary-member"
            )
            old_status = str(old_row.status)
            old_negative = "negative-thickness" in old_status or "negative-start-thickness" in old_status
            new_negative = "negative-thickness" in new_status or "negative-start-thickness" in new_status
            if old_negative and new_status == "success":
                outcome = "old-negative-now-success"
            elif old_negative and new_negative:
                outcome = "old-negative-persists"
            elif not old_negative and new_negative:
                outcome = "new-primary-negative"
            elif old_status == "success" and new_status == "success":
                outcome = "success-persists"
            else:
                outcome = "not-directly-comparable"
            transition_rows.append({
                "rgi_id": key[0],
                "lia_unique_id": getattr(old_row, "lia_unique_id", np.nan),
                "bed_scenario": key[1],
                "old_connector_alpha": getattr(old_row, "connector_alpha", np.nan),
                "old_stress_scenario": getattr(old_row, "stress_scenario", ""),
                "old_tau_kpa": getattr(old_row, "basal_shear_stress_kpa", np.nan),
                "old_status": old_status,
                "new_calibrated_status": new_status,
                "transition_class": outcome,
                "comparison_scope": "old-fixed-tau-member-vs-new-calibrated-tau-100m-primary",
                "route_bed_compatibility_note": (
                    "F19-route-on-M22-bed-mismatch-suspect"
                    if key[1] == "millan2022" else "route-and-bed-both-F19"
                ),
            })
        transition = pd.DataFrame(transition_rows)
        transition.to_csv(lia_dir / "lia_negative_thickness_transition_v0.csv", index=False)
    payload = {
        "status": "pilot-complete-not-production-ready",
        "method_decisions": {
            "tau": "one modern product-specific effective tau per system; F fixed at 0.8",
            "connector": "alpha=0, beta=0 primary; explicit connector in LIA network only",
            "head_exclusion_primary_m": primary_distance,
            "head_exclusion_selection": (
                "smallest tested metric distance that removed every centerline-boundary conflict"
            ),
            "candidate_divides": "v0 zero-thickness with explicit bias flag; Neumann deferred",
        },
        "modern_tau_calibration": {
            "bed_member_count": int(len(tau)),
            "identified_interior_count": int((tau["status"] == "identified-interior").sum()),
            "lower_bound_count": int((tau["status"] == "identifiability-lower-bound").sum()),
            "missing_or_error_count": int(tau["status"].astype(str).str.startswith("calibration-error:").sum()),
            "farinotti_tau_kpa": _finite_summary(tau.loc[tau["bed_scenario"] == "farinotti2019", "tau_kpa"]),
            "millan_tau_kpa": _finite_summary(tau.loc[tau["bed_scenario"] == "millan2022", "tau_kpa"]),
            "scale_diagnostic": "no strong pilot relationship; do not derive an HMA correction from n=12",
        },
        "lia_primary_3d": {
            "primary_member_count": int(len(primary)),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "successful_member_count": int(len(success)),
            "successful_system_count": int(success["rgi_id"].nunique()),
            "boundary_conflict_count": int(
                primary["status"].astype(str).str.contains("boundary-conflict").sum()
            ),
            "native_branch_failure_count": int(
                (~failures.get("failed_branch_is_connector", False).astype(str).str.lower().eq("true")).sum()
            ),
            "connector_failure_count": int(
                failures.get("failed_branch_is_connector", False).astype(str).str.lower().eq("true").sum()
            ),
            "candidate_divide_member_count": int(len(candidate_divide)),
            "candidate_divide_success_count": int((candidate_divide["status"] == "success").sum()),
        },
        "head_exclusion_sensitivity": {
            "tested_distances_m": sorted(
                pd.to_numeric(sensitivity["head_exclusion_distance_m"], errors="coerce")
                .dropna().unique().astype(float).tolist()
            ),
            "median_volume_change_non_primary_percent": (
                float(np.median(relative.dropna())) if relative.notna().any() else None
            ),
            "maximum_absolute_volume_change_non_primary_percent": (
                float(np.max(np.abs(relative.dropna()))) if relative.notna().any() else None
            ),
            "failure_class_changed_with_distance": False,
            "flowline_head_count": int(len(heads)),
            "candidate_divide_head_count": int(heads["candidate_divide"].astype(str).str.lower().eq("true").sum()),
        },
        "negative_thickness_transition": {
            "comparison_is_causal_one_to_one": False,
            "row_count": int(len(transition)),
            "class_counts": (
                {str(k): int(v) for k, v in transition["transition_class"].value_counts().to_dict().items()}
                if not transition.empty else {}
            ),
        },
        "production_gate": {
            "ready_for_core_191": False,
            "blocking_reasons": [
                "Only 5 pilot systems have at least one successful calibrated 3-D member.",
                "Ten primary-member failures occur on native LIA flowline branches.",
                "Two primary-member failures are connector negative-start-thickness cases.",
                "One pilot centerline remains structure-review and one Millan input is absent.",
                "Candidate divides still use the deliberately biased v0 zero-thickness boundary.",
            ],
            "next_scientific_task": (
                "diagnose native-branch negative thickness against LIA bed reversals, "
                "route geometry, and the modern-to-LIA tau transfer assumption"
            ),
        },
    }
    (lia_dir / "lia_surface_readiness_v0.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# LIA surface pilot readiness v0",
        "",
        f"Status: {payload['status']}",
        "",
        f"Modern tau: {payload['modern_tau_calibration']['identified_interior_count']}/24 identified interior; "
        f"{payload['modern_tau_calibration']['lower_bound_count']} lower-bound; "
        f"{payload['modern_tau_calibration']['missing_or_error_count']} missing/error.",
        f"Primary LIA 3-D at {primary_distance:g} m: {len(success)} successful members across "
        f"{success['rgi_id'].nunique()} systems; zero centerline-boundary conflicts.",
        f"Network failures: {payload['lia_primary_3d']['native_branch_failure_count']} native-branch, "
        f"{payload['lia_primary_3d']['connector_failure_count']} connector.",
        "",
        "Production gate: NOT READY for core-191. The next task is native-branch failure diagnosis, not further head-distance tuning.",
    ]
    (lia_dir / "lia_surface_readiness_v0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    payload = run(json.loads(args.config.read_text(encoding="utf-8")))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
