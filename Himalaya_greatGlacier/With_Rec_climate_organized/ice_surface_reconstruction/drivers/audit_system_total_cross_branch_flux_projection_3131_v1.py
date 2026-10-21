"""Read-only system-total-conserving cross-branch flux projection for 3131."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment") != "system-total-conserving-cross-branch-flux-projection-3131-v1":
        raise ValueError("unexpected-experiment")
    if int(config.get("system_id", -1)) != 3131:
        raise ValueError("only-system-3131-is-authorized")
    output_root = resolve(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"versioned-output-exists:{output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    nodes = pd.read_csv(resolve(config["source_nodes"]))
    budget = json.loads(resolve(config["source_area_budget"]).read_text(encoding="utf-8"))
    groups = {
        int(branch): frame.sort_values("node_index").copy()
        for branch, frame in nodes.groupby("branch_index", sort=True)
    }
    outlets = {branch: float(frame["apparent_flux_native"].iloc[-1]) for branch, frame in groups.items()}
    negative = {branch: value for branch, value in outlets.items() if value < 0.0}
    positive = {branch: value for branch, value in outlets.items() if value > 0.0}
    deficit = -sum(negative.values())
    capacity = sum(positive.values())
    capacity_pass = capacity + 1e-12 >= deficit
    system_scale = max(max(abs(value) for value in nodes["apparent_flux_native"]), 1.0)
    floor = system_scale * float(config["activation_floor_relative_to_system_flux_scale"])

    donor_deductions = {
        branch: (deficit * value / capacity if capacity > 0.0 else np.nan)
        for branch, value in positive.items()
    }
    projected_frames: list[pd.DataFrame] = []
    branch_rows: list[dict] = []
    for branch, frame in groups.items():
        original = frame["apparent_flux_native"].to_numpy(float)
        correction = np.zeros_like(original)
        role = "unchanged-zero-outlet"
        target_outlet = outlets[branch]
        if outlets[branch] < 0.0:
            role = "negative-outlet-receiver"
            target_outlet = 0.0
            lower = np.where(original < 0.0, floor - original, 0.0)
            correction = np.maximum.accumulate(lower)
            correction[-1] = -outlets[branch]
        elif outlets[branch] > 0.0:
            role = "positive-outlet-donor"
            deduction = donor_deductions[branch]
            target_outlet = outlets[branch] - deduction
            # Minimal monotone endpoint correction: the nonnegative deduction
            # magnitude is zero upstream and reaches its prescribed endpoint.
            correction[-1] = -deduction

        projected = original + correction
        original_increment = np.diff(np.r_[0.0, original])
        projected_increment = np.diff(np.r_[0.0, projected])
        surface = frame["surface_h_m"].to_numpy(float)
        reverse_count = int(np.count_nonzero(np.diff(surface) > 1e-6))
        frame["projected_apparent_flux_native"] = projected
        frame["cross_branch_correction_native"] = correction
        frame["original_local_increment_native"] = original_increment
        frame["projected_local_increment_native"] = projected_increment
        frame["projection_role"] = role
        frame["projected_positive_flux"] = projected > 0.0
        projected_frames.append(frame)
        branch_rows.append(
            {
                "system_id": 3131,
                "branch_index": branch,
                "role": role,
                "node_count": int(len(frame)),
                "original_outlet_flux_native": outlets[branch],
                "target_outlet_flux_native": target_outlet,
                "projected_outlet_flux_native": float(projected[-1]),
                "outlet_target_error": abs(float(projected[-1] - target_outlet)),
                "minimum_projected_flux_native": float(projected.min()),
                "nonnegative_projected_flux": bool(np.all(projected >= 0.0)),
                "correction_monotone_in_required_direction": bool(
                    np.all(np.diff(correction) >= -1e-12)
                    if role == "negative-outlet-receiver"
                    else np.all(np.diff(-correction) >= -1e-12)
                ),
                "local_increment_sum_original": float(original_increment.sum()),
                "local_increment_sum_projected": float(projected_increment.sum()),
                "surface_reverse_slope_count_before": reverse_count,
                "surface_reverse_slope_count_after": reverse_count,
                "surface_or_bed_modified": False,
            }
        )

    projected_nodes = pd.concat(projected_frames, ignore_index=True)
    branches = pd.DataFrame(branch_rows).sort_values("branch_index")
    original_total = float(sum(outlets.values()))
    projected_total = float(branches["projected_outlet_flux_native"].sum())
    total_error = abs(projected_total - original_total)
    projected_active_area = float(
        projected_nodes.loc[projected_nodes["projected_positive_flux"], "node_plan_area_m2"].sum()
    )
    outline_area = float(budget["outline_area_m2"])
    area_error = abs(projected_active_area - outline_area) / outline_area
    tolerance = float(config["maximum_projected_active_area_relative_error"])
    total_tolerance = float(config["maximum_system_outlet_flux_absolute_error"])
    feasible = bool(
        capacity_pass
        and branches["nonnegative_projected_flux"].all()
        and branches["correction_monotone_in_required_direction"].all()
        and total_error <= total_tolerance
        and area_error <= tolerance
        and int(branches["surface_reverse_slope_count_after"].sum()) == 0
    )
    summary = {
        "status": "read-only-feasibility-audit-complete",
        "experiment": config["experiment"],
        "system_id": 3131,
        "mathematically_feasible": feasible,
        "negative_outlet_branches": negative,
        "positive_outlet_branches": positive,
        "negative_outlet_deficit_native": deficit,
        "positive_outlet_capacity_native": capacity,
        "positive_capacity_margin_native": capacity - deficit,
        "positive_capacity_pass": capacity_pass,
        "donor_deductions_native": donor_deductions,
        "original_system_total_branch_outlet_flux_native": original_total,
        "projected_system_total_branch_outlet_flux_native": projected_total,
        "system_total_flux_absolute_error": total_error,
        "original_active_area_m2": float(budget["model_active_source_area_m2"]),
        "projected_active_area_m2": projected_active_area,
        "outline_area_m2": outline_area,
        "projected_active_area_relative_error": area_error,
        "area_gate_tolerance": tolerance,
        "area_gate_pass": bool(area_error <= tolerance),
        "minimum_projected_node_flux_native": float(projected_nodes["projected_apparent_flux_native"].min()),
        "all_node_flux_nonnegative": bool((projected_nodes["projected_apparent_flux_native"] >= 0.0).all()),
        "surface_reverse_slope_count": int(branches["surface_reverse_slope_count_after"].sum()),
        "outline_width_bed_surface_climate_stage2_modified": False,
        "oggm_run": False,
        "decision": (
            "register-one-3131-standard-inversion-static-zeroMB-pilot"
            if feasible
            else "close-system-total-cross-branch-projection-mechanism"
        ),
        "config": str(config_path.resolve()),
    }
    projected_nodes.to_csv(output_root / "system_3131_node_projection.csv", index=False)
    branches.to_csv(output_root / "system_3131_branch_projection.csv", index=False)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
