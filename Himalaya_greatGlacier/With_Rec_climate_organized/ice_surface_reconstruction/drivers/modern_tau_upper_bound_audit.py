"""Read-only extended tau audit for upper-bound modern calibrations.

This diagnostic preserves the production calibration objective and exclusions,
but evaluates selected bed members on an isolated, wider tau interval.  It does
not modify the supplied config or the existing calibration products.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .modern_tau_calibration import calibrate_system


DEFAULT_RGI_IDS = (
    "RGI60-15.01152",
    "RGI60-15.04415",
    "RGI60-15.07173",
)
BED_SCENARIO = "millan2022"
AUDIT_SUBDIR = "modern_tau_upper_bound_audit_v1"


def _audit_config(config: dict, lower_kpa: float, upper_kpa: float, step_kpa: float) -> dict:
    """Return an in-memory calibration config for the extended scan."""
    if not lower_kpa < upper_kpa:
        raise ValueError("lower_kpa must be less than upper_kpa")
    if step_kpa <= 0:
        raise ValueError("step_kpa must be positive")
    values = np.arange(lower_kpa, upper_kpa + 0.5 * step_kpa, step_kpa)
    values = values[values <= upper_kpa + 1e-9]
    if values.size == 0 or not np.isclose(values[-1], upper_kpa):
        values = np.append(values, upper_kpa)

    audit = deepcopy(config)
    calibration = audit["modern_tau_calibration"]
    calibration["tau_bounds_kpa"] = [float(lower_kpa), float(upper_kpa)]
    calibration["coarse_grid_kpa"] = [float(value) for value in values]
    calibration["baseline_tau_kpa"] = float(lower_kpa)
    return audit


def _route(status: str, touches_upper: bool) -> str:
    if touches_upper or status == "identifiability-upper-bound":
        return "hold-from-production-extend-upper-bound-again"
    if status == "identified-interior":
        return "use-extended-interior-optimum-after-production-config-review"
    if status == "weak-identifiability":
        return "retain-member-with-extended-near-optimal-uncertainty"
    return "calibration-review-required"


def run(
    config: dict,
    rgi_ids: tuple[str, ...] = DEFAULT_RGI_IDS,
    lower_kpa: float = 180.0,
    upper_kpa: float = 260.0,
    step_kpa: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    audit_config = _audit_config(config, lower_kpa, upper_kpa, step_kpa)
    output_dir = resolve(config["output_dir"]) / AUDIT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    production_dir = (
        resolve(config["output_dir"])
        / str(config["modern_tau_calibration"]["output_subdir"])
    )
    production = pd.read_csv(production_dir / "modern_tau_calibration_v0.csv")

    summaries: list[dict] = []
    evaluations: list[dict] = []
    for rgi_id in rgi_ids:
        summary, rows = calibrate_system(audit_config, rgi_id, BED_SCENARIO)
        original = production[
            (production["rgi_id"].astype(str) == rgi_id)
            & (production["bed_scenario"].astype(str) == BED_SCENARIO)
        ]
        if len(original) != 1:
            raise ValueError(f"Expected one production calibration row for {rgi_id}")
        original_row = original.iloc[0]
        current_loss = float(summary["baseline_loss"])
        current_rmse = float(summary["baseline_thickness_rmse_m"])
        best_loss = float(summary["loss"])
        best_rmse = float(summary["thickness_rmse_m"])
        near_upper = float(summary["near_optimal_tau_max_kpa"])
        touches_upper = bool(near_upper >= upper_kpa - 1e-6)
        baseline_matches = bool(
            np.isclose(current_loss, float(original_row["loss"]), rtol=0.0, atol=1e-12)
        )
        summaries.append({
            "rgi_id": rgi_id,
            "bed_scenario": BED_SCENARIO,
            "production_status": str(original_row["status"]),
            "production_tau_kpa": float(original_row["tau_kpa"]),
            "production_loss": float(original_row["loss"]),
            "production_thickness_rmse_m": float(original_row["thickness_rmse_m"]),
            "audit_status": str(summary["status"]),
            "audit_best_tau_kpa": float(summary["tau_kpa"]),
            "audit_best_loss": best_loss,
            "audit_best_thickness_rmse_m": best_rmse,
            "audit_near_optimal_tau_min_kpa": float(summary["near_optimal_tau_min_kpa"]),
            "audit_near_optimal_tau_max_kpa": near_upper,
            "audit_near_optimal_touches_upper_bound": touches_upper,
            "loss_improvement_vs_180_fraction": (current_loss - best_loss) / current_loss,
            "rmse_improvement_vs_180_m": current_rmse - best_rmse,
            "rmse_improvement_vs_180_fraction": (current_rmse - best_rmse) / current_rmse,
            "usable_point_count": int(summary["usable_point_count"]),
            "production_180_replay_exact": baseline_matches,
            "production_route_recommendation": _route(str(summary["status"]), touches_upper),
        })
        evaluations.extend(
            {"rgi_id": rgi_id, "bed_scenario": BED_SCENARIO, **row} for row in rows
        )

    summary_df = pd.DataFrame(summaries)
    evaluation_df = pd.DataFrame(evaluations).sort_values(["rgi_id", "tau_kpa"])
    summary_df.to_csv(output_dir / "modern_tau_upper_bound_audit_v1.csv", index=False)
    evaluation_df.to_csv(output_dir / "modern_tau_upper_bound_loss_curves_v1.csv", index=False)
    payload = {
        "status": "completed",
        "scope": "diagnostic-only-no-production-config-or-output-modification",
        "bed_scenario": BED_SCENARIO,
        "rgi_ids": list(rgi_ids),
        "scan_bounds_kpa": [float(lower_kpa), float(upper_kpa)],
        "coarse_step_kpa": float(step_kpa),
        "objective": "unchanged-interior-modern-inversion-thickness-rmse-normalized-by-median",
        "boundary_and_minimum_thickness_exclusions": "inherited-unchanged-from-batch2-config",
        "status_counts": summary_df["audit_status"].value_counts().to_dict(),
        "near_optimal_upper_bound_count": int(
            summary_df["audit_near_optimal_touches_upper_bound"].sum()
        ),
        "production_180_replay_all_exact": bool(summary_df["production_180_replay_exact"].all()),
    }
    (output_dir / "modern_tau_upper_bound_audit_summary_v1.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return summary_df, evaluation_df, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rgi-id", action="append", dest="rgi_ids")
    parser.add_argument("--lower-kpa", type=float, default=180.0)
    parser.add_argument("--upper-kpa", type=float, default=260.0)
    parser.add_argument("--step-kpa", type=float, default=5.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ids = tuple(args.rgi_ids) if args.rgi_ids else DEFAULT_RGI_IDS
    summary, _, _ = run(config, ids, args.lower_kpa, args.upper_kpa, args.step_kpa)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
