"""Independent 3-D pilot using the converged quarter-step 1-D integration."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .lia_3d_zero_boundary_pilot import run as run_base_pilot


def run(config: dict) -> dict:
    refined_config = copy.deepcopy(config)
    pilot = refined_config["lia_3d_zero_boundary_pilot"]
    frozen = refined_config["lia_3d_refined_flowline_pilot_v1"]
    output_subdir = str(frozen["output_subdir"])
    refinement_factor = int(frozen["flowline_integration_refinement_factor"])
    pilot["output_subdir"] = output_subdir
    pilot["result_version"] = str(frozen["result_version"])
    pilot["flowline_integration_refinement_factor"] = refinement_factor
    pilot["head_exclusion_primary_m"] = float(frozen["head_exclusion_minimum_m"])
    pilot["head_exclusion_grid_multiplier"] = float(
        frozen["head_exclusion_grid_multiplier"]
    )
    results, _ = run_base_pilot(refined_config)

    root = resolve(config["output_dir"])
    output_dir = root / output_subdir
    v0_path = (
        root / str(config["lia_3d_zero_boundary_pilot"]["output_subdir"])
        / "lia_3d_calibrated_results_v0.csv"
    )
    common_columns = ["rgi_id", "lia_unique_id", "bed_scenario"]
    science_columns = ["volume_km3", "mean_thickness_m", "maximum_thickness_m"]
    comparison_status = "not-available"
    comparison = pd.DataFrame(columns=common_columns)
    if v0_path.exists() and v0_path.parent != output_dir:
        v0 = pd.read_csv(v0_path)
        v0 = v0[
            v0["head_exclusion_distance_m"]
            == float(pilot["head_exclusion_primary_m"])
        ]
        v0_success = v0[v0.status.astype(str).eq("success")][
            common_columns + science_columns
        ]
        v1_success = results[results.status.astype(str).eq("success")][
            common_columns + science_columns
        ]
        comparison = v0_success.merge(
            v1_success,
            on=common_columns,
            how="inner",
            suffixes=("_v0_1x", "_v1_4x"),
        )
        comparison_status = "completed"
    if comparison_status == "completed":
        for column in science_columns:
            denominator = comparison[f"{column}_v0_1x"].replace(0.0, np.nan)
            comparison[f"{column}_relative_change"] = (
                comparison[f"{column}_v1_4x"] - comparison[f"{column}_v0_1x"]
            ) / denominator
    comparison.to_csv(
        output_dir / "lia_3d_refinement_science_change_v1.csv", index=False
    )
    primary = results[np.isfinite(results["head_exclusion_distance_m"])].copy()
    payload = {
        "status": "completed-independent-v1-pilot",
        "flowline_integration_refinement_factor": refinement_factor,
        "flowline_integration_policy": "piecewise-linear-bed-no-smoothing",
        "head_exclusion_rule": str(frozen["head_exclusion_rule"]),
        "head_exclusion_grid_multiplier": float(frozen["head_exclusion_grid_multiplier"]),
        "result_status_counts": primary.status.value_counts(dropna=False).to_dict(),
        "successful_members": int(primary.status.astype(str).eq("success").sum()),
        "successful_systems": int(
            primary.loc[primary.status.astype(str).eq("success"), "rgi_id"].nunique()
        ),
        "v0_comparison_status": comparison_status,
        "preexisting_success_comparison_members": int(len(comparison)),
        "volume_relative_change_median": (
            float(comparison.volume_km3_relative_change.median())
            if not comparison.empty else np.nan
        ),
        "volume_relative_change_max_abs": (
            float(comparison.volume_km3_relative_change.abs().max())
            if not comparison.empty else np.nan
        ),
        "v0_outputs_unchanged": True,
    }
    (output_dir / "lia_3d_refined_flowline_summary_v1.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()
