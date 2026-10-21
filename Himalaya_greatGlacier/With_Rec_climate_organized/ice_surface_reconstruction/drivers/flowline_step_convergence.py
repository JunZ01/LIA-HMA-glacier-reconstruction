"""Flowline step-length convergence for calibrated LIA network failures.

The test refines every branch of the existing network by piecewise-linear
interpolation.  It changes neither topology nor the bed polyline and is not a
smoothing or repair operation.  Production members remain untouched.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import (
    refine_flowline_branch,
    refine_flowline_network,
    solve_flowline_network,
)
from ..paths import resolve
from .diagnose_lia_failures import _build_context, _scenario, replay_branch
from .lia_centerline_pilot import _configure_oggm


REFINEMENT_FACTORS = (1, 2, 4, 8)


refine_branch = refine_flowline_branch
refine_network = refine_flowline_network


def _network_diagnostic(build, tau_kpa: float, factor: int) -> dict:
    branches = refine_network(build.branches, factor)
    scenario = _scenario(tau_kpa)
    result = solve_flowline_network(
        branches, scenario, outlet_node_ids=build.outlet_node_ids
    )
    maximum_step = 0.0
    maximum_slope = 0.0
    for branch in branches:
        dx = np.diff(np.asarray(branch.distance_m, dtype=float))
        dz = np.diff(np.asarray(branch.bed_elevation_m, dtype=float))
        maximum_step = max(maximum_step, float(np.max(np.abs(dz))))
        maximum_slope = max(
            maximum_slope,
            float(np.max(np.degrees(np.arctan2(np.abs(dz), dx)))),
        )
    record = {
        "refinement_factor": factor,
        "status": result.status,
        "network_success": bool(result.success),
        "maximum_network_segment_bed_step_m": maximum_step,
        "maximum_network_segment_slope_deg": maximum_slope,
        "failed_branch_id": str(result.diagnostics.get("failed_branch_id", "")),
        "failed_index": result.diagnostics.get("failed_index", np.nan),
        "failed_distance_m": np.nan,
        "failed_segment_dx_m": np.nan,
        "failed_bed_step_m": np.nan,
        "failed_segment_slope_deg": np.nan,
        "failed_thickness_m": np.nan,
        "failure_discriminant": np.nan,
    }
    if result.success:
        record["maximum_solution_thickness_m"] = float(max(
            np.max(solution.thickness_m)
            for solution in result.branch_solutions.values()
        ))
        return record
    branch_id = record["failed_branch_id"]
    if not branch_id:
        return record
    branch = next(item for item in branches if item.branch_id == branch_id)
    inherited = result.node_surface_m.get(
        branch.downstream_node_id, float(branch.bed_elevation_m[0])
    )
    replay, _, failed_index = replay_branch(branch, inherited, scenario)
    if failed_index is None:
        return record
    failed = replay.iloc[-1]
    record.update({
        "failed_index": int(failed_index),
        "failed_distance_m": float(failed.distance_m),
        "failed_segment_dx_m": (
            float(branch.distance_m[failed_index] - branch.distance_m[failed_index - 1])
            if failed_index > 0 else 0.0
        ),
        "failed_bed_step_m": float(failed.bed_step_m) if np.isfinite(failed.bed_step_m) else np.nan,
        "failed_segment_slope_deg": (
            float(np.degrees(np.arctan2(abs(failed.bed_step_m), branch.distance_m[failed_index] - branch.distance_m[failed_index - 1])))
            if failed_index > 0 and np.isfinite(failed.bed_step_m) else np.nan
        ),
        "failed_thickness_m": float(failed.thickness_m) if np.isfinite(failed.thickness_m) else np.nan,
        "failure_discriminant": float(failed.discriminant) if np.isfinite(failed.discriminant) else np.nan,
    })
    return record


def _classify_member(group: pd.DataFrame) -> dict:
    by_factor = group.set_index("refinement_factor")
    statuses = [bool(by_factor.loc[factor, "network_success"]) for factor in REFINEMENT_FACTORS]
    branches = [str(by_factor.loc[factor, "failed_branch_id"]) for factor in REFINEMENT_FACTORS]
    distances = [float(by_factor.loc[factor, "failed_distance_m"]) for factor in REFINEMENT_FACTORS]
    if statuses[0] and all(statuses):
        classification = "success-at-all-resolutions"
    elif not statuses[0] and all(statuses[1:]):
        classification = "resolution-sensitive-disappears-at-half-step"
    elif not statuses[0] and not statuses[1] and all(statuses[2:]):
        classification = "resolution-sensitive-disappears-at-quarter-step"
    elif not statuses[0] and not statuses[1] and not statuses[2] and statuses[3]:
        classification = "resolution-sensitive-disappears-at-eighth-step"
    elif len(set(statuses)) > 1 and not statuses[-1]:
        classification = "non-monotonic-resolution-response"
    elif not any(statuses):
        finite = [value for value in distances if np.isfinite(value)]
        same_branch = len(set(branches)) == 1
        distance_spread = float(np.ptp(finite)) if finite else np.nan
        classification = (
            "resolution-robust-same-branch"
            if same_branch else "resolution-robust-failure-migrates"
        )
    else:
        classification = "mixed-resolution-response"
    return {
        "convergence_class": classification,
        "current_success": statuses[0],
        "half_step_success": statuses[1],
        "quarter_step_success": statuses[2],
        "eighth_step_success": statuses[3],
        "failed_branch_stable": len(set(branches)) == 1,
        "failed_distance_spread_m": (
            float(np.ptp([value for value in distances if np.isfinite(value)]))
            if any(np.isfinite(distances)) else np.nan
        ),
    }


def _original_node_values(build, result, factor: int) -> tuple[np.ndarray, np.ndarray]:
    surfaces: list[np.ndarray] = []
    thicknesses: list[np.ndarray] = []
    for original in build.branches:
        solution = result.branch_solutions[original.branch_id]
        indices = np.arange(0, len(solution.surface_elevation_m), factor, dtype=int)
        if indices[-1] != len(solution.surface_elevation_m) - 1:
            indices = np.append(indices, len(solution.surface_elevation_m) - 1)
        if len(indices) != len(original.distance_m):
            raise ValueError(f"original-node mapping mismatch: {original.branch_id}")
        surfaces.append(np.asarray(solution.surface_elevation_m)[indices])
        thicknesses.append(np.asarray(solution.thickness_m)[indices])
    return np.concatenate(surfaces), np.concatenate(thicknesses)


def _successful_baseline_convergence(
    config: dict,
    root: Path,
    output_dir: Path,
    temp_root: Path,
    cache: dict,
) -> pd.DataFrame:
    pilot_path = (
        root / str(config["lia_3d_zero_boundary_pilot"]["output_subdir"])
        / "lia_3d_calibrated_results_v0.csv"
    )
    pilot = pd.read_csv(pilot_path)
    success = pilot[
        (pilot["head_exclusion_distance_m"] == 100.0)
        & pilot["status"].astype(str).eq("success")
    ]
    rows: list[dict] = []
    for item in success.itertuples(index=False):
        key = (str(item.rgi_id), int(item.lia_unique_id), str(item.bed_scenario))
        if key not in cache:
            cache[key] = _build_context(
                config, key[0], key[1], key[2],
                Path(tempfile.mkdtemp(prefix=f"{key[0]}_{key[2]}_", dir=temp_root)),
            )
        build = cache[key][-1]
        scenario = _scenario(float(item.basal_shear_stress_kpa))
        solutions = {
            factor: solve_flowline_network(
                refine_network(build.branches, factor),
                scenario,
                outlet_node_ids=build.outlet_node_ids,
            )
            for factor in REFINEMENT_FACTORS
        }
        values = {
            factor: _original_node_values(build, result, factor)
            for factor, result in solutions.items() if result.success
        }
        reference = values.get(8)
        baseline = values.get(1)
        for factor in REFINEMENT_FACTORS:
            result = solutions[factor]
            record = {
                "member_id": str(item.member_id),
                "rgi_id": key[0],
                "bed_scenario": key[2],
                "refinement_factor": factor,
                "status": result.status,
                "surface_rmse_vs_1x_m": np.nan,
                "surface_max_abs_vs_1x_m": np.nan,
                "surface_rmse_vs_8x_m": np.nan,
                "surface_max_abs_vs_8x_m": np.nan,
                "thickness_rmse_vs_8x_m": np.nan,
            }
            if result.success and factor in values:
                surface, thickness = values[factor]
                if baseline is not None:
                    diff = surface - baseline[0]
                    record["surface_rmse_vs_1x_m"] = float(np.sqrt(np.mean(diff ** 2)))
                    record["surface_max_abs_vs_1x_m"] = float(np.max(np.abs(diff)))
                if reference is not None:
                    diff = surface - reference[0]
                    hdiff = thickness - reference[1]
                    record["surface_rmse_vs_8x_m"] = float(np.sqrt(np.mean(diff ** 2)))
                    record["surface_max_abs_vs_8x_m"] = float(np.max(np.abs(diff)))
                    record["thickness_rmse_vs_8x_m"] = float(np.sqrt(np.mean(hdiff ** 2)))
            rows.append(record)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "successful_member_step_convergence_v1.csv", index=False)
    return frame


def run(config: dict) -> dict:
    _configure_oggm(config)
    root = resolve(config["output_dir"])
    v0 = pd.read_csv(root / "lia_failure_diagnostics_v0" / "lia_failure_diagnostics_v0.csv")
    output_dir = root / "lia_failure_diagnostics_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root = output_dir / "_step_convergence_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    rows: list[dict] = []
    for item in v0.itertuples(index=False):
        key = (str(item.rgi_id), int(item.lia_unique_id), str(item.bed_scenario))
        if key not in cache:
            cache[key] = _build_context(
                config, key[0], key[1], key[2],
                Path(tempfile.mkdtemp(prefix=f"{key[0]}_{key[2]}_", dir=temp_root)),
            )
        build = cache[key][-1]
        for factor in REFINEMENT_FACTORS:
            rows.append({
                "member_id": f"{key[0]}__{key[2]}",
                "rgi_id": key[0],
                "lia_unique_id": key[1],
                "bed_scenario": key[2],
                "tau_kpa": float(item.tau_kpa),
                "original_diagnostic_class": str(item.diagnostic_class),
                **_network_diagnostic(build, float(item.tau_kpa), factor),
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "flowline_step_convergence_detail_v1.csv", index=False)
    summary_rows = []
    for member_id, group in detail.groupby("member_id", sort=True):
        first = group.iloc[0]
        summary_rows.append({
            "member_id": member_id,
            "rgi_id": first.rgi_id,
            "bed_scenario": first.bed_scenario,
            "original_diagnostic_class": first.original_diagnostic_class,
            **_classify_member(group),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "flowline_step_convergence_summary_v1.csv", index=False)
    successful = _successful_baseline_convergence(
        config, root, output_dir, temp_root, cache
    )
    quarter = successful[successful["refinement_factor"] == 4]
    payload = {
        "status": "diagnostic-only-no-production-change",
        "refinement_factors": list(REFINEMENT_FACTORS),
        "interpolation": "piecewise-linear-through-all-original-bed-nodes",
        "smoothing_applied": False,
        "member_count": int(len(summary)),
        "convergence_class_counts": summary.convergence_class.value_counts().to_dict(),
        "successful_baseline_member_count": int(successful.member_id.nunique()),
        "successful_baseline_quarter_vs_eighth_surface_rmse_median_m": (
            float(quarter.surface_rmse_vs_8x_m.median()) if not quarter.empty else np.nan
        ),
        "successful_baseline_quarter_vs_eighth_surface_max_abs_max_m": (
            float(quarter.surface_max_abs_vs_8x_m.max()) if not quarter.empty else np.nan
        ),
        "production_fields_unchanged": True,
    }
    (output_dir / "flowline_step_convergence_summary_v1.json").write_text(
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
