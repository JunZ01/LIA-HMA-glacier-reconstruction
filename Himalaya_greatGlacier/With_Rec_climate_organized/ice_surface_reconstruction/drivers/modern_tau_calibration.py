"""Automatic modern effective-stress calibration for paleo surface runs.

The calibration estimates one effective ``tau`` per glacier and bed product by
matching the perfect-plasticity solution to modern inversion thickness on the
interior of the observed flowline tree.  It is deliberately separate from
OGGM's ``glen_a``: the result is a geometry-transfer parameter for the paleo
surface initial state, not a climate or ice-flow calibration.

Modern boundary points and explicit connector pixels are excluded from the
objective.  Every failed solve remains in the evaluation table; it is never
converted into a numerical penalty without recording the failure class.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.ndimage import binary_erosion, distance_transform_edt

from ..core import SurfaceScenario, solve_flowline
from ..paths import resolve
from .gridded_bed import load_gridded_bed
from .oggm_flowline_io import load_flowline_inputs
from .oggm_network_adapter import FlowlineInput


_THICK_SUFFIX = {
    "farinotti2019": "f19_fs0_full_v0",
    "millan2022": "m22_fs0_full_v0",
}


def _read_pickle(path: Path):
    with gzip.open(path, "rb") as source:
        return pickle.load(source)


def _real_inputs(
    inputs: list[FlowlineInput], raw_flowlines: list, inversion: list
) -> tuple[list[FlowlineInput], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Attach the modern inversion bed and return truth by native flowline."""
    truth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rebuilt: list[FlowlineInput] = []
    for item in inputs:
        i = item.flowline_index
        surface = np.asarray(raw_flowlines[i].surface_h, dtype=float)
        thickness = np.asarray(inversion[i]["thick"], dtype=float)
        if surface.shape != thickness.shape or not np.isfinite(thickness).all():
            raise ValueError(f"non-finite or mismatched modern inversion on flowline {i}")
        truth[i] = (surface, thickness)
        rebuilt.append(
            FlowlineInput(
                flowline_index=i,
                flows_to=item.flows_to,
                flows_to_indice=item.flows_to_indice,
                bed_elevation_m=surface - thickness,
                reference_surface_m=surface,
                dx_meter=item.dx_meter,
                map_dx_m=item.map_dx_m,
                coords_xy=item.coords_xy,
            )
        )
    return rebuilt, truth


def _interior_truth_points(
    build,
    solution,
    truth: dict[int, tuple[np.ndarray, np.ndarray]],
    modern_mask: np.ndarray,
    map_dx_m: float,
    boundary_exclusion_m: float,
    minimum_thickness_m: float,
    coords_by_flowline: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return model/truth thickness at unique, interior OGGM points."""
    boundary = modern_mask & ~binary_erosion(modern_mask, border_value=0)
    distance_m = distance_transform_edt(~boundary) * map_dx_m
    provenance = {item.branch_id: item for item in build.provenance}
    points: dict[tuple[int, int], tuple[float, float, float]] = {}
    excluded_boundary = 0
    excluded_nonfinite = 0
    for branch_id, branch_solution in solution.branch_solutions.items():
        item = provenance.get(branch_id)
        if item is None:  # connector has no modern truth
            continue
        surface_truth, thickness_truth = truth[item.source_flowline_index]
        for k, model_thickness in enumerate(branch_solution.thickness_m):
            native = item.native_index_end - k
            coord = coords_by_flowline[item.source_flowline_index][native]
            col = int(round(float(coord[0])))
            row = int(round(float(coord[1])))
            if not (0 <= row < modern_mask.shape[0] and 0 <= col < modern_mask.shape[1]):
                excluded_nonfinite += 1
                continue
            if not modern_mask[row, col]:
                excluded_boundary += 1
                continue
            if distance_m[row, col] < boundary_exclusion_m:
                excluded_boundary += 1
                continue
            observed = float(thickness_truth[native])
            if not np.isfinite(model_thickness) or not np.isfinite(observed):
                excluded_nonfinite += 1
                continue
            if observed < minimum_thickness_m:
                continue
            key = (item.source_flowline_index, native)
            points[key] = (float(model_thickness), observed, float(distance_m[row, col]))
    if points:
        model = np.asarray([v[0] for v in points.values()], dtype=float)
        observed = np.asarray([v[1] for v in points.values()], dtype=float)
    else:
        model = np.empty(0, dtype=float)
        observed = np.empty(0, dtype=float)
    return model, observed, {
        "usable_point_count": int(model.size),
        "excluded_boundary_point_count": int(excluded_boundary),
        "excluded_nonfinite_point_count": int(excluded_nonfinite),
    }


def _evaluate(
    tau_kpa: float,
    inputs: list[FlowlineInput],
    truth,
    modern_mask,
    map_dx_m: float,
    config: dict,
    coords_by_flowline: dict[int, np.ndarray],
) -> dict:
    shape = float(config["shape_factor"])
    scenario = SurfaceScenario("tau-calibration", tau_kpa * 1000.0, shape)
    solutions = {}
    failed_id = ""
    failed_index = np.nan
    failed_status = ""
    for item in inputs:
        distance = np.arange(len(item.bed_elevation_m), dtype=float) * item.dx_meter
        surface_truth, _ = truth[item.flowline_index]
        solution = solve_flowline(
            distance,
            np.asarray(item.bed_elevation_m, dtype=float)[::-1],
            scenario,
            start_surface_m=float(surface_truth[-1]),
        )
        solutions[item.flowline_index] = solution
        if not solution.success:
            failed_id = f"fl{item.flowline_index}"
            failed_index = solution.diagnostics.get("failed_index", np.nan)
            failed_status = solution.status
            break
    row = {
        "tau_kpa": float(tau_kpa),
        "kappa_kpa": float(tau_kpa / shape),
        "status": "success" if not failed_status else f"profile-failed:{failed_status}",
        "failed_branch_id": failed_id,
        "failed_index": failed_index,
    }
    if failed_status:
        row.update({"loss": np.nan, "thickness_rmse_m": np.nan, "usable_point_count": 0})
        return row
    boundary = modern_mask & ~binary_erosion(modern_mask, border_value=0)
    distance_to_boundary_m = distance_transform_edt(~boundary) * map_dx_m
    model_values: list[float] = []
    observed_values: list[float] = []
    excluded_boundary = 0
    excluded_nonfinite = 0
    for item in inputs:
        solution = solutions[item.flowline_index]
        _, observed_thickness = truth[item.flowline_index]
        coords = coords_by_flowline[item.flowline_index]
        for native in range(len(observed_thickness)):
            coord = coords[native]
            col = int(round(float(coord[0])))
            row_index = int(round(float(coord[1])))
            if not (
                0 <= row_index < modern_mask.shape[0]
                and 0 <= col < modern_mask.shape[1]
                and modern_mask[row_index, col]
            ):
                excluded_nonfinite += 1
                continue
            if distance_to_boundary_m[row_index, col] < float(config["boundary_exclusion_m"]):
                excluded_boundary += 1
                continue
            observed = float(observed_thickness[native])
            model = float(solution.thickness_m[len(observed_thickness) - 1 - native])
            if not np.isfinite(observed) or not np.isfinite(model):
                excluded_nonfinite += 1
                continue
            if observed < float(config["minimum_thickness_m"]):
                continue
            model_values.append(model)
            observed_values.append(observed)
    model = np.asarray(model_values, dtype=float)
    observed = np.asarray(observed_values, dtype=float)
    metrics = {
        "usable_point_count": int(model.size),
        "excluded_boundary_point_count": int(excluded_boundary),
        "excluded_nonfinite_point_count": int(excluded_nonfinite),
    }
    if model.size < int(config["minimum_points"]):
        row.update({"loss": np.nan, "thickness_rmse_m": np.nan, **metrics})
        row["status"] = "insufficient-interior-points"
        return row
    rmse = float(np.sqrt(np.mean((model - observed) ** 2)))
    scale = max(float(np.median(observed)), 1.0)
    row.update({"loss": rmse / scale, "thickness_rmse_m": rmse, **metrics})
    return row


def calibrate_system(
    config: dict, rgi_id: str, bed_scenario: str
) -> tuple[dict, list[dict]]:
    root = resolve(config["stage0_gdir_root"])
    gdir = root / rgi_id[:-6] / rgi_id[:-3] / rgi_id
    inputs, raw_flowlines = load_flowline_inputs(gdir)
    inversion = _read_pickle(gdir / f"inversion_output_{_THICK_SUFFIX[bed_scenario]}.pkl")
    rebuilt, truth = _real_inputs(inputs, raw_flowlines, inversion)
    gridded = load_gridded_bed(gdir, bed_scenario)
    coords_by_flowline = {item.flowline_index: item.coords_xy for item in rebuilt}
    modern_mask = np.asarray(gridded.glacier_mask, dtype=bool)
    calibration = config["modern_tau_calibration"]
    total_centerline_length_km = float(
        sum((len(item.bed_elevation_m) - 1) * item.dx_meter for item in rebuilt)
        / 1000.0
    )
    main_lengths = [
        (len(item.bed_elevation_m) - 1) * item.dx_meter
        for item in rebuilt
        if item.flows_to is None
    ]
    main_centerline_length_km = (
        float(max(main_lengths) / 1000.0) if main_lengths else np.nan
    )
    coarse = [float(value) for value in calibration["coarse_grid_kpa"]]
    evaluations: list[dict] = []
    cache: dict[float, dict] = {}

    def evaluate(value: float) -> dict:
        value = round(float(value), 6)
        if value not in cache:
            cache[value] = _evaluate(
                value,
                rebuilt,
                truth,
                modern_mask,
                gridded.map_dx_m,
                calibration,
                coords_by_flowline,
            )
            evaluations.append(cache[value])
        return cache[value]

    for value in coarse:
        evaluate(value)
    valid = [row for row in evaluations if np.isfinite(row.get("loss", np.nan))]
    if not valid:
        return {
            "rgi_id": rgi_id,
            "bed_scenario": bed_scenario,
            "status": "no-valid-tau-evaluation",
            "tau_kpa": np.nan,
            "kappa_kpa": np.nan,
            "usable_point_count": 0,
            "modern_area_km2": float(modern_mask.sum() * gridded.map_dx_m**2 / 1e6),
            "modern_elevation_span_m": float(np.ptp(np.concatenate([v[0] for v in truth.values()]))),
            "modern_total_centerline_length_km": total_centerline_length_km,
            "modern_main_centerline_length_km": main_centerline_length_km,
        }, evaluations
    best_coarse = min(valid, key=lambda row: row["loss"])
    values = sorted(row["tau_kpa"] for row in valid)
    lo = max(min(values), best_coarse["tau_kpa"] - float(calibration["refinement_half_width_kpa"]))
    hi = min(max(values), best_coarse["tau_kpa"] + float(calibration["refinement_half_width_kpa"]))
    if hi > lo:
        result = minimize_scalar(
            lambda value: evaluate(value).get("loss", np.inf),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": float(calibration["refinement_xatol_kpa"])},
        )
        evaluate(float(result.x))
    valid = [row for row in evaluations if np.isfinite(row.get("loss", np.nan))]
    best = min(valid, key=lambda row: row["loss"])
    baseline = evaluate(float(calibration["baseline_tau_kpa"]))
    tau = float(best["tau_kpa"])
    lower_bound = float(calibration["tau_bounds_kpa"][0])
    upper_bound = float(calibration["tau_bounds_kpa"][1])
    tolerance = float(calibration["near_optimal_fraction"])
    near = [row["tau_kpa"] for row in valid if row["loss"] <= best["loss"] * (1.0 + tolerance)]
    if tau <= lower_bound + 1e-6:
        status = "identifiability-lower-bound"
    elif tau >= upper_bound - 1e-6:
        status = "identifiability-upper-bound"
    elif near and max(near) - min(near) >= float(calibration["weak_interval_kpa"]):
        status = "weak-identifiability"
    else:
        status = "identified-interior"
    all_thickness = np.concatenate([v[1] for v in truth.values()])
    summary = {
        "rgi_id": rgi_id,
        "bed_scenario": bed_scenario,
        "status": status,
        "tau_kpa": tau,
        "kappa_kpa": tau / float(calibration["shape_factor"]),
        "loss": float(best["loss"]),
        "thickness_rmse_m": float(best["thickness_rmse_m"]),
        "usable_point_count": int(best["usable_point_count"]),
        "near_optimal_tau_min_kpa": float(min(near)) if near else np.nan,
        "near_optimal_tau_max_kpa": float(max(near)) if near else np.nan,
        "modern_area_km2": float(modern_mask.sum() * gridded.map_dx_m**2 / 1e6),
        "modern_elevation_span_m": float(np.ptp(np.concatenate([v[0] for v in truth.values()]))),
        "modern_median_thickness_m": float(np.median(all_thickness)),
        "boundary_exclusion_m": float(calibration["boundary_exclusion_m"]),
        "modern_total_centerline_length_km": total_centerline_length_km,
        "modern_main_centerline_length_km": main_centerline_length_km,
        "baseline_tau_kpa": float(calibration["baseline_tau_kpa"]),
        "baseline_status": str(baseline["status"]),
        "baseline_loss": float(baseline["loss"]) if np.isfinite(baseline.get("loss", np.nan)) else np.nan,
        "baseline_thickness_rmse_m": (
            float(baseline["thickness_rmse_m"])
            if np.isfinite(baseline.get("thickness_rmse_m", np.nan)) else np.nan
        ),
        "calibration_loss_improvement_fraction": (
            (float(baseline["loss"]) - float(best["loss"])) / float(baseline["loss"])
            if np.isfinite(baseline.get("loss", np.nan)) and float(baseline["loss"]) > 0
            else np.nan
        ),
    }
    return summary, evaluations


def _linear_regression(x: np.ndarray, y: np.ndarray) -> dict:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[valid]
    y = np.asarray(y, dtype=float)[valid]
    if x.size < 3 or np.ptp(x) == 0:
        return {"n": int(x.size), "slope": np.nan, "intercept": np.nan, "r2": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / denominator if denominator > 0 else np.nan
    return {"n": int(x.size), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def run(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = config["modern_tau_calibration"]
    out = resolve(config["output_dir"]) / str(calibration["output_subdir"])
    out.mkdir(parents=True, exist_ok=True)
    pilot = pd.read_csv(resolve(config["pilot_selection_path"]))
    rgi_ids = sorted(set(pilot["main_rgi_id"].astype(str)))
    summaries: list[dict] = []
    evaluations: list[dict] = []
    for rgi_id in rgi_ids:
        for bed in config["bed_scenarios"]:
            try:
                summary, rows = calibrate_system(config, rgi_id, bed)
            except (FileNotFoundError, ValueError, KeyError) as exc:
                summary = {"rgi_id": rgi_id, "bed_scenario": bed, "status": f"calibration-error:{type(exc).__name__}", "failure_reason": str(exc)}
                rows = []
            summaries.append(summary)
            evaluations.extend({"rgi_id": rgi_id, "bed_scenario": bed, **row} for row in rows)
    summary_df = pd.DataFrame(summaries)
    eval_df = pd.DataFrame(evaluations)
    summary_df.to_csv(out / "modern_tau_calibration_v0.csv", index=False)
    eval_df.to_csv(out / "modern_tau_calibration_loss_curves_v0.csv", index=False)
    curve_dir = out / "loss_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for summary_row in summary_df.itertuples(index=False):
        rgi_id = str(summary_row.rgi_id)
        bed = str(summary_row.bed_scenario)
        group = eval_df[
            (eval_df["rgi_id"] == rgi_id) & (eval_df["bed_scenario"] == bed)
        ].copy()
        if group.empty:
            group = pd.DataFrame([{
                "rgi_id": rgi_id,
                "bed_scenario": bed,
                "status": str(summary_row.status),
                "failure_reason": getattr(summary_row, "failure_reason", ""),
                "tau_kpa": np.nan,
                "loss": np.nan,
            }])
        group.sort_values("tau_kpa").to_csv(
            curve_dir / f"{rgi_id}_{bed}_loss_curve_v0.csv", index=False
        )
    payload = {
        "status": "completed",
        "objective": "interior-modern-inversion-thickness-rmse-normalized-by-median",
        "tau_is_geometry_transfer_only": True,
        "shape_factor": float(calibration["shape_factor"]),
        "system_count": int(summary_df["rgi_id"].nunique()),
        "bed_result_count": int(len(summary_df)),
        "status_counts": summary_df["status"].value_counts(dropna=False).to_dict(),
    }
    (out / "modern_tau_calibration_summary_v0.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not summary_df.empty:
        scale = summary_df.assign(
            log10_modern_area_km2=np.log10(
                summary_df["modern_area_km2"].clip(lower=1e-9)
            )
        )
        scale.to_csv(out / "modern_tau_scale_diagnostic_v0.csv", index=False)
        regressions = {}
        for bed, group in scale[np.isfinite(scale["kappa_kpa"])].groupby("bed_scenario"):
            regressions[bed] = {
                "kappa_vs_log10_area": _linear_regression(
                    group["log10_modern_area_km2"].to_numpy(),
                    group["kappa_kpa"].to_numpy(),
                ),
                "kappa_vs_elevation_span_km": _linear_regression(
                    group["modern_elevation_span_m"].to_numpy() / 1000.0,
                    group["kappa_kpa"].to_numpy(),
                ),
            }
        (out / "modern_tau_scale_regression_v0.json").write_text(
            json.dumps({
                "status": "pilot-smoke-not-transfer-correction",
                "warning": "12 systems are insufficient for an HMA scaling law",
                "regressions": regressions,
            }, indent=2),
            encoding="utf-8",
        )
    return summary_df, eval_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    summary, _ = run(json.loads(args.config.read_text(encoding="utf-8")))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
