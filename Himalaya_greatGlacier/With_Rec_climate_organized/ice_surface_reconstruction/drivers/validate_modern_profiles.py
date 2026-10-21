"""Modern pseudo-paleo flowline validation for the surface operator."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


from ..core import SurfaceScenario, solve_flowline
from ..paths import resolve


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:-6] / rgi_id[:-3] / rgi_id


def read_pickle(path: Path):
    with gzip.open(path, "rb") as source:
        return pickle.load(source)


def legacy_profile(
    distance_m: np.ndarray,
    bed_elevation_m: np.ndarray,
    tau_pa: float,
    shape_factor: float,
    *,
    rho: float = 900.0,
    gravity: float = 9.81,
    max_surface_step_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the capped iteration used by the archived prototype."""

    bed = np.asarray(bed_elevation_m, dtype=float)
    distance = np.asarray(distance_m, dtype=float)
    surface = bed.copy()
    thickness = np.zeros_like(bed)
    bed_slope = np.gradient(bed, distance)
    slope_angle = np.arctan(np.abs(bed_slope))
    for i in range(1, len(bed)):
        dx = distance[i] - distance[i - 1]
        sin_alpha = max(np.sin(slope_angle[i]), np.sin(np.radians(1.0)))
        current = min(tau_pa / (rho * gravity * sin_alpha * shape_factor), 1500.0)
        for _ in range(10):
            average = max((thickness[i - 1] + current) / 2.0, 1.0)
            surface_step = min(
                tau_pa * dx / (rho * gravity * shape_factor * average),
                max_surface_step_m,
            )
            updated = surface[i - 1] + surface_step - bed[i]
            if abs(updated - current) < 1.0:
                current = updated
                break
            current = updated
        thickness[i] = np.clip(current, 0.0, 1000.0)
        surface[i] = bed[i] + thickness[i]
    return surface, thickness


def metrics(
    truth_surface: np.ndarray,
    truth_thickness: np.ndarray,
    model_surface: np.ndarray,
    model_thickness: np.ndarray,
) -> dict:
    valid = (
        np.isfinite(truth_surface)
        & np.isfinite(truth_thickness)
        & np.isfinite(model_surface)
        & np.isfinite(model_thickness)
    )
    if not valid.any():
        return {
            "surface_rmse_m": np.nan,
            "surface_median_bias_m": np.nan,
            "thickness_rmse_m": np.nan,
            "thickness_median_bias_m": np.nan,
            "mean_thickness_ratio": np.nan,
            "maximum_thickness_ratio": np.nan,
            "mean_thickness_ratio_status": "not-evaluated-no-valid-pairs",
            "maximum_thickness_ratio_status": "not-evaluated-no-valid-pairs",
        }
    surface_error = model_surface[valid] - truth_surface[valid]
    thickness_error = model_thickness[valid] - truth_thickness[valid]
    truth_mean = float(np.mean(truth_thickness[valid]))
    truth_maximum = float(np.max(truth_thickness[valid]))
    mean_ratio_valid = not np.isclose(truth_mean, 0.0)
    maximum_ratio_valid = not np.isclose(truth_maximum, 0.0)
    return {
        "surface_rmse_m": float(np.sqrt(np.mean(surface_error**2))),
        "surface_median_bias_m": float(np.median(surface_error)),
        "thickness_rmse_m": float(np.sqrt(np.mean(thickness_error**2))),
        "thickness_median_bias_m": float(np.median(thickness_error)),
        "mean_thickness_ratio": (
            float(np.mean(model_thickness[valid]) / truth_mean)
            if mean_ratio_valid else np.nan
        ),
        "maximum_thickness_ratio": (
            float(np.max(model_thickness[valid]) / truth_maximum)
            if maximum_ratio_valid else np.nan
        ),
        "mean_thickness_ratio_status": (
            "evaluated" if mean_ratio_valid else "not-evaluated-zero-truth-mean"
        ),
        "maximum_thickness_ratio_status": (
            "evaluated"
            if maximum_ratio_valid else "not-evaluated-zero-truth-maximum"
        ),
    }


def main_flowline_index(flowlines: list) -> int:
    candidates = [i for i, flowline in enumerate(flowlines) if flowline.flows_to is None]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one main flowline, found {len(candidates)}")
    return candidates[0]


def run(config: dict) -> pd.DataFrame:
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot = pd.read_csv(resolve(config["pilot_selection_path"]))
    rgi_ids = set(pilot["main_rgi_id"].astype(str))
    glathida_matches = output_dir / "glathida_summary_target_matches.csv"
    if glathida_matches.exists():
        rgi_ids.update(pd.read_csv(glathida_matches)["RGIId"].astype(str))

    root = resolve(config["stage0_gdir_root"])
    suffixes = {
        "farinotti2019": "f19_fs0_full_v0",
        "millan2022": "m22_fs0_full_v0",
    }
    rows: list[dict] = []
    for rgi_id in sorted(rgi_ids):
        directory = gdir_path(root, rgi_id)
        flowlines = read_pickle(directory / "inversion_flowlines.pkl")
        index = main_flowline_index(flowlines)
        flowline = flowlines[index]
        for bed_scenario in config["bed_scenarios"]:
            inversion_path = directory / f"inversion_output_{suffixes[bed_scenario]}.pkl"
            if not inversion_path.exists():
                rows.append(
                    {
                        "rgi_id": rgi_id,
                        "bed_scenario": bed_scenario,
                        "surface_method": "not-run",
                        "stress_shape_scenario": "not-run",
                        "tau_kpa": np.nan,
                        "shape_factor": np.nan,
                        "status": "missing-bed-scenario",
                        "flowline_points": len(flowline.surface_h),
                        "flowline_length_m": float(
                            (len(flowline.surface_h) - 1) * flowline.dx_meter
                        ),
                        **metrics(
                            np.asarray(flowline.surface_h, dtype=float),
                            np.zeros(len(flowline.surface_h)),
                            np.full(len(flowline.surface_h), np.nan),
                            np.full(len(flowline.surface_h), np.nan),
                        ),
                    }
                )
                continue
            inversion = read_pickle(inversion_path)[index]
            truth_surface = np.asarray(flowline.surface_h, dtype=float)[::-1]
            truth_thickness = np.asarray(inversion["thick"], dtype=float)[::-1]
            bed = truth_surface - truth_thickness
            distance = np.arange(len(bed), dtype=float) * float(flowline.dx_meter)
            for scenario_config in config["scenarios"]:
                shape_factor = float(
                    np.clip(
                        config["baseline_shape_factor"]
                        * scenario_config["shape_factor_multiplier"],
                        0.445,
                        1.0,
                    )
                )
                tau_pa = float(scenario_config["basal_shear_stress_kpa"]) * 1000.0
                scenario = SurfaceScenario(
                    scenario_config["scenario_id"], tau_pa, shape_factor
                )
                solution = solve_flowline(distance, bed, scenario)
                if solution.success:
                    result_metrics = metrics(
                        truth_surface,
                        truth_thickness,
                        solution.surface_elevation_m,
                        solution.thickness_m,
                    )
                else:
                    result_metrics = metrics(
                        truth_surface,
                        truth_thickness,
                        np.full_like(truth_surface, np.nan),
                        np.full_like(truth_thickness, np.nan),
                    )
                rows.append(
                    {
                        "rgi_id": rgi_id,
                        "bed_scenario": bed_scenario,
                        "surface_method": "new-exact-perfect-plasticity",
                        "stress_shape_scenario": scenario.scenario_id,
                        "tau_kpa": tau_pa / 1000.0,
                        "shape_factor": shape_factor,
                        "status": solution.status,
                        "flowline_points": len(distance),
                        "flowline_length_m": float(distance[-1]),
                        **result_metrics,
                    }
                )

                legacy_surface, legacy_thickness = legacy_profile(
                    distance, bed, tau_pa, shape_factor
                )
                rows.append(
                    {
                        "rgi_id": rgi_id,
                        "bed_scenario": bed_scenario,
                        "surface_method": "archived-capped-iteration",
                        "stress_shape_scenario": scenario.scenario_id,
                        "tau_kpa": tau_pa / 1000.0,
                        "shape_factor": shape_factor,
                        "status": "success-with-hard-caps",
                        "flowline_points": len(distance),
                        "flowline_length_m": float(distance[-1]),
                        **metrics(
                            truth_surface,
                            truth_thickness,
                            legacy_surface,
                            legacy_thickness,
                        ),
                    }
                )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "modern_profile_validation.csv", index=False)
    best = (
        result[np.isfinite(result["surface_rmse_m"])]
        .sort_values("surface_rmse_m")
        .groupby(["rgi_id", "bed_scenario", "surface_method"], as_index=False)
        .first()
    )
    best.to_csv(output_dir / "modern_profile_best_scenarios.csv", index=False)
    summary = (
        best.groupby(["bed_scenario", "surface_method"], as_index=False)
        .agg(
            glacier_count=("rgi_id", "nunique"),
            median_surface_rmse_m=("surface_rmse_m", "median"),
            median_thickness_rmse_m=("thickness_rmse_m", "median"),
            median_thickness_ratio=("mean_thickness_ratio", "median"),
        )
    )
    summary.to_csv(output_dir / "modern_profile_validation_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["main-only-baseline", "full-network"],
        default="main-only-baseline",
        help=(
            "main-only-baseline (default) reproduces the single-main-flowline "
            "Level B baseline number-for-number and writes the existing CSVs. "
            "full-network runs the whole tree in BOTH junction-transfer policies "
            "via full_network_validation and writes only new versioned files "
            "(never overwrites the baseline)."
        ),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.mode == "main-only-baseline":
        summary = run(config)
        print(summary.to_string(index=False))
        return
    # full-network: delegate to the dedicated 3-mode driver.
    from .full_network_validation import run as run_full_network

    result = run_full_network(config)
    print("=== collapsed control ===")
    print(result["collapsed_summary"].to_string(index=False))
    print("=== explicit connector ===")
    print(result["explicit_summary"].to_string(index=False))
    print("=== collapsed vs connector outcome ===")
    print(result["diff"]["outcome"].value_counts().to_string())


if __name__ == "__main__":
    main()
