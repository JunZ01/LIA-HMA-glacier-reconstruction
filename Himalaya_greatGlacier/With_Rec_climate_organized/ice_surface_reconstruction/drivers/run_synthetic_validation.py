"""Run reproducible synthetic checks for the paleo-surface operators."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


from ..core import (
    SurfaceScenario,
    harmonic_extend_thickness,
    solve_flowline,
    surface_quality_metrics,
)
from ..paths import resolve
from .validate_modern_profiles import legacy_profile


def profile_error(model: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    difference = np.asarray(model) - np.asarray(reference)
    return float(np.sqrt(np.mean(difference**2))), float(np.max(np.abs(difference)))


def run(config: dict) -> dict:
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = SurfaceScenario("synthetic-central", 100_000.0, 0.8)
    cases = {
        "flat": lambda x: np.zeros_like(x),
        "linear-rising": lambda x: 2500.0 + 0.04 * x,
        "undulating-rising": lambda x: 2500.0 + 0.04 * x + 25.0 * np.sin(x / 900.0),
    }
    rows: list[dict] = []
    for case_id, bed_function in cases.items():
        fine_distance = np.arange(0.0, 12_000.0 + 1.0, 1.0)
        fine_bed = bed_function(fine_distance)
        reference = solve_flowline(fine_distance, fine_bed, scenario)
        coarse_distance = np.arange(0.0, 12_000.0 + 50.0, 50.0)
        coarse_bed = bed_function(coarse_distance)
        reference_surface = np.interp(
            coarse_distance, fine_distance, reference.surface_elevation_m
        )
        reference_thickness = reference_surface - coarse_bed

        start = time.perf_counter()
        exact = solve_flowline(coarse_distance, coarse_bed, scenario)
        exact_seconds = time.perf_counter() - start
        if exact.success:
            surface_rmse, surface_max = profile_error(
                exact.surface_elevation_m, reference_surface
            )
            thickness_rmse, thickness_max = profile_error(
                exact.thickness_m, reference_thickness
            )
        else:
            surface_rmse = surface_max = thickness_rmse = thickness_max = np.nan
        rows.append(
            {
                "case_id": case_id,
                "method": "new-exact-perfect-plasticity",
                "status": exact.status,
                "surface_rmse_m": surface_rmse,
                "surface_max_abs_error_m": surface_max,
                "thickness_rmse_m": thickness_rmse,
                "thickness_max_abs_error_m": thickness_max,
                "runtime_seconds": exact_seconds,
            }
        )

        start = time.perf_counter()
        old_surface, old_thickness = legacy_profile(
            coarse_distance,
            coarse_bed,
            scenario.basal_shear_stress_pa,
            scenario.shape_factor,
        )
        old_seconds = time.perf_counter() - start
        surface_rmse, surface_max = profile_error(old_surface, reference_surface)
        thickness_rmse, thickness_max = profile_error(
            old_thickness, reference_thickness
        )
        rows.append(
            {
                "case_id": case_id,
                "method": "archived-capped-iteration",
                "status": "success-with-hard-caps",
                "surface_rmse_m": surface_rmse,
                "surface_max_abs_error_m": surface_max,
                "thickness_rmse_m": thickness_rmse,
                "thickness_max_abs_error_m": thickness_max,
                "runtime_seconds": old_seconds,
            }
        )

    yy, xx = np.mgrid[-25:26, -50:51]
    mask = (xx / 50.0) ** 2 + (yy / 25.0) ** 2 <= 1.0
    center_rows = np.full(99, 25, dtype=int)
    center_cols = np.arange(1, 100, dtype=int)
    center_values = 150.0 * np.sin(np.linspace(0.0, np.pi, 99))
    thickness, extension = harmonic_extend_thickness(
        mask, center_rows, center_cols, center_values
    )
    extension_qc = surface_quality_metrics(
        mask,
        np.zeros_like(thickness),
        thickness,
        centerline_rows=center_rows,
        centerline_cols=center_cols,
        centerline_reference_m=center_values,
    )

    profiles = pd.DataFrame(rows)
    profiles.to_csv(output_dir / "synthetic_profile_validation.csv", index=False)
    report = {
        "validation_id": config["validation_id"],
        "profile_case_count": len(cases),
        "all_exact_cases_success": bool(
            profiles.loc[
                profiles["method"] == "new-exact-perfect-plasticity", "status"
            ].eq("success").all()
        ),
        "profile_results": rows,
        "harmonic_extension": {**extension, **extension_qc},
    }
    temporary = output_dir / "synthetic_validation.json.tmp"
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(output_dir / "synthetic_validation.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
