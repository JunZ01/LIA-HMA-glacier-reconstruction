"""Numerically scaled v2 runner for the unchanged Cycle-3 v1 mechanism.

Version 1 supplied the exact volume equality to SLSQP in raw m2/m3 units and
stopped with status 8 before any static score existed.  This implementation
only divides both sides of that same equality by total assigned area.  It does
not change an input, objective, constraint, tolerance, or gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from . import v3_true_mainline_area_equivalent_rectangular_preflight_v1 as v1


_polygon_transverse_widths_m = v1.polygon_transverse_widths_m


def _width_adapter(component_geometry, x_m, y_m):
    return _polygon_transverse_widths_m(
        np.column_stack([
            np.asarray(x_m, dtype=float),
            np.asarray(y_m, dtype=float),
        ]),
        component_geometry,
    )


def _solve_thickness_scaled(
    *,
    bed_h_m: np.ndarray,
    sampled_thickness_m: np.ndarray,
    assigned_area_m2: np.ndarray,
    target_volume_m3: float,
    options: dict,
):
    decimals = int(options["sample_rounding_decimals"])
    bed = np.round(np.asarray(bed_h_m, dtype=float), decimals)
    sampled_h = np.maximum(
        np.round(np.asarray(sampled_thickness_m, dtype=float), decimals), 0.0
    )
    area = np.asarray(assigned_area_m2, dtype=float)
    if np.any(area <= 0):
        raise ValueError("mainline-control-volume-has-nonpositive-area")
    minimum_surface = np.maximum.accumulate(bed[::-1])[::-1]
    minimum_volume = float(np.dot(area, minimum_surface - bed))
    tolerance_fraction = float(options["volume_constraint_tolerance_fraction"])
    if target_volume_m3 - minimum_volume < -tolerance_fraction * target_volume_m3:
        raise ValueError(
            "monotonic-bed-minimum-volume-exceeds-source-volume:"
            f"minimum={minimum_volume}:source={target_volume_m3}"
        )
    feasible_surface = minimum_surface + max(
        target_volume_m3 - minimum_volume, 0.0
    ) / float(area.sum())
    target_surface = bed + sampled_h
    weights = area / float(np.median(area))
    difference = np.zeros((len(area) - 1, len(area)), dtype=float)
    index = np.arange(len(area) - 1)
    difference[index, index] = 1.0
    difference[index, index + 1] = -1.0

    # Same exact volume equality as v1, expressed as an area-weighted mean
    # surface elevation to avoid mixing O(1e5) coefficients with O(1e12) RHS.
    normalized_area = area / float(area.sum())
    target_mean_surface = (
        target_volume_m3 + float(np.dot(area, bed))
    ) / float(area.sum())
    result = minimize(
        lambda candidate: 0.5 * float(np.sum(
            weights * np.square(candidate - target_surface)
        )),
        feasible_surface,
        jac=lambda candidate: weights * (candidate - target_surface),
        bounds=[(float(value), None) for value in bed],
        constraints=[
            LinearConstraint(difference, 0.0, np.inf),
            LinearConstraint(
                normalized_area[np.newaxis, :],
                target_mean_surface,
                target_mean_surface,
            ),
        ],
        method=str(options["method"]),
        options={
            "maxiter": int(options["maximum_iterations"]),
            "ftol": float(options["function_tolerance"]),
            "disp": False,
        },
    )
    surface = np.asarray(result.x, dtype=float)
    thickness = surface - bed
    volume_residual = float(np.dot(area, thickness) - target_volume_m3)
    constraint_tolerance = float(options["constraint_tolerance_m"])
    reverse = np.diff(surface) > constraint_tolerance
    if not result.success:
        raise RuntimeError(
            f"thickness-optimization-failed:{result.status}:{result.message}"
        )
    if np.any(reverse):
        raise RuntimeError(
            f"thickness-optimization-reverse-slope-count:{int(reverse.sum())}"
        )
    if float(np.min(thickness)) < -constraint_tolerance:
        raise RuntimeError(
            f"thickness-optimization-negative-thickness:{float(np.min(thickness))}"
        )
    if abs(volume_residual) > tolerance_fraction * target_volume_m3:
        raise RuntimeError(
            f"thickness-optimization-volume-residual:{volume_residual}"
        )
    return np.maximum(thickness, 0.0), {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iteration_count": int(result.nit),
        "objective_value": float(result.fun),
        "minimum_feasible_volume_m3": minimum_volume,
        "source_minus_minimum_feasible_volume_m3": float(
            target_volume_m3 - minimum_volume
        ),
        "volume_constraint_residual_m3": volume_residual,
        "minimum_surface_constraint_m": float(np.min(surface - bed)),
        "minimum_monotonic_constraint_m": float(
            np.min(surface[:-1] - surface[1:])
        ),
        "sample_rounding_decimals": decimals,
        "volume_equality_scaling": (
            "area coefficients and target divided by total assigned area"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    v1.polygon_transverse_widths_m = _width_adapter
    v1._solve_thickness = _solve_thickness_scaled
    print(json.dumps(
        v1.run(args.config), indent=2, ensure_ascii=True, allow_nan=False
    ))


if __name__ == "__main__":
    main()
