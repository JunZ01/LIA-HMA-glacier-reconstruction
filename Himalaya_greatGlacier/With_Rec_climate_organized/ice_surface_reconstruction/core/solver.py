"""Small, auditable perfect-plasticity surface solver.

The solver is deliberately independent of OGGM and climate forcing. It creates
candidate LIA flowline states from a fixed bed and a fixed footprint. OGGM is
used later for state conversion and transient forward integration.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.sparse.linalg import LinearOperator, cg


@dataclass(frozen=True)
class SurfaceScenario:
    """Physical scenario used by the one-dimensional stress balance."""

    scenario_id: str
    basal_shear_stress_pa: float | np.ndarray
    shape_factor: float | np.ndarray
    ice_density_kg_m3: float = 900.0
    gravity_m_s2: float = 9.81
    maximum_thickness_m: float = 2000.0

    def validate(self) -> None:
        tau = np.asarray(self.basal_shear_stress_pa, dtype=float)
        shape = np.asarray(self.shape_factor, dtype=float)
        if not np.isfinite(tau).all() or np.any(tau <= 0):
            raise ValueError("Basal shear stress must be positive")
        if not np.isfinite(shape).all() or np.any((shape <= 0) | (shape > 1)):
            raise ValueError("Shape factor must be in (0, 1]")
        if self.ice_density_kg_m3 <= 0 or self.gravity_m_s2 <= 0:
            raise ValueError("Ice density and gravity must be positive")


@dataclass
class FlowlineSolution:
    """Result and diagnostics for one flowline profile."""

    distance_m: np.ndarray
    bed_elevation_m: np.ndarray
    surface_elevation_m: np.ndarray
    thickness_m: np.ndarray
    surface_slope: np.ndarray
    driving_shear_stress_pa: np.ndarray
    status: str
    diagnostics: dict

    @property
    def success(self) -> bool:
        return self.status == "success"


def _as_profile(value: float | np.ndarray, size: int, name: str) -> np.ndarray:
    profile = np.asarray(value, dtype=float)
    if profile.ndim == 0:
        profile = np.full(size, float(profile))
    if profile.shape != (size,):
        raise ValueError(f"{name} must be scalar or length {size}")
    return profile


def _segment_values_at_nodes(segment_values: np.ndarray) -> np.ndarray:
    """Map downstream-to-upstream segment values onto their upstream nodes."""
    values = np.asarray(segment_values, dtype=float)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("At least one segment value is required")
    return np.concatenate(([values[0]], values))


def flowline_stress_diagnostics(
    distance_m: np.ndarray,
    surface_elevation_m: np.ndarray,
    thickness_m: np.ndarray,
    scenario: SurfaceScenario,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return diagnostics using the same segment balance as the solver.

    Values at node ``i > 0`` describe the segment ending at that node; node 0
    repeats the first segment because no downstream segment exists. The driving
    stress uses segment-mean thickness and the small-slope form integrated by
    :func:`solve_flowline`. For varying stress or shape factor, the effective
    segment shape maps the integrated ``tau / F`` back to segment-mean ``tau``.
    """
    distance = np.asarray(distance_m, dtype=float)
    surface = np.asarray(surface_elevation_m, dtype=float)
    thickness = np.asarray(thickness_m, dtype=float)
    if not (distance.ndim == 1 and distance.shape == surface.shape == thickness.shape):
        raise ValueError("Flowline diagnostic arrays must be paired and one-dimensional")
    if distance.size < 2 or not np.all(np.diff(distance) > 0):
        raise ValueError("Flowline diagnostics require increasing two-point distance")

    tau = _as_profile(scenario.basal_shear_stress_pa, distance.size, "basal_shear_stress_pa")
    shape = _as_profile(scenario.shape_factor, distance.size, "shape_factor")
    segment_slope = np.diff(surface) / np.diff(distance)
    segment_thickness = 0.5 * (thickness[:-1] + thickness[1:])
    segment_tau = 0.5 * (tau[:-1] + tau[1:])
    segment_kappa = 0.5 * (tau[:-1] / shape[:-1] + tau[1:] / shape[1:])
    effective_shape = segment_tau / segment_kappa
    segment_driving = (
        scenario.ice_density_kg_m3
        * scenario.gravity_m_s2
        * segment_thickness
        * np.abs(segment_slope)
        * effective_shape
    )
    return (
        _segment_values_at_nodes(segment_slope),
        _segment_values_at_nodes(segment_driving),
        _segment_values_at_nodes(segment_driving - segment_tau),
    )


def _trapezoidal_integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(values, coordinates))
    return float(np.trapz(values, coordinates))


def solve_flowline(
    distance_m: np.ndarray,
    bed_elevation_m: np.ndarray,
    scenario: SurfaceScenario,
    *,
    start_surface_m: float | None = None,
    discriminant_tolerance_m2: float = 1e-8,
    thickness_tolerance_m: float = 1e-8,
    discriminant_tolerance: float | None = None,
) -> FlowlineSolution:
    """Solve the Benn-Hulton finite-difference stress balance upstream.

    ``distance_m`` must run from a terminus or tributary junction upstream.
    The start surface is the bed at a free terminus, or a parent-flowline
    surface at a tributary junction. No negative-discriminant or thickness
    fallback is applied: those cases are explicit scientific failures.
    """

    scenario.validate()
    if discriminant_tolerance is not None:
        # Compatibility with the v0 keyword, which served both tolerances.
        discriminant_tolerance_m2 = float(discriminant_tolerance)
        thickness_tolerance_m = float(discriminant_tolerance)
    if discriminant_tolerance_m2 < 0 or thickness_tolerance_m < 0:
        raise ValueError("Solver tolerances must be non-negative")
    distance = np.asarray(distance_m, dtype=float)
    bed = np.asarray(bed_elevation_m, dtype=float)
    if distance.ndim != 1 or bed.ndim != 1 or distance.size != bed.size:
        raise ValueError("Distance and bed must be one-dimensional and paired")
    if distance.size < 2:
        raise ValueError("At least two flowline points are required")
    if not np.isfinite(distance).all() or not np.isfinite(bed).all():
        raise ValueError("Distance and bed must be finite")
    if not np.all(np.diff(distance) > 0):
        raise ValueError("Distance must be strictly increasing upstream")

    n = distance.size
    tau = _as_profile(scenario.basal_shear_stress_pa, n, "basal_shear_stress_pa")
    shape = _as_profile(scenario.shape_factor, n, "shape_factor")
    if np.any(tau <= 0) or np.any((shape <= 0) | (shape > 1)):
        raise ValueError("Invalid stress or shape-factor profile")

    surface = np.full(n, np.nan, dtype=float)
    thickness = np.full(n, np.nan, dtype=float)
    surface[0] = bed[0] if start_surface_m is None else float(start_surface_m)
    thickness[0] = surface[0] - bed[0]
    if thickness[0] < -thickness_tolerance_m:
        return _failed_solution(distance, bed, "negative-start-thickness", 0)
    if thickness[0] > scenario.maximum_thickness_m:
        return _failed_solution(distance, bed, "start-thickness-exceeded", 0)

    for i in range(1, n):
        dx = distance[i] - distance[i - 1]
        zsum = bed[i - 1] + bed[i]
        if tau[i] == tau[i - 1] and shape[i] == shape[i - 1]:
            # Preserve the v0 scalar operation order bit-for-bit.
            flux_term = 2.0 * dx * tau[i] / (
                scenario.ice_density_kg_m3 * scenario.gravity_m_s2 * shape[i]
            )
        else:
            segment_kappa = 0.5 * (
                tau[i - 1] / shape[i - 1] + tau[i] / shape[i]
            )
            flux_term = 2.0 * dx * segment_kappa / (
                scenario.ice_density_kg_m3 * scenario.gravity_m_s2
            )
        # (S_i-S_prev) * (S_i+S_prev-z_prev-z_i) = flux_term
        constant = surface[i - 1] * (zsum - surface[i - 1]) - flux_term
        discriminant = zsum * zsum - 4.0 * constant
        roundoff_tolerance_m2 = 32.0 * np.finfo(float).eps * (
            abs(zsum * zsum) + abs(4.0 * constant)
        )
        effective_tolerance_m2 = max(
            discriminant_tolerance_m2, roundoff_tolerance_m2
        )
        if discriminant < -effective_tolerance_m2:
            return _failed_solution(distance, bed, "negative-discriminant", i)
        root = max(discriminant, 0.0) ** 0.5
        surface[i] = 0.5 * (zsum + root)
        thickness[i] = surface[i] - bed[i]
        if thickness[i] < -thickness_tolerance_m:
            return _failed_solution(distance, bed, "negative-thickness", i)
        if thickness[i] > scenario.maximum_thickness_m:
            return _failed_solution(distance, bed, "maximum-thickness-exceeded", i)

    surface_slope, driving, residual = flowline_stress_diagnostics(
        distance, surface, thickness, scenario
    )
    return FlowlineSolution(
        distance_m=distance,
        bed_elevation_m=bed,
        surface_elevation_m=surface,
        thickness_m=np.maximum(thickness, 0.0),
        surface_slope=surface_slope,
        driving_shear_stress_pa=driving,
        status="success",
        diagnostics={
            "maximum_surface_stress_residual_pa": float(np.nanmax(np.abs(residual))),
            "median_surface_stress_residual_pa": float(np.nanmedian(np.abs(residual))),
            "maximum_thickness_m": float(np.nanmax(thickness)),
            "volume_proxy_m2": _trapezoidal_integral(thickness, distance),
            "stress_diagnostic_discretization": (
                "upstream-segment-slope-mean-thickness-small-slope"
            ),
        },
    )


def _failed_solution(
    distance: np.ndarray, bed: np.ndarray, status: str, failed_index: int
) -> FlowlineSolution:
    nan = np.full(distance.shape, np.nan, dtype=float)
    return FlowlineSolution(
        distance_m=distance,
        bed_elevation_m=bed,
        surface_elevation_m=nan,
        thickness_m=nan,
        surface_slope=nan,
        driving_shear_stress_pa=nan,
        status=status,
        diagnostics={"failed_index": int(failed_index)},
    )


def harmonic_extend_thickness(
    mask: np.ndarray,
    centerline_rows: np.ndarray,
    centerline_cols: np.ndarray,
    centerline_thickness_m: np.ndarray,
    *,
    centerline_surface_m: np.ndarray | None = None,
    duplicate_tolerance_m: float = 1e-3,
    boundary_tolerance_m: float = 1e-3,
    max_iterations: int = 5000,
    tolerance_m: float = 1e-3,
    method: str = "jacobi",
    cg_relative_tolerance: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """Extend flowline thickness smoothly inside a fixed footprint.

    The boundary is Dirichlet zero and flowline cells are Dirichlet at their
    solved values. Interior cells satisfy a discrete Laplace equation. This is
    an explicit, reproducible extension operator; it does not claim to solve a
    two-dimensional stress balance.

    A flowline network commonly maps more than one branch to the same raster
    cell near a junction. Those constraints are explicitly consolidated. A
    disagreement larger than ``duplicate_tolerance_m`` is a scientific
    failure, rather than an order-dependent array assignment. Optional surface
    values are used for this comparison because junction continuity is a
    surface constraint; thickness is the fallback when no bed is available.
    """

    inside = np.asarray(mask, dtype=bool)
    if inside.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    rows = np.asarray(centerline_rows, dtype=int)
    cols = np.asarray(centerline_cols, dtype=int)
    values = np.asarray(centerline_thickness_m, dtype=float)
    if not (rows.shape == cols.shape == values.shape):
        raise ValueError("Centerline indices and values must have equal shape")
    if np.any(rows < 0) or np.any(rows >= inside.shape[0]) or np.any(cols < 0) or np.any(cols >= inside.shape[1]):
        raise ValueError("Centerline index outside raster")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Centerline thickness must be finite and non-negative")
    if not np.all(inside[rows, cols]):
        raise ValueError("centerline-constraint-outside-mask")
    if duplicate_tolerance_m < 0 or boundary_tolerance_m < 0:
        raise ValueError("Constraint tolerances must be non-negative")
    if (
        isinstance(max_iterations, bool)
        or int(max_iterations) != max_iterations
        or int(max_iterations) < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("tolerance_m must be finite and positive")
    surface_values = values if centerline_surface_m is None else np.asarray(
        centerline_surface_m, dtype=float
    )
    if surface_values.shape != values.shape:
        raise ValueError("centerline_surface_m must match centerline_thickness_m")
    if not np.isfinite(surface_values).all():
        raise ValueError("Centerline surface values must be finite")

    boundary = inside & ~binary_erosion(inside, border_value=0)

    flat = np.ravel_multi_index((rows, cols), inside.shape)
    unique_flat, inverse = np.unique(flat, return_inverse=True)
    duplicate_count = int(len(flat) - len(unique_flat))
    max_duplicate_spread = 0.0
    consolidated_rows = np.empty(len(unique_flat), dtype=int)
    consolidated_cols = np.empty(len(unique_flat), dtype=int)
    consolidated_values = np.empty(len(unique_flat), dtype=float)
    member_order = np.argsort(inverse, kind="stable")
    group_counts = np.bincount(inverse, minlength=len(unique_flat))
    group_offsets = np.concatenate(([0], np.cumsum(group_counts)))
    for group_index in range(len(unique_flat)):
        members = member_order[
            group_offsets[group_index]:group_offsets[group_index + 1]
        ]
        spread = float(np.ptp(surface_values[members]))
        max_duplicate_spread = max(max_duplicate_spread, spread)
        if spread > duplicate_tolerance_m:
            failed = np.full(inside.shape, np.nan, dtype=float)
            failed[~inside] = 0.0
            return failed, {
                "status": "centerline-cell-conflict",
                "failure_reason": "duplicate centerline surface constraints exceed tolerance",
                "conflict_cell_count": 1,
                "conflict_row": int(rows[members[0]]),
                "conflict_col": int(cols[members[0]]),
                "maximum_duplicate_surface_spread_m": spread,
                "duplicate_tolerance_m": float(duplicate_tolerance_m),
            }
        consolidated_rows[group_index] = rows[members[0]]
        consolidated_cols[group_index] = cols[members[0]]
        consolidated_values[group_index] = float(np.mean(values[members]))

    boundary_constraints = boundary[consolidated_rows, consolidated_cols]
    conflicts = boundary_constraints & (
        consolidated_values > boundary_tolerance_m
    )
    if np.any(conflicts):
        bad = int(np.flatnonzero(conflicts)[0])
        failed = np.full(inside.shape, np.nan, dtype=float)
        failed[~inside] = 0.0
        return failed, {
            "status": "centerline-boundary-conflict",
            "failure_reason": "positive centerline thickness lies on a zero-thickness boundary",
            "conflict_cell_count": int(np.sum(conflicts)),
            "conflict_row": int(consolidated_rows[bad]),
            "conflict_col": int(consolidated_cols[bad]),
            "conflict_constraint_thickness_m": float(consolidated_values[bad]),
            "maximum_boundary_constraint_thickness_m": float(
                np.max(consolidated_values[conflicts])
            ),
            "duplicate_centerline_cell_count": duplicate_count,
            "maximum_duplicate_surface_spread_m": max_duplicate_spread,
            "boundary_tolerance_m": float(boundary_tolerance_m),
        }

    if method not in {"jacobi", "conjugate-gradient"}:
        raise ValueError(f"unsupported-harmonic-method:{method}")
    if not np.isfinite(cg_relative_tolerance) or cg_relative_tolerance <= 0:
        raise ValueError("cg_relative_tolerance must be finite and positive")

    fixed = boundary.copy()
    result = np.zeros(inside.shape, dtype=float)
    result[consolidated_rows, consolidated_cols] = consolidated_values
    fixed[consolidated_rows, consolidated_cols] = True
    padded_inside = np.pad(
        inside.astype(float), 1, mode="constant", constant_values=0.0
    )
    neighbour_count = (
        padded_inside[:-2, 1:-1]
        + padded_inside[2:, 1:-1]
        + padded_inside[1:-1, :-2]
        + padded_inside[1:-1, 2:]
    )
    unknown = inside & ~fixed & (neighbour_count > 0)

    if method == "jacobi":
        previous = result.copy()
        for iteration in range(1, max_iterations + 1):
            padded = np.pad(previous, 1, mode="constant", constant_values=0.0)
            neighbour_sum = (
                padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
            )
            updated = previous.copy()
            updated[unknown] = neighbour_sum[unknown] / neighbour_count[unknown]
            updated[boundary] = 0.0
            updated[consolidated_rows, consolidated_cols] = consolidated_values
            delta = float(np.nanmax(np.abs(updated - previous)))
            previous = updated
            if delta <= tolerance_m:
                break
        output = previous
        status = "converged" if delta <= tolerance_m else "iteration-limit"
        padded_output = np.pad(
            output, 1, mode="constant", constant_values=0.0
        )
        output_neighbour_sum = (
            padded_output[:-2, 1:-1]
            + padded_output[2:, 1:-1]
            + padded_output[1:-1, :-2]
            + padded_output[1:-1, 2:]
        )
        solver_info = 0 if status == "converged" else int(iteration)
    else:
        unknown_count = int(unknown.sum())
        if unknown_count:
            diagonal = np.asarray(neighbour_count[unknown], dtype=float)
            fixed_field = result.copy()
            padded_fixed = np.pad(
                fixed_field, 1, mode="constant", constant_values=0.0
            )
            fixed_neighbour_sum = (
                padded_fixed[:-2, 1:-1]
                + padded_fixed[2:, 1:-1]
                + padded_fixed[1:-1, :-2]
                + padded_fixed[1:-1, 2:]
            )
            right_hand_side = np.asarray(
                fixed_neighbour_sum[unknown], dtype=float
            )

            def matvec(vector):
                field = np.zeros(inside.shape, dtype=float)
                field[unknown] = np.asarray(vector, dtype=float)
                padded = np.pad(field, 1, mode="constant", constant_values=0.0)
                neighbour_sum = (
                    padded[:-2, 1:-1]
                    + padded[2:, 1:-1]
                    + padded[1:-1, :-2]
                    + padded[1:-1, 2:]
                )
                return diagonal * vector - neighbour_sum[unknown]

            operator = LinearOperator(
                (unknown_count, unknown_count), matvec=matvec, dtype=float
            )
            preconditioner = LinearOperator(
                (unknown_count, unknown_count),
                matvec=lambda vector: np.asarray(vector, dtype=float) / diagonal,
                dtype=float,
            )
            callback_count = [0]

            def count_iteration(_):
                callback_count[0] += 1

            tolerance_keyword = (
                "rtol" if "rtol" in inspect.signature(cg).parameters else "tol"
            )
            cg_options = {
                tolerance_keyword: float(cg_relative_tolerance),
                "atol": 0.0,
                "maxiter": int(max_iterations),
                "M": preconditioner,
                "callback": count_iteration,
            }
            solved, solver_info = cg(
                operator,
                right_hand_side,
                x0=np.zeros(unknown_count, dtype=float),
                **cg_options,
            )
            iteration = callback_count[0]
            output = result.copy()
            output[unknown] = solved
        else:
            output = result.copy()
            solver_info = 0
            iteration = 0
        delta = np.nan
        if solver_info < 0:
            status = "solver-failure"
        elif solver_info > 0:
            status = "iteration-limit"
        elif not np.isfinite(output[unknown]).all():
            status = "solver-non-finite"
        else:
            status = "converged"

    output[~inside] = 0.0
    padded_output = np.pad(output, 1, mode="constant", constant_values=0.0)
    output_neighbour_sum = (
        padded_output[:-2, 1:-1]
        + padded_output[2:, 1:-1]
        + padded_output[1:-1, :-2]
        + padded_output[1:-1, 2:]
    )
    equation_residual = (
        neighbour_count[unknown] * output[unknown]
        - output_neighbour_sum[unknown]
    )
    maximum_equation_residual = (
        float(np.max(np.abs(equation_residual)))
        if equation_residual.size else 0.0
    )
    maximum_harmonic_defect = (
        float(np.max(np.abs(
            equation_residual / neighbour_count[unknown]
        ))) if equation_residual.size else 0.0
    )
    fixed_field = np.zeros(inside.shape, dtype=float)
    fixed_field[fixed] = output[fixed]
    padded_fixed = np.pad(fixed_field, 1, mode="constant", constant_values=0.0)
    fixed_neighbour_sum = (
        padded_fixed[:-2, 1:-1]
        + padded_fixed[2:, 1:-1]
        + padded_fixed[1:-1, :-2]
        + padded_fixed[1:-1, 2:]
    )[unknown]
    residual_l2 = float(np.linalg.norm(equation_residual))
    relative_l2_residual = residual_l2 / max(
        float(np.linalg.norm(fixed_neighbour_sum)), np.finfo(float).tiny
    )
    return output, {
        "status": status,
        "method": method,
        "iterations": int(iteration),
        "maximum_update_m": delta,
        "maximum_laplace_residual_m": maximum_harmonic_defect,
        "maximum_equation_residual_m": maximum_equation_residual,
        "maximum_harmonic_defect_m": maximum_harmonic_defect,
        "laplace_residual_l2_m": residual_l2,
        "relative_l2_residual": relative_l2_residual,
        "solver_info": int(solver_info),
        "cg_relative_tolerance": (
            float(cg_relative_tolerance)
            if method == "conjugate-gradient" else np.nan
        ),
        "boundary_cell_count": int(boundary.sum()),
        "centerline_cell_count": int(len(consolidated_values)),
        "input_centerline_cell_count": int(len(values)),
        "duplicate_centerline_cell_count": duplicate_count,
        "maximum_duplicate_surface_spread_m": max_duplicate_spread,
    }


def surface_quality_metrics(
    mask: np.ndarray,
    bed_elevation_m: np.ndarray,
    thickness_m: np.ndarray,
    *,
    centerline_rows: np.ndarray | None = None,
    centerline_cols: np.ndarray | None = None,
    centerline_reference_m: np.ndarray | None = None,
) -> dict:
    """Return QC metrics without silently repairing a candidate surface."""

    inside = np.asarray(mask, dtype=bool)
    bed = np.asarray(bed_elevation_m, dtype=float)
    thickness = np.asarray(thickness_m, dtype=float)
    if inside.shape != bed.shape or inside.shape != thickness.shape:
        raise ValueError("Mask, bed and thickness shapes must match")
    boundary = inside & ~binary_erosion(inside, border_value=0)
    valid = np.isfinite(thickness[inside]) & np.isfinite(bed[inside])
    centerline_rmse = np.nan
    if centerline_reference_m is not None:
        sampled = thickness[np.asarray(centerline_rows), np.asarray(centerline_cols)]
        reference = np.asarray(centerline_reference_m, dtype=float)
        centerline_rmse = float(np.sqrt(np.mean((sampled - reference) ** 2)))
    return {
        "finite_fraction_inside": float(valid.mean()) if valid.size else 0.0,
        "negative_thickness_cells": int(np.sum(thickness[inside] < 0)),
        "boundary_max_thickness_m": float(np.nanmax(thickness[boundary])) if boundary.any() else 0.0,
        "centerline_rmse_m": centerline_rmse,
        "volume_proxy_m3": float(np.nansum(np.maximum(thickness[inside], 0.0))),
    }
