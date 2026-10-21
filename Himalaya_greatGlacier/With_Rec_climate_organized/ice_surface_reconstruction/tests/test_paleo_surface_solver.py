from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_ROOT = Path(__file__).resolve().parents[2]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from ice_surface_reconstruction.core import (  # noqa: E402
    SurfaceScenario,
    harmonic_extend_thickness,
    solve_flowline,
    surface_quality_metrics,
)


def test_flat_bed_profile_matches_continuous_limit() -> None:
    distance = np.linspace(0.0, 10_000.0, 1001)
    scenario = SurfaceScenario("central", 100_000.0, 0.8)
    solution = solve_flowline(distance, np.zeros_like(distance), scenario)
    expected = np.sqrt(
        2.0 * scenario.basal_shear_stress_pa * distance
        / (scenario.ice_density_kg_m3 * scenario.gravity_m_s2 * scenario.shape_factor)
    )
    assert solution.success
    assert np.nanmax(np.abs(solution.surface_elevation_m - expected)) < 1.0
    assert solution.diagnostics["maximum_surface_stress_residual_pa"] < 1e-6


def test_scalar_tau_keeps_v0_recurrence_exactly() -> None:
    distance = np.array([0.0, 75.0, 210.0, 410.0])
    bed = np.array([3200.0, 3210.0, 3225.0, 3250.0])
    scenario = SurfaceScenario("scalar", 93_000.0, 0.83)
    solution = solve_flowline(distance, bed, scenario)
    expected = np.empty_like(distance)
    expected[0] = bed[0]
    for i in range(1, len(distance)):
        dx = distance[i] - distance[i - 1]
        zsum = bed[i - 1] + bed[i]
        flux = 2.0 * dx * scenario.basal_shear_stress_pa / (
            scenario.ice_density_kg_m3
            * scenario.gravity_m_s2
            * scenario.shape_factor
        )
        constant = expected[i - 1] * (zsum - expected[i - 1]) - flux
        expected[i] = 0.5 * (zsum + np.sqrt(zsum**2 - 4.0 * constant))
    np.testing.assert_array_equal(solution.surface_elevation_m, expected)


def test_nonuniform_tau_over_shape_uses_trapezoidal_segment_average() -> None:
    distance = np.array([0.0, 100.0])
    bed = np.zeros(2)
    tau = np.array([60_000.0, 120_000.0])
    shape = np.array([0.6, 0.8])
    scenario = SurfaceScenario("profile", tau, shape)
    solution = solve_flowline(distance, bed, scenario)
    segment_kappa = 0.5 * (tau[0] / shape[0] + tau[1] / shape[1])
    expected_surface = np.sqrt(
        2.0 * distance[1] * segment_kappa
        / (scenario.ice_density_kg_m3 * scenario.gravity_m_s2)
    )
    assert solution.success
    assert solution.surface_elevation_m[1] == pytest.approx(expected_surface)
    assert solution.diagnostics["maximum_surface_stress_residual_pa"] < 1e-6


def test_high_altitude_translation_does_not_change_thickness() -> None:
    distance = np.array([0.0, 57.0, 171.0, 390.0])
    bed = np.array([0.0, 7.0, 13.0, 32.0])
    scenario = SurfaceScenario("translation", 100_000.0, 0.8)
    low = solve_flowline(distance, bed, scenario)
    high = solve_flowline(distance, bed + 6000.0, scenario)
    assert low.success and high.success
    np.testing.assert_allclose(high.thickness_m, low.thickness_m, atol=1e-10)


def test_nonfinite_bed_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        solve_flowline(
            np.array([0.0, 100.0]),
            np.array([0.0, np.nan]),
            SurfaceScenario("nan", 100_000.0, 0.8),
        )


def test_negative_input_is_not_repaired() -> None:
    distance = np.arange(3, dtype=float)
    solution = solve_flowline(
        distance,
        np.zeros(3),
        SurfaceScenario("test", 100_000.0, 0.8),
        start_surface_m=-1.0,
    )
    assert solution.status == "negative-start-thickness"


def test_tributary_start_inherits_parent_surface() -> None:
    distance = np.linspace(0.0, 2000.0, 101)
    bed = np.linspace(2500.0, 2700.0, 101)
    solution = solve_flowline(
        distance,
        bed,
        SurfaceScenario("test", 80_000.0, 0.8),
        start_surface_m=2600.0,
    )
    assert solution.success
    assert solution.surface_elevation_m[0] == 2600.0


def test_harmonic_extension_preserves_boundary_and_centerline() -> None:
    mask = np.ones((21, 21), dtype=bool)
    rows = np.arange(1, 20)
    cols = np.full(rows.shape, 10)
    values = np.linspace(0.0, 100.0, len(rows))
    thickness, diagnostics = harmonic_extend_thickness(mask, rows, cols, values)
    metrics = surface_quality_metrics(
        mask,
        np.zeros_like(thickness),
        thickness,
        centerline_rows=rows,
        centerline_cols=cols,
        centerline_reference_m=values,
    )
    assert diagnostics["status"] == "converged"
    assert metrics["boundary_max_thickness_m"] == 0.0
    assert metrics["centerline_rmse_m"] == 0.0
    assert metrics["finite_fraction_inside"] == 1.0


def test_harmonic_extension_consolidates_consistent_duplicate_cells() -> None:
    mask = np.ones((11, 11), dtype=bool)
    thickness, diagnostics = harmonic_extend_thickness(
        mask,
        np.array([5, 5]),
        np.array([5, 5]),
        np.array([100.0, 100.0005]),
        centerline_surface_m=np.array([2600.0, 2600.0005]),
        duplicate_tolerance_m=0.001,
    )
    assert diagnostics["status"] == "converged"
    assert diagnostics["duplicate_centerline_cell_count"] == 1
    assert diagnostics["centerline_cell_count"] == 1
    assert np.isclose(thickness[5, 5], 100.00025)


def test_harmonic_extension_rejects_duplicate_surface_conflict() -> None:
    mask = np.ones((11, 11), dtype=bool)
    thickness, diagnostics = harmonic_extend_thickness(
        mask,
        np.array([5, 5]),
        np.array([5, 5]),
        np.array([100.0, 100.0]),
        centerline_surface_m=np.array([2600.0, 2601.0]),
        duplicate_tolerance_m=0.01,
    )
    assert diagnostics["status"] == "centerline-cell-conflict"
    assert diagnostics["conflict_row"] == 5
    assert diagnostics["conflict_col"] == 5
    assert np.isnan(thickness[mask]).all()


def test_harmonic_extension_rejects_positive_zero_boundary_constraint() -> None:
    mask = np.ones((11, 11), dtype=bool)
    _, diagnostics = harmonic_extend_thickness(
        mask,
        np.array([0]),
        np.array([5]),
        np.array([10.0]),
    )
    assert diagnostics["status"] == "centerline-boundary-conflict"


def test_conjugate_gradient_matches_tightly_converged_jacobi_solution() -> None:
    mask = np.ones((31, 31), dtype=bool)
    rows = np.arange(2, 29)
    cols = np.full(rows.shape, 15)
    values = np.linspace(10.0, 120.0, len(rows))
    jacobi, jacobi_diagnostics = harmonic_extend_thickness(
        mask,
        rows,
        cols,
        values,
        max_iterations=100_000,
        tolerance_m=1e-9,
    )
    conjugate_gradient, cg_diagnostics = harmonic_extend_thickness(
        mask,
        rows,
        cols,
        values,
        method="conjugate-gradient",
        max_iterations=10_000,
        cg_relative_tolerance=1e-12,
    )
    assert jacobi_diagnostics["status"] == "converged"
    assert cg_diagnostics["status"] == "converged"
    assert cg_diagnostics["method"] == "conjugate-gradient"
    assert cg_diagnostics["maximum_laplace_residual_m"] < 1e-8
    np.testing.assert_allclose(conjugate_gradient, jacobi, atol=1e-6)


def test_harmonic_extension_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unsupported-harmonic-method"):
        harmonic_extend_thickness(
            np.ones((5, 5), dtype=bool),
            np.asarray([2]),
            np.asarray([2]),
            np.asarray([10.0]),
            method="not-a-method",
        )


def test_harmonic_extension_rejects_constraint_outside_mask() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    with pytest.raises(ValueError, match="centerline-constraint-outside-mask"):
        harmonic_extend_thickness(
            mask,
            np.asarray([0]),
            np.asarray([0]),
            np.asarray([100.0]),
            method="conjugate-gradient",
        )


def test_conjugate_gradient_iteration_limit_is_not_success() -> None:
    mask = np.ones((31, 31), dtype=bool)
    _, diagnostics = harmonic_extend_thickness(
        mask,
        np.asarray([15]),
        np.asarray([15]),
        np.asarray([100.0]),
        method="conjugate-gradient",
        max_iterations=1,
        cg_relative_tolerance=1e-14,
    )
    assert diagnostics["status"] == "iteration-limit"
    assert diagnostics["solver_info"] == 1
