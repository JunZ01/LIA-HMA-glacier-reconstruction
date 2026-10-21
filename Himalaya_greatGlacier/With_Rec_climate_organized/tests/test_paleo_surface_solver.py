from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paleo_surface import (  # noqa: E402
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
