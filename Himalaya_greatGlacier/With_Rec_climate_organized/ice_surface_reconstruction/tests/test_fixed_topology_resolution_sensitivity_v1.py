from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from ice_surface_reconstruction.drivers.fixed_topology_resolution_sensitivity_v1 import (
    PhysicalConstraints,
    comparison_table,
    densify_physical_constraints,
    filter_physical_constraints_by_boundary_distance,
    physical_constraint_sha256,
    rasterize_physical_constraints,
)
from ice_surface_reconstruction.drivers.lia_system_grid import SystemGrid


def _grid(tmp_path, *, dx=100.0, x0=0.0, y0=0.0, size=20) -> SystemGrid:
    return SystemGrid(
        crs="EPSG:32645",
        x0=x0,
        y0=y0,
        nx=size,
        ny=size,
        dx=dx,
        dy=dx,
        json_path=tmp_path / "grid.json",
    )


def _constraints(source_ids, xy, thickness, surface) -> PhysicalConstraints:
    xy = np.asarray(xy, dtype=float)
    return PhysicalConstraints(
        source_ids=np.asarray(source_ids, dtype=str),
        x_m=xy[:, 0],
        y_m=xy[:, 1],
        thickness_m=np.asarray(thickness, dtype=float),
        surface_m=np.asarray(surface, dtype=float),
    )


def test_metric_boundary_filter_is_independent_of_target_grid() -> None:
    constraints = _constraints(
        ["a", "a", "a"],
        [(99.0, 500.0), (100.0, 500.0), (101.0, 500.0)],
        [10.0, 11.0, 12.0],
        [110.0, 111.0, 112.0],
    )
    polygon = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    retained, metadata = filter_physical_constraints_by_boundary_distance(
        constraints, polygon.boundary, 100.0
    )
    np.testing.assert_allclose(retained.x_m, [100.0, 101.0])
    assert metadata["physical_excluded_constraint_count"] == 1
    assert metadata["retained_physical_constraint_sha256"] == (
        physical_constraint_sha256(retained)
    )


def test_same_branch_collision_selects_nearest_point(tmp_path) -> None:
    constraints = _constraints(
        ["a", "a"], [(5.0, 5.0), (20.0, 20.0)], [10.0, 50.0], [110.0, 150.0]
    )
    rows, cols, thickness, surface, metadata = rasterize_physical_constraints(
        constraints, _grid(tmp_path), duplicate_surface_tolerance_m=0.001
    )
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(cols, [0])
    np.testing.assert_allclose(thickness, [10.0])
    np.testing.assert_allclose(surface, [110.0])
    assert metadata["same_branch_repeated_cell_count"] == 1


def test_cross_branch_comparison_uses_surface_and_never_averages(tmp_path) -> None:
    constraints = _constraints(
        ["b", "a"], [(5.0, 5.0), (5.0, 5.0)], [50.0, 10.0], [100.0005, 100.0]
    )
    _, _, thickness, surface, metadata = rasterize_physical_constraints(
        constraints, _grid(tmp_path), duplicate_surface_tolerance_m=0.001
    )
    # Equal-distance tie is resolved by source id; thickness is not averaged.
    np.testing.assert_allclose(thickness, [10.0])
    np.testing.assert_allclose(surface, [100.0])
    assert metadata["cross_branch_shared_cell_count"] == 1


def test_cross_branch_surface_conflict_is_hard_failure(tmp_path) -> None:
    constraints = _constraints(
        ["a", "b"], [(5.0, 5.0), (5.0, 5.0)], [10.0, 10.0], [100.0, 101.0]
    )
    with pytest.raises(
        ValueError, match="fixed-topology-physical-cross-branch-conflict"
    ):
        rasterize_physical_constraints(
            constraints, _grid(tmp_path), duplicate_surface_tolerance_m=0.001
        )


def test_noncoincident_cross_branch_cell_collision_selects_nearest(tmp_path) -> None:
    constraints = _constraints(
        ["a", "b"], [(5.0, 5.0), (30.0, 30.0)], [10.0, 50.0], [100.0, 150.0]
    )
    _, _, thickness, _, metadata = rasterize_physical_constraints(
        constraints, _grid(tmp_path), duplicate_surface_tolerance_m=0.001
    )
    np.testing.assert_allclose(thickness, [10.0])
    assert metadata["cross_branch_shared_cell_count"] == 1
    assert metadata["noncoincident_cross_branch_shared_cell_count"] == 1


def test_edge_nested_tie_uses_registered_greater_index_policy(tmp_path) -> None:
    constraints = _constraints(["a"], [(50.0, 50.0)], [10.0], [100.0])
    rows, cols, _, _, metadata = rasterize_physical_constraints(
        constraints,
        _grid(tmp_path, dx=50.0, x0=25.0, y0=25.0),
        duplicate_surface_tolerance_m=0.001,
    )
    np.testing.assert_array_equal(rows, [1])
    np.testing.assert_array_equal(cols, [1])
    assert metadata["physical_to_raster_tie_policy"].startswith("half-open")
    assert metadata["maximum_physical_to_cell_center_distance_m"] == pytest.approx(
        np.sqrt(2.0) * 25.0
    )


def test_densify_interpolates_fixed_geometry_thickness_and_surface() -> None:
    constraints = _constraints(
        ["a", "a"], [(0.0, 0.0), (100.0, 0.0)], [10.0, 30.0], [100.0, 140.0]
    )
    dense = densify_physical_constraints(constraints, maximum_spacing_m=25.0)
    np.testing.assert_allclose(dense.x_m, [0.0, 25.0, 50.0, 75.0, 100.0])
    np.testing.assert_allclose(dense.thickness_m, [10.0, 15.0, 20.0, 25.0, 30.0])
    np.testing.assert_allclose(dense.surface_m, [100.0, 110.0, 120.0, 130.0, 140.0])


def _comparison_rows(changes, *, hash_value="same") -> pd.DataFrame:
    rows = []
    for factor, value in zip((2, 4, 8), changes):
        rows.append({
            "rgi_id": "RGI60-15.00001",
            "lia_unique_id": 1,
            "component_index": 0,
            "bed_scenario": "farinotti2019",
            "resolution_factor": factor,
            "status": "success",
            "volume_km3": value,
            "mean_thickness_m": value * 100.0,
            "vector_area_normalized_mean_thickness_m": value * 100.0,
            "maximum_thickness_m": value * 300.0,
            "fixed_topology_sha256": hash_value,
            "fixed_physical_constraint_sha256": "anchors",
            "retained_physical_constraint_sha256": "constraints",
            "constraint_sampling_policy": "anchor-only",
        })
    return pd.DataFrame(rows)


def test_comparison_selects_factor2_only_when_both_steps_converge() -> None:
    comparison = comparison_table(_comparison_rows([1.0, 1.002, 1.003]))
    assert comparison.loc[0, "selected_minimum_resolution_factor"] == 2
    assert comparison.loc[0, "fixed_topology_resolution_verdict"] == (
        "fixed-topology-factor2-converged-through-factor8"
    )


def test_comparison_rejects_hash_mismatch_even_when_values_converge() -> None:
    rows = _comparison_rows([1.0, 1.0, 1.0])
    rows.loc[rows["resolution_factor"].eq(8), "fixed_topology_sha256"] = "changed"
    comparison = comparison_table(rows)
    assert comparison.loc[0, "fixed_topology_resolution_verdict"] == (
        "invalid-confounded-run-hash-mismatch"
    )
    assert np.isnan(comparison.loc[0, "selected_minimum_resolution_factor"])
