"""Tests for LIA 3-D pilot adapter helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

from ice_surface_reconstruction.drivers.gridded_bed import GriddedBed
from ice_surface_reconstruction.drivers.lia_3d_zero_boundary_pilot import (
    collect_network_constraints,
    classify_flowline_heads,
    filter_constraints_by_boundary_distance,
    flowline_objects_to_inputs,
    lia_gridded_bed,
)
from ice_surface_reconstruction.drivers.oggm_network_adapter import BranchProvenance


def test_lia_gridded_bed_replaces_only_path_mask() -> None:
    source = GriddedBed(
        bed_m=np.arange(25, dtype=float).reshape(5, 5),
        glacier_mask=np.zeros((5, 5), dtype=bool),
        slope=None,
        map_dx_m=10.0,
        bed_scenario="test",
    )
    lia_mask = np.ones((5, 5), dtype=bool)
    result = lia_gridded_bed(source, lia_mask)
    np.testing.assert_array_equal(result.bed_m, source.bed_m)
    np.testing.assert_array_equal(result.glacier_mask, lia_mask)
    assert result.slope.shape == source.bed_m.shape


def test_flowline_object_conversion_samples_scenario_bed() -> None:
    parent = SimpleNamespace(
        line=LineString([(1, 1), (2, 1), (3, 1)]),
        surface_h=np.array([110.0, 105.0, 100.0]),
        dx_meter=20.0,
        flows_to=None,
    )
    tributary = SimpleNamespace(
        line=LineString([(2, 3), (2, 2)]),
        surface_h=np.array([120.0, 110.0]),
        dx_meter=20.0,
        flows_to=parent,
    )
    tributary.__dict__["flows_to_point"] = Point(2, 1)
    bed_values = np.arange(25, dtype=float).reshape(5, 5)
    bed = GriddedBed(
        bed_values, np.ones((5, 5), bool), np.zeros((5, 5)), 10.0, "test"
    )
    inputs = flowline_objects_to_inputs([tributary, parent], bed)
    assert inputs[0].flows_to == 1
    assert inputs[0].flows_to_indice == 1
    np.testing.assert_allclose(inputs[1].bed_elevation_m, [6.0, 7.0, 8.0])


def test_same_branch_subpixel_repeats_are_aggregated() -> None:
    inputs = [SimpleNamespace(
        flowline_index=0,
        coords_xy=np.asarray([[0.1, 0.1], [0.4, 0.4], [1.0, 1.0]]),
    )]
    build = SimpleNamespace(
        provenance=[BranchProvenance("fl0", 0, 0, 2)],
        connector_audit=[],
    )
    solution = SimpleNamespace(branch_solutions={
        "fl0": SimpleNamespace(
            thickness_m=np.asarray([10.0, 11.0, 12.0]),
            surface_elevation_m=np.asarray([110.0, 111.0, 112.0]),
        )
    })
    rows, cols, _, _, meta = collect_network_constraints(inputs, build, solution)
    assert len(rows) == 2
    assert len(cols) == 2
    assert meta["input_constraint_count"] == 3
    assert meta["same_branch_repeated_cell_count"] == 1


def test_distinct_branch_raster_conflict_is_not_averaged() -> None:
    inputs = [
        SimpleNamespace(flowline_index=0, coords_xy=np.asarray([[1.0, 1.0]])),
        SimpleNamespace(flowline_index=1, coords_xy=np.asarray([[1.0, 1.0]])),
    ]
    build = SimpleNamespace(
        provenance=[
            BranchProvenance("fl0", 0, 0, 0),
            BranchProvenance("fl1", 1, 0, 0),
        ],
        connector_audit=[],
    )
    solution = SimpleNamespace(branch_solutions={
        "fl0": SimpleNamespace(
            thickness_m=np.asarray([10.0]), surface_elevation_m=np.asarray([110.0])
        ),
        "fl1": SimpleNamespace(
            thickness_m=np.asarray([20.0]), surface_elevation_m=np.asarray([130.0])
        ),
    })
    with pytest.raises(ValueError, match="rasterized-cross-branch-cell-conflict"):
        collect_network_constraints(inputs, build, solution)


def test_small_cross_branch_raster_collision_keeps_nearest_sample() -> None:
    inputs = [
        SimpleNamespace(flowline_index=0, coords_xy=np.asarray([[1.10, 1.10]])),
        SimpleNamespace(flowline_index=1, coords_xy=np.asarray([[1.40, 1.40]])),
    ]
    build = SimpleNamespace(
        provenance=[
            BranchProvenance("fl0", 0, 0, 0),
            BranchProvenance("fl1", 1, 0, 0),
        ],
        connector_audit=[],
    )
    solution = SimpleNamespace(branch_solutions={
        "fl0": SimpleNamespace(
            thickness_m=np.asarray([10.0]), surface_elevation_m=np.asarray([110.0])
        ),
        "fl1": SimpleNamespace(
            thickness_m=np.asarray([20.0]), surface_elevation_m=np.asarray([112.0])
        ),
    })
    _, _, thickness, surface, meta = collect_network_constraints(
        inputs,
        build,
        solution,
        cross_branch_surface_tolerance_m=3.0,
    )
    np.testing.assert_allclose(thickness, [10.0])
    np.testing.assert_allclose(surface, [110.0])
    assert meta["cross_branch_shared_cell_count"] == 1
    assert meta["accepted_cross_branch_collision_count"] == 1
    assert meta["maximum_accepted_cross_branch_surface_spread_m"] == 2.0


def test_boundary_filter_uses_metric_cell_center_distance() -> None:
    class IdentityGrid:
        @staticmethod
        def ij_to_crs(cols, rows):
            return np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)

    boundary = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]).boundary
    rows, cols, thickness, surface, metrics = filter_constraints_by_boundary_distance(
        np.asarray([1, 2, 5]),
        np.asarray([5, 5, 5]),
        np.asarray([30.0, 40.0, 50.0]),
        np.asarray([130.0, 140.0, 150.0]),
        grid=IdentityGrid(),
        boundary_geometry=boundary,
        domain_mask=np.ones((11, 11), dtype=bool),
        exclusion_distance_m=2.0,
    )
    np.testing.assert_array_equal(rows, [2, 5])
    np.testing.assert_array_equal(cols, [5, 5])
    np.testing.assert_allclose(thickness, [40.0, 50.0])
    np.testing.assert_allclose(surface, [140.0, 150.0])
    assert metrics["excluded_constraint_count"] == 1
    assert metrics["maximum_excluded_thickness_m"] == 30.0
    assert metrics["minimum_retained_boundary_distance_m"] == 2.0


def test_boundary_filter_uses_harmonic_raster_boundary_contract() -> None:
    class IdentityGrid:
        @staticmethod
        def ij_to_crs(cols, rows):
            return np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)

    boundary = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]).boundary
    domain_mask = np.ones((11, 11), dtype=bool)
    domain_mask[1, 5] = False
    rows, _, thickness, _, metrics = filter_constraints_by_boundary_distance(
        np.asarray([2, 5]),
        np.asarray([5, 5]),
        np.asarray([40.0, 50.0]),
        np.asarray([140.0, 150.0]),
        grid=IdentityGrid(),
        boundary_geometry=boundary,
        domain_mask=domain_mask,
        exclusion_distance_m=2.0,
    )
    np.testing.assert_array_equal(rows, [5])
    np.testing.assert_allclose(thickness, [50.0])
    assert metrics["metric_excluded_constraint_count"] == 0
    assert metrics["raster_boundary_excluded_constraint_count"] == 1


def test_flowline_head_gets_nearest_candidate_divide_label() -> None:
    class IdentityGrid:
        @staticmethod
        def ij_to_crs(cols, rows):
            return np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)

    flowlines = [SimpleNamespace(line=LineString([(9.5, 5.0), (5.0, 5.0)]))]
    segments = gpd.GeoDataFrame(
        {
            "boundary_class": [
                "outer-margin-zero-thickness-provisional",
                "ice-divide-candidate-review",
            ]
        },
        geometry=[LineString([(0, 0), (0, 10)]), LineString([(10, 0), (10, 10)])],
        crs="EPSG:3857",
    )
    rows = classify_flowline_heads(
        flowlines,
        grid=IdentityGrid(),
        boundary_segments=segments,
        exclusion_candidates_m=[0.25, 1.0],
    )
    assert rows[0]["candidate_divide"]
    assert rows[0]["head_distance_to_boundary_m"] == pytest.approx(0.5)
    assert not rows[0]["excluded_at_0.25m"]
    assert rows[0]["excluded_at_1m"]
