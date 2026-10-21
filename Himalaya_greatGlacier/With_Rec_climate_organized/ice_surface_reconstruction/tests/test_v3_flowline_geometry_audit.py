from __future__ import annotations

import numpy as np

from ice_surface_reconstruction.drivers.v3_flowline_geometry_audit import (
    conservative_node_assignment,
    flowline_control_lengths_m,
    maximum_cumulative_hypsometry_error_fraction,
    polygon_transverse_widths_m,
)
from shapely.geometry import Polygon


def test_flowline_control_lengths_close_polyline_length() -> None:
    coordinates = np.asarray([[0.0, 0.0], [10.0, 0.0], [30.0, 0.0]])
    lengths = flowline_control_lengths_m(coordinates)
    np.testing.assert_allclose(lengths, [5.0, 15.0, 10.0])
    assert lengths.sum() == 30.0


def test_conservative_assignment_preserves_area_and_volume() -> None:
    result = conservative_node_assignment(
        cell_xy_m=np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]),
        cell_thickness_m=np.asarray([10.0, 20.0, 30.0]),
        cell_surface_m=np.asarray([110.0, 120.0, 130.0]),
        node_xy_m=np.asarray([[0.0, 0.0], [20.0, 0.0]]),
        node_thickness_m=np.asarray([10.0, 30.0]),
        node_control_length_m=np.asarray([10.0, 10.0]),
        cell_area_m2=100.0,
        maximum_node_to_ice_distance_m=15.0,
    )
    assert result["node_area_m2"].sum() == 300.0
    assert result["node_volume_m3"].sum() == 6000.0
    assert np.sum(result["section_m2"] * [10.0, 10.0]) == 6000.0
    np.testing.assert_allclose(result["node_surface_minimum_m"], [110.0, 130.0])
    np.testing.assert_allclose(result["node_surface_maximum_m"], [120.0, 130.0])
    np.testing.assert_allclose(
        result["node_surface_moment_m3"] / result["node_area_m2"],
        [115.0, 130.0],
    )


def test_assignment_distinguishes_excluded_and_unrepresented_nodes() -> None:
    result = conservative_node_assignment(
        cell_xy_m=np.asarray([[0.0, 0.0], [20.0, 0.0]]),
        cell_thickness_m=np.asarray([10.0, 30.0]),
        cell_surface_m=np.asarray([110.0, 130.0]),
        node_xy_m=np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]),
        node_thickness_m=np.asarray([10.0, 20.0, 30.0]),
        node_control_length_m=np.asarray([5.0, 10.0, 5.0]),
        cell_area_m2=100.0,
        maximum_node_to_ice_distance_m=15.0,
        eligible_node_mask=np.asarray([True, False, True]),
    )
    assert result["excluded_positive_node_count"] == 1
    assert result["unrepresented_positive_node_count"] == 0


def test_polygon_transverse_widths_use_local_normal_intersection() -> None:
    polygon = Polygon([(0.0, -5.0), (20.0, -5.0), (20.0, 5.0), (0.0, 5.0)])
    widths = polygon_transverse_widths_m(
        np.asarray([[2.0, 0.0], [10.0, 0.0], [18.0, 0.0]]), polygon
    )
    np.testing.assert_allclose(widths, 10.0)


def test_hypsometry_error_is_zero_for_identical_weighted_states() -> None:
    error = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=np.asarray([100.0, 200.0]),
        source_area_m2=np.asarray([10.0, 20.0]),
        node_surface_m=np.asarray([100.0, 200.0]),
        model_area_m2=np.asarray([10.0, 20.0]),
        bin_size_m=50.0,
    )
    assert error == 0.0
