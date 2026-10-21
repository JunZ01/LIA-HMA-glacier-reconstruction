"""Tests for cross-boundary ridge evidence sampling."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon

from ice_surface_reconstruction.drivers.lia_boundary_ridge_evidence import (
    _classify_segment,
    sample_segment_ridge_evidence,
)


def test_synthetic_ridge_has_two_sided_and_geometry_support() -> None:
    yy, xx = np.mgrid[0:11, 0:11]
    topo = 100.0 - 2.0 * np.abs(xx - 5.0)
    grid = {"x0y0": [0.0, 0.0], "dxdy": [1.0, 1.0]}
    segment = LineString([(5.0, 1.0), (5.0, 9.0)])
    target = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])
    neighbor = Polygon([(5, 0), (10, 0), (10, 10), (5, 10)])
    samples = sample_segment_ridge_evidence(
        segment, topo, grid, target, neighbor,
        sample_spacing_m=1.0, offset_pixels=[2.0], minimum_relief_m=3.0,
    )
    assert samples["offset_2px_two_sided_descent"].all()
    assert samples["offset_2px_opposite_polygon_sides"].all()
    label, geometry_support, ridge_support = _classify_segment(
        samples, 2.0, 0.7, 0.5, 0.2
    )
    assert label == "ridge-supported-divide-candidate"
    assert geometry_support == 1.0
    assert ridge_support == 1.0


def test_flat_boundary_is_not_promoted_to_supported_divide() -> None:
    topo = np.ones((11, 11), dtype=float) * 100.0
    grid = {"x0y0": [0.0, 0.0], "dxdy": [1.0, 1.0]}
    segment = LineString([(5.0, 1.0), (5.0, 9.0)])
    target = Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])
    neighbor = Polygon([(5, 0), (10, 0), (10, 10), (5, 10)])
    samples = sample_segment_ridge_evidence(
        segment, topo, grid, target, neighbor,
        sample_spacing_m=1.0, offset_pixels=[2.0], minimum_relief_m=3.0,
    )
    label, _, ridge_support = _classify_segment(
        samples, 2.0, 0.7, 0.5, 0.2
    )
    assert label == "weak-ridge-divide-review"
    assert ridge_support == 0.0
