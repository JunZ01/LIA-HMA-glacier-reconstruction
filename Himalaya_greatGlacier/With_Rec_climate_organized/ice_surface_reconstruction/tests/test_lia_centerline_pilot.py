"""Unit tests for LIA centerline raster and coordinate preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import Affine
from shapely.geometry import Polygon

from ice_surface_reconstruction.drivers.lia_centerline_pilot import (
    _glacier_exterior,
    _line_distance_metrics,
    _map_to_pixel_geometry,
    _pixel_to_map_line,
    _prepare_oggm_lia_geometry,
    build_routing_decisions,
)
from shapely.geometry import LineString


def test_map_pixel_round_trip_for_line() -> None:
    transform = Affine(10.0, 0.0, 95.0, 0.0, -10.0, 205.0)
    line = LineString([(0.0, 0.0), (2.0, 3.0)])
    mapped = _pixel_to_map_line(line, transform)
    restored = _map_to_pixel_geometry(mapped, transform)
    np.testing.assert_allclose(np.asarray(restored.coords), np.asarray(line.coords))


@pytest.mark.filterwarnings(
    "ignore:numpy.ndarray size changed, may indicate binary incompatibility:RuntimeWarning"
)
def test_rasterized_polygon_respects_hole() -> None:
    import salem

    grid = salem.Grid(
        proj=salem.wgs84, nxny=(7, 7), dxdy=(1.0, 1.0),
        x0y0=(0.0, 0.0), pixel_ref="center",
    )
    polygon = Polygon(
        [(0, 0), (6, 0), (6, 6), (0, 6)],
        holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]],
    )
    _, mask, _ = _prepare_oggm_lia_geometry(polygon, grid)
    assert mask[1, 1]
    assert not mask[3, 3]


def test_glacier_exterior_includes_outer_and_hole_boundaries() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:8, 1:8] = True
    mask[4, 4] = False
    exterior = _glacier_exterior(mask)
    assert exterior[1, 4]
    assert exterior[4, 3]
    assert not exterior[2, 2]


def test_network_distance_distinguishes_parallel_routes() -> None:
    line_a = LineString([(0.0, 0.0), (1000.0, 0.0)])
    line_b = LineString([(0.0, 100.0), (1000.0, 100.0)])
    metrics = _line_distance_metrics([line_a], [line_b])
    assert metrics["hausdorff_distance_m"] == 100.0
    assert metrics["mean_symmetric_network_distance_m"] == 100.0


def test_routing_decision_flags_only_above_one_pixel() -> None:
    comparisons = pd.DataFrame([
        {
            "rgi_id": "stable",
            "elevation_field_a": "raw-bed-farinotti2019",
            "elevation_field_b": "routing-taper-132m-bed-farinotti2019",
            "mean_symmetric_network_distance_m": 20.0,
        },
        {
            "rgi_id": "review",
            "elevation_field_a": "raw-bed-farinotti2019",
            "elevation_field_b": "routing-taper-132m-bed-farinotti2019",
            "mean_symmetric_network_distance_m": 80.0,
        },
    ])
    decisions = build_routing_decisions(
        ["stable", "review"], comparisons,
        {"stable": 66.0, "review": 66.0}, 1.0, 132.0,
    ).set_index("rgi_id")
    assert decisions.loc["stable", "centerline_route_status"] == "stable-pilot"
    assert (
        decisions.loc["review", "centerline_route_status"]
        == "centerline-structure-review"
    )
