"""Tests for explicit LIA boundary partitioning."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon

from ice_surface_reconstruction.drivers.lia_boundary_preflight import (
    classify_boundary_segments,
)


def test_boundary_partition_closes_and_marks_neighbor_side() -> None:
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    neighbor_boundary = LineString([(10, 0), (10, 10)])
    rows, metrics = classify_boundary_segments(
        polygon, [neighbor_boundary], tolerance_m=0.1
    )
    classes = {row["boundary_class"] for row in rows}
    assert "ice-divide-candidate-review" in classes
    assert "outer-margin-zero-thickness-provisional" in classes
    assert abs(metrics["candidate_divide_length_m"] - 10.0) < 0.3
    assert abs(metrics["exterior_partition_closure_error_m"]) < 1e-9


def test_nunatak_is_separate_zero_thickness_class() -> None:
    polygon = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]],
    )
    rows, metrics = classify_boundary_segments(polygon, [], tolerance_m=1.0)
    nunatak = [
        row for row in rows
        if row["boundary_class"] == "nunatak-zero-thickness-provisional"
    ]
    assert len(nunatak) == 1
    assert np.isclose(metrics["nunatak_boundary_length_m"], 8.0)
    assert abs(metrics["exterior_partition_closure_error_m"]) < 1e-9
