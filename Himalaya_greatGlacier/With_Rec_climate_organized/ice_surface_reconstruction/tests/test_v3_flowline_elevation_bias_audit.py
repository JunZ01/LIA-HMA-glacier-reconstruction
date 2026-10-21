from __future__ import annotations

import numpy as np

from ice_surface_reconstruction.drivers.v3_flowline_elevation_bias_audit import (
    summarize_node_bias,
    weighted_quantiles,
)


def test_weighted_quantiles_respect_dominant_area() -> None:
    result = weighted_quantiles(
        np.asarray([-10.0, 20.0]),
        np.asarray([1.0, 9.0]),
        (0.5,),
    )
    assert result[0] > 15.0


def test_summarize_node_bias_uses_assigned_area() -> None:
    rows = [
        {
            "mapping_active": True,
            "assigned_area_m2": 100.0,
            "node_minus_strip_mean_surface_m": -10.0,
            "strip_surface_range_m": 20.0,
            "strip_surface_standard_deviation_m": 5.0,
        },
        {
            "mapping_active": True,
            "assigned_area_m2": 300.0,
            "node_minus_strip_mean_surface_m": 10.0,
            "strip_surface_range_m": 40.0,
            "strip_surface_standard_deviation_m": 10.0,
        },
        {
            "mapping_active": False,
            "assigned_area_m2": 0.0,
            "node_minus_strip_mean_surface_m": np.nan,
            "strip_surface_range_m": np.nan,
            "strip_surface_standard_deviation_m": np.nan,
        },
    ]
    result = summarize_node_bias(rows)
    assert result["node_minus_strip_mean_area_weighted_mean_m"] == 5.0
    assert result["node_minus_strip_mean_area_weighted_absolute_mean_m"] == 10.0
    assert result["node_lower_than_strip_mean_area_fraction"] == 0.25
    assert result["node_higher_than_strip_mean_area_fraction"] == 0.75
