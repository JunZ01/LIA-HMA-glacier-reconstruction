from __future__ import annotations

import numpy as np
import pandas as pd

from ice_surface_reconstruction.drivers.v3_strip_mean_rectangular_pipeline import (
    derive_strip_mean_rectangular_branch,
    select_dynamic_sample,
)


def test_strip_mean_branch_preserves_area_volume_and_mean_surface() -> None:
    rows = [
        {
            "node_index": 0,
            "mapping_active": False,
            "assigned_area_m2": 0.0,
            "assigned_volume_m3": 0.0,
            "source_surface_m": 120.0,
            "strip_mean_surface_m": np.nan,
        },
        {
            "node_index": 1,
            "mapping_active": True,
            "assigned_area_m2": 200.0,
            "assigned_volume_m3": 2000.0,
            "source_surface_m": 110.0,
            "strip_mean_surface_m": 115.0,
        },
        {
            "node_index": 2,
            "mapping_active": True,
            "assigned_area_m2": 400.0,
            "assigned_volume_m3": 8000.0,
            "source_surface_m": 100.0,
            "strip_mean_surface_m": 105.0,
        },
    ]
    result = derive_strip_mean_rectangular_branch(rows, node_dx_m=100.0)
    assert result["represented_area_m2"] == 600.0
    assert result["represented_volume_m3"] == 10000.0
    assert result["area_error_fraction"] == 0.0
    assert result["volume_error_fraction"] == 0.0
    np.testing.assert_allclose(result["thickness_m"], [0.0, 10.0, 20.0])
    np.testing.assert_allclose(result["surface_h_m"], [120.0, 115.0, 105.0])
    np.testing.assert_allclose(result["bed_h_m"], [120.0, 105.0, 85.0])
    assert np.all(result["surface_width_m"] > 0)


def test_dynamic_sample_contains_six_distinct_one_component_systems() -> None:
    frame = pd.DataFrame({
        "lia_unique_id": np.arange(1, 11),
        "status": ["static-pass"] * 10,
        "component_count": [1] * 10,
        "branch_count": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "source_area_km2": np.arange(10.0, 20.0),
        "strip_mean_hypsometry_error_fraction": np.linspace(0.005, 0.1, 10),
    })
    sample = select_dynamic_sample(frame, count=6)
    assert len(sample) == 6
    assert sample["lia_unique_id"].nunique() == 6
    assert sample["selection_reason"].notna().all()


def test_zero_control_branch_is_retained_as_zero_section() -> None:
    rows = [
        {
            "node_index": index,
            "mapping_active": False,
            "assigned_area_m2": 0.0,
            "assigned_volume_m3": 0.0,
            "source_surface_m": 120.0 - index,
            "strip_mean_surface_m": np.nan,
            "polygon_transverse_width_m": 50.0,
        }
        for index in range(3)
    ]
    result = derive_strip_mean_rectangular_branch(rows, node_dx_m=75.0)
    assert result["zero_control_branch"]
    np.testing.assert_allclose(result["section_m2"], 0.0)
    np.testing.assert_allclose(result["surface_width_m"], 50.0)
    assert result["represented_area_m2"] == 0.0
    assert result["represented_volume_m3"] == 0.0
