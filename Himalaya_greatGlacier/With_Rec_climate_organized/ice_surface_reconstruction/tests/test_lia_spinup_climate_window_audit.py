from __future__ import annotations

import pandas as pd

from ice_surface_reconstruction.drivers.lia_spinup_climate_window_audit import (
    choose_spinup_window,
    rolling_window_table,
)


def test_rolling_window_table_uses_complete_consecutive_windows() -> None:
    annual = pd.Series([4.0, 3.0, 2.0, 1.0], index=[1501, 1502, 1503, 1504])
    result = rolling_window_table(annual, 3)
    assert result.to_dict("records") == [
        {"start_year": 1501, "end_year": 1503, "mean_temperature_c": 3.0},
        {"start_year": 1502, "end_year": 1504, "mean_temperature_c": 2.0},
    ]


def test_choose_spinup_window_replaces_warm_target_with_coldest() -> None:
    windows = pd.DataFrame(
        {
            "start_year": [1501, 1502, 1503],
            "end_year": [1530, 1531, 1532],
            "mean_temperature_c": [1.0, 0.5, 1.5],
        }
    )
    result = choose_spinup_window(1.2, windows)
    assert result["decision"] == "replace"
    assert result["selected_start_year"] == 1502
    assert result["selected_end_year"] == 1531


def test_choose_spinup_window_keeps_cold_end_target() -> None:
    windows = pd.DataFrame(
        {
            "start_year": [1501, 1502, 1503],
            "end_year": [1530, 1531, 1532],
            "mean_temperature_c": [1.0, 0.5, 1.5],
        }
    )
    result = choose_spinup_window(0.4, windows)
    assert result["decision"] == "keep"
    assert result["selected_start_year"] is None
    assert result["selected_mean_temperature_c"] == 0.4
