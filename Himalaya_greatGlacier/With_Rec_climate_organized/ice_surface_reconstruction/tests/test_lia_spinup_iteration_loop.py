from __future__ import annotations

import pandas as pd

from ice_surface_reconstruction.drivers.lia_spinup_iteration_loop import summarize_loop


def test_volume_trend_bracket_requests_bisection() -> None:
    trials = pd.DataFrame([
        {
            "lia_unique_id": 7,
            "prcp_mode": "off",
            "delta_t_c": -1.0,
            "length_relative_error": 0.02,
            "last_100yr_volume_trend_m3_per_year": -1.0,
            "final_volume_m3": 100.0,
        },
        {
            "lia_unique_id": 7,
            "prcp_mode": "off",
            "delta_t_c": 0.0,
            "length_relative_error": 0.03,
            "last_100yr_volume_trend_m3_per_year": 1.0,
            "final_volume_m3": 100.0,
        },
    ])
    trials["volume_trend_fraction_per_year"] = trials["last_100yr_volume_trend_m3_per_year"] / trials["final_volume_m3"]
    trials["volume_trend_fraction_per_century"] = 100.0 * trials["volume_trend_fraction_per_year"]
    trials["geometry_pass"] = True
    result = summarize_loop(trials, volume_trend_threshold_per_century=0.02).iloc[0]
    assert result["status"] == "needs-volume-bisection"
    assert result["next_delta_t_c"] == -0.5


def test_geometry_failure_is_reported() -> None:
    trials = pd.DataFrame([
        {
            "lia_unique_id": 8,
            "prcp_mode": "off",
            "delta_t_c": 0.0,
            "length_relative_error": 0.3,
            "last_100yr_volume_trend_m3_per_year": 0.0,
            "final_volume_m3": 100.0,
            "volume_trend_fraction_per_year": 0.0,
            "volume_trend_fraction_per_century": 0.0,
            "geometry_pass": False,
        },
    ])
    result = summarize_loop(trials).iloc[0]
    assert result["status"] == "no-length-match"


def test_joint_boundary_uses_nearest_length_and_volume_passes() -> None:
    trials = pd.DataFrame([
        {
            "lia_unique_id": 9,
            "prcp_mode": "off",
            "delta_t_c": 0.5,
            "length_relative_error": 0.08,
            "volume_trend_fraction_per_century": 0.05,
        },
        {
            "lia_unique_id": 9,
            "prcp_mode": "off",
            "delta_t_c": 0.6,
            "length_relative_error": 0.12,
            "volume_trend_fraction_per_century": 0.01,
        },
    ])
    result = summarize_loop(trials).iloc[0]
    assert result["status"] == "needs-joint-boundary-bisection"
    assert result["next_delta_t_c"] == 0.55
