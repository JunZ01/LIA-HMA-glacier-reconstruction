"""Retry Candidate v2 with corrected full-candidate station year labels."""

from __future__ import annotations

import build_temperature_background_candidate_v2_allowed_years_v3 as compatibility


base = compatibility.base
original_grid_points_to_stations = base.forced.grid_points_to_stations


def grid_points_to_stations_with_matching_years(fields, lat, lon, mask, stations):
    previous_years = base.forced.ANALYSIS_YEARS
    if len(fields) == len(base.OUTPUT_YEARS):
        base.forced.ANALYSIS_YEARS = base.OUTPUT_YEARS
    try:
        return original_grid_points_to_stations(fields, lat, lon, mask, stations)
    finally:
        base.forced.ANALYSIS_YEARS = previous_years


base.forced.grid_points_to_stations = grid_points_to_stations_with_matching_years
base.OUTPUT = base.LOOP / "temperature_background_candidate_v2_allowed_years_v4"


if __name__ == "__main__":
    raise SystemExit(base.main())
