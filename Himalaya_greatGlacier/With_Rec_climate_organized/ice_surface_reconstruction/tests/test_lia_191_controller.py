from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..drivers.lia_191_controller import (
    _workflow_state_from_preflight,
    nearest_valid_climate_pixel,
    parameter_fingerprint,
    score_saved_task,
)


def test_parameter_fingerprint_is_order_independent() -> None:
    left = {"system": 7, "parameters": {"delta_t": 0.5, "melt_f": 4.0}}
    right = {"parameters": {"melt_f": 4.0, "delta_t": 0.5}, "system": 7}
    assert parameter_fingerprint(left) == parameter_fingerprint(right)


def test_nearest_valid_climate_pixel_skips_incomplete_nearest_cell() -> None:
    dataset = xr.Dataset(
        {
            "temp": (
                ("time", "lat", "lon"),
                np.array([[[np.nan, 1.0]], [[np.nan, 2.0]]]),
            ),
            "prcp": (
                ("time", "lat", "lon"),
                np.array([[[np.nan, 3.0]], [[np.nan, 4.0]]]),
            ),
        },
        coords={"time": [0, 1], "lat": [30.0], "lon": [90.0, 90.5]},
    )
    lat_index, lon_index, distance_m = nearest_valid_climate_pixel(
        dataset, longitude=90.0, latitude=30.0
    )
    assert lat_index == 0
    assert lon_index == 1
    assert 40_000 < distance_m < 50_000


def test_blank_existing_decision_preserves_dynamic_review_state() -> None:
    row = pd.Series(
        {
            "existing_decision": np.nan,
            "preflight_state": "blocked_dynamic_review",
        }
    )
    assert _workflow_state_from_preflight(row) == "blocked_dynamic_review"


def _config(tolerance):
    return {
        "workflow": {"spinup_stability_tolerance": 0.02},
        "acceptance": {
            "endpoint_tolerance_fraction_of_separation": tolerance,
            "corridor_buffer_fraction": 0.2,
        },
    }


def test_score_waits_when_tolerance_is_pending() -> None:
    task = {"task_id": "x", "lia_unique_id": 1, "task_state": "completed"}
    summary = {
        "status": "ok",
        "targets": {"lia_length_m": 12000.0, "rgi_length_m": 10000.0},
        "spinup_control": {"converged": True, "final_stability_fraction": 0.01},
        "observation_domain": {"node_count": 240},
    }
    series = pd.DataFrame(
        {"phase": ["forward", "forward"], "year": [1500, 2000], "length_m": [11900, 10100], "observation_domain_overflow": [False, False]}
    )
    score = score_saved_task(_config(None), task, summary, series)
    assert score["accepted"] is False
    assert score["acceptance_state"] == "pending-tolerance"
    assert score["e_lia_sep"] == 0.05
    assert score["e_rgi_sep"] == 0.05


def test_score_accepts_joint_separation_gate() -> None:
    task = {"task_id": "x", "lia_unique_id": 1, "task_state": "completed"}
    summary = {
        "status": "ok",
        "targets": {"lia_length_m": 12000.0, "rgi_length_m": 10000.0},
        "spinup_control": {"converged": True, "final_stability_fraction": 0.01},
        "observation_domain": {"node_count": 240},
    }
    series = pd.DataFrame(
        {"phase": ["forward", "forward"], "year": [1500, 2000], "length_m": [11900, 10100], "observation_domain_overflow": [False, False]}
    )
    score = score_saved_task(_config(0.2), task, summary, series)
    assert score["accepted"] is True
    assert score["objective_sep"] == 0.1
