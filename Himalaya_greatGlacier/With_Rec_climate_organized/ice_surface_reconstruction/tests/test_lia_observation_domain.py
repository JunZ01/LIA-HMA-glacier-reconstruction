from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..drivers.lia_191_controller import score_saved_task
from ..drivers.lia_observation_domain import (
    infer_saved_series_overflow,
    observation_domain_metrics,
    saved_series_observation_domain_assessment,
)


def test_observation_length_is_undefined_when_thick_ice_enters_buffer() -> None:
    metrics = observation_domain_metrics(
        np.array([30.0, 20.0, 15.0, 11.0, 0.0]),
        dx_m=50.0,
        observation_domain_node_count=3,
    )

    assert metrics["observation_domain_overflow"] is True
    assert metrics["observation_domain_overflow_node_count"] == 1
    assert metrics["first_observation_domain_overflow_index"] == 3
    assert math.isnan(metrics["observation_length_m"])
    assert metrics["observation_domain_censored_length_m"] == 150.0


def test_legacy_series_infers_overflow_from_ten_metre_length() -> None:
    series = pd.DataFrame(
        {
            "length_gt_0m": [650.0, 1050.0],
            "length_gt_10m": [600.0, 800.0],
            "ice_node_count": [13, 21],
            "last_ice_index": [12, 20],
            "terminal_thickness_m": [5.0, 5.0],
        }
    )

    overflow = infer_saved_series_overflow(series, 13)

    assert overflow.tolist() == [False, True]


def test_controller_hard_rejects_observation_domain_overflow() -> None:
    config = {
        "workflow": {"spinup_stability_tolerance": 0.02},
        "acceptance": {
            "endpoint_tolerance_fraction_of_separation": 0.2,
            "corridor_buffer_fraction": 0.2,
        },
    }
    task = {
        "task_id": "overflow",
        "lia_unique_id": 3897,
        "task_state": "completed",
    }
    summary = {
        "status": "ok",
        "targets": {"lia_length_m": 12000.0, "rgi_length_m": 10000.0},
        "spinup_control": {"converged": True, "final_stability_fraction": 0.01},
        "observation_domain": {"node_count": 240},
    }
    series = pd.DataFrame(
        {
            "phase": ["forward", "forward"],
            "year": [1500, 2000],
            "length_m": [11900.0, 10100.0],
            "observation_domain_overflow": [False, True],
        }
    )

    score = score_saved_task(config, task, summary, series)

    assert score["accepted"] is False
    assert score["length_observation_valid"] is False
    assert score["acceptance_state"] == "observation-domain-overflow"
    assert math.isnan(score["objective_sep"])
    assert math.isnan(score["forward_peak_m"])
    assert math.isnan(score["forward_2000_m"])

def test_legacy_outside_domain_without_thickness_is_unresolved() -> None:
    series = pd.DataFrame(
        {
            "last_ice_index": [20],
            "terminal_thickness_m": [5.0],
            "ice_node_count": [21],
            "length_gt_10m": [600.0],
        }
    )

    assessment = saved_series_observation_domain_assessment(
        series, observation_domain_node_count=13, node_dx_m=50.0
    )

    assert assessment["overflow"] is False
    assert assessment["unresolved"] is True
    assert assessment["valid"] is False
    assert assessment["state"] == "observation-domain-unresolved"

def test_controller_hard_rejects_unassessed_observation_domain() -> None:
    config = {
        "workflow": {"spinup_stability_tolerance": 0.02},
        "acceptance": {
            "endpoint_tolerance_fraction_of_separation": 0.2,
            "corridor_buffer_fraction": 0.2,
        },
    }
    task = {"task_id": "unassessed", "lia_unique_id": 1, "task_state": "completed"}
    summary = {
        "status": "ok",
        "targets": {"lia_length_m": 12000.0, "rgi_length_m": 10000.0},
        "spinup_control": {"converged": True, "final_stability_fraction": 0.01},
    }
    series = pd.DataFrame(
        {"phase": ["forward", "forward"], "year": [1500, 2000], "length_m": [11900.0, 10100.0]}
    )

    score = score_saved_task(config, task, summary, series)

    assert score["accepted"] is False
    assert score["length_observation_valid"] is False
    assert score["acceptance_state"] == "observation-domain-unassessed"
    assert math.isnan(score["objective_sep"])

