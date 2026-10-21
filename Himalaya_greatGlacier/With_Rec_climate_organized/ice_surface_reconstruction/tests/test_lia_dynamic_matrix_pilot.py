from __future__ import annotations

import pytest

from ice_surface_reconstruction.drivers.lia_dynamic_matrix_pilot import (
    normalize_resolution_cases,
    resolution_case_directory,
    result_row_from_audit,
)


def test_resolution_cases_decouple_map_and_flowline_spacing() -> None:
    pilot = {
        "resolution_cases": [
            {"map_dx_m": 50.0, "flowline_dx": 2.0},
            {"map_dx_m": 100.0, "flowline_dx": 1.0},
            {
                "map_dx_m": 100.0,
                "flowline_dx": 2.0,
                "run_dynamics": False,
            },
        ]
    }

    cases = normalize_resolution_cases(pilot, default_flowline_dx=2.0)

    assert [case["flowline_node_dx_m"] for case in cases] == [100.0, 100.0, 200.0]
    assert [case["run_dynamics"] for case in cases] == [True, True, False]
    assert resolution_case_directory(100.0, 1.0) == "map_100m_flowline_1"
    assert (
        resolution_case_directory(17.0, 2.0, run_dynamics=False)
        == "map_17m_flowline_2_mapping_only"
    )


def test_resolution_cases_keep_legacy_map_dx_configuration() -> None:
    cases = normalize_resolution_cases(
        {"map_dx_m": [35.0, 50.0]}, default_flowline_dx=2.0
    )

    assert [(case["map_dx_m"], case["flowline_dx"]) for case in cases] == [
        (35.0, 2.0),
        (50.0, 2.0),
    ]


def test_resolution_cases_reject_nonpositive_spacing() -> None:
    with pytest.raises(ValueError, match="flowline-dx-must-be-positive"):
        normalize_resolution_cases(
            {"resolution_cases": [{"map_dx_m": 100.0, "flowline_dx": 0.0}]},
            default_flowline_dx=2.0,
        )


def test_mapping_only_result_has_no_dynamic_metrics() -> None:
    audit = {
        "lia_unique_id": 553,
        "rgi_id": "RGI60-15.02846",
        "component_index": 0,
        "map_dx_m": 17.0,
        "flowline_dx": 2.0,
        "flowline_node_dx_m": 34.0,
        "run_dynamics": False,
        "routing_surface": "lia-surface",
        "flowline_build": {
            "network": {
                "flowline_count": 3,
                "junction_count": 2,
                "negative_junction_surface_count": 1,
            }
        },
        "area_error_fraction": 0.01,
        "volume_error_fraction": 0.0,
        "downstream_free_distance_m": 5000.0,
        "zero_mb_checkpoints": [],
        "geometry_preprocessing_seconds": 2.0,
        "cross_section_mapping_seconds": 1.0,
        "zero_mb_dynamics_seconds": 0.0,
        "wall_time_seconds": 3.0,
    }

    row = result_row_from_audit(audit)

    assert row["status"] == "ok"
    assert row["run_dynamics"] is False
    assert row["final_zero_mb_year"] is None
    assert row["negative_tributary_creation_m3"] is None
