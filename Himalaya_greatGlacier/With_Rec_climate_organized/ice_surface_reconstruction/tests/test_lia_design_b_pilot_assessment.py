from __future__ import annotations

import pytest

from ice_surface_reconstruction.drivers.lia_design_b_pilot_assessment import (
    evaluate_v3_target_gate,
    relative_error,
)


def _row() -> dict:
    row = {
        "target_area_km2": 20.0,
        "target_volume_km3": 1.0,
        "target_main_flowline_length_m": 10000.0,
        "simulated_area_km2": 20.0,
        "simulated_volume_km3": 1.0,
        "simulated_main_length_m": 10000.0,
    }
    for quantile in (10, 25, 50, 75, 90):
        row[f"target_surface_q{quantile:02d}_m"] = 5000.0 + quantile
        row[f"simulated_surface_q{quantile:02d}_m"] = 5000.0 + quantile
    return row


def test_relative_error_is_signed() -> None:
    assert relative_error(12.0, 10.0) == pytest.approx(0.2)
    assert relative_error(8.0, 10.0) == pytest.approx(-0.2)


def test_full_gate_rejects_gross_volume_mismatch() -> None:
    row = _row()
    row["simulated_volume_km3"] = 2.4
    result = evaluate_v3_target_gate(row)
    assert result["v3_area_gate_pass"]
    assert result["v3_hypsometry_gate_pass"]
    assert not result["v3_volume_gate_pass"]
    assert not result["v3_gross_volume_screen_pass"]
    assert not result["v3_full_lia_target_gate_pass"]


def test_full_gate_accepts_consistent_state() -> None:
    result = evaluate_v3_target_gate(_row())
    assert result["v3_full_lia_target_gate_pass"]
