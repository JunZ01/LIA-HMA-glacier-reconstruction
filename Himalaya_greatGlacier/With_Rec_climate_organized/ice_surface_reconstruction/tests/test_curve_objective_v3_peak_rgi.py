from __future__ import annotations

import pandas as pd

from ice_surface_reconstruction.drivers.curve_objective_v3_peak_rgi import (
    CurvePeakRgiObjectiveConfig,
    score_curve_peak_rgi,
)


def _series(forward_lengths: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "phase": ["spinup", "spinup", "spinup", "forward", "forward", "forward"],
            "year": [0, 300, 500, 1470, 1800, 2000],
            "length_m": [14000.0, 14100.0, 14150.0, *forward_lengths],
            "volume_m3": [1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9],
        }
    )


def test_v3_accepts_peak_near_lia_and_2000_minimum() -> None:
    result = score_curve_peak_rgi(
        _series([14150.0, 15950.0, 14150.0]),
        lia_length_m=16019.207928275264,
        rgi_length_m=13880.49003465458,
    )

    assert result["accepted_curve_v3"] is True
    assert result["failure_reasons_v3"] == ""
    assert result["peak_year"] == 1800
    assert result["minimum_at_2000"] is True
    assert result["pass_peak_lia"] is True
    assert result["pass_rgi_endpoint"] is True


def test_v3_rejects_when_2000_is_not_forward_minimum() -> None:
    result = score_curve_peak_rgi(
        _series([22550.0, 23050.0, 22800.0]),
        lia_length_m=24146.83971067518,
        rgi_length_m=22357.119036079332,
    )

    assert result["pass_peak_lia"] is True
    assert result["pass_rgi_endpoint"] is True
    assert result["pass_forward_corridor"] is True
    assert result["pass_spinup_stability"] is True
    assert result["pass_minimum_at_2000"] is False
    assert result["accepted_curve_v3"] is False
    assert result["failure_reasons_v3"] == "minimum_at_2000"


def test_v3_can_disable_2000_minimum_gate_for_sensitivity() -> None:
    result = score_curve_peak_rgi(
        _series([22550.0, 23050.0, 22800.0]),
        lia_length_m=24146.83971067518,
        rgi_length_m=22357.119036079332,
        config=CurvePeakRgiObjectiveConfig(require_minimum_at_2000=False),
    )

    assert result["minimum_at_2000"] is False
    assert result["pass_minimum_at_2000"] is True
    assert result["accepted_curve_v3"] is True