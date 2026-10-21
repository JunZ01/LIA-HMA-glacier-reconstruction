from __future__ import annotations

import pandas as pd

from ice_surface_reconstruction.drivers.diagnose_lia_191_curve_loop import diagnose_chains


def _row(peak: float, endpoint: float, stability: float = 0.01) -> dict:
    lia, rgi = 12_000.0, 10_000.0
    separation = lia - rgi
    return {
        "task_id": "test", "lia_unique_id": 1, "rgi_id": "RGI60-15.00001",
        "delta_t": 0.0, "summary_status": "ok", "technical_failure": False,
        "spinup_converged": True, "spinup_stability_fraction": stability,
        "lia_target_m": lia, "rgi_target_m": rgi,
        "lia_rgi_separation_m": separation, "forward_peak_m": peak,
        "forward_2000_m": endpoint, "e_lia_sep": abs(peak - lia) / separation,
        "e_rgi_sep": abs(endpoint - rgi) / separation,
        "amplitude_ratio_to_target": (peak - endpoint) / separation,
        "max_corridor_exceedance_m": 0.0,
    }


def test_spinup_length_is_not_required_or_scored() -> None:
    row = diagnose_chains(pd.DataFrame([_row(11_900.0, 10_100.0)])).iloc[0]
    assert row["curve_class"] == "joint_endpoint_candidate"
    assert bool(row["spinup_length_used_as_lia_constraint"]) is False


def test_peak_match_with_long_endpoint_is_insufficient_retreat() -> None:
    row = diagnose_chains(pd.DataFrame([_row(12_100.0, 10_800.0)])).iloc[0]
    assert row["curve_class"] == "insufficient_retreat_endpoint_too_long"


def test_inward_misses_are_amplitude_deficit() -> None:
    row = diagnose_chains(pd.DataFrame([_row(11_400.0, 10_600.0)])).iloc[0]
    assert row["curve_class"] == "amplitude_deficit"


def test_unstable_spinup_is_not_interpreted_as_curve_failure() -> None:
    row = diagnose_chains(pd.DataFrame([_row(11_900.0, 10_100.0, 0.03)])).iloc[0]
    assert row["curve_class"] == "spinup_not_converged"
