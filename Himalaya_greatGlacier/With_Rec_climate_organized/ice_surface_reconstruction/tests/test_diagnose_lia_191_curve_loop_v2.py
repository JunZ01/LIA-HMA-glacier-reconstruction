from __future__ import annotations

import pandas as pd

from ice_surface_reconstruction.drivers.diagnose_lia_191_curve_loop_v2 import diagnose_chains


def test_nonconverged_summary_is_not_a_technical_failure() -> None:
    lia, rgi = 12_000.0, 10_000.0
    separation = lia - rgi
    scores = pd.DataFrame([{
        "task_id": "test", "lia_unique_id": 1, "rgi_id": "RGI60-15.00001",
        "delta_t": 0.0, "summary_status": "non_converged",
        "technical_failure": False, "spinup_converged": False,
        "spinup_stability_fraction": 0.03, "lia_target_m": lia,
        "rgi_target_m": rgi, "lia_rgi_separation_m": separation,
        "forward_peak_m": 11_900.0, "forward_2000_m": 10_100.0,
        "e_lia_sep": 0.05, "e_rgi_sep": 0.05,
        "amplitude_ratio_to_target": 0.9, "max_corridor_exceedance_m": 0.0,
    }])
    row = diagnose_chains(scores).iloc[0]
    assert row["curve_class"] == "spinup_not_converged"
    assert bool(row["technical_failure_bool"]) is False
