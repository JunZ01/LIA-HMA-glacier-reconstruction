"""Adjudicate hourly versus six-hour system-2005 climate convergence."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
HOURLY = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_topology_adaptive_split_system2005_climate_1y_hourly_formal_v2/summary.json"
SIX = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_topology_adaptive_split_system2005_climate_1y_6hour_v1/summary.json"
DAILY = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_topology_adaptive_split_system2005_climate_1y_daily_v1/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_topology_adaptive_split_system2005_timestep_convergence_v1"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    hourly = json.loads(HOURLY.read_text(encoding="utf-8"))
    six = json.loads(SIX.read_text(encoding="utf-8"))
    daily = json.loads(DAILY.read_text(encoding="utf-8"))
    volume_rel = abs(six["final_volume_m3"] - hourly["final_volume_m3"]) / hourly["final_volume_m3"]
    area_rel = abs(six["final_area_m2"] - hourly["final_area_m2"]) / hourly["final_area_m2"]
    length_delta = abs(six["root"]["length_m"] - hourly["root"]["length_m"])
    topology_equal = (
        six["final_support_state"]["disconnected_tributary_branches"]
        == hourly["final_support_state"]["disconnected_tributary_branches"]
        and six["final_support_state"]["internal_gap_branches"]
        == hourly["final_support_state"]["internal_gap_branches"]
    )
    checks = {
        "hourly_pass": hourly["status"] == "climate-screen-pass",
        "six_hour_pass": six["status"] == "climate-screen-pass",
        "volume_relative_difference_le_0p1pct": volume_rel <= 0.001,
        "area_relative_difference_le_0p1pct": area_rel <= 0.001,
        "length_difference_le_one_50m_node": length_delta <= 50.0,
        "support_topology_equal": topology_equal,
        "daily_rejected": daily["status"] == "climate-technical-or-scientific-fail",
    }
    summary = {
        "schema": "lia-191-network-topology-adaptive-split-system2005-timestep-convergence-v1",
        "status": "six-hour-converged-authorized-for-50y" if all(checks.values()) else "six-hour-not-authorized",
        "system_id": 2005,
        "hourly_maximum_timestep_seconds": 3600.0,
        "six_hour_maximum_timestep_seconds": 21600.0,
        "daily_maximum_timestep_seconds": 86400.0,
        "volume_relative_difference": volume_rel,
        "area_relative_difference": area_rel,
        "root_length_difference_m": length_delta,
        "support_topology_equal": topology_equal,
        "daily_failure": daily.get("error"),
        "checks": checks,
        "next_action": "run one 50y steady screen at six-hour maximum timestep" if all(checks.values()) else "use hourly or close",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
