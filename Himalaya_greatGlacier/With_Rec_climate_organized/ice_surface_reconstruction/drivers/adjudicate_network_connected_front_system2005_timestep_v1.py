"""Adjudicate 1 h versus 24 h connected-front zero-MB convergence."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
ONE_HOUR = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_system2005_zero_0p1y_v2/summary.json"
DAY = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_system2005_zero_0p1y_dt24h_v2/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_system2005_timestep_adjudication_v1"


def main() -> None:
    one = json.loads(ONE_HOUR.read_text(encoding="utf-8"))
    day = json.loads(DAY.read_text(encoding="utf-8"))
    rel_area = abs(float(day["final_area_m2"]) - float(one["final_area_m2"])) / float(one["final_area_m2"])
    gate = 1e-4
    passed = (
        one["status"] == "zero-mb-pass"
        and day["status"] == "zero-mb-pass"
        and rel_area <= gate
        and int(day["maximum_internal_active_gap_count"]) == 0
        and float(day["maximum_numerical_zero_canonicalization_volume_m3"]) == 0.0
    )
    summary = {
        "schema": "lia-191-network-connected-front-system2005-timestep-adjudication-v1",
        "status": "timestep-convergence-pass" if passed else "timestep-convergence-fail-mechanism-closed",
        "system_id": 2005,
        "one_hour_summary": str(ONE_HOUR),
        "twenty_four_hour_summary": str(DAY),
        "one_hour_final_area_m2": one["final_area_m2"],
        "twenty_four_hour_final_area_m2": day["final_area_m2"],
        "relative_final_area_difference": rel_area,
        "relative_difference_gate": gate,
        "selected_timestep_seconds": 86400.0 if passed else None,
        "five_year_authorized": passed,
        "steady_authorized": False,
        "historical_authorized": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
