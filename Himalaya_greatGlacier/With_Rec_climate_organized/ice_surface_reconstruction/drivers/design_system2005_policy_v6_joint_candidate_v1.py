"""Design one bounded year-9 temperature/state joint-shooting candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUT = RESULTS / "lia_191_system2005_policy_v6_joint_candidate_design_v1"
COLD = RESULTS / "lia_191_network_topology_adaptive_split_system2005_steady_50y_coldm0p125_6hour_v1/summary.json"
WARM = RESULTS / "lia_191_network_topology_adaptive_split_system2005_steady_50y_6hour_v1/summary.json"
REPRO = RESULTS / "lia_191_system2005_policy_v6_year9_reproduction_v1/summary.json"


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1024 * 1024), b""):
            d.update(b)
    return d.hexdigest().upper()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    cold = json.loads(COLD.read_text(encoding="utf-8"))
    warm = json.loads(WARM.read_text(encoding="utf-8"))
    repro = json.loads(REPRO.read_text(encoding="utf-8"))
    if repro.get("status") != "year9-independent-reproduction-pass":
        raise RuntimeError("year9 reproduction not pass")
    cold_t = float(cold["stage2"]["delta_t_c"])
    warm_t = float(warm["stage2"]["delta_t_c"])
    cold_v = float(cold["final_volume_m3"])
    warm_v = float(warm["final_volume_m3"])
    cold_l = float(cold["root"]["length_m"])
    warm_l = float(warm["root"]["length_m"])
    slope_v = (warm_v - cold_v) / (warm_t - cold_t)
    slope_l = (warm_l - cold_l) / (warm_t - cold_t)
    selected_t = -0.25
    scale = 1.02
    year9_v = float(repro["observed"]["volume_m3"])
    predicted_final_v = cold_v + slope_v * (selected_t - cold_t)
    predicted_initial_v = year9_v * scale
    predicted_dv = (predicted_final_v - predicted_initial_v) / predicted_initial_v
    predicted_length = cold_l + slope_l * (selected_t - cold_t)
    colder_grid_t = -0.3125
    colder_final_v = cold_v + slope_v * (colder_grid_t - cold_t)
    colder_predicted_dv = (colder_final_v - predicted_initial_v) / predicted_initial_v
    checks = {
        "cold_and_warm_completed_50y": float(cold["years"]) == 50.0 and float(warm["years"]) == 50.0,
        "same_original_initial_volume": float(cold["initial_volume_m3"]) == float(warm["initial_volume_m3"]),
        "same_registered_physics_except_temperature": all(
            cold["stage2"][key] == warm["stage2"][key]
            for key in ("melt_f", "prcp_fac", "glen_a", "fs", "temp_melt_c", "base_temp_bias_c", "prcp_anomaly_mode")
        ),
        "year9_reproduction_pass": True,
        "selected_temperature_on_0p0625K_grid": abs(selected_t / 0.0625 - round(selected_t / 0.0625)) < 1e-12,
        "state_volume_scale_at_cap": scale == 1.02,
        "selected_linear_volume_prediction_within_2pct": abs(predicted_dv) <= 0.02,
        "next_colder_grid_prediction_exceeds_2pct": abs(colder_predicted_dv) > 0.02,
    }
    summary = {
        "schema": "lia-191-system2005-policy-v6-joint-candidate-design-v1",
        "status": "unique-bounded-joint-candidate-selected" if all(checks.values()) else "design-fail",
        "system_id": 2005,
        "candidate": {
            "source_state": "year9-natural-state",
            "delta_t_c": selected_t,
            "active_section_scale": scale,
            "scale_scope": "all-and-only-positive-section-control-volumes",
            "bed_width_topology_climate_parameters_except_delta_t_unchanged": True,
            "temperature_grid_increment_k": 0.0625,
        },
        "linear_response_diagnostic": {
            "temperature_endpoints_c": [cold_t, warm_t],
            "final_volume_slope_m3_per_k": slope_v,
            "final_length_slope_m_per_k": slope_l,
            "predicted_initial_volume_after_scale_m3": predicted_initial_v,
            "predicted_final_volume_m3": predicted_final_v,
            "predicted_volume_change_fraction": predicted_dv,
            "predicted_final_length_m_without_scale_effect": predicted_length,
            "next_colder_grid_temperature_c": colder_grid_t,
            "next_colder_predicted_volume_change_fraction": colder_predicted_dv,
            "inference_is_design_only_not_acceptance": True,
        },
        "stage_gates": {
            "static_first": True,
            "static_volume_scale_exact": 1.02,
            "area_and_support_unchanged": True,
            "hypsometry_cdf_max": 0.02,
            "no_new_reverse_slope": True,
            "preflight_years": 10.0,
            "preflight_absolute_volume_change_max": 0.015,
            "preflight_connected_length_min_m": 25500.0,
            "formal_years": 50.0,
            "formal_absolute_volume_change_max": 0.02,
            "formal_connected_length_min_m": 25450.125918037134,
            "formal_free_downstream_min_m": 5000.0,
            "formal_candidate_count_max": 1,
        },
        "checks": checks,
        "source_sha256": {
            "cold_50y": sha256(COLD), "warm_50y": sha256(WARM), "year9_reproduction": sha256(REPRO)
        },
        "simulation_launched": False,
        "historical_run": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] != "design-fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
