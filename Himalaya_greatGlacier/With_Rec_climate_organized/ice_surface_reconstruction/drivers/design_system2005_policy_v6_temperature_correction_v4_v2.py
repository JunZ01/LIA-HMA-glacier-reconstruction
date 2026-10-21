"""Select one temperature-only correction for the frozen system-2005 v2 state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
V2 = RESULTS / "lia_191_system2005_policy_v6_joint_candidate_v2/summary.json"
V3 = RESULTS / "lia_191_system2005_policy_v6_joint_candidate_v3/summary.json"
COLD = RESULTS / "lia_191_network_topology_adaptive_split_system2005_steady_50y_coldm0p125_6hour_v1/summary.json"
WARM = RESULTS / "lia_191_network_topology_adaptive_split_system2005_steady_50y_6hour_v1/summary.json"
STATE = RESULTS / "lia_191_system2005_policy_v6_joint_candidate_v2/joint_candidate_initial_state.pkl.gz"
OUT = RESULTS / "lia_191_system2005_policy_v6_temperature_correction_v4_design_v2"
MIN_LENGTH_M = 25450.125918037134


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    cold = json.loads(COLD.read_text(encoding="utf-8"))
    warm = json.loads(WARM.read_text(encoding="utf-8"))
    if not STATE.exists():
        raise FileNotFoundError(STATE)

    source_t = -0.25
    selected_t = -0.21875
    next_warmer_t = -0.1875
    cold_t = float(cold["stage2"]["delta_t_c"])
    warm_t = float(warm["stage2"]["delta_t_c"])
    volume_slope = (
        float(warm["volume_change_fraction"]) - float(cold["volume_change_fraction"])
    ) / (warm_t - cold_t)
    length_slope = (
        float(warm["root"]["length_m"]) - float(cold["root"]["length_m"])
    ) / (warm_t - cold_t)
    source_dv = float(v2["formal"]["volume_change_fraction"])
    source_length = float(v2["formal"]["final_length_m"])
    predicted_dv = source_dv + volume_slope * (selected_t - source_t)
    predicted_length = source_length + length_slope * (selected_t - source_t)
    predicted_lattice_length = 50.0 * round(predicted_length / 50.0)
    next_dv = source_dv + volume_slope * (next_warmer_t - source_t)
    next_length = source_length + length_slope * (next_warmer_t - source_t)
    next_lattice_length = 50.0 * round(next_length / 50.0)

    checks = {
        "v2_complete_volume_only_failure": (
            v2.get("status") == "formal-50y-fail-mechanism-closed"
            and v2["formal"]["checks"]["volume_change"] is False
            and v2["formal"]["checks"]["connected_length"] is True
        ),
        "v3_complete_and_uniform_lift_family_closed": (
            v3.get("status") == "formal-50y-fail-mechanism-closed"
            and v3["formal"]["checks"]["volume_change"] is False
            and v3["formal"]["checks"]["connected_length"] is True
        ),
        "v2_state_hash_fixed": sha256(STATE)
        == "1B8D9E606538F69E5FE19BEAAD75ACB6DF0C8DD77731F3703BDC8700F1C4DA24",
        "candidate_state_is_unmodified_v2": True,
        "selected_temperature_is_single_half_step": abs(selected_t - source_t - 0.03125) < 1e-12,
        "predicted_volume_inside_2pct": abs(predicted_dv) <= 0.02,
        "predicted_lattice_length_above_gate": predicted_lattice_length >= MIN_LENGTH_M,
        "next_warmer_lattice_length_below_gate": next_lattice_length < MIN_LENGTH_M,
    }
    status = "unique-temperature-only-candidate-selected" if all(checks.values()) else "design-fail"
    summary = {
        "schema": "lia-191-system2005-policy-v6-temperature-correction-v4-design-v2",
        "status": status,
        "system_id": 2005,
        "candidate": {
            "initial_state": str(STATE.relative_to(ROOT)).replace("\\", "/"),
            "initial_state_sha256": sha256(STATE),
            "initial_state_modified": False,
            "delta_t_c": selected_t,
            "other_stage2_modified": False,
            "historical_authorized": False,
        },
        "response": {
            "source_delta_t_c": source_t,
            "source_volume_change_fraction": source_dv,
            "source_final_length_m": source_length,
            "volume_change_slope_per_k": volume_slope,
            "length_slope_m_per_k": length_slope,
            "predicted_volume_change_fraction": predicted_dv,
            "predicted_continuous_length_m": predicted_length,
            "predicted_lattice_length_m": predicted_lattice_length,
            "next_warmer_delta_t_c": next_warmer_t,
            "next_warmer_predicted_volume_change_fraction": next_dv,
            "next_warmer_predicted_continuous_length_m": next_length,
            "next_warmer_predicted_lattice_length_m": next_lattice_length,
            "prediction_is_design_not_acceptance": True,
        },
        "checks": checks,
        "stages": {
            "preflight_years": 10,
            "formal_years": 50,
            "candidate_count_max": 1,
            "historical": False,
        },
        "simulation_launched": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

