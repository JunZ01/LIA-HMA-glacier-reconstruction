"""Read-only feasibility audit for the unchanged natural system-2005 state.

This audit never imports OGGM and never advances model time.  It verifies the
bitwise-reproduced natural year-9 state, excludes thickness-modified descendants,
and evaluates the registered temperature response on the frozen 0.0625 K grid.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
STATE = ROOT / "LIA_CURRENT_STATE.json"
POLICY = RESULTS / "lia_191_system2005_no_initial_thickness_policy_v7/summary.json"
NATURAL = RESULTS / "lia_191_system2005_policy_v6_year9_materialization_v1/summary.json"
REPRO = RESULTS / "lia_191_system2005_policy_v6_year9_reproduction_v1/summary.json"
COLD = RESULTS / "lia_191_network_topology_adaptive_split_system2005_steady_50y_coldm0p125_6hour_v1/summary.json"
WARM = RESULTS / "lia_191_network_topology_adaptive_split_system2005_steady_50y_6hour_v1/summary.json"
OUT = RESULTS / "lia_191_system2005_policy_v7_unchanged_natural_state_feasibility_v1"

TEMPERATURE_GRID = tuple(-0.3125 + 0.0625 * index for index in range(7))
MINIMUM_LENGTH_M = 25450.125918037134
VOLUME_CHANGE_LIMIT = 0.02


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    controller = read_json(STATE)
    policy = read_json(POLICY)
    natural = read_json(NATURAL)
    reproduction = read_json(REPRO)
    cold = read_json(COLD)
    warm = read_json(WARM)

    state_path = ROOT / natural["state"]
    cold_t = float(cold["stage2"]["delta_t_c"])
    warm_t = float(warm["stage2"]["delta_t_c"])
    cold_volume = float(cold["final_volume_m3"])
    warm_volume = float(warm["final_volume_m3"])
    cold_length = float(cold["root"]["length_m"])
    warm_length = float(warm["root"]["length_m"])
    initial_volume = float(natural["observed"]["volume_m3"])
    volume_slope = (warm_volume - cold_volume) / (warm_t - cold_t)
    length_slope = (warm_length - cold_length) / (warm_t - cold_t)

    grid = []
    for temperature in TEMPERATURE_GRID:
        final_volume = cold_volume + volume_slope * (temperature - cold_t)
        volume_change = final_volume / initial_volume - 1.0
        continuous_length = cold_length + length_slope * (temperature - cold_t)
        lattice_length = 50.0 * round(continuous_length / 50.0)
        grid.append(
            {
                "delta_t_c": temperature,
                "predicted_final_volume_m3": final_volume,
                "predicted_volume_change_fraction": volume_change,
                "predicted_continuous_length_m": continuous_length,
                "predicted_lattice_length_m": lattice_length,
                "volume_gate": abs(volume_change) <= VOLUME_CHANGE_LIMIT,
                "connected_length_gate": lattice_length >= MINIMUM_LENGTH_M,
                "joint_gate": (
                    abs(volume_change) <= VOLUME_CHANGE_LIMIT
                    and lattice_length >= MINIMUM_LENGTH_M
                ),
            }
        )

    volume_safe = [row for row in grid if row["volume_gate"]]
    best_volume_safe = max(volume_safe, key=lambda row: row["predicted_lattice_length_m"])
    joint = [row for row in grid if row["joint_gate"]]
    checks = {
        "controller_is_policy_v7_hold": (
            controller.get("current_focus", {}).get("mainline_family")
            == "policy-v7-unchanged-natural-state-only"
            and controller.get("next_action", {}).get("launch_authorized") is False
        ),
        "initial_thickness_modification_prohibited": (
            policy.get("user_decision") == "initial-thickness-modification-prohibited"
        ),
        "natural_materialization_pass": natural.get("status") == "year9-materialization-pass-reproduction-pending",
        "natural_reproduction_pass": reproduction.get("status") == "year9-independent-reproduction-pass",
        "natural_state_hash_matches": sha256(state_path) == natural["state_sha256"],
        "five_array_families_bitwise_reproduced": reproduction["checks"]["five_flowline_array_families_bitwise_equal"],
        "natural_connected_length_passes": float(natural["observed"]["length_m"]) >= MINIMUM_LENGTH_M,
        "natural_state_has_no_internal_gap": int(natural["observed"]["internal_gap_branch_count"]) == 0,
        "registered_temperature_endpoints_share_original_initial_volume": (
            float(cold["initial_volume_m3"]) == float(warm["initial_volume_m3"])
        ),
        "temperature_only_grid_has_no_joint_candidate": len(joint) == 0,
    }
    status = "natural-state-pass-temperature-only-infeasible" if all(checks.values()) else "audit-fail"
    summary = {
        "schema": "lia-191-system2005-policy-v7-unchanged-natural-state-feasibility-v1",
        "status": status,
        "system_id": 2005,
        "simulation_launched": False,
        "historical_run": False,
        "natural_state": {
            "path": str(state_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(state_path),
            "source_year": 9,
            "initial_thickness_modified": False,
            "volume_m3": initial_volume,
            "area_m2": float(natural["observed"]["area_m2"]),
            "connected_root_length_m": float(natural["observed"]["length_m"]),
            "free_downstream_m": float(natural["observed"]["free_downstream_m"]),
            "internal_gap_branch_count": int(natural["observed"]["internal_gap_branch_count"]),
        },
        "excluded_descendants": {
            "v2_uniform_lift_m": float(policy["ineligible_technical_outputs"]["candidate_v2_uniform_lift_m"]),
            "v3_uniform_lift_m": float(policy["ineligible_technical_outputs"]["candidate_v3_uniform_lift_m"]),
            "v4_inherits_v2_lift": bool(policy["ineligible_technical_outputs"]["candidate_v4_inherits_v2_lift"]),
        },
        "temperature_response": {
            "cold_endpoint_delta_t_c": cold_t,
            "warm_endpoint_delta_t_c": warm_t,
            "final_volume_slope_m3_per_k": volume_slope,
            "final_length_slope_m_per_k": length_slope,
            "grid_increment_k": 0.0625,
            "grid": grid,
            "joint_candidate_count": len(joint),
            "best_volume_safe_point": best_volume_safe,
            "remaining_length_deficit_m": max(
                0.0, MINIMUM_LENGTH_M - best_volume_safe["predicted_lattice_length_m"]
            ),
            "prediction_is_read_only_design_evidence_not_acceptance": True,
        },
        "checks": checks,
        "decision": {
            "temperature_only_mechanism": "closed-by-read-only-joint-gate",
            "natural_year9_state_remains_eligible": True,
            "next_mechanism_required": "natural-year9-state-plus-one-orthogonal-dynamic-control",
            "recommended_control": "glen_a_multiplier",
            "reason": "deformation rate can change downstream redistribution and length without directly altering the initial thickness or prescribing a mass source",
            "launch_authorized": False,
            "user_decision_required_before_physical_parameter_change": True,
        },
        "source_sha256": {
            "policy": sha256(POLICY),
            "natural_materialization": sha256(NATURAL),
            "natural_reproduction": sha256(REPRO),
            "cold_50y": sha256(COLD),
            "warm_50y": sha256(WARM),
        },
    }

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status != "audit-fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
