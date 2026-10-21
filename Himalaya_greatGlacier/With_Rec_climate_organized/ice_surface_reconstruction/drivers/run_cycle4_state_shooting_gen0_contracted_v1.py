"""Run the one authorized Section-172 contracted Cycle-4 generation 0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import run_cycle4_state_shooting_gen0_v1 as core


REPO_ROOT = Path(__file__).resolve().parents[4]
_ORIGINAL_MAXIMIN_LHS = core._maximin_lhs


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _contracted_maximin_lhs(n: int, d: int, seed: int, designs: int):
    sample, audit = _ORIGINAL_MAXIMIN_LHS(n, d, seed, designs)
    sample = sample.copy()
    # The base orchestrator maps u_3 over parent +/-0.5 K.  Mapping
    # u_3 -> 0.5+0.5*u_3 gives exactly [parent,parent+0.5 K].
    sample[:, 2] = 0.5 + 0.5 * sample[:, 2]
    audit["third_coordinate_contraction"] = "u3_contracted=0.5+0.5*u3"
    return sample, audit


def _contracted_sensitivity(rows, representative, master):
    valid = rows[rows["valid"].astype(bool)].copy()
    result = {
        "system_id": int(representative["system_id"]),
        "valid_member_count": int(len(valid)),
        "method": "centered least-squares Jacobian using contracted-prior normalized theta and whitened residual vector",
    }
    if len(valid) < 4:
        result.update({"status": "insufficient-valid-members", "effective_rank": None, "singular_values": []})
        return result
    q_low, q_high = map(float, master["q_v_range"])
    z_low, z_high = map(float, master["z_shape_range"])
    dt_low = float(representative["parent_delta_t_c"])
    dt_high = min(dt_low + 0.5, 1.5)
    centers = np.asarray([(q_low + q_high) / 2, (z_low + z_high) / 2, (dt_low + dt_high) / 2])
    half = np.asarray([(q_high - q_low) / 2, (z_high - z_low) / 2, (dt_high - dt_low) / 2])
    x = (valid[["q_v", "z_shape", "delta_t_c"]].to_numpy(float) - centers) / half
    y = valid[["normalized_residual_peak", "normalized_residual_2000", "normalized_residual_stability"]].to_numpy(float)
    x -= np.mean(x, axis=0)
    y -= np.mean(y, axis=0)
    jacobian, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    threshold = max(float(master["svd_absolute_rank_threshold"]), float(master["svd_relative_rank_threshold"]) * float(singular[0]))
    result.update({"status": "ok", "jacobian_theta_by_observation": jacobian.tolist(), "singular_values": singular.tolist(), "rank_threshold": threshold, "effective_rank": int(np.count_nonzero(singular > threshold))})
    return result


def run(config_path: Path):
    overlay = json.loads(config_path.read_text(encoding="utf-8"))
    master = json.loads(_resolve(overlay["base_config"]).read_text(encoding="utf-8"))
    master["experiment_id"] = overlay["experiment_id"]
    master["q_v_range"] = overlay["q_v_range"]
    master["z_shape_range"] = overlay["z_shape_range"]
    master["contracted_delta_t_policy"] = overlay["delta_t_policy"]
    for representative in master["representatives"]:
        sid = str(representative["system_id"])
        representative["lhs_seed"] = int(overlay["seed_overrides"][sid])
        representative["parent_physics"].update(overlay["representative_overrides"][sid]["parent_physics"])
    master["paths"].update(overlay["path_overrides"])
    master["pre_run_contract"] = {
        "prior_contraction_ordinal": 1,
        "another_contraction_authorized": False,
        "gen1_authorized": False,
        "overlay_config": str(config_path),
    }
    work_root = _resolve(master["paths"]["workdir_root"])
    work_root.mkdir(parents=True, exist_ok=True)
    resolved = work_root / "resolved_master_config.json"
    resolved.write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")
    core._maximin_lhs = _contracted_maximin_lhs
    core._sensitivity = _contracted_sensitivity
    decision = core.run(resolved)
    output_root = _resolve(master["paths"]["output_root"])
    audit_path = output_root / "lhs_design_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for representative in master["representatives"]:
            sid = str(representative["system_id"])
            low = float(representative["parent_delta_t_c"])
            audit[sid]["delta_t_range_c"] = [low, min(low + 0.5, 1.5)]
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    decision_path = output_root / "generation_decision.json"
    decision["cycle4_closed"] = bool(decision["overall_invalid_fraction_of_planned_lower_bound"] > master["overall_invalid_stop_fraction"])
    decision["closure_reason"] = "repeated-invalid-fraction-over-30-percent" if decision["cycle4_closed"] else None
    decision["gen1_started"] = False
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "Himalaya_greatGlacier" / "With_Rec_climate_organized" / "ice_surface_reconstruction" / "config" / "cycle4_state_shooting_gen0_contracted_three_representatives_v1.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
