"""Generic real-OGGM Cycle-4 state-shooting member runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import run_cycle4_state_shooting_3850_timing_v1 as core


REPO_ROOT = Path(__file__).resolve().parents[4]
DRIVER_VERSION = "cycle4-state-shooting-member-v1"
_ORIGINAL_SELECT_INPUTS = core._select_inputs
_ACTIVE_CONFIG = None


def _select_inputs(config):
    ref, stage2, pixel = _ORIGINAL_SELECT_INPUTS(config)
    ref = ref.copy()
    stage2 = stage2.copy()
    ref["lia_length_m"] = ref["lia_registered_length_m"]
    ref["rgi_length_m"] = ref["rgi_registered_length_m"]
    physics = config["parent_physics"]
    stage2["melt_f"] = float(physics["melt_f"])
    stage2["prcp_fac"] = float(physics["prcp_fac"])
    stage2["glen_a"] = float(physics["glen_a"])
    stage2["fs"] = float(physics["fs"])
    stage2["temp_bias"] = float(physics["temp_bias"])

    task_path = Path(config["parent_task_path"])
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if int(task["lia_unique_id"]) != int(config["system_id"]):
        raise RuntimeError("parent-task-system-mismatch")
    if str(task["rgi_id"]) != str(config["rgi_id"]):
        raise RuntimeError("parent-task-rgi-mismatch")
    for key, task_key in (("melt_f", "melt_f"), ("glen_a", "glen_a")):
        if task_key in task and float(task[task_key]) != float(physics[key]):
            raise RuntimeError(f"parent-task-physics-mismatch:{key}")
    return ref, stage2, pixel


def _identity_derived_geometry(baseline, candidate, *, active_thickness_epsilon_m):
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        raise RuntimeError("missing-active-member-config")
    epsilon = float(active_thickness_epsilon_m)
    base_active = np.asarray(baseline.thick, dtype=float) > epsilon
    candidate_active = np.asarray(candidate.thick, dtype=float) > epsilon
    frozen = {
        "bed_h_exact": core._arrays_equal(baseline.bed_h, candidate.bed_h),
        "widths_m_exact": core._arrays_equal(baseline.widths_m, candidate.widths_m),
        "dx_exact": baseline.dx == candidate.dx,
        "map_dx_exact": baseline.map_dx == candidate.map_dx,
        "nx_exact": baseline.nx == candidate.nx,
        "active_prefix_mask_exact": core._arrays_equal(base_active, candidate_active),
    }
    identity_requested = (
        float(_ACTIVE_CONFIG["q_v"]) == 1.0
        and float(_ACTIVE_CONFIG["z_shape"]) == 0.0
    )
    identity_arrays = {
        name: core._arrays_equal(getattr(baseline, name), getattr(candidate, name))
        for name in ("bed_h", "widths_m", "surface_h", "thick", "section")
    }
    identity_pass = all(identity_arrays.values()) if identity_requested else None
    passed = all(frozen.values()) and (identity_pass is not False)
    return {
        "passed": passed,
        "contract": "analytic state derived from exact Core baseline with frozen bed/width/dx/active prefix",
        "theta": {
            "q_v": float(_ACTIVE_CONFIG["q_v"]),
            "z_shape": float(_ACTIVE_CONFIG["z_shape"]),
            "delta_t_c": float(_ACTIVE_CONFIG["delta_t_c"]),
        },
        "frozen_geometry_checks": frozen,
        "identity_requested": identity_requested,
        "identity_pass": identity_pass,
        "identity_arrays": identity_arrays,
        "baseline_volume_m3": float(baseline.volume_m3),
        "candidate_volume_m3": float(candidate.volume_m3),
        "realized_q_v": float(candidate.volume_m3 / baseline.volume_m3),
        "baseline_area_m2": float(baseline.area_m2),
        "candidate_area_m2": float(candidate.area_m2),
    }


def run(config_path: Path):
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = json.loads(config_path.read_text(encoding="utf-8"))
    core.DRIVER_VERSION = DRIVER_VERSION
    core._select_inputs = _select_inputs
    core.geometry_reproduction = _identity_derived_geometry
    decision = core.run(config_path)
    output_dir = Path(_ACTIVE_CONFIG["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["member_id"] = _ACTIVE_CONFIG["member_id"]
        summary["generation"] = int(_ACTIVE_CONFIG["generation"])
        summary["parent_task_path"] = _ACTIVE_CONFIG["parent_task_path"]
        summary["parent_delta_t_c"] = float(_ACTIVE_CONFIG["parent_delta_t_c"])
        summary["geometry_provenance_pass"] = bool(summary.pop("geometry_identity_pass"))
        if summary.get("forward_observation_domain_valid"):
            summary["formal_endpoint_objective_sep"] = float(summary["e_lia_sep"]) + float(summary["e_rgi_sep"])
            residual = np.asarray(summary["whitened_residual"], dtype=float)
            summary["whitened_objective_l2"] = float(np.linalg.norm(residual))
            summary["whitened_objective_squared"] = float(residual @ residual)
        else:
            summary["formal_endpoint_objective_sep"] = None
            summary["whitened_objective_l2"] = None
            summary["whitened_objective_squared"] = None
        core._write_json(summary_path, summary)
    decision["member_id"] = _ACTIVE_CONFIG["member_id"]
    decision["generation"] = int(_ACTIVE_CONFIG["generation"])
    decision["historical_chain_kind"] = "Cycle4 Gen0 real-OGGM member"
    core._write_json(output_dir / "decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(json.dumps(core._json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
