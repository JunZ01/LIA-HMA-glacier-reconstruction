"""Orchestrate deterministic maximin-LHS Cycle-4 generation 0."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.run_cycle4_state_shooting_member_v1"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _maximin_lhs(n: int, d: int, seed: int, designs: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    best = None
    best_distance = -math.inf
    best_index = -1
    for index in range(int(designs)):
        sample = np.empty((n, d), dtype=float)
        for column in range(d):
            sample[:, column] = (rng.permutation(n) + rng.random(n)) / n
        diff = sample[:, None, :] - sample[None, :, :]
        distance = np.sqrt(np.sum(diff * diff, axis=2))
        distance[np.diag_indices(n)] = np.inf
        minimum = float(np.min(distance))
        if minimum > best_distance:
            best = sample.copy()
            best_distance = minimum
            best_index = index
    return best, {"seed": int(seed), "candidate_designs": int(designs), "selected_design_index": best_index, "minimum_normalized_pair_distance": best_distance}


def _scale(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return float(lower) + values * (float(upper) - float(lower))


def _member_config(master: dict[str, Any], representative: dict[str, Any], member_id: str, theta: list[float], member_dir: Path, workdir: Path) -> dict[str, Any]:
    contract = master["member_contract"]
    parent_task = _resolve(representative["parent_task_path"])
    return {
        "experiment_id": master["experiment_id"],
        "generation": int(master["generation"]),
        "member_id": member_id,
        "system_id": int(representative["system_id"]),
        "rgi_id": representative["rgi_id"],
        "parent_task_path": str(parent_task),
        "parent_delta_t_c": float(representative["parent_delta_t_c"]),
        "parent_physics": representative["parent_physics"],
        "ensemble_role": contract["ensemble_role"],
        "climate_id": contract["climate_id"],
        "q_v": float(theta[0]),
        "z_shape": float(theta[1]),
        "delta_t_c": float(theta[2]),
        "temp_melt_c": float(representative["parent_physics"]["temp_melt_c"]),
        "prcp_mode": contract["prcp_mode"],
        "shooting_window_start": contract["shooting_window_start"],
        "shooting_window_end": contract["shooting_window_end"],
        "shooting_years": contract["shooting_years"],
        "forward_start_year": contract["forward_start_year"],
        "forward_end_year": contract["forward_end_year"],
        "zero_mb_checkpoints_years": contract["zero_mb_checkpoints_years"],
        "fixed_dt_days": contract["fixed_dt_days"],
        "geometry": contract["geometry"],
        "gates": contract["gates"],
        "paths": {
            "facts": str(_resolve(master["paths"]["facts"])),
            "elevation_band_root": str(_resolve(master["paths"]["elevation_band_root"])),
            "references": str(_resolve(master["paths"]["references"])),
            "stage2_members": str(_resolve(master["paths"]["stage2_members"])),
            "climate_pixels": str(_resolve(master["paths"]["climate_pixels"])),
            "reconstruction_climate": str(_resolve(master["paths"]["reconstruction_climate"])),
            "source_gdir_root": str(_resolve(master["paths"]["source_gdir_root"])),
            "workdir": str(workdir),
            "output_dir": str(member_dir),
        },
    }


def _run_member(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    command = [sys.executable, "-m", MODULE, "--config", str(task["config_path"])]
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=360)
        returncode = int(result.returncode)
        stdout = result.stdout
        stderr = result.stderr
        exception = None if returncode == 0 else f"subprocess-returncode:{returncode}"
    except subprocess.TimeoutExpired as exc:
        returncode = -9
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        exception = "subprocess-timeout-360s"
    task["member_dir"].mkdir(parents=True, exist_ok=True)
    (task["member_dir"] / "runner_stdout.log").write_text(stdout, encoding="utf-8")
    (task["member_dir"] / "runner_stderr.log").write_text(stderr, encoding="utf-8")
    return {"member_id": task["member_id"], "system_id": task["system_id"], "returncode": returncode, "exception": exception, "controller_runtime_s": time.perf_counter() - started, "member_dir": str(task["member_dir"])}


def _summarize_member(task: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    member_dir = Path(execution["member_dir"])
    summary_path = member_dir / "summary.json"
    decision_path = member_dir / "decision.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
    technically_valid = bool(
        execution["returncode"] == 0
        and summary
        and summary.get("geometry_provenance_pass")
        and summary.get("static_pass")
        and summary.get("zero_mb_pass")
        and summary.get("shooting_valid")
        and summary.get("forward_observation_domain_valid")
    )
    row = {
        "system_id": task["system_id"], "member_id": task["member_id"],
        "q_v": task["theta"][0], "z_shape": task["theta"][1], "delta_t_c": task["theta"][2],
        "valid": technically_valid, "returncode": execution["returncode"],
        "exception": execution["exception"] or decision.get("reason"),
        "controller_runtime_s": execution["controller_runtime_s"],
        "signed_stability": summary.get("signed_stability"),
        "shooting_stable": summary.get("shooting_stable"),
        "peak_length_m": summary.get("forward_peak_length_m"),
        "length_2000_m": summary.get("forward_2000_length_m"),
        "e_lia_total": summary.get("e_lia_total"), "e_rgi_total": summary.get("e_rgi_total"),
        "e_lia_sep": summary.get("e_lia_sep"), "e_rgi_sep": summary.get("e_rgi_sep"),
        "formal_endpoint_objective_sep": summary.get("formal_endpoint_objective_sep"),
        "whitened_objective_l2": summary.get("whitened_objective_l2"),
        "formal_pass": summary.get("formal_dual_endpoint_and_stability_pass", False),
        "normalized_residual_peak": None, "normalized_residual_2000": None, "normalized_residual_stability": None,
        "initial_area_m2": None, "initial_volume_m3": None,
        "shooting_end_area_m2": None, "shooting_end_volume_m3": None,
        "forward_end_area_m2": None, "forward_end_volume_m3": None,
        "member_dir": str(member_dir),
    }
    residual = summary.get("whitened_residual")
    if residual and len(residual) == 3:
        row["normalized_residual_peak"], row["normalized_residual_2000"], row["normalized_residual_stability"] = residual
    static_path = member_dir / "static_summary.json"
    if static_path.exists():
        static = json.loads(static_path.read_text(encoding="utf-8"))
        row["initial_area_m2"] = static.get("metrics", {}).get("candidate_area_m2")
        row["initial_volume_m3"] = static.get("metrics", {}).get("candidate_volume_m3")
    for filename, prefix in (("shooting_timeseries.csv", "shooting_end"), ("forward_timeseries.csv", "forward_end")):
        path = member_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
            if len(frame):
                row[f"{prefix}_area_m2"] = float(frame.iloc[-1]["area_m2"])
                row[f"{prefix}_volume_m3"] = float(frame.iloc[-1]["volume_m3"])
    return row


def _sensitivity(rows: pd.DataFrame, representative: dict[str, Any], master: dict[str, Any]) -> dict[str, Any]:
    valid = rows[rows["valid"].astype(bool)].copy()
    result = {"system_id": int(representative["system_id"]), "valid_member_count": int(len(valid)), "method": "centered least-squares Jacobian from prior-half-range normalized theta to whitened residual vector"}
    if len(valid) < 4:
        result.update({"status": "insufficient-valid-members", "effective_rank": None, "singular_values": []})
        return result
    dt_low = max(master["global_delta_t_range_c"][0], representative["parent_delta_t_c"] - 0.5)
    dt_high = min(master["global_delta_t_range_c"][1], representative["parent_delta_t_c"] + 0.5)
    centers = np.asarray([1.0, 0.0, (dt_low + dt_high) / 2.0])
    half = np.asarray([0.3, 1.0, (dt_high - dt_low) / 2.0])
    x = (valid[["q_v", "z_shape", "delta_t_c"]].to_numpy(float) - centers) / half
    y = valid[["normalized_residual_peak", "normalized_residual_2000", "normalized_residual_stability"]].to_numpy(float)
    x -= np.mean(x, axis=0)
    y -= np.mean(y, axis=0)
    jacobian, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    threshold = max(float(master["svd_absolute_rank_threshold"]), float(master["svd_relative_rank_threshold"]) * float(singular[0]))
    result.update({"status": "ok", "jacobian_theta_by_observation": jacobian.tolist(), "singular_values": singular.tolist(), "rank_threshold": threshold, "effective_rank": int(np.count_nonzero(singular > threshold))})
    return result


def run(config_path: Path) -> dict[str, Any]:
    master = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = _resolve(master["paths"]["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refuse-nonempty-versioned-output:{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    workdir_root = _resolve(master["paths"]["workdir_root"])
    workdir_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "resolved_master_config.json", master)

    all_tasks = []
    design_audits = {}
    for rep in master["representatives"]:
        sid = int(rep["system_id"])
        sample, audit = _maximin_lhs(int(master["members_per_system"]), 3, int(rep["lhs_seed"]), int(master["maximin_lhs_candidate_designs"]))
        dt_low = max(master["global_delta_t_range_c"][0], rep["parent_delta_t_c"] - 0.5)
        dt_high = min(master["global_delta_t_range_c"][1], rep["parent_delta_t_c"] + 0.5)
        theta = np.column_stack((_scale(sample[:, 0], *master["q_v_range"]), _scale(sample[:, 1], *master["z_shape_range"]), _scale(sample[:, 2], dt_low, dt_high)))
        audit.update({"system_id": sid, "delta_t_range_c": [dt_low, dt_high], "q_v_range": master["q_v_range"], "z_shape_range": master["z_shape_range"]})
        design_audits[str(sid)] = audit
        for index, values in enumerate(theta):
            member_id = f"system_{sid}_gen0_member_{index:02d}"
            member_dir = output_root / "members" / f"system_{sid}" / f"member_{index:02d}"
            workdir = workdir_root / f"system_{sid}" / f"member_{index:02d}"
            cfg = _member_config(master, rep, member_id, values.tolist(), member_dir, workdir)
            config_member_path = output_root / "member_configs" / f"{member_id}.json"
            _write_json(config_member_path, cfg)
            all_tasks.append({"system_id": sid, "member_id": member_id, "theta": values.tolist(), "config_path": config_member_path, "member_dir": member_dir})
    _write_json(output_root / "lhs_design_audit.json", design_audits)
    pd.DataFrame([{**{"system_id": task["system_id"], "member_id": task["member_id"]}, **dict(zip(("q_v", "z_shape", "delta_t_c"), task["theta"])), "config_path": str(task["config_path"])} for task in all_tasks]).to_csv(output_root / "member_manifest.csv", index=False)

    executions = []
    rows = []
    stopped = False
    max_workers = min(12, int(master["max_workers"]))
    for start in range(0, len(all_tasks), max_workers):
        wave = all_tasks[start:start + max_workers]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_run_member, wave))
        executions.extend(results)
        rows.extend(_summarize_member(task, result) for task, result in zip(wave, results))
        invalid_count = sum(not bool(row["valid"]) for row in rows)
        if invalid_count / len(all_tasks) > float(master["overall_invalid_stop_fraction"]):
            stopped = True
            break

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "member_summary.csv", index=False)
    system_summaries = []
    sensitivities = []
    for rep in master["representatives"]:
        sid = int(rep["system_id"])
        subset = frame[frame["system_id"].eq(sid)]
        valid = subset[subset["valid"].astype(bool)]
        best = None if valid.empty else valid.loc[valid["formal_endpoint_objective_sep"].astype(float).idxmin()]
        sensitivity = _sensitivity(subset, rep, master)
        sensitivities.append(sensitivity)
        best_objective = None if best is None else float(best["formal_endpoint_objective_sep"])
        system_summaries.append({"system_id": sid, "planned": int(master["members_per_system"]), "completed": int(len(subset)), "valid": int(len(valid)), "invalid": int(len(subset) - len(valid)), "valid_fraction": None if len(subset) == 0 else float(len(valid) / len(subset)), "best_member_id": None if best is None else str(best["member_id"]), "best_formal_objective_sep": best_objective, "parent_formal_objective_sep": float(rep["parent_formal_objective_sep"]), "strictly_improves_parent": None if best_objective is None else bool(best_objective < float(rep["parent_formal_objective_sep"])), "formal_pass_count": int(valid["formal_pass"].astype(bool).sum()) if len(valid) else 0, "sensitivity_effective_rank": sensitivity.get("effective_rank")})
    pd.DataFrame(system_summaries).to_csv(output_root / "system_summary.csv", index=False)
    _write_json(output_root / "whitened_sensitivity_svd.json", sensitivities)

    invalid = int((~frame["valid"].astype(bool)).sum()) if len(frame) else 0
    overall_invalid_fraction = float(invalid / len(frame)) if len(frame) else 1.0
    decision = {"experiment_id": master["experiment_id"], "generation": 0, "gen1_started": False, "planned_members": len(all_tasks), "completed_members": len(frame), "valid_members": int(frame["valid"].astype(bool).sum()) if len(frame) else 0, "invalid_members": invalid, "overall_invalid_fraction_completed": overall_invalid_fraction, "overall_invalid_fraction_of_planned_lower_bound": float(invalid / len(all_tasks)), "invalid_stop_threshold": float(master["overall_invalid_stop_fraction"]), "stopped_early": stopped, "status": "stopped-invalid-fraction" if stopped else "gen0-completed", "system_summaries": system_summaries}
    _write_json(output_root / "generation_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Himalaya_greatGlacier" / "With_Rec_climate_organized" / "ice_surface_reconstruction" / "config" / "cycle4_state_shooting_gen0_three_representatives_v1.json")
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
