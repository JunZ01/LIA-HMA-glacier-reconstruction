"""Adaptive 600-second zero-MB diagnostic for system 2005.

The trajectory starts from the immutable static geology and lands exactly on
60, 120, 300, and 600 second checkpoints.  Area is reported at four thickness
thresholds, but no area threshold is silently promoted to a formal pass gate.
No state is serialized or used by a longer run.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

from .torch_planform_2d_system2005_adaptive_relaxation_60s_v1 import (
    CHECK_INTERVAL,
    ERROR_GATE,
    GLEN_N,
    GRAVITY_M_S2,
    GROWTH_ERROR_THRESHOLD,
    ICE_DENSITY_KG_M3,
    MAXIMUM_ACCEPTED_STEPS,
    MAXIMUM_CONSECUTIVE_REJECTIONS,
    MAXIMUM_DT_SECONDS,
    MINIMUM_DT_SECONDS,
    SOLVER_MAXITER,
    SOLVER_RTOL,
    SYSTEM_ID,
    _boundary_maximum,
    _sha256,
    _step,
)


TARGET_SECONDS = 600.0
CHECKPOINT_SECONDS = (60.0, 120.0, 300.0, 600.0)
THICKNESS_THRESHOLDS_M = (0.0, 0.1, 1.0, 10.0)


def _area_metrics(state: torch.Tensor, cell_area: float) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for threshold in THICKNESS_THRESHOLDS_M:
        mask = state > threshold
        count = int(torch.count_nonzero(mask).item())
        volume = float(torch.sum(torch.where(mask, state, torch.zeros_like(state))).item()) * cell_area
        result[f"gt_{threshold:g}m"] = {
            "cell_count": count,
            "area_m2": count * cell_area,
            "volume_m3": volume,
        }
    return result


def _with_relative_area_change(
    metrics: dict[str, dict[str, float | int]],
    initial: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    for key, values in metrics.items():
        initial_area = float(initial[key]["area_m2"])
        values["area_relative_change_from_initial"] = (
            (float(values["area_m2"]) - initial_area) / initial_area if initial_area > 0 else 0.0
        )
        initial_volume = float(initial[key]["volume_m3"])
        values["volume_relative_change_from_initial"] = (
            (float(values["volume_m3"]) - initial_volume) / initial_volume if initial_volume > 0 else 0.0
        )
    return metrics


def run(
    geology: Path,
    static_summary_path: Path,
    convergence_path: Path,
    relaxation_60s_path: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    relaxation_60s = json.loads(relaxation_60s_path.read_text(encoding="utf-8"))
    if static_summary.get("status") != "static-pass":
        raise RuntimeError("source-static-not-pass")
    if convergence.get("status") != "initial-dt-selected-adaptive-60s-eligible":
        raise RuntimeError("initial-dt-not-selected")
    if relaxation_60s.get("status") != "adaptive-60s-pass-longer-run-not-authorized":
        raise RuntimeError("adaptive-60s-prerequisite-not-pass")
    initial_dt = float(convergence["selected_initial_dt_seconds"])
    source_hash_before = _sha256(geology)
    if source_hash_before.lower() != str(static_summary["geology_nc_sha256"]).lower():
        raise RuntimeError("geology-hash-mismatch")
    with Dataset(geology, "r") as dataset:
        bed_np = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        thickness_np = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        dx = float(getattr(dataset, "dx_m"))
    if not np.isfinite(bed_np).all() or not np.isfinite(thickness_np).all() or np.any(thickness_np < 0.0):
        raise RuntimeError("invalid-initial-grid")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("registered-sm120-gpu-unavailable")

    glen_a = float(static_summary["stage2_frozen"]["glen_a"])
    gamma = 2.0 * glen_a * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N / (GLEN_N + 2.0)
    initial_state = torch.from_numpy(thickness_np).to(device="cuda:0", dtype=torch.float64)
    state = initial_state.clone()
    bed = torch.from_numpy(bed_np).to(device="cuda:0", dtype=torch.float64)
    cell_area = dx**2
    initial_volume = float(torch.sum(initial_state).item()) * cell_area
    closure_tolerance = max(1e-6, initial_volume * 1e-12)
    initial_metrics = _area_metrics(initial_state, cell_area)

    accepted_steps = 0
    rejected_steps = 0
    consecutive_rejections = 0
    solver_calls = 0
    physical_time = 0.0
    proposed_dt = initial_dt
    minimum_accepted_dt = float("inf")
    maximum_accepted_dt = 0.0
    maximum_accepted_error = 0.0
    maximum_step_closure = 0.0
    maximum_step_change = 0.0
    checkpoint_index = 0
    checkpoint_metrics: dict[str, object] = {
        "0": {
            "physical_time_s": 0.0,
            "area_metrics": _with_relative_area_change(initial_metrics, initial_metrics),
        }
    }
    history: list[dict[str, object]] = []
    wall_started = time.perf_counter()

    while physical_time < TARGET_SECONDS and accepted_steps < MAXIMUM_ACCEPTED_STEPS:
        next_checkpoint = CHECKPOINT_SECONDS[checkpoint_index]
        checkpoint_remaining = next_checkpoint - physical_time
        dt = min(proposed_dt, checkpoint_remaining)
        checkpoint_limited = dt < proposed_dt - 1e-15
        if dt <= 0.0:
            raise RuntimeError("nonpositive-adaptive-dt")
        trial_started = time.perf_counter()
        full_state, full_diag = _step(state, bed, dx=dx, gamma=gamma, dt=dt)
        half_state, half_diag = _step(state, bed, dx=dx, gamma=gamma, dt=0.5 * dt)
        half2_state, half2_diag = _step(half_state, bed, dx=dx, gamma=gamma, dt=0.5 * dt)
        torch.cuda.synchronize()
        solver_calls += 3
        full_delta = full_state - state
        half_delta = half_state - state
        half2_delta = half2_state - half_state
        candidate_delta = half2_state - state
        denominator = max(float(torch.linalg.vector_norm(candidate_delta).item()), 1e-30)
        error_relative_l2 = float(torch.linalg.vector_norm(half2_state - full_state).item()) / denominator
        closures = [
            float(torch.sum(full_delta).item()) * cell_area,
            float(torch.sum(half_delta).item()) * cell_area,
            float(torch.sum(half2_delta).item()) * cell_area,
        ]
        trial_max_closure = max(abs(value) for value in closures)
        numerical_pass = (
            bool(full_diag["solver_converged"])
            and bool(half_diag["solver_converged"])
            and bool(half2_diag["solver_converged"])
            and bool(full_diag["state_finite"])
            and bool(half_diag["state_finite"])
            and bool(half2_diag["state_finite"])
            and int(full_diag["negative_cell_count"]) == 0
            and int(half_diag["negative_cell_count"]) == 0
            and int(half2_diag["negative_cell_count"]) == 0
            and trial_max_closure <= closure_tolerance
            and _boundary_maximum(half2_state) <= 10.0
        )
        accepted = numerical_pass and error_relative_l2 <= ERROR_GATE
        record = {
            "trial_index": len(history) + 1,
            "accepted": accepted,
            "physical_time_before_s": physical_time,
            "dt_s": dt,
            "checkpoint_limited": checkpoint_limited,
            "error_relative_l2": error_relative_l2,
            "trial_max_step_closure_m3": trial_max_closure,
            "maximum_candidate_change_m": float(torch.max(torch.abs(candidate_delta)).item()),
            "full_iterations": int(full_diag["solver_iterations"]),
            "half1_iterations": int(half_diag["solver_iterations"]),
            "half2_iterations": int(half2_diag["solver_iterations"]),
            "numerical_pass": numerical_pass,
            "wall_seconds": time.perf_counter() - trial_started,
        }
        history.append(record)

        if accepted:
            state = half2_state
            physical_time += dt
            if abs(physical_time - next_checkpoint) <= 1e-10:
                physical_time = next_checkpoint
            accepted_steps += 1
            consecutive_rejections = 0
            minimum_accepted_dt = min(minimum_accepted_dt, dt)
            maximum_accepted_dt = max(maximum_accepted_dt, dt)
            maximum_accepted_error = max(maximum_accepted_error, error_relative_l2)
            maximum_step_closure = max(maximum_step_closure, trial_max_closure)
            maximum_step_change = max(maximum_step_change, float(record["maximum_candidate_change_m"]))
            if error_relative_l2 <= GROWTH_ERROR_THRESHOLD:
                growth_base = proposed_dt if checkpoint_limited else dt
                proposed_dt = min(2.0 * growth_base, MAXIMUM_DT_SECONDS)
            elif not checkpoint_limited:
                proposed_dt = dt

            if physical_time == next_checkpoint:
                metrics = _with_relative_area_change(_area_metrics(state, cell_area), initial_metrics)
                checkpoint_metrics[f"{next_checkpoint:g}"] = {
                    "physical_time_s": next_checkpoint,
                    "accepted_steps": accepted_steps,
                    "rejected_steps": rejected_steps,
                    "current_proposed_dt_s": proposed_dt,
                    "maximum_thickness_m": float(torch.max(state).item()),
                    "area_metrics": metrics,
                }
                checkpoint_index += 1
                progress = {
                    "status": "running" if physical_time < TARGET_SECONDS else "finishing",
                    "physical_time_s": physical_time,
                    "accepted_steps": accepted_steps,
                    "rejected_steps": rejected_steps,
                    "next_dt_s": proposed_dt,
                }
                (output / "progress.json").write_text(
                    json.dumps(progress, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
                print(json.dumps(progress, sort_keys=True), flush=True)
                if checkpoint_index == len(CHECKPOINT_SECONDS):
                    break
        else:
            rejected_steps += 1
            consecutive_rejections += 1
            proposed_dt = 0.5 * dt
            if proposed_dt < MINIMUM_DT_SECONDS or consecutive_rejections > MAXIMUM_CONSECUTIVE_REJECTIONS:
                break

    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    final_volume = float(torch.sum(state).item()) * cell_area
    cumulative_closure = final_volume - initial_volume
    final_change = state - initial_state
    source_hash_after = _sha256(geology)
    final_checks = {
        "target_600s_completed": abs(physical_time - TARGET_SECONDS) <= 1e-10,
        "all_checkpoints_recorded": all(f"{checkpoint:g}" in checkpoint_metrics for checkpoint in CHECKPOINT_SECONDS),
        "cumulative_mass_closure_pass": abs(cumulative_closure) <= closure_tolerance,
        "final_state_finite": bool(torch.isfinite(state).all()),
        "final_negative_cell_count_zero": int(torch.count_nonzero(state < 0).item()) == 0,
        "final_boundary_thickness_pass": _boundary_maximum(state) <= 10.0,
        "source_immutable": source_hash_after == source_hash_before,
        "accepted_error_gate_respected": all(
            (not bool(item["accepted"])) or float(item["error_relative_l2"]) <= ERROR_GATE for item in history
        ),
    }
    passed = all(bool(value) for value in final_checks.values())
    history_path = output / "step_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()) if history else [])
        if history:
            writer.writeheader()
            writer.writerows(history)
    summary: dict[str, object] = {
        "schema": "lia-191-torch-planform2d-system2005-adaptive-relaxation-600s-v1",
        "status": "adaptive-600s-pass-extension-not-authorized" if passed else "adaptive-600s-fail-stop",
        "system_id": SYSTEM_ID,
        "target_seconds": TARGET_SECONDS,
        "completed_seconds": physical_time,
        "checkpoint_seconds": list(CHECKPOINT_SECONDS),
        "thickness_thresholds_m": list(THICKNESS_THRESHOLDS_M),
        "area_diagnostics_are_not_formal_gate": True,
        "initial_dt_seconds": initial_dt,
        "minimum_accepted_dt_seconds": minimum_accepted_dt if accepted_steps else None,
        "maximum_accepted_dt_seconds": maximum_accepted_dt if accepted_steps else None,
        "maximum_dt_policy_seconds": MAXIMUM_DT_SECONDS,
        "maximum_dt_policy_reached": maximum_accepted_dt >= MAXIMUM_DT_SECONDS,
        "error_gate_relative_l2": ERROR_GATE,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "solver_calls": solver_calls,
        "wall_seconds": wall_seconds,
        "simulated_seconds_per_wall_second": physical_time / max(wall_seconds, 1e-12),
        "maximum_accepted_error_relative_l2": maximum_accepted_error,
        "maximum_step_closure_m3": maximum_step_closure,
        "cumulative_mass_closure_m3": cumulative_closure,
        "mass_closure_tolerance_m3": closure_tolerance,
        "maximum_single_accepted_change_m": maximum_step_change,
        "maximum_final_change_from_initial_m": float(torch.max(torch.abs(final_change)).item()),
        "checkpoint_metrics": checkpoint_metrics,
        "final_checks": final_checks,
        "source_geology_sha256_before": source_hash_before,
        "source_geology_sha256_after": source_hash_after,
        "step_history_csv": str(history_path),
        "runtime_clipping": False,
        "runtime_limiter": False,
        "mass_balance": "zero",
        "state_saved": False,
        "zero_mb_longer_run": False,
        "steady_run": False,
        "historical_run": False,
        "next_authorized_stage": None,
        "adjudication_required": "review-thresholded-area-and-30s-step-behavior-before-extension",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "progress.json").write_text(
        json.dumps({"status": summary["status"], "physical_time_s": physical_time}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--convergence-summary", type=Path, required=True)
    parser.add_argument("--relaxation-60s-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.geology,
                arguments.static_summary,
                arguments.convergence_summary,
                arguments.relaxation_60s_summary,
                arguments.output,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
