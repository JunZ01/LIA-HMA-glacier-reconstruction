"""Adaptive 60-second zero-MB relaxation for system 2005.

Each trial compares one full frozen-diffusivity step with two recomputed half
steps.  The two-half-step state is accepted only when their relative L2
difference is at most five percent and all conservation/domain gates pass.
No clipping, limiting, climate forcing, long run, or final-state serialization
is allowed.
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

from .torch_planform_2d_frozen_diffusivity_core_v2 import frozen_diffusivity_step_chunked
from .torch_planform_2d_system2005_static_grid_smoke_v1 import (
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
    SYSTEM_ID,
    _sha256,
)


TARGET_SECONDS = 60.0
ERROR_GATE = 0.05
GROWTH_ERROR_THRESHOLD = ERROR_GATE / 4.0
MINIMUM_DT_SECONDS = 0.001
MAXIMUM_DT_SECONDS = 30.0
MAXIMUM_ACCEPTED_STEPS = 10000
MAXIMUM_CONSECUTIVE_REJECTIONS = 20
SOLVER_RTOL = 1e-11
SOLVER_MAXITER = 4000
CHECK_INTERVAL = 8


def _boundary_maximum(state: torch.Tensor) -> float:
    return float(
        torch.max(
            torch.stack(
                [
                    torch.max(state[0, :]),
                    torch.max(state[-1, :]),
                    torch.max(state[:, 0]),
                    torch.max(state[:, -1]),
                ]
            )
        ).item()
    )


def _step(state: torch.Tensor, bed: torch.Tensor, *, dx: float, gamma: float, dt: float):
    return frozen_diffusivity_step_chunked(
        state,
        bed,
        dx=dx,
        dy=dx,
        glen_n=GLEN_N,
        gamma=gamma,
        dt=dt,
        rtol=SOLVER_RTOL,
        maxiter=SOLVER_MAXITER,
        check_interval=CHECK_INTERVAL,
    )


def run(geology: Path, static_summary_path: Path, convergence_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    if static_summary.get("status") != "static-pass":
        raise RuntimeError("source-static-not-pass")
    if convergence.get("status") != "initial-dt-selected-adaptive-60s-eligible":
        raise RuntimeError("initial-dt-not-selected")
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
    initial_active_cells = int(torch.count_nonzero(initial_state > 0).item())

    accepted_steps = 0
    rejected_steps = 0
    consecutive_rejections = 0
    solver_calls = 0
    elapsed_physical = 0.0
    proposed_dt = initial_dt
    minimum_accepted_dt = float("inf")
    maximum_accepted_dt = 0.0
    maximum_accepted_error = 0.0
    maximum_step_closure = 0.0
    maximum_step_change = 0.0
    history: list[dict[str, object]] = []
    wall_started = time.perf_counter()

    while elapsed_physical < TARGET_SECONDS and accepted_steps < MAXIMUM_ACCEPTED_STEPS:
        remaining = TARGET_SECONDS - elapsed_physical
        dt = min(proposed_dt, remaining)
        if dt <= 0.0:
            break
        trial_started = time.perf_counter()
        full_state, full_diag = _step(state, bed, dx=dx, gamma=gamma, dt=dt)
        half_state, half_diag = _step(state, bed, dx=dx, gamma=gamma, dt=0.5 * dt)
        half2_state, half2_diag = _step(half_state, bed, dx=dx, gamma=gamma, dt=0.5 * dt)
        torch.cuda.synchronize()
        solver_calls += 3
        full_delta = full_state - state
        half_delta = half_state - state
        half2_delta = half2_state - half_state
        accepted_candidate_delta = half2_state - state
        denominator = max(float(torch.linalg.vector_norm(accepted_candidate_delta).item()), 1e-30)
        error_relative_l2 = float(torch.linalg.vector_norm(half2_state - full_state).item()) / denominator
        closure_full = float(torch.sum(full_delta).item()) * cell_area
        closure_half = float(torch.sum(half_delta).item()) * cell_area
        closure_half2 = float(torch.sum(half2_delta).item()) * cell_area
        trial_max_closure = max(abs(closure_full), abs(closure_half), abs(closure_half2))
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
        trial_seconds = time.perf_counter() - trial_started
        record = {
            "trial_index": len(history) + 1,
            "accepted": accepted,
            "physical_time_before_s": elapsed_physical,
            "dt_s": dt,
            "error_relative_l2": error_relative_l2,
            "trial_max_step_closure_m3": trial_max_closure,
            "maximum_candidate_change_m": float(torch.max(torch.abs(accepted_candidate_delta)).item()),
            "full_iterations": int(full_diag["solver_iterations"]),
            "half1_iterations": int(half_diag["solver_iterations"]),
            "half2_iterations": int(half2_diag["solver_iterations"]),
            "numerical_pass": numerical_pass,
            "wall_seconds": trial_seconds,
        }
        history.append(record)

        if accepted:
            state = half2_state
            elapsed_physical += dt
            accepted_steps += 1
            consecutive_rejections = 0
            minimum_accepted_dt = min(minimum_accepted_dt, dt)
            maximum_accepted_dt = max(maximum_accepted_dt, dt)
            maximum_accepted_error = max(maximum_accepted_error, error_relative_l2)
            maximum_step_closure = max(maximum_step_closure, trial_max_closure)
            maximum_step_change = max(maximum_step_change, float(record["maximum_candidate_change_m"]))
            if error_relative_l2 <= GROWTH_ERROR_THRESHOLD:
                proposed_dt = min(2.0 * dt, MAXIMUM_DT_SECONDS)
            else:
                proposed_dt = dt
        else:
            rejected_steps += 1
            consecutive_rejections += 1
            proposed_dt = 0.5 * dt
            if proposed_dt < MINIMUM_DT_SECONDS and remaining > MINIMUM_DT_SECONDS:
                break
            if consecutive_rejections > MAXIMUM_CONSECUTIVE_REJECTIONS:
                break

        if accepted_steps > 0 and accepted_steps % 25 == 0:
            progress = {
                "status": "running",
                "physical_time_s": elapsed_physical,
                "accepted_steps": accepted_steps,
                "rejected_steps": rejected_steps,
                "next_dt_s": proposed_dt,
                "last_error_relative_l2": error_relative_l2,
            }
            (output / "progress.json").write_text(
                json.dumps(progress, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(json.dumps(progress, sort_keys=True), flush=True)

    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    final_volume = float(torch.sum(state).item()) * cell_area
    cumulative_closure = final_volume - initial_volume
    final_active_cells = int(torch.count_nonzero(state > 0).item())
    final_change = state - initial_state
    source_hash_after = _sha256(geology)
    completed_target = abs(elapsed_physical - TARGET_SECONDS) <= 1e-10
    final_checks = {
        "target_60s_completed": completed_target,
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
        "schema": "lia-191-torch-planform2d-system2005-adaptive-relaxation-60s-v1",
        "status": "adaptive-60s-pass-longer-run-not-authorized" if passed else "adaptive-60s-fail-stop",
        "system_id": SYSTEM_ID,
        "target_seconds": TARGET_SECONDS,
        "completed_seconds": elapsed_physical,
        "initial_dt_seconds": initial_dt,
        "minimum_accepted_dt_seconds": minimum_accepted_dt if accepted_steps else None,
        "maximum_accepted_dt_seconds": maximum_accepted_dt if accepted_steps else None,
        "error_gate_relative_l2": ERROR_GATE,
        "growth_error_threshold": GROWTH_ERROR_THRESHOLD,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "solver_calls": solver_calls,
        "wall_seconds": wall_seconds,
        "simulated_seconds_per_wall_second": elapsed_physical / max(wall_seconds, 1e-12),
        "maximum_accepted_error_relative_l2": maximum_accepted_error,
        "maximum_step_closure_m3": maximum_step_closure,
        "cumulative_mass_closure_m3": cumulative_closure,
        "mass_closure_tolerance_m3": closure_tolerance,
        "maximum_single_accepted_change_m": maximum_step_change,
        "maximum_final_change_from_initial_m": float(torch.max(torch.abs(final_change)).item()),
        "initial_active_cell_count": initial_active_cells,
        "final_active_cell_count": final_active_cells,
        "final_minimum_thickness_m": float(torch.min(state).item()),
        "final_maximum_thickness_m": float(torch.max(state).item()),
        "final_boundary_maximum_thickness_m": _boundary_maximum(state),
        "final_checks": final_checks,
        "source_geology_sha256_before": source_hash_before,
        "source_geology_sha256_after": source_hash_after,
        "step_history_csv": str(history_path),
        "runtime_clipping": False,
        "runtime_limiter": False,
        "mass_balance": "zero",
        "state_saved": False,
        "zero_mb_0p1y_run": False,
        "steady_run": False,
        "historical_run": False,
        "next_authorized_stage": None,
        "adjudication_required": "review-adaptive-throughput-and-geometry-relaxation-before-any-extension",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "progress.json").write_text(
        json.dumps({"status": summary["status"], "physical_time_s": elapsed_physical}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--convergence-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.geology, arguments.static_summary, arguments.convergence_summary, arguments.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
