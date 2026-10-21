"""Formal five-year zero-mass-balance GPU run for system 2005.

This is one end-to-end trajectory from the immutable planform-2D source.  It
uses the validated full-step versus two-half-step error controller, permits a
candidate timestep ceiling of one day, and serializes recoverable checkpoints
plus the final state.  Rejected trials are never committed to model state.
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
    MAXIMUM_CONSECUTIVE_REJECTIONS,
    MINIMUM_DT_SECONDS,
    SOLVER_MAXITER,
    SOLVER_RTOL,
    SYSTEM_ID,
    _boundary_maximum,
    _sha256,
    _step,
)
from .torch_planform_2d_system2005_adaptive_relaxation_600s_v1 import (
    THICKNESS_THRESHOLDS_M,
    _area_metrics,
    _with_relative_area_change,
)


SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.0 * SECONDS_PER_DAY
TARGET_YEARS = 5.0
TARGET_SECONDS = TARGET_YEARS * SECONDS_PER_YEAR
MAXIMUM_DT_SECONDS = SECONDS_PER_DAY
MAXIMUM_ACCEPTED_STEPS = 200_000
FORMAL_VOLUME_DRIFT_LIMIT = 0.001
CHECKPOINTS = (
    ("7d", 7.0 * SECONDS_PER_DAY),
    ("30d", 30.0 * SECONDS_PER_DAY),
    ("0p1y", 0.1 * SECONDS_PER_YEAR),
    ("1y", 1.0 * SECONDS_PER_YEAR),
    ("2y", 2.0 * SECONDS_PER_YEAR),
    ("3y", 3.0 * SECONDS_PER_YEAR),
    ("4y", 4.0 * SECONDS_PER_YEAR),
    ("5y", 5.0 * SECONDS_PER_YEAR),
)


def _write_state(
    path: Path,
    state: torch.Tensor,
    bed_np: np.ndarray,
    *,
    dx: float,
    elapsed_seconds: float,
    proposed_dt_seconds: float,
) -> str:
    thickness = state.detach().cpu().numpy()
    np.savez_compressed(
        path,
        thickness_m=thickness,
        bed_m=bed_np,
        dx_m=np.asarray(dx, dtype=np.float64),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
        proposed_dt_seconds=np.asarray(proposed_dt_seconds, dtype=np.float64),
    )
    return _sha256(path)


def run(
    geology: Path,
    static_summary_path: Path,
    convergence_path: Path,
    relaxation_60s_path: Path,
    relaxation_7d_path: Path,
    output: Path,
) -> dict[str, object]:
    static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    relaxation_60s = json.loads(relaxation_60s_path.read_text(encoding="utf-8"))
    relaxation_7d = json.loads(relaxation_7d_path.read_text(encoding="utf-8"))
    if static_summary.get("status") != "static-pass":
        raise RuntimeError("source-static-not-pass")
    if convergence.get("status") != "initial-dt-selected-adaptive-60s-eligible":
        raise RuntimeError("initial-dt-not-selected")
    if relaxation_60s.get("status") != "adaptive-60s-pass-longer-run-not-authorized":
        raise RuntimeError("adaptive-60s-prerequisite-not-pass")
    if relaxation_7d.get("status") != "adaptive-604800s-pass-extension-not-authorized":
        raise RuntimeError("adaptive-7d-prerequisite-not-pass")
    if output.exists():
        raise FileExistsError(output)

    source_hash_before = _sha256(geology)
    if source_hash_before.lower() != str(static_summary["geology_nc_sha256"]).lower():
        raise RuntimeError("geology-hash-mismatch")
    with Dataset(geology, "r") as dataset:
        bed_np = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        thickness_np = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        dx = float(getattr(dataset, "dx_m"))
    if not np.isfinite(bed_np).all() or not np.isfinite(thickness_np).all():
        raise RuntimeError("nonfinite-initial-grid")
    if np.any(thickness_np < 0.0):
        raise RuntimeError("negative-initial-thickness")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("registered-sm120-gpu-unavailable")

    output.mkdir(parents=True, exist_ok=False)
    initial_dt = float(convergence["selected_initial_dt_seconds"])
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
    elapsed = 0.0
    proposed_dt = initial_dt
    minimum_accepted_dt = float("inf")
    maximum_accepted_dt = 0.0
    maximum_accepted_error = 0.0
    maximum_formal_step_closure = 0.0
    maximum_trial_step_closure = 0.0
    maximum_step_change = 0.0
    checkpoint_index = 0
    checkpoint_metrics: dict[str, object] = {
        "0": {
            "elapsed_seconds": 0.0,
            "elapsed_years": 0.0,
            "volume_relative_change": 0.0,
            "area_metrics": _with_relative_area_change(initial_metrics, initial_metrics),
        }
    }
    history: list[dict[str, object]] = []
    wall_started = time.perf_counter()

    while elapsed < TARGET_SECONDS and accepted_steps < MAXIMUM_ACCEPTED_STEPS:
        checkpoint_label, next_checkpoint = CHECKPOINTS[checkpoint_index]
        checkpoint_remaining = next_checkpoint - elapsed
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
        full_closure = float(torch.sum(full_delta).item()) * cell_area
        half1_closure = float(torch.sum(half_delta).item()) * cell_area
        half2_closure = float(torch.sum(half2_delta).item()) * cell_area
        formal_step_closure = max(abs(half1_closure), abs(half2_closure))
        trial_step_closure = max(formal_step_closure, abs(full_closure))
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
            and formal_step_closure <= closure_tolerance
            and _boundary_maximum(half2_state) <= 10.0
        )
        accepted = numerical_pass and error_relative_l2 <= ERROR_GATE
        history.append(
            {
                "trial_index": len(history) + 1,
                "accepted": accepted,
                "elapsed_before_seconds": elapsed,
                "elapsed_before_years": elapsed / SECONDS_PER_YEAR,
                "dt_seconds": dt,
                "checkpoint_limited": checkpoint_limited,
                "error_relative_l2": error_relative_l2,
                "formal_max_substep_closure_m3": formal_step_closure,
                "trial_max_closure_including_comparator_m3": trial_step_closure,
                "maximum_candidate_change_m": float(torch.max(torch.abs(candidate_delta)).item()),
                "full_iterations": int(full_diag["solver_iterations"]),
                "half1_iterations": int(half_diag["solver_iterations"]),
                "half2_iterations": int(half2_diag["solver_iterations"]),
                "numerical_pass": numerical_pass,
                "wall_seconds": time.perf_counter() - trial_started,
            }
        )

        if accepted:
            state = half2_state
            elapsed += dt
            if abs(elapsed - next_checkpoint) <= 1e-8:
                elapsed = next_checkpoint
            accepted_steps += 1
            consecutive_rejections = 0
            minimum_accepted_dt = min(minimum_accepted_dt, dt)
            maximum_accepted_dt = max(maximum_accepted_dt, dt)
            maximum_accepted_error = max(maximum_accepted_error, error_relative_l2)
            maximum_formal_step_closure = max(maximum_formal_step_closure, formal_step_closure)
            maximum_trial_step_closure = max(maximum_trial_step_closure, trial_step_closure)
            maximum_step_change = max(
                maximum_step_change,
                float(history[-1]["maximum_candidate_change_m"]),
            )
            if error_relative_l2 <= GROWTH_ERROR_THRESHOLD:
                growth_base = proposed_dt if checkpoint_limited else dt
                proposed_dt = min(2.0 * growth_base, MAXIMUM_DT_SECONDS)
            elif not checkpoint_limited:
                proposed_dt = dt

            if elapsed == next_checkpoint:
                current_volume = float(torch.sum(state).item()) * cell_area
                volume_relative_change = (current_volume - initial_volume) / initial_volume
                metrics = _with_relative_area_change(_area_metrics(state, cell_area), initial_metrics)
                state_path = output / f"checkpoint_state_{checkpoint_label}.npz"
                state_sha = _write_state(
                    state_path,
                    state,
                    bed_np,
                    dx=dx,
                    elapsed_seconds=elapsed,
                    proposed_dt_seconds=proposed_dt,
                )
                checkpoint_metrics[checkpoint_label] = {
                    "elapsed_seconds": elapsed,
                    "elapsed_years": elapsed / SECONDS_PER_YEAR,
                    "accepted_steps": accepted_steps,
                    "rejected_steps": rejected_steps,
                    "next_dt_seconds": proposed_dt,
                    "maximum_thickness_m": float(torch.max(state).item()),
                    "volume_relative_change": volume_relative_change,
                    "area_metrics": metrics,
                    "state_file": str(state_path),
                    "state_sha256": state_sha,
                }
                checkpoint_index += 1
                progress = {
                    "status": "running" if elapsed < TARGET_SECONDS else "finishing",
                    "elapsed_seconds": elapsed,
                    "elapsed_years": elapsed / SECONDS_PER_YEAR,
                    "accepted_steps": accepted_steps,
                    "rejected_steps": rejected_steps,
                    "next_dt_seconds": proposed_dt,
                }
                (output / "progress.json").write_text(
                    json.dumps(progress, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(progress, sort_keys=True), flush=True)
                if checkpoint_index == len(CHECKPOINTS):
                    break
        else:
            rejected_steps += 1
            consecutive_rejections += 1
            proposed_dt = 0.5 * dt
            if proposed_dt < MINIMUM_DT_SECONDS:
                break
            if consecutive_rejections > MAXIMUM_CONSECUTIVE_REJECTIONS:
                break

    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    final_volume = float(torch.sum(state).item()) * cell_area
    volume_relative_change = (final_volume - initial_volume) / initial_volume
    cumulative_closure = final_volume - initial_volume
    source_hash_after = _sha256(geology)

    history_path = output / "step_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()) if history else [])
        if history:
            writer.writeheader()
            writer.writerows(history)

    target_complete = abs(elapsed - TARGET_SECONDS) <= 1e-8
    state_path = output / "final_state_5y.npz"
    final_state_sha = None
    if target_complete:
        final_state_sha = _write_state(
            state_path,
            state,
            bed_np,
            dx=dx,
            elapsed_seconds=elapsed,
            proposed_dt_seconds=proposed_dt,
        )
    checkpoint_0p1 = checkpoint_metrics.get("0p1y")
    checkpoint_0p1_drift = (
        None if checkpoint_0p1 is None else float(checkpoint_0p1["volume_relative_change"])
    )
    final_checks = {
        "target_5y_completed": target_complete,
        "all_checkpoints_recorded": all(label in checkpoint_metrics for label, _ in CHECKPOINTS),
        "formal_0p1y_volume_drift_pass": (
            checkpoint_0p1_drift is not None
            and abs(checkpoint_0p1_drift) <= FORMAL_VOLUME_DRIFT_LIMIT
        ),
        "formal_5y_volume_drift_pass": abs(volume_relative_change) <= FORMAL_VOLUME_DRIFT_LIMIT,
        "formal_max_absolute_per_substep_closure_pass": (
            maximum_formal_step_closure <= closure_tolerance
        ),
        "cumulative_mass_closure_pass": abs(cumulative_closure) <= closure_tolerance,
        "final_state_finite": bool(torch.isfinite(state).all()),
        "final_negative_cell_count_zero": int(torch.count_nonzero(state < 0).item()) == 0,
        "final_boundary_thickness_pass": _boundary_maximum(state) <= 10.0,
        "source_immutable": source_hash_after == source_hash_before,
        "accepted_error_gate_respected": all(
            (not bool(item["accepted"])) or float(item["error_relative_l2"]) <= ERROR_GATE
            for item in history
        ),
        "final_state_saved": final_state_sha is not None and state_path.exists(),
    }
    passed = all(final_checks.values())
    summary: dict[str, object] = {
        "schema": "lia-191-torch-planform2d-system2005-zero-mb-5y-formal-v1",
        "status": "formal-zero-mb-5y-pass" if passed else "formal-zero-mb-5y-fail-stop",
        "system_id": SYSTEM_ID,
        "target_years": TARGET_YEARS,
        "seconds_per_year": SECONDS_PER_YEAR,
        "target_seconds": TARGET_SECONDS,
        "completed_seconds": elapsed,
        "completed_years": elapsed / SECONDS_PER_YEAR,
        "checkpoint_contract": [
            {"label": label, "seconds": seconds} for label, seconds in CHECKPOINTS
        ],
        "initial_dt_seconds": initial_dt,
        "minimum_accepted_dt_seconds": minimum_accepted_dt if accepted_steps else None,
        "maximum_accepted_dt_seconds": maximum_accepted_dt if accepted_steps else None,
        "candidate_dt_ceiling_seconds": MAXIMUM_DT_SECONDS,
        "error_gate_relative_l2": ERROR_GATE,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "solver_calls": solver_calls,
        "wall_seconds": wall_seconds,
        "simulated_seconds_per_wall_second": elapsed / max(wall_seconds, 1e-12),
        "maximum_accepted_error_relative_l2": maximum_accepted_error,
        "maximum_formal_per_substep_closure_m3": maximum_formal_step_closure,
        "maximum_trial_closure_including_comparator_m3": maximum_trial_step_closure,
        "cumulative_mass_closure_m3": cumulative_closure,
        "mass_closure_tolerance_m3": closure_tolerance,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "final_volume_relative_change": volume_relative_change,
        "formal_volume_drift_limit": FORMAL_VOLUME_DRIFT_LIMIT,
        "maximum_single_accepted_change_m": maximum_step_change,
        "maximum_final_change_from_initial_m": float(
            torch.max(torch.abs(state - initial_state)).item()
        ),
        "final_boundary_maximum_thickness_m": _boundary_maximum(state),
        "thickness_thresholds_m": list(THICKNESS_THRESHOLDS_M),
        "area_diagnostics_are_not_formal_gate": True,
        "checkpoint_metrics": checkpoint_metrics,
        "final_checks": final_checks,
        "source_geology_sha256_before": source_hash_before,
        "source_geology_sha256_after": source_hash_after,
        "source_prerequisite_7d_status": relaxation_7d["status"],
        "step_history_csv": str(history_path),
        "final_state_file": str(state_path) if final_state_sha is not None else None,
        "final_state_sha256": final_state_sha,
        "runtime_clipping": False,
        "runtime_limiter": False,
        "mass_balance": "zero",
        "state_saved": final_state_sha is not None,
        "steady_run": False,
        "historical_run": False,
        "next_authorized_stage": None,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "progress.json").write_text(
        json.dumps(
            {"status": summary["status"], "elapsed_years": summary["completed_years"]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--convergence-summary", type=Path, required=True)
    parser.add_argument("--relaxation-60s-summary", type=Path, required=True)
    parser.add_argument("--relaxation-7d-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.geology,
                arguments.static_summary,
                arguments.convergence_summary,
                arguments.relaxation_60s_summary,
                arguments.relaxation_7d_summary,
                arguments.output,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
