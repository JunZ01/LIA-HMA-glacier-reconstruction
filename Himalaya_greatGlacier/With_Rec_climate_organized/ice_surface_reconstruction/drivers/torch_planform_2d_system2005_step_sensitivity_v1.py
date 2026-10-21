"""Short timestep and chunked-PCG benchmark for system 2005.

Exactly four independent in-memory zero-MB steps start from the same static
state at 0.1, 1, 10, and 30 seconds.  No result is chained into the next,
saved as a glacier state, or extended to 0.1 years.
"""

from __future__ import annotations

import argparse
import json
import math
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
    _maximum_errors,
    _sha256,
)


TIMESTEPS_SECONDS = (0.1, 1.0, 10.0, 30.0)
SECONDS_PER_YEAR = 31_556_926.0
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


def run(
    geology: Path,
    static_summary_path: Path,
    single_step_summary_path: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
    baseline = json.loads(single_step_summary_path.read_text(encoding="utf-8"))
    if static_summary.get("status") != "static-pass":
        raise RuntimeError("source-static-not-pass")
    if baseline.get("status") != "single-zero-step-pass-0p1y-not-authorized":
        raise RuntimeError("baseline-single-step-not-pass")
    if int(static_summary.get("system_id", -1)) != SYSTEM_ID or int(baseline.get("system_id", -1)) != SYSTEM_ID:
        raise RuntimeError("system-id-mismatch")
    source_hash_before = _sha256(geology)
    if source_hash_before.lower() != str(static_summary["geology_nc_sha256"]).lower():
        raise RuntimeError("geology-hash-mismatch")

    with Dataset(geology, "r") as dataset:
        bed = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        thickness = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        dx = float(getattr(dataset, "dx_m"))
    if not np.isfinite(bed).all() or not np.isfinite(thickness).all() or np.any(thickness < 0.0):
        raise RuntimeError("invalid-initial-grid")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("registered-sm120-gpu-unavailable")

    glen_a = float(static_summary["stage2_frozen"]["glen_a"])
    gamma = 2.0 * glen_a * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N / (GLEN_N + 2.0)
    cpu_h0 = torch.from_numpy(thickness).to(dtype=torch.float64)
    cpu_bed = torch.from_numpy(bed).to(dtype=torch.float64)
    gpu_h0 = cpu_h0.to("cuda:0")
    gpu_bed = cpu_bed.to("cuda:0")
    cell_area = dx**2
    initial_volume = float(torch.sum(cpu_h0).item()) * cell_area
    closure_tolerance = max(1e-6, initial_volume * 1e-12)

    states: dict[float, torch.Tensor] = {}
    records: dict[str, dict[str, object]] = {}
    for dt in TIMESTEPS_SECONDS:
        torch.cuda.synchronize()
        started = time.perf_counter()
        state, diagnostics = frozen_diffusivity_step_chunked(
            gpu_h0,
            gpu_bed,
            dx=dx,
            dy=dx,
            glen_n=GLEN_N,
            gamma=gamma,
            dt=dt,
            rtol=SOLVER_RTOL,
            maxiter=SOLVER_MAXITER,
            check_interval=CHECK_INTERVAL,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        states[dt] = state
        delta = state - gpu_h0
        final_volume = float(torch.sum(state).item()) * cell_area
        closure = final_volume - initial_volume
        flat_index = int(torch.argmax(torch.abs(delta)).item())
        row = flat_index // state.shape[1]
        column = flat_index % state.shape[1]
        checks = {
            "solver_converged": bool(diagnostics["solver_converged"]),
            "finite": bool(diagnostics["state_finite"]),
            "negative_cell_count_zero": int(diagnostics["negative_cell_count"]) == 0,
            "mass_closure_pass": abs(closure) <= closure_tolerance,
            "boundary_thickness_pass": _boundary_maximum(state) <= 10.0,
        }
        records[f"{dt:g}"] = {
            "dt_seconds": dt,
            "gpu_seconds": elapsed,
            "maximum_absolute_change_m": float(torch.max(torch.abs(delta)).item()),
            "maximum_absolute_tendency_m_s": float(torch.max(torch.abs(delta / dt)).item()),
            "mass_closure_m3": closure,
            "mass_closure_tolerance_m3": closure_tolerance,
            "maximum_boundary_thickness_m": _boundary_maximum(state),
            "maximum_change_location_row_column": [row, column],
            "initial_thickness_at_maximum_change_m": float(gpu_h0[row, column].item()),
            "bed_at_maximum_change_m": float(gpu_bed[row, column].item()),
            "diagnostics": diagnostics,
            "checks": checks,
            "passed_numerical_gates": all(bool(value) for value in checks.values()),
        }

    reference_tendency = (states[0.1] - gpu_h0) / 0.1
    reference_norm = max(float(torch.linalg.vector_norm(reference_tendency).item()), 1e-30)
    for dt in TIMESTEPS_SECONDS:
        tendency = (states[dt] - gpu_h0) / dt
        difference = tendency - reference_tendency
        records[f"{dt:g}"]["tendency_l2_relative_to_0p1s"] = float(
            torch.linalg.vector_norm(difference).item()
        ) / reference_norm
        records[f"{dt:g}"]["tendency_max_abs_difference_from_0p1s_m_s"] = float(
            torch.max(torch.abs(difference)).item()
        )

    cpu_start = time.perf_counter()
    cpu_30, cpu_30_diagnostics = frozen_diffusivity_step_chunked(
        cpu_h0,
        cpu_bed,
        dx=dx,
        dy=dx,
        glen_n=GLEN_N,
        gamma=gamma,
        dt=30.0,
        rtol=SOLVER_RTOL,
        maxiter=SOLVER_MAXITER,
        check_interval=CHECK_INTERVAL,
    )
    cpu_30_seconds = time.perf_counter() - cpu_start
    parity_30 = _maximum_errors(cpu_30, states[30.0])
    baseline_gpu_seconds = float(baseline["gpu_seconds"])
    optimized_gpu_seconds = float(records["30"]["gpu_seconds"])
    projected_steps_0p1y_at_30s = int(math.ceil(0.1 * SECONDS_PER_YEAR / 30.0))
    projected_gpu_hours_0p1y_at_30s = projected_steps_0p1y_at_30s * optimized_gpu_seconds / 3600.0
    source_hash_after = _sha256(geology)
    global_checks = {
        "all_timestep_numerical_gates_pass": all(
            bool(records[f"{dt:g}"]["passed_numerical_gates"]) for dt in TIMESTEPS_SECONDS
        ),
        "optimized_30s_cpu_converged": bool(cpu_30_diagnostics["solver_converged"]),
        "optimized_30s_cpu_gpu_parity_pass": parity_30["maximum_relative_error"] <= 1e-10,
        "source_immutable": source_hash_after == source_hash_before,
    }
    completed = all(bool(value) for value in global_checks.values())
    summary: dict[str, object] = {
        "schema": "lia-191-torch-planform2d-system2005-step-sensitivity-v1",
        "status": "sensitivity-complete-0p1y-not-authorized" if completed else "sensitivity-fail-stop",
        "system_id": SYSTEM_ID,
        "timesteps_seconds": list(TIMESTEPS_SECONDS),
        "each_step_restarts_from_identical_static_state": True,
        "solver_rtol": SOLVER_RTOL,
        "solver_maxiter": SOLVER_MAXITER,
        "solver_check_interval": CHECK_INTERVAL,
        "records": records,
        "optimized_30s_cpu_seconds": cpu_30_seconds,
        "optimized_30s_cpu_diagnostics": cpu_30_diagnostics,
        "optimized_30s_cpu_gpu_errors": parity_30,
        "baseline_v1_gpu_seconds": baseline_gpu_seconds,
        "optimized_v2_gpu_seconds_30s": optimized_gpu_seconds,
        "gpu_speedup_v2_over_v1_30s": baseline_gpu_seconds / max(optimized_gpu_seconds, 1e-12),
        "optimized_cpu_to_gpu_speedup_30s": cpu_30_seconds / max(optimized_gpu_seconds, 1e-12),
        "projected_steps_0p1y_at_30s": projected_steps_0p1y_at_30s,
        "projected_gpu_hours_0p1y_at_30s": projected_gpu_hours_0p1y_at_30s,
        "global_checks": global_checks,
        "source_geology_sha256_before": source_hash_before,
        "source_geology_sha256_after": source_hash_after,
        "runtime_clipping": False,
        "runtime_limiter": False,
        "state_saved": False,
        "steps_chained": False,
        "zero_mb_0p1y_run": False,
        "steady_run": False,
        "historical_run": False,
        "next_authorized_stage": None,
        "adjudication_required": "choose-performance-optimization-and-accuracy-contract-before-any-0p1y-run",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--single-step-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.geology, arguments.static_summary, arguments.single_step_summary, arguments.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
