"""One in-memory 30-second zero-MB step for system 2005's 2-D grid.

The step uses the frozen-diffusivity conservative PyTorch backend.  CPU and
GPU execute the same operator for numerical comparison.  No clipping,
limiting, state serialization, repeated stepping, steady state, or historical
climate is permitted here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

from .torch_planform_2d_frozen_diffusivity_core_v1 import frozen_diffusivity_step
from .torch_planform_2d_system2005_static_grid_smoke_v1 import (
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
    SYSTEM_ID,
    _maximum_errors,
    _sha256,
)


DT_SECONDS = 30.0
SOLVER_RTOL = 1e-11
SOLVER_MAXITER = 4000


def _maximum_boundary_thickness(state: torch.Tensor) -> float:
    return float(
        torch.max(
            torch.stack(
                (
                    torch.max(state[0, :]),
                    torch.max(state[-1, :]),
                    torch.max(state[:, 0]),
                    torch.max(state[:, -1]),
                )
            )
        ).item()
    )


def run(
    geology: Path,
    static_summary_path: Path,
    static_smoke_path: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
    static_smoke = json.loads(static_smoke_path.read_text(encoding="utf-8"))
    if static_summary.get("status") != "static-pass":
        raise RuntimeError("source-static-not-pass")
    if static_smoke.get("status") != "static-gpu-smoke-pass-single-zero-mb-step-eligible":
        raise RuntimeError("static-gpu-smoke-not-pass")
    if int(static_summary.get("system_id", -1)) != SYSTEM_ID or int(static_smoke.get("system_id", -1)) != SYSTEM_ID:
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
    initial_volume_m3 = float(torch.sum(cpu_h0).item()) * cell_area
    closure_tolerance_m3 = max(1e-6, initial_volume_m3 * 1e-12)

    torch.cuda.synchronize()
    gpu_start = time.perf_counter()
    gpu_h1, gpu_diagnostics = frozen_diffusivity_step(
        gpu_h0,
        gpu_bed,
        dx=dx,
        dy=dx,
        glen_n=GLEN_N,
        gamma=gamma,
        dt=DT_SECONDS,
        rtol=SOLVER_RTOL,
        maxiter=SOLVER_MAXITER,
    )
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - gpu_start

    cpu_start = time.perf_counter()
    cpu_h1, cpu_diagnostics = frozen_diffusivity_step(
        cpu_h0,
        cpu_bed,
        dx=dx,
        dy=dx,
        glen_n=GLEN_N,
        gamma=gamma,
        dt=DT_SECONDS,
        rtol=SOLVER_RTOL,
        maxiter=SOLVER_MAXITER,
    )
    cpu_seconds = time.perf_counter() - cpu_start

    parity = _maximum_errors(cpu_h1, gpu_h1)
    final_volume_gpu_m3 = float(torch.sum(gpu_h1).item()) * cell_area
    final_volume_cpu_m3 = float(torch.sum(cpu_h1).item()) * cell_area
    gpu_closure_m3 = final_volume_gpu_m3 - initial_volume_m3
    cpu_closure_m3 = final_volume_cpu_m3 - initial_volume_m3
    maximum_change_m = float(torch.max(torch.abs(gpu_h1 - gpu_h0)).item())
    maximum_boundary_thickness_m = _maximum_boundary_thickness(gpu_h1)
    source_hash_after = _sha256(geology)
    checks = {
        "gpu_solver_converged": bool(gpu_diagnostics["solver_converged"]),
        "cpu_solver_converged": bool(cpu_diagnostics["solver_converged"]),
        "gpu_state_finite": bool(gpu_diagnostics["state_finite"]),
        "cpu_state_finite": bool(cpu_diagnostics["state_finite"]),
        "gpu_negative_cell_count_zero": int(gpu_diagnostics["negative_cell_count"]) == 0,
        "cpu_negative_cell_count_zero": int(cpu_diagnostics["negative_cell_count"]) == 0,
        "gpu_mass_closure_pass": abs(gpu_closure_m3) <= closure_tolerance_m3,
        "cpu_mass_closure_pass": abs(cpu_closure_m3) <= closure_tolerance_m3,
        "cpu_gpu_state_relative_error_pass": parity["maximum_relative_error"] <= 1e-10,
        "state_change_nonzero": maximum_change_m > 0.0,
        "boundary_thickness_pass": maximum_boundary_thickness_m <= 10.0,
        "source_immutable": source_hash_after == source_hash_before,
    }
    passed = all(bool(value) for value in checks.values())
    summary: dict[str, object] = {
        "schema": "lia-191-torch-planform2d-system2005-single-zero-step-v1",
        "status": "single-zero-step-pass-0p1y-not-authorized" if passed else "single-zero-step-fail-stop",
        "system_id": SYSTEM_ID,
        "mechanism": "torch-frozen-diffusivity-conservative-no-flux-pcg",
        "dt_seconds": DT_SECONDS,
        "mass_balance": "zero",
        "solver_rtol": SOLVER_RTOL,
        "solver_maxiter": SOLVER_MAXITER,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
        "shape": list(thickness.shape),
        "dx_m": dx,
        "glen_a": glen_a,
        "glen_n": GLEN_N,
        "gamma": gamma,
        "initial_volume_m3": initial_volume_m3,
        "final_volume_gpu_m3": final_volume_gpu_m3,
        "final_volume_cpu_m3": final_volume_cpu_m3,
        "gpu_mass_closure_m3": gpu_closure_m3,
        "cpu_mass_closure_m3": cpu_closure_m3,
        "mass_closure_tolerance_m3": closure_tolerance_m3,
        "maximum_state_change_m": maximum_change_m,
        "maximum_boundary_thickness_m": maximum_boundary_thickness_m,
        "cpu_gpu_state_errors": parity,
        "gpu_seconds": gpu_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_to_gpu_speedup_full_step": cpu_seconds / max(gpu_seconds, 1e-12),
        "gpu_diagnostics": gpu_diagnostics,
        "cpu_diagnostics": cpu_diagnostics,
        "checks": checks,
        "source_geology_sha256_before": source_hash_before,
        "source_geology_sha256_after": source_hash_after,
        "runtime_clipping": False,
        "runtime_limiter": False,
        "state_saved": False,
        "repeated_steps_run": False,
        "zero_mb_0p1y_run": False,
        "steady_run": False,
        "historical_run": False,
        "next_authorized_stage": "zero-mb-0p1y-short-benchmark" if passed else None,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--static-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.geology, arguments.static_summary, arguments.static_smoke, arguments.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
