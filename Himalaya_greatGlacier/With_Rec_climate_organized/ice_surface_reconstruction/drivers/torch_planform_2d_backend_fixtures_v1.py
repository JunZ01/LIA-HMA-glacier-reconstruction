"""Pre-real-system fixtures for the PyTorch 2-D frozen-diffusivity backend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .torch_planform_2d_frozen_diffusivity_core_v1 import (
    frozen_diffusivity_step,
    upstream_face_diffusivities,
)


def _zero_diffusivity_fixture(device: torch.device) -> dict[str, object]:
    dtype = torch.float64
    thickness = torch.full((24, 28), 70.0, dtype=dtype, device=device)
    bed = torch.full_like(thickness, 3100.0)
    updated, diagnostics = frozen_diffusivity_step(
        thickness,
        bed,
        dx=100.0,
        dy=100.0,
        glen_n=3.0,
        gamma=7.0e-13,
        dt=31_556_926.0,
    )
    bitwise_identity = bool(torch.equal(updated, thickness))
    passed = bitwise_identity and diagnostics["solver_converged"] and diagnostics["solver_iterations"] == 0
    return {
        "passed": passed,
        "bitwise_identity": bitwise_identity,
        "diagnostics": diagnostics,
    }


def _analytic_planar_slope_fixture(device: torch.device) -> dict[str, object]:
    dtype = torch.float64
    ny, nx = 20, 26
    dx = dy = 100.0
    slope = 0.025
    thickness = torch.full((ny, nx), 80.0, dtype=dtype, device=device)
    x = torch.arange(nx, dtype=dtype, device=device) * dx
    surface = 3600.0 - slope * x[None, :].expand(ny, nx)
    gamma = 7.0e-13
    east, south, symmetry = upstream_face_diffusivities(
        thickness,
        surface,
        dx=dx,
        dy=dy,
        glen_n=3.0,
        gamma=gamma,
    )
    expected = gamma * 80.0**5 * slope**2
    interior = east[1:-1, 1:-1]
    relative_error = float(torch.max(torch.abs(interior - expected)).item()) / max(abs(expected), 1e-30)
    south_max = float(torch.max(torch.abs(south)).item())
    passed = relative_error <= 1e-12 and south_max <= max(abs(expected) * 1e-12, 1e-20)
    return {
        "passed": passed,
        "expected_east_diffusivity_m2_s": expected,
        "relative_error": relative_error,
        "south_max_abs_m2_s": south_max,
        "face_symmetry": symmetry,
    }


def _manufactured_conservation_fixture(device: torch.device) -> dict[str, object]:
    dtype = torch.float64
    ny, nx = 42, 48
    dx = dy = 100.0
    y = torch.linspace(-1.0, 1.0, ny, dtype=dtype, device=device)[:, None]
    x = torch.linspace(-1.0, 1.0, nx, dtype=dtype, device=device)[None, :]
    thickness = 75.0 + 8.0 * torch.exp(-3.0 * (x**2 + y**2))
    bed = 3300.0 - 180.0 * x - 110.0 * y + 15.0 * torch.sin(2.0 * x) * torch.cos(3.0 * y)
    updated, diagnostics = frozen_diffusivity_step(
        thickness,
        bed,
        dx=dx,
        dy=dy,
        glen_n=3.0,
        gamma=7.0e-13,
        dt=86400.0 * 5.0,
        rtol=1e-12,
    )
    initial_sum = float(torch.sum(thickness).item())
    final_sum = float(torch.sum(updated).item())
    closure = final_sum - initial_sum
    tolerance = max(1e-10, abs(initial_sum) * 1e-13)
    changed = float(torch.max(torch.abs(updated - thickness)).item())
    passed = (
        diagnostics["solver_converged"]
        and diagnostics["state_finite"]
        and diagnostics["negative_cell_count"] == 0
        and abs(closure) <= tolerance
        and changed > 1e-12
    )
    return {
        "passed": passed,
        "initial_thickness_sum_m": initial_sum,
        "final_thickness_sum_m": final_sum,
        "closure_m": closure,
        "closure_tolerance_m": tolerance,
        "maximum_state_change_m": changed,
        "diagnostics": diagnostics,
    }


def _cpu_gpu_parity_fixture() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"passed": False, "reason": "cuda-unavailable"}
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    dtype = torch.float64
    ny, nx = 30, 34
    y = torch.linspace(-1.0, 1.0, ny, dtype=dtype)[:, None]
    x = torch.linspace(-1.0, 1.0, nx, dtype=dtype)[None, :]
    thickness = 65.0 + 5.0 * torch.cos(2.0 * x) * torch.cos(2.0 * y)
    bed = 3200.0 - 120.0 * x - 90.0 * y
    kwargs = dict(dx=100.0, dy=100.0, glen_n=3.0, gamma=7.0e-13, dt=172800.0, rtol=1e-12)
    cpu_state, cpu_diag = frozen_diffusivity_step(thickness.to(cpu), bed.to(cpu), **kwargs)
    gpu_state, gpu_diag = frozen_diffusivity_step(thickness.to(gpu), bed.to(gpu), **kwargs)
    gpu_on_cpu = gpu_state.cpu()
    max_absolute_error = float(torch.max(torch.abs(cpu_state - gpu_on_cpu)).item())
    scale = max(float(torch.max(torch.abs(cpu_state)).item()), 1.0)
    relative_error = max_absolute_error / scale
    passed = (
        cpu_diag["solver_converged"]
        and gpu_diag["solver_converged"]
        and relative_error <= 1e-11
        and bool(torch.isfinite(gpu_state).all())
    )
    return {
        "passed": passed,
        "max_absolute_error_m": max_absolute_error,
        "max_relative_error": relative_error,
        "cpu_diagnostics": cpu_diag,
        "gpu_diagnostics": gpu_diag,
    }


def run(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("cuda-unavailable")
    device = torch.device("cuda:0")
    fixtures = {
        "zero_diffusivity_identity_gpu": _zero_diffusivity_fixture(device),
        "analytic_planar_slope_gpu": _analytic_planar_slope_fixture(device),
        "manufactured_mass_conservation_gpu": _manufactured_conservation_fixture(device),
        "cpu_gpu_parity": _cpu_gpu_parity_fixture(),
    }
    passed = all(bool(item["passed"]) for item in fixtures.values())
    summary: dict[str, object] = {
        "schema": "lia-191-torch-planform2d-backend-fixtures-v1",
        "status": "fixture-pass-system2005-static-grid-authorized" if passed else "fixture-fail-mechanism-closed",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
        "default_dtype": str(torch.get_default_dtype()),
        "fixtures": fixtures,
        "system2005_state_loaded": False,
        "dynamics_run": False,
        "zero_mb_run": False,
        "steady_run": False,
        "historical_run": False,
        "next_authorized_stage": "system2005-static-grid-gpu-operator-smoke" if passed else None,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
