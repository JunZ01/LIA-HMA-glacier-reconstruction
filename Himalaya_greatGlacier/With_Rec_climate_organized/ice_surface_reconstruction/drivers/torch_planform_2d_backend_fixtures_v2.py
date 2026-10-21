"""Corrected pre-real-system fixtures for the PyTorch 2-D backend.

Version 2 supersedes only the planar-slope assertion in v1: a transverse
face diffusivity may be nonzero while its flux is exactly zero because the
surface drop across that face is zero.  The backend and all tolerances are
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .torch_planform_2d_backend_fixtures_v1 import (
    _cpu_gpu_parity_fixture,
    _manufactured_conservation_fixture,
    _zero_diffusivity_fixture,
)
from .torch_planform_2d_frozen_diffusivity_core_v1 import upstream_face_diffusivities


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
    east_relative_error = float(torch.max(torch.abs(east[1:-1, 1:-1] - expected)).item()) / max(
        abs(expected), 1e-30
    )
    south_relative_error = float(torch.max(torch.abs(south - expected)).item()) / max(
        abs(expected), 1e-30
    )
    south_surface_drop = surface[:-1, :] - surface[1:, :]
    south_flux_max = float(torch.max(torch.abs(south * south_surface_drop)).item())
    passed = (
        east_relative_error <= 1e-12
        and south_relative_error <= 1e-12
        and south_flux_max <= max(abs(expected) * 1e-12, 1e-20)
    )
    return {
        "passed": passed,
        "expected_face_diffusivity_m2_s": expected,
        "east_relative_error": east_relative_error,
        "south_relative_error": south_relative_error,
        "south_flux_max_abs_m3_s": south_flux_max,
        "face_symmetry": symmetry,
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
        "schema": "lia-191-torch-planform2d-backend-fixtures-v2",
        "status": "fixture-pass-system2005-static-grid-authorized" if passed else "fixture-fail-mechanism-closed",
        "supersedes": "lia-191-torch-planform2d-backend-fixtures-v1-planar-slope-assertion-only",
        "scientific_operator_changed": False,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
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
