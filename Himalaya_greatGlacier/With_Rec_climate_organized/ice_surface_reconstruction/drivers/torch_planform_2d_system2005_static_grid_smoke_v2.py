"""Type-safe read-only PyTorch GPU operator smoke for system 2005.

This supersedes v1 before execution: the only correction is conversion of a
NumPy boolean grid-spacing check to a native Python bool.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

from .torch_planform_2d_frozen_diffusivity_core_v1 import (
    apply_graph_laplacian,
    upstream_face_diffusivities,
)
from .torch_planform_2d_system2005_static_grid_smoke_v1 import (
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
    SYSTEM_ID,
    _maximum_errors,
    _sha256,
)


def run(geology: Path, static_summary_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
    if static_summary.get("status") != "static-pass" or int(static_summary.get("system_id", -1)) != SYSTEM_ID:
        raise RuntimeError("source-static-summary-not-pass")
    expected_hash = str(static_summary["geology_nc_sha256"]).lower()
    source_hash_before = _sha256(geology)
    if source_hash_before.lower() != expected_hash:
        raise RuntimeError("geology-hash-mismatch-before")

    with Dataset(geology, "r") as dataset:
        if int(getattr(dataset, "system_id")) != SYSTEM_ID:
            raise RuntimeError("geology-system-id-mismatch")
        bed = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        thickness = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        surface = np.asarray(dataset.variables["usurf"][:], dtype=np.float64)
        x = np.asarray(dataset.variables["x"][:], dtype=np.float64)
        y = np.asarray(dataset.variables["y"][:], dtype=np.float64)
        dx_attribute = float(getattr(dataset, "dx_m"))

    dx_x = float(np.median(np.abs(np.diff(x))))
    dx_y = float(np.median(np.abs(np.diff(y))))
    grid_checks = {
        "shape_matches_static_summary": bool(list(thickness.shape) == list(static_summary["padded_shape"])),
        "bed_finite": bool(np.isfinite(bed).all()),
        "thickness_finite": bool(np.isfinite(thickness).all()),
        "surface_finite": bool(np.isfinite(surface).all()),
        "thickness_nonnegative": bool((thickness >= 0.0).all()),
        "dx_attribute_matches_coordinates": bool(
            abs(dx_attribute - dx_x) <= 1e-9 and abs(dx_attribute - dx_y) <= 1e-9
        ),
        "surface_closure_max_m": float(np.max(np.abs(surface - bed - thickness))),
    }
    if not all(bool(value) for key, value in grid_checks.items() if key != "surface_closure_max_m"):
        raise RuntimeError("static-grid-precheck-failed")

    glen_a = float(static_summary["stage2_frozen"]["glen_a"])
    gamma = 2.0 * glen_a * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N / (GLEN_N + 2.0)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("registered-sm120-gpu-unavailable")
    cpu_h = torch.from_numpy(thickness).to(dtype=torch.float64)
    cpu_s = torch.from_numpy(surface).to(dtype=torch.float64)
    gpu_h = cpu_h.to("cuda:0")
    gpu_s = cpu_s.to("cuda:0")

    cpu_start = time.perf_counter()
    cpu_east, cpu_south, cpu_symmetry = upstream_face_diffusivities(
        cpu_h, cpu_s, dx=dx_attribute, dy=dx_attribute, glen_n=GLEN_N, gamma=gamma
    )
    cpu_laplacian = apply_graph_laplacian(cpu_s, cpu_east / dx_attribute**2, cpu_south / dx_attribute**2)
    cpu_seconds = time.perf_counter() - cpu_start

    torch.cuda.synchronize()
    gpu_start = time.perf_counter()
    gpu_east, gpu_south, gpu_symmetry = upstream_face_diffusivities(
        gpu_h, gpu_s, dx=dx_attribute, dy=dx_attribute, glen_n=GLEN_N, gamma=gamma
    )
    gpu_laplacian = apply_graph_laplacian(gpu_s, gpu_east / dx_attribute**2, gpu_south / dx_attribute**2)
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - gpu_start

    east_errors = _maximum_errors(cpu_east, gpu_east)
    south_errors = _maximum_errors(cpu_south, gpu_south)
    laplacian_errors = _maximum_errors(cpu_laplacian, gpu_laplacian)
    laplacian_sum = float(torch.sum(gpu_laplacian).item())
    laplacian_l1 = float(torch.sum(torch.abs(gpu_laplacian)).item())
    neutrality_tolerance = max(1e-10, laplacian_l1 * 1e-12)
    checks = {
        "gpu_state_finite": bool(torch.isfinite(gpu_h).all() and torch.isfinite(gpu_s).all()),
        "gpu_faces_finite": bool(torch.isfinite(gpu_east).all() and torch.isfinite(gpu_south).all()),
        "gpu_laplacian_finite": bool(torch.isfinite(gpu_laplacian).all()),
        "face_conductance_nonnegative": bool((gpu_east >= 0).all() and (gpu_south >= 0).all()),
        "face_operator_nontrivial": bool(torch.max(gpu_east) > 0 and torch.max(gpu_south) > 0),
        "east_cpu_gpu_relative_error_pass": east_errors["maximum_relative_error"] <= 1e-11,
        "south_cpu_gpu_relative_error_pass": south_errors["maximum_relative_error"] <= 1e-11,
        "laplacian_cpu_gpu_relative_error_pass": laplacian_errors["maximum_relative_error"] <= 1e-11,
        "laplacian_mass_neutrality_pass": abs(laplacian_sum) <= neutrality_tolerance,
        "east_face_symmetry_pass": float(gpu_symmetry["east_max_relative_difference"]) <= 1e-12,
        "south_face_symmetry_pass": float(gpu_symmetry["south_max_relative_difference"]) <= 1e-12,
    }
    source_hash_after = _sha256(geology)
    checks["source_immutable"] = source_hash_after == source_hash_before
    passed = all(bool(value) for value in checks.values())
    summary: dict[str, object] = {
        "schema": "lia-191-torch-planform2d-system2005-static-grid-smoke-v2",
        "status": "static-gpu-smoke-pass-single-zero-mb-step-eligible" if passed else "static-gpu-smoke-fail-stop",
        "supersedes": "v1-unexecuted-numpy-bool-identity-precheck",
        "scientific_operator_changed": False,
        "system_id": SYSTEM_ID,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
        "shape": list(thickness.shape),
        "dx_m": dx_attribute,
        "glen_a": glen_a,
        "glen_n": GLEN_N,
        "gamma": gamma,
        "grid_checks": grid_checks,
        "checks": checks,
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
        "cpu_to_gpu_speedup_operator_only": cpu_seconds / max(gpu_seconds, 1e-12),
        "east_cpu_gpu_errors": east_errors,
        "south_cpu_gpu_errors": south_errors,
        "laplacian_cpu_gpu_errors": laplacian_errors,
        "laplacian_sum": laplacian_sum,
        "laplacian_l1": laplacian_l1,
        "laplacian_mass_neutrality_tolerance": neutrality_tolerance,
        "cpu_face_symmetry": cpu_symmetry,
        "gpu_face_symmetry": gpu_symmetry,
        "maximum_east_diffusivity_m2_s": float(torch.max(gpu_east).item()),
        "maximum_south_diffusivity_m2_s": float(torch.max(gpu_south).item()),
        "source_geology_sha256_before": source_hash_before,
        "source_geology_sha256_after": source_hash_after,
        "state_updated": False,
        "dynamics_run": False,
        "zero_mb_run": False,
        "steady_run": False,
        "historical_run": False,
        "next_authorized_stage": "single-zero-mb-frozen-diffusivity-step" if passed else None,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.geology, arguments.static_summary, arguments.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
