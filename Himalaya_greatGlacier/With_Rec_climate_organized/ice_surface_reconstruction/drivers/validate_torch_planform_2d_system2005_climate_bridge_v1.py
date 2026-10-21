"""Validate the exported system-2005 MonthlyTIModel bridge on the GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

from .torch_planform_2d_monthly_ti_bridge_v1 import TorchMonthlyTIBridge


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    package_path: Path,
    reference_path: Path,
    export_summary_path: Path,
    geology_path: Path,
    output: Path,
) -> dict[str, object]:
    export = json.loads(export_summary_path.read_text(encoding="utf-8"))
    if export.get("status") != "climate-bridge-export-pass":
        raise RuntimeError("export-prerequisite-not-pass")
    if sha256(package_path) != export["package_sha256"]:
        raise RuntimeError("package-hash-mismatch")
    if sha256(reference_path) != export["reference_sha256"]:
        raise RuntimeError("reference-hash-mismatch")
    if sha256(geology_path) != export["geology_source_sha256"]:
        raise RuntimeError("geology-hash-mismatch")
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("registered-sm120-gpu-unavailable")

    bridge = TorchMonthlyTIBridge(package_path, device="cuda:0")
    with np.load(reference_path) as reference:
        validation_heights = np.asarray(reference["validation_heights_m"], dtype=np.float64)
        authoritative = np.asarray(reference["reference_monthly_mb_m_ice_s"], dtype=np.float64)
        active_indices = np.asarray(reference["active_flat_indices"], dtype=np.int64)
        active_heights = np.asarray(reference["active_surface_heights_m"], dtype=np.float64)
        selected_month_indices = np.asarray(reference["selected_month_indices"], dtype=np.int64)
        active_authoritative = np.asarray(
            reference["active_reference_monthly_mb_m_ice_s"], dtype=np.float64
        )

    started = time.perf_counter()
    height_tensor = torch.from_numpy(validation_heights).to(device="cuda:0", dtype=torch.float64)
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for month_index in range(authoritative.shape[0]):
        evaluated = bridge.monthly_mb(height_tensor, month_index).cpu().numpy()
        difference = np.abs(evaluated - authoritative[month_index])
        maximum_absolute_error = max(maximum_absolute_error, float(np.max(difference)))
        maximum_relative_error = max(
            maximum_relative_error,
            float(np.max(difference / np.maximum(np.abs(authoritative[month_index]), 1e-30))),
        )

    active_height_tensor = torch.from_numpy(active_heights).to(device="cuda:0", dtype=torch.float64)
    maximum_active_absolute_error = 0.0
    maximum_active_relative_error = 0.0
    for row, month_index in enumerate(selected_month_indices):
        evaluated = bridge.monthly_mb(active_height_tensor, int(month_index)).cpu().numpy()
        difference = np.abs(evaluated - active_authoritative[row])
        maximum_active_absolute_error = max(maximum_active_absolute_error, float(np.max(difference)))
        maximum_active_relative_error = max(
            maximum_active_relative_error,
            float(np.max(difference / np.maximum(np.abs(active_authoritative[row]), 1e-30))),
        )

    with Dataset(geology_path, "r") as dataset:
        bed_np = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        thickness_np = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        dx = float(getattr(dataset, "dx_m"))
    surface_np = bed_np + thickness_np
    actual_active_indices = np.flatnonzero((thickness_np > bridge.active_thickness_tolerance_m).ravel())
    if not np.array_equal(actual_active_indices, active_indices):
        raise RuntimeError("active-index-reference-mismatch")
    if not np.array_equal(surface_np.ravel()[active_indices], active_heights):
        raise RuntimeError("active-height-reference-mismatch")

    state = torch.from_numpy(thickness_np).to(device="cuda:0", dtype=torch.float64)
    bed = torch.from_numpy(bed_np).to(device="cuda:0", dtype=torch.float64)
    dt_seconds = 21_600.0
    updated, diagnostics = bridge.apply_connected_ice_source(
        state,
        bed,
        month_index=int(selected_month_indices[0]),
        dt_seconds=dt_seconds,
    )
    torch.cuda.synchronize()
    updated_np = updated.cpu().numpy()
    expected_active = np.maximum(
        thickness_np.ravel()[active_indices] + dt_seconds * active_authoritative[0],
        0.0,
    )
    actual_active = updated_np.ravel()[active_indices]
    source_state_maximum_absolute_error = float(np.max(np.abs(actual_active - expected_active)))
    expected_change_volume = float(
        np.sum(expected_active - thickness_np.ravel()[active_indices]) * dx**2
    )
    actual_change_volume = float(np.sum(updated_np - thickness_np) * dx**2)
    source_volume_absolute_error = abs(actual_change_volume - expected_change_volume)
    source_volume_tolerance = max(1e-6, abs(expected_change_volume) * 1e-12)
    inactive = thickness_np <= bridge.active_thickness_tolerance_m
    inactive_bitwise_unchanged = np.array_equal(updated_np[inactive], thickness_np[inactive])

    checks = {
        "all_600_months_gpu_parity": maximum_absolute_error <= 1e-18 and maximum_relative_error <= 1e-10,
        "actual_grid_selected_month_gpu_parity": (
            maximum_active_absolute_error <= 1e-18 and maximum_active_relative_error <= 1e-10
        ),
        "source_state_gpu_cpu_parity": source_state_maximum_absolute_error <= 1e-12,
        "source_volume_budget_parity": source_volume_absolute_error <= source_volume_tolerance,
        "remote_zero_cells_unchanged": bool(diagnostics["remote_zero_unchanged"]) and inactive_bitwise_unchanged,
        "source_state_finite": bool(diagnostics["state_finite"]),
        "source_state_nonnegative": int(diagnostics["negative_cell_count"]) == 0,
        "source_files_immutable": (
            sha256(package_path) == export["package_sha256"]
            and sha256(reference_path) == export["reference_sha256"]
            and sha256(geology_path) == export["geology_source_sha256"]
        ),
    }
    summary = {
        "schema": "lia-191-torch-planform2d-system2005-climate-bridge-validation-v1",
        "status": (
            "climate-bridge-gpu-validation-pass-50y-not-run"
            if all(checks.values())
            else "climate-bridge-gpu-validation-fail-stop"
        ),
        "system_id": 2005,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "month_count": int(authoritative.shape[0]),
        "validation_height_count": int(validation_heights.size),
        "actual_active_grid_cell_count": int(active_indices.size),
        "selected_actual_grid_month_indices": selected_month_indices.tolist(),
        "maximum_gpu_reference_absolute_error_m_ice_s": maximum_absolute_error,
        "maximum_gpu_reference_relative_error": maximum_relative_error,
        "maximum_actual_grid_absolute_error_m_ice_s": maximum_active_absolute_error,
        "maximum_actual_grid_relative_error": maximum_active_relative_error,
        "source_application_dt_seconds": dt_seconds,
        "source_state_maximum_absolute_error_m": source_state_maximum_absolute_error,
        "expected_source_volume_change_m3": expected_change_volume,
        "actual_source_volume_change_m3": actual_change_volume,
        "source_volume_absolute_error_m3": source_volume_absolute_error,
        "source_volume_tolerance_m3": source_volume_tolerance,
        "source_diagnostics": diagnostics,
        "checks": checks,
        "wall_seconds": time.perf_counter() - started,
        "package_file": str(package_path),
        "package_sha256": sha256(package_path),
        "reference_file": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "geology_file": str(geology_path),
        "geology_sha256": sha256(geology_path),
        "dynamics_run": False,
        "steady_run": False,
        "historical_run": False,
        "next_authorized_stage": None,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--export-summary", type=Path, required=True)
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = run(
        arguments.package,
        arguments.reference,
        arguments.export_summary,
        arguments.geology,
        arguments.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "climate-bridge-gpu-validation-pass-50y-not-run":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
