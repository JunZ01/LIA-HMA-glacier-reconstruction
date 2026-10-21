"""Formal 50-year planform-2D PyTorch steady run for LIA system 2005.

The trajectory starts from the immutable padded V3/F19 geology, uses the
validated frozen-diffusivity GPU core and the exact exported OGGM MonthlyTI
bridge, and advances one uninterrupted 50-year B-window repeat.  The source
and diffusion are Strang split.  A full step is compared with two half steps;
rejected trials are never committed.  Monthly and formal checkpoint boundaries
are hit exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from netCDF4 import Dataset

from .torch_planform_2d_monthly_ti_bridge_v1 import TorchMonthlyTIBridge
from .torch_planform_2d_system2005_adaptive_relaxation_60s_v1 import (
    ERROR_GATE,
    GLEN_N,
    GRAVITY_M_S2,
    GROWTH_ERROR_THRESHOLD,
    ICE_DENSITY_KG_M3,
    MAXIMUM_CONSECUTIVE_REJECTIONS,
    MINIMUM_DT_SECONDS,
    SYSTEM_ID,
    _boundary_maximum,
    _step,
)
from .torch_planform_2d_system2005_adaptive_relaxation_600s_v1 import (
    THICKNESS_THRESHOLDS_M,
    _area_metrics,
    _with_relative_area_change,
)


TARGET_YEARS = 50
MAXIMUM_DT_SECONDS = 86_400.0
MAXIMUM_ACCEPTED_STEPS = 500_000
STEADY_VOLUME_CHANGE_GATE = 0.02
LENGTH_RELATIVE_ERROR_GATE = 0.10
FREE_DOMAIN_GATE_M = 5_000.0
LENGTH_ICE_THRESHOLD_M = 10.0
CUMULATIVE_BUDGET_RELATIVE_GATE = 1.0e-10
RECOVERY_SCHEMA = "lia-191-torch-planform2d-system2005-steady50-recovery-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _control_lengths(coordinates: np.ndarray) -> np.ndarray:
    segments = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    result = np.zeros(len(coordinates), dtype=np.float64)
    if len(coordinates) == 1:
        return result
    result[0] = 0.5 * segments[0]
    result[-1] = 0.5 * segments[-1]
    if len(coordinates) > 2:
        result[1:-1] = 0.5 * (segments[:-1] + segments[1:])
    return result


def _sample_regular_grid(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    points_xy: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a regular grid without adding a SciPy dependency."""
    if x.ndim != 1 or y.ndim != 1 or values.shape != (len(y), len(x)):
        raise ValueError("invalid-regular-grid")
    if len(x) < 2 or len(y) < 2:
        raise ValueError("undersized-regular-grid")
    work = values
    x_work = x
    y_work = y
    if x[1] < x[0]:
        x_work = x[::-1]
        work = work[:, ::-1]
    if y[1] < y[0]:
        y_work = y[::-1]
        work = work[::-1, :]
    dx = float(x_work[1] - x_work[0])
    dy = float(y_work[1] - y_work[0])
    if not np.allclose(np.diff(x_work), dx) or not np.allclose(np.diff(y_work), dy):
        raise ValueError("nonuniform-regular-grid")
    fx = (points_xy[:, 0] - x_work[0]) / dx
    fy = (points_xy[:, 1] - y_work[0]) / dy
    valid = (fx >= 0.0) & (fx <= len(x_work) - 1) & (fy >= 0.0) & (fy <= len(y_work) - 1)
    result = np.full(len(points_xy), np.nan, dtype=np.float64)
    ix0 = np.floor(np.clip(fx, 0.0, len(x_work) - 1)).astype(int)
    iy0 = np.floor(np.clip(fy, 0.0, len(y_work) - 1)).astype(int)
    ix1 = np.minimum(ix0 + 1, len(x_work) - 1)
    iy1 = np.minimum(iy0 + 1, len(y_work) - 1)
    wx = fx - ix0
    wy = fy - iy0
    sampled = (
        (1.0 - wx) * (1.0 - wy) * work[iy0, ix0]
        + wx * (1.0 - wy) * work[iy0, ix1]
        + (1.0 - wx) * wy * work[iy1, ix0]
        + wx * wy * work[iy1, ix1]
    )
    result[valid] = sampled[valid]
    return result


def _path_metrics(
    thickness: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    path_xy: np.ndarray,
    control_lengths: np.ndarray,
    registered_length_m: float,
) -> dict[str, Any]:
    sampled = _sample_regular_grid(thickness, x, y, path_xy)
    active = np.isfinite(sampled) & (sampled > LENGTH_ICE_THRESHOLD_M)
    active_indices = np.flatnonzero(active)
    observed_length = float(np.sum(control_lengths[active]))
    if len(active_indices):
        first = int(active_indices[0])
        last = int(active_indices[-1])
        gaps = np.flatnonzero(~active[first : last + 1]) + first
    else:
        first = None
        last = None
        gaps = np.asarray([], dtype=int)
    return {
        "observed_active_control_volume_length_m": observed_length,
        "registered_lia_length_m": registered_length_m,
        "registered_length_relative_error": abs(observed_length - registered_length_m)
        / registered_length_m,
        "root_first_active_node": first,
        "root_last_active_node": last,
        "root_internal_gap_count_diagnostic_only": int(len(gaps)),
        "root_internal_gap_indices_diagnostic_only": gaps.tolist(),
        "root_gap_policy": "classify-and-report; not a numerical failure for the 2D state",
        "sampled_root_maximum_thickness_m": float(np.nanmax(sampled)),
    }


def _free_domain_metrics(thickness: np.ndarray, dx: float) -> dict[str, float | int]:
    active = np.argwhere(thickness > LENGTH_ICE_THRESHOLD_M)
    if len(active) == 0:
        return {
            "active_cell_count_gt_10m": 0,
            "minimum_active_cell_center_to_grid_edge_m": float("inf"),
        }
    ny, nx = thickness.shape
    rows = active[:, 0]
    cols = active[:, 1]
    distance_cells = np.minimum.reduce(
        [rows + 0.5, ny - rows - 0.5, cols + 0.5, nx - cols - 0.5]
    )
    return {
        "active_cell_count_gt_10m": int(len(active)),
        "minimum_active_cell_center_to_grid_edge_m": float(np.min(distance_cells) * dx),
    }


def _valid_diffusion(diagnostic: dict[str, Any]) -> bool:
    return (
        bool(diagnostic.get("solver_converged", False))
        and bool(diagnostic.get("state_finite", False))
        and int(diagnostic.get("negative_cell_count", 1)) == 0
    )


def _safe_diffusion(
    state: torch.Tensor,
    bed: torch.Tensor,
    *,
    dx: float,
    gamma: float,
    dt: float,
) -> tuple[torch.Tensor, dict[str, Any], bool]:
    try:
        result, diagnostic = _step(state, bed, dx=dx, gamma=gamma, dt=dt)
    except ValueError as error:
        if str(error) != "negative-input-thickness":
            raise
        diagnostic = {
            "solver_converged": False,
            "state_finite": bool(torch.isfinite(state).all()),
            "negative_cell_count": int(torch.count_nonzero(state < 0).item()),
            "solver_iterations": 0,
            "rejection_reason": "negative-uncommitted-intermediate-input",
        }
        return state.clone(), diagnostic, True
    return result, diagnostic, False


def _split_step(
    state: torch.Tensor,
    bed: torch.Tensor,
    bridge: TorchMonthlyTIBridge,
    *,
    month_index: int,
    dx: float,
    gamma: float,
    dt: float,
    cell_area: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source1, source1_diag = bridge.apply_connected_ice_source(
        state, bed, month_index=month_index, dt_seconds=0.5 * dt
    )
    diffused, diffusion_diag, negative_rejection = _safe_diffusion(
        source1, bed, dx=dx, gamma=gamma, dt=dt
    )
    diffusion_closure = float(torch.sum(diffused - source1).item()) * cell_area
    if _valid_diffusion(diffusion_diag):
        result, source2_diag = bridge.apply_connected_ice_source(
            diffused, bed, month_index=month_index, dt_seconds=0.5 * dt
        )
    else:
        result = diffused
        source2_diag = {
            "active_cell_count": 0,
            "remote_zero_cell_count": int(result.numel()),
            "remote_zero_unchanged": True,
            "prevented_overmelt_sum_m": 0.0,
            "accepted_thickness_change_sum_m": 0.0,
            "state_finite": bool(torch.isfinite(result).all()),
            "negative_cell_count": int(torch.count_nonzero(result < 0).item()),
        }
    source_volume = cell_area * (
        float(source1_diag["accepted_thickness_change_sum_m"])
        + float(source2_diag["accepted_thickness_change_sum_m"])
    )
    return result, {
        "numerical_pass": _valid_diffusion(diffusion_diag)
        and bool(source1_diag["state_finite"])
        and bool(source2_diag["state_finite"])
        and int(source1_diag["negative_cell_count"]) == 0
        and int(source2_diag["negative_cell_count"]) == 0
        and bool(source1_diag["remote_zero_unchanged"])
        and bool(source2_diag["remote_zero_unchanged"]),
        "diffusion": diffusion_diag,
        "negative_intermediate_trial_rejection": negative_rejection,
        "diffusion_mass_closure_m3": diffusion_closure,
        "source_volume_change_m3": source_volume,
        "prevented_overmelt_volume_m3": cell_area
        * (
            float(source1_diag["prevented_overmelt_sum_m"])
            + float(source2_diag["prevented_overmelt_sum_m"])
        ),
        "remote_zero_unchanged": bool(source1_diag["remote_zero_unchanged"])
        and bool(source2_diag["remote_zero_unchanged"]),
    }


def _write_state(
    path: Path,
    state: torch.Tensor,
    bed_np: np.ndarray,
    *,
    dx: float,
    elapsed_seconds: float,
    proposed_dt_seconds: float,
    counters: dict[str, float | int],
) -> str:
    thickness = state.detach().cpu().numpy()
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        thickness_m=thickness,
        bed_m=bed_np,
        dx_m=np.asarray(dx, dtype=np.float64),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
        proposed_dt_seconds=np.asarray(proposed_dt_seconds, dtype=np.float64),
        counters_json=np.asarray(json.dumps(counters, sort_keys=True)),
        schema=np.asarray(RECOVERY_SCHEMA),
    )
    os.replace(temporary, path)
    return _sha256(path)


def _load_inputs(
    geology: Path,
    static_summary_path: Path,
    zero_summary_path: Path,
    climate_validation_path: Path,
    climate_package: Path,
    observation_path: Path,
) -> dict[str, Any]:
    static = json.loads(static_summary_path.read_text(encoding="utf-8"))
    zero = json.loads(zero_summary_path.read_text(encoding="utf-8"))
    climate = json.loads(climate_validation_path.read_text(encoding="utf-8"))
    if static.get("status") != "static-pass":
        raise RuntimeError("source-static-not-pass")
    if zero.get("status") != "formal-zero-mb-5y-pass":
        raise RuntimeError("formal-zero-mb-5y-not-pass")
    if climate.get("status") != "climate-bridge-gpu-validation-pass-50y-not-run":
        raise RuntimeError("climate-bridge-not-pass")
    geology_hash = _sha256(geology)
    if geology_hash.lower() != str(static["geology_nc_sha256"]).lower():
        raise RuntimeError("geology-static-hash-mismatch")
    if geology_hash.lower() != str(climate["geology_sha256"]).lower():
        raise RuntimeError("geology-climate-hash-mismatch")
    package_hash = _sha256(climate_package)
    if package_hash.lower() != str(climate["package_sha256"]).lower():
        raise RuntimeError("climate-package-hash-mismatch")
    with Dataset(geology, "r") as dataset:
        bed_np = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        thickness_np = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        x = np.asarray(dataset.variables["x"][:], dtype=np.float64)
        y = np.asarray(dataset.variables["y"][:], dtype=np.float64)
        dx = float(getattr(dataset, "dx_m"))
    if not np.isfinite(bed_np).all() or not np.isfinite(thickness_np).all():
        raise RuntimeError("nonfinite-initial-grid")
    if np.any(thickness_np < 0.0):
        raise RuntimeError("negative-initial-thickness")
    with observation_path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    path_xy = np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in rows])
    controls = _control_lengths(path_xy)
    registered_length = float(static["registered_lia_length_m"])
    initial_path = _path_metrics(
        thickness_np, x, y, path_xy, controls, registered_length
    )
    expected_initial_length = float(static["observed_active_control_volume_length_m"])
    if abs(initial_path["observed_active_control_volume_length_m"] - expected_initial_length) > 1e-6:
        raise RuntimeError("initial-root-length-reproduction-fail")
    return {
        "static": static,
        "zero": zero,
        "climate": climate,
        "geology_hash": geology_hash,
        "package_hash": package_hash,
        "observation_path_hash": _sha256(observation_path),
        "bed_np": bed_np,
        "thickness_np": thickness_np,
        "x": x,
        "y": y,
        "dx": dx,
        "path_xy": path_xy,
        "controls": controls,
        "registered_length": registered_length,
        "initial_path": initial_path,
    }


def run(
    geology: Path,
    static_summary_path: Path,
    zero_summary_path: Path,
    climate_validation_path: Path,
    climate_package: Path,
    observation_path: Path,
    output: Path,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    loaded = _load_inputs(
        geology,
        static_summary_path,
        zero_summary_path,
        climate_validation_path,
        climate_package,
        observation_path,
    )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("registered-sm120-gpu-unavailable")
    bridge = TorchMonthlyTIBridge(climate_package, device="cuda:0")
    if bridge.month_count != TARGET_YEARS * 12:
        raise RuntimeError("climate-month-count-not-600")
    if not math.isclose(12.0 * bridge.sec_in_month, bridge.sec_in_year, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("climate-calendar-inconsistent")

    preflight = {
        "schema": "lia-191-torch-planform2d-system2005-steady50-monthly-ti-preflight-v1",
        "status": "formal-steady50-preflight-pass",
        "system_id": SYSTEM_ID,
        "target_years": TARGET_YEARS,
        "climate_month_count": bridge.month_count,
        "initial_root_path_metrics": loaded["initial_path"],
        "initial_free_domain_metrics": _free_domain_metrics(
            loaded["thickness_np"], loaded["dx"]
        ),
        "geology_sha256": loaded["geology_hash"],
        "climate_package_sha256": loaded["package_hash"],
        "observation_path_sha256": loaded["observation_path_hash"],
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "dynamics_run": False,
    }
    if preflight_only:
        return preflight
    if output.exists():
        raise FileExistsError(output)

    output.mkdir(parents=True, exist_ok=False)
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    config_snapshot = {
        "schema": "lia-191-torch-planform2d-system2005-steady50-monthly-ti-config-v1",
        "system_id": SYSTEM_ID,
        "target_years": TARGET_YEARS,
        "source_geology": str(geology),
        "source_static_summary": str(static_summary_path),
        "formal_zero_mb_5y_summary": str(zero_summary_path),
        "climate_validation_summary": str(climate_validation_path),
        "climate_package": str(climate_package),
        "observation_path": str(observation_path),
        "stage2_frozen": loaded["static"]["stage2_frozen"],
        "adaptive": {
            "initial_dt_seconds": 0.03,
            "maximum_dt_seconds": MAXIMUM_DT_SECONDS,
            "minimum_dt_seconds": MINIMUM_DT_SECONDS,
            "error_gate_relative_l2": ERROR_GATE,
            "growth_error_threshold": GROWTH_ERROR_THRESHOLD,
            "maximum_consecutive_rejections": MAXIMUM_CONSECUTIVE_REJECTIONS,
            "operator_split": "monthly-source-half/diffusion/source-half",
            "monthly_boundaries_exact": True,
        },
        "formal_gates": {
            "volume_change_relative": STEADY_VOLUME_CHANGE_GATE,
            "registered_length_relative_error": LENGTH_RELATIVE_ERROR_GATE,
            "free_domain_m": FREE_DOMAIN_GATE_M,
            "length_ice_threshold_m": LENGTH_ICE_THRESHOLD_M,
            "adaptive_error_relative_l2": ERROR_GATE,
            "diffusion_per_substep_mass_closure_m3": "max(1e-6, initial_volume*1e-12)",
            "cumulative_mass_budget_relative": CUMULATIVE_BUDGET_RELATIVE_GATE,
        },
        "root_gap_policy": "diagnostic-only; not a 2D steady hard gate",
        "runtime_dynamics_clipping": False,
        "source_overmelt_limited_to_available_ice": True,
        "remote_zero_mass_balance_accumulation": False,
    }
    (output / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    bed_np = loaded["bed_np"]
    thickness_np = loaded["thickness_np"]
    dx = float(loaded["dx"])
    cell_area = dx**2
    bed = torch.as_tensor(bed_np, dtype=torch.float64, device="cuda:0")
    initial_state = torch.as_tensor(thickness_np, dtype=torch.float64, device="cuda:0")
    state = initial_state.clone()
    glen_a = float(loaded["static"]["stage2_frozen"]["glen_a"])
    gamma = 2.0 * glen_a * (ICE_DENSITY_KG_M3 * GRAVITY_M_S2) ** GLEN_N / (GLEN_N + 2.0)
    initial_volume = float(torch.sum(state).item()) * cell_area
    per_substep_closure_tolerance = max(1e-6, initial_volume * 1e-12)
    cumulative_budget_tolerance = max(1e-6, initial_volume * CUMULATIVE_BUDGET_RELATIVE_GATE)
    initial_area_metrics = _area_metrics(state, cell_area)
    target_seconds = TARGET_YEARS * bridge.sec_in_year
    checkpoint_pairs = [("7d", 7.0 * 86_400.0), ("30d", 30.0 * 86_400.0), ("0p1y", 0.1 * bridge.sec_in_year)]
    checkpoint_pairs.extend((f"{year}y", year * bridge.sec_in_year) for year in range(1, TARGET_YEARS + 1))

    accepted_steps = 0
    rejected_steps = 0
    consecutive_rejections = 0
    solver_calls = 0
    negative_intermediate_rejections = 0
    elapsed = 0.0
    proposed_dt = 0.03
    minimum_accepted_dt = float("inf")
    maximum_accepted_dt = 0.0
    maximum_accepted_error = 0.0
    maximum_diffusion_substep_closure = 0.0
    maximum_accepted_budget_residual = 0.0
    maximum_step_change = 0.0
    cumulative_source_volume = 0.0
    cumulative_prevented_overmelt_volume = 0.0
    checkpoint_index = 0
    checkpoint_metrics: dict[str, Any] = {}
    wall_started = time.perf_counter()
    history_path = output / "step_history.csv"
    fieldnames = [
        "trial_index", "accepted", "elapsed_before_years", "month_index", "dt_seconds",
        "checkpoint_limited", "month_limited", "error_relative_l2",
        "formal_max_diffusion_substep_closure_m3", "accepted_candidate_budget_residual_m3",
        "candidate_source_volume_change_m3", "prevented_overmelt_volume_m3",
        "maximum_candidate_change_m", "full_iterations", "half1_iterations", "half2_iterations",
        "numerical_pass", "wall_seconds",
    ]

    with history_path.open("w", newline="", encoding="utf-8") as history_stream:
        writer = csv.DictWriter(history_stream, fieldnames=fieldnames)
        writer.writeheader()
        trial_index = 0
        while elapsed < target_seconds and accepted_steps < MAXIMUM_ACCEPTED_STEPS:
            label, next_checkpoint = checkpoint_pairs[checkpoint_index]
            checkpoint_remaining = next_checkpoint - elapsed
            month_index = min(int(math.floor(elapsed / bridge.sec_in_month + 1e-12)), bridge.month_count - 1)
            next_month_boundary = (month_index + 1) * bridge.sec_in_month
            month_remaining = next_month_boundary - elapsed
            dt = min(proposed_dt, checkpoint_remaining, month_remaining, target_seconds - elapsed)
            checkpoint_limited = dt < proposed_dt - 1e-12 and checkpoint_remaining <= month_remaining + 1e-9
            month_limited = dt < proposed_dt - 1e-12 and month_remaining < checkpoint_remaining - 1e-9
            if dt <= 0.0:
                raise RuntimeError("nonpositive-adaptive-dt")

            trial_started = time.perf_counter()
            full_state, full_diag = _split_step(
                state, bed, bridge, month_index=month_index, dx=dx, gamma=gamma,
                dt=dt, cell_area=cell_area,
            )
            half_state, half1_diag = _split_step(
                state, bed, bridge, month_index=month_index, dx=dx, gamma=gamma,
                dt=0.5 * dt, cell_area=cell_area,
            )
            if bool(half1_diag["numerical_pass"]):
                half2_state, half2_diag = _split_step(
                    half_state, bed, bridge, month_index=month_index, dx=dx, gamma=gamma,
                    dt=0.5 * dt, cell_area=cell_area,
                )
            else:
                half2_state = half_state
                half2_diag = {
                    "numerical_pass": False,
                    "diffusion": {"solver_iterations": 0},
                    "negative_intermediate_trial_rejection": False,
                    "diffusion_mass_closure_m3": 0.0,
                    "source_volume_change_m3": 0.0,
                    "prevented_overmelt_volume_m3": 0.0,
                    "remote_zero_unchanged": True,
                }
            torch.cuda.synchronize()
            solver_calls += 3 if bool(half1_diag["numerical_pass"]) else 2
            negative_intermediate_rejections += sum(
                int(diag["negative_intermediate_trial_rejection"])
                for diag in (full_diag, half1_diag, half2_diag)
            )

            candidate_delta = half2_state - state
            denominator = max(float(torch.linalg.vector_norm(candidate_delta).item()), 1e-30)
            error_relative_l2 = float(torch.linalg.vector_norm(half2_state - full_state).item()) / denominator
            formal_closure = max(
                abs(float(half1_diag["diffusion_mass_closure_m3"])),
                abs(float(half2_diag["diffusion_mass_closure_m3"])),
            )
            candidate_source = float(half1_diag["source_volume_change_m3"]) + float(half2_diag["source_volume_change_m3"])
            candidate_volume_delta = float(torch.sum(candidate_delta).item()) * cell_area
            candidate_budget_residual = candidate_volume_delta - candidate_source
            prevented_overmelt = float(half1_diag["prevented_overmelt_volume_m3"]) + float(half2_diag["prevented_overmelt_volume_m3"])
            numerical_pass = (
                bool(full_diag["numerical_pass"])
                and bool(half1_diag["numerical_pass"])
                and bool(half2_diag["numerical_pass"])
                and bool(full_diag["remote_zero_unchanged"])
                and bool(half1_diag["remote_zero_unchanged"])
                and bool(half2_diag["remote_zero_unchanged"])
                and formal_closure <= per_substep_closure_tolerance
                and _boundary_maximum(half2_state) <= LENGTH_ICE_THRESHOLD_M
            )
            accepted = numerical_pass and error_relative_l2 <= ERROR_GATE
            trial_index += 1
            writer.writerow({
                "trial_index": trial_index,
                "accepted": accepted,
                "elapsed_before_years": elapsed / bridge.sec_in_year,
                "month_index": month_index,
                "dt_seconds": dt,
                "checkpoint_limited": checkpoint_limited,
                "month_limited": month_limited,
                "error_relative_l2": error_relative_l2,
                "formal_max_diffusion_substep_closure_m3": formal_closure,
                "accepted_candidate_budget_residual_m3": candidate_budget_residual,
                "candidate_source_volume_change_m3": candidate_source,
                "prevented_overmelt_volume_m3": prevented_overmelt,
                "maximum_candidate_change_m": float(torch.max(torch.abs(candidate_delta)).item()),
                "full_iterations": int(full_diag["diffusion"].get("solver_iterations", 0)),
                "half1_iterations": int(half1_diag["diffusion"].get("solver_iterations", 0)),
                "half2_iterations": int(half2_diag["diffusion"].get("solver_iterations", 0)),
                "numerical_pass": numerical_pass,
                "wall_seconds": time.perf_counter() - trial_started,
            })

            if accepted:
                state = half2_state
                elapsed += dt
                if abs(elapsed - next_checkpoint) <= 1e-7:
                    elapsed = next_checkpoint
                if abs(elapsed - next_month_boundary) <= 1e-7:
                    elapsed = next_month_boundary
                accepted_steps += 1
                consecutive_rejections = 0
                cumulative_source_volume += candidate_source
                cumulative_prevented_overmelt_volume += prevented_overmelt
                minimum_accepted_dt = min(minimum_accepted_dt, dt)
                maximum_accepted_dt = max(maximum_accepted_dt, dt)
                maximum_accepted_error = max(maximum_accepted_error, error_relative_l2)
                maximum_diffusion_substep_closure = max(maximum_diffusion_substep_closure, formal_closure)
                maximum_accepted_budget_residual = max(maximum_accepted_budget_residual, abs(candidate_budget_residual))
                maximum_step_change = max(maximum_step_change, float(torch.max(torch.abs(candidate_delta)).item()))
                if error_relative_l2 <= GROWTH_ERROR_THRESHOLD:
                    growth_base = proposed_dt if (checkpoint_limited or month_limited) else dt
                    proposed_dt = min(2.0 * growth_base, MAXIMUM_DT_SECONDS)
                elif not checkpoint_limited and not month_limited:
                    proposed_dt = dt

                if elapsed == next_checkpoint:
                    state_np = state.detach().cpu().numpy()
                    current_volume = float(torch.sum(state).item()) * cell_area
                    physical_volume_change = (current_volume - initial_volume) / initial_volume
                    cumulative_budget = current_volume - initial_volume - cumulative_source_volume
                    path_metrics = _path_metrics(
                        state_np, loaded["x"], loaded["y"], loaded["path_xy"],
                        loaded["controls"], loaded["registered_length"],
                    )
                    domain_metrics = _free_domain_metrics(state_np, dx)
                    checkpoint_metrics[label] = {
                        "elapsed_years": elapsed / bridge.sec_in_year,
                        "accepted_steps": accepted_steps,
                        "rejected_steps": rejected_steps,
                        "next_dt_seconds": proposed_dt,
                        "volume_relative_change": physical_volume_change,
                        "cumulative_source_volume_change_m3": cumulative_source_volume,
                        "cumulative_mass_budget_residual_m3": cumulative_budget,
                        "maximum_thickness_m": float(torch.max(state).item()),
                        "area_metrics": _with_relative_area_change(_area_metrics(state, cell_area), initial_area_metrics),
                        "root_path_metrics": path_metrics,
                        "free_domain_metrics": domain_metrics,
                    }
                    counters = {
                        "accepted_steps": accepted_steps,
                        "rejected_steps": rejected_steps,
                        "solver_calls": solver_calls,
                        "cumulative_source_volume_m3": cumulative_source_volume,
                        "cumulative_prevented_overmelt_volume_m3": cumulative_prevented_overmelt_volume,
                    }
                    recovery_sha = _write_state(
                        output / "recovery_latest.npz", state, bed_np, dx=dx,
                        elapsed_seconds=elapsed, proposed_dt_seconds=proposed_dt, counters=counters,
                    )
                    checkpoint_metrics[label]["recovery_sha256"] = recovery_sha
                    checkpoint_index += 1
                    progress = {
                        "status": "running" if elapsed < target_seconds else "finishing",
                        "elapsed_years": elapsed / bridge.sec_in_year,
                        "accepted_steps": accepted_steps,
                        "rejected_steps": rejected_steps,
                        "simulated_years_per_wall_hour": (elapsed / bridge.sec_in_year)
                        / max((time.perf_counter() - wall_started) / 3600.0, 1e-12),
                        "volume_relative_change": physical_volume_change,
                        "registered_length_relative_error": path_metrics["registered_length_relative_error"],
                        "free_domain_m": domain_metrics["minimum_active_cell_center_to_grid_edge_m"],
                    }
                    (output / "progress.json").write_text(
                        json.dumps(progress, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                    )
                    history_stream.flush()
                    print(json.dumps(progress, sort_keys=True), flush=True)
                    if checkpoint_index == len(checkpoint_pairs):
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
    final_np = state.detach().cpu().numpy()
    final_volume = float(torch.sum(state).item()) * cell_area
    volume_relative_change = (final_volume - initial_volume) / initial_volume
    cumulative_budget_residual = final_volume - initial_volume - cumulative_source_volume
    final_path = _path_metrics(
        final_np, loaded["x"], loaded["y"], loaded["path_xy"], loaded["controls"], loaded["registered_length"]
    )
    final_domain = _free_domain_metrics(final_np, dx)
    target_complete = abs(elapsed - target_seconds) <= 1e-7
    final_state_path = output / "final_state_50y.npz"
    final_state_sha = None
    if target_complete:
        final_state_sha = _write_state(
            final_state_path, state, bed_np, dx=dx, elapsed_seconds=elapsed,
            proposed_dt_seconds=proposed_dt,
            counters={
                "accepted_steps": accepted_steps,
                "rejected_steps": rejected_steps,
                "solver_calls": solver_calls,
                "cumulative_source_volume_m3": cumulative_source_volume,
                "cumulative_prevented_overmelt_volume_m3": cumulative_prevented_overmelt_volume,
            },
        )
    source_hashes_after = {
        "geology": _sha256(geology),
        "climate_package": _sha256(climate_package),
        "observation_path": _sha256(observation_path),
    }
    final_checks = {
        "target_50y_completed": target_complete,
        "all_checkpoints_recorded": checkpoint_index == len(checkpoint_pairs),
        "steady_volume_change_pass": abs(volume_relative_change) <= STEADY_VOLUME_CHANGE_GATE,
        "registered_length_error_pass": final_path["registered_length_relative_error"] <= LENGTH_RELATIVE_ERROR_GATE,
        "free_domain_pass": final_domain["minimum_active_cell_center_to_grid_edge_m"] >= FREE_DOMAIN_GATE_M,
        "cumulative_mass_budget_pass": abs(cumulative_budget_residual) <= cumulative_budget_tolerance,
        "formal_max_diffusion_substep_closure_pass": maximum_diffusion_substep_closure <= per_substep_closure_tolerance,
        "accepted_error_gate_respected": maximum_accepted_error <= ERROR_GATE,
        "final_state_finite": bool(torch.isfinite(state).all()),
        "final_negative_cell_count_zero": int(torch.count_nonzero(state < 0).item()) == 0,
        "final_boundary_thickness_pass": _boundary_maximum(state) <= LENGTH_ICE_THRESHOLD_M,
        "source_files_immutable": source_hashes_after == {
            "geology": loaded["geology_hash"],
            "climate_package": loaded["package_hash"],
            "observation_path": loaded["observation_path_hash"],
        },
        "final_state_saved": final_state_sha is not None and final_state_path.exists(),
    }
    passed = all(final_checks.values())
    summary = {
        "schema": "lia-191-torch-planform2d-system2005-steady50-monthly-ti-formal-v1",
        "status": "formal-steady50-pass" if passed else "formal-steady50-fail-stop",
        "system_id": SYSTEM_ID,
        "target_years": TARGET_YEARS,
        "completed_years": elapsed / bridge.sec_in_year,
        "climate_months_completed": elapsed / bridge.sec_in_month,
        "initial_dt_seconds": 0.03,
        "minimum_accepted_dt_seconds": minimum_accepted_dt if accepted_steps else None,
        "maximum_accepted_dt_seconds": maximum_accepted_dt if accepted_steps else None,
        "candidate_dt_ceiling_seconds": MAXIMUM_DT_SECONDS,
        "error_gate_relative_l2": ERROR_GATE,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "solver_calls": solver_calls,
        "negative_intermediate_trial_rejections": negative_intermediate_rejections,
        "wall_seconds": wall_seconds,
        "simulated_years_per_wall_hour": (elapsed / bridge.sec_in_year) / max(wall_seconds / 3600.0, 1e-12),
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "final_volume_relative_change": volume_relative_change,
        "steady_volume_change_gate": STEADY_VOLUME_CHANGE_GATE,
        "cumulative_climate_source_volume_change_m3": cumulative_source_volume,
        "cumulative_prevented_overmelt_volume_m3": cumulative_prevented_overmelt_volume,
        "cumulative_mass_budget_residual_m3": cumulative_budget_residual,
        "cumulative_mass_budget_tolerance_m3": cumulative_budget_tolerance,
        "maximum_diffusion_substep_closure_m3": maximum_diffusion_substep_closure,
        "diffusion_substep_closure_tolerance_m3": per_substep_closure_tolerance,
        "maximum_accepted_budget_residual_m3": maximum_accepted_budget_residual,
        "maximum_accepted_error_relative_l2": maximum_accepted_error,
        "maximum_single_accepted_change_m": maximum_step_change,
        "maximum_final_change_from_initial_m": float(torch.max(torch.abs(state - initial_state)).item()),
        "final_root_path_metrics": final_path,
        "final_free_domain_metrics": final_domain,
        "final_boundary_maximum_thickness_m": _boundary_maximum(state),
        "thickness_thresholds_m": list(THICKNESS_THRESHOLDS_M),
        "checkpoint_metrics": checkpoint_metrics,
        "final_checks": final_checks,
        "source_hashes_before": {
            "geology": loaded["geology_hash"],
            "climate_package": loaded["package_hash"],
            "observation_path": loaded["observation_path_hash"],
        },
        "source_hashes_after": source_hashes_after,
        "step_history_csv": str(history_path),
        "recovery_state_file": str(output / "recovery_latest.npz"),
        "final_state_file": str(final_state_path) if final_state_sha else None,
        "final_state_sha256": final_state_sha,
        "runtime_dynamics_clipping": False,
        "runtime_limiter": False,
        "source_overmelt_limited_to_available_ice": True,
        "remote_zero_mass_balance_accumulation": False,
        "mass_balance": "exact-exported-oggm-monthly-ti-repeat-600-months",
        "root_internal_gaps_are_formal_gate": False,
        "steady_run": True,
        "historical_run": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "progress.json").write_text(
        json.dumps({"status": summary["status"], "elapsed_years": summary["completed_years"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--zero-summary", type=Path, required=True)
    parser.add_argument("--climate-validation", type=Path, required=True)
    parser.add_argument("--climate-package", type=Path, required=True)
    parser.add_argument("--observation-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    result = run(
        arguments.geology,
        arguments.static_summary,
        arguments.zero_summary,
        arguments.climate_validation,
        arguments.climate_package,
        arguments.observation_path,
        arguments.output,
        preflight_only=arguments.preflight_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
