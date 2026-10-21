"""Resume the passed system-14 semi-implicit state from 0.1 to 5 years."""

from __future__ import annotations

import gzip
import json
import pickle

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .planform_2d_frozen_diffusivity_semiimplicit_core_v2 import frozen_diffusivity_step
from .planform_2d_frozen_diffusivity_system14_zero_0p1y_v1 import (
    GLEN_A, INITIAL_STEP_SECONDS, MINIMUM_STEP_SECONDS, NETWORK, PROJECT,
    REGISTERED_LENGTH_M, STATE, control_lengths, write_json,
)
from .planform_2d_frozen_diffusivity_system14_zero_0p1y_v3 import (
    NEGATIVE_VOLUME_BUDGET_M3, NUMERICAL_ZERO_M,
)


CHECKPOINT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_planform2d_frozen_diffusivity_system14_zero_0p1y_v3/state_0p1y.npz"
CHECKPOINT_SUMMARY = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_planform2d_frozen_diffusivity_system14_zero_0p1y_v3/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_planform2d_frozen_diffusivity_system14_zero_5y_v1"
TARGET_YEAR = 5.0


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    checkpoint_summary = json.loads(CHECKPOINT_SUMMARY.read_text(encoding="utf-8"))
    if checkpoint_summary.get("status") != "zero-mb-0p1y-pass":
        raise RuntimeError("zero-0p1y-gate-not-passed")
    source_state = np.load(STATE)
    checkpoint = np.load(CHECKPOINT)
    bed = np.asarray(checkpoint["bed_elevation_m"], float)
    thickness = np.asarray(checkpoint["thickness_m"], float)
    x, y = np.asarray(checkpoint["x_m"], float), np.asarray(checkpoint["y_m"], float)
    dx = float(checkpoint["dx_m"])
    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.massbalance import ScalarMassBalance
    from oggm.core.sia2d import Upstream2D
    cfg.initialize(logging_level="WARNING")
    model = Upstream2D(
        bed, init_ice_thick=thickness, dx=dx,
        mb_model=ScalarMassBalance(0.0), glen_a=GLEN_A, ice_thick_filter=None,
    )
    elapsed = float(checkpoint["elapsed_seconds"])
    model.t = elapsed
    target_seconds = TARGET_YEAR * SEC_IN_YEAR
    original_h = np.asarray(source_state["thickness_m"], float)
    volume0 = float(np.sum(original_h) * dx * dx)
    closure_gate = max(1e-6, volume0 * 1e-12)
    used_steps = []
    rejected = 0
    max_residual = 0.0
    max_negative_volume = float(-np.sum(thickness[thickness < 0]) * model.dxdy)
    max_zeroed_cells = 0
    failure = None
    last_rejection = None
    while elapsed < target_seconds:
        remaining = target_seconds - elapsed
        proposed = min(INITIAL_STEP_SECONDS, remaining)
        floor = min(MINIMUM_STEP_SECONDS, remaining)
        accepted = False
        while proposed >= floor:
            try:
                candidate, diagnostics = frozen_diffusivity_step(
                    model, proposed, numerical_zero_m=NUMERICAL_ZERO_M,
                    tol=1e-12, maxiter=4000,
                )
                residual = abs(float(np.sum(candidate - model.ice_thick) * model.dxdy))
                negative_volume = float(-np.sum(candidate[candidate < 0]) * model.dxdy)
                valid = (
                    diagnostics["solver_info"] == 0
                    and diagnostics["matrix_finite"]
                    and diagnostics["matrix_positive_diagonal"]
                    and diagnostics["matrix_symmetric_max_abs"] == 0.0
                    and np.isfinite(candidate).all()
                    and float(np.min(candidate)) >= -NUMERICAL_ZERO_M
                    and negative_volume <= NEGATIVE_VOLUME_BUDGET_M3
                    and residual <= closure_gate
                )
                last_rejection = None if valid else {
                    "reason": "candidate-hard-gate", "dt_seconds": proposed,
                    "minimum_thickness_m": float(np.min(candidate)),
                    "negative_volume_m3": negative_volume,
                    "volume_residual_m3": residual,
                    "solver_info": diagnostics["solver_info"],
                }
            except Exception as exc:
                valid = False
                last_rejection = {"reason": f"{type(exc).__name__}:{exc}", "dt_seconds": proposed}
            if valid:
                model.ice_thick = candidate
                model.t += proposed
                elapsed += proposed
                used_steps.append(proposed)
                max_residual = max(max_residual, residual)
                max_negative_volume = max(max_negative_volume, negative_volume)
                max_zeroed_cells = max(max_zeroed_cells, diagnostics["diffusivity_evaluation_zeroed_negative_cell_count"])
                accepted = True
                break
            rejected += 1
            proposed /= 2.0
        if not accepted:
            failure = "no-conservative-numerical-zero-safe-step-at-or-above-30s"
            break
        if len(used_steps) % 5 == 0 or elapsed >= target_seconds:
            write_json(OUTPUT / "progress.json", {
                "elapsed_year": elapsed / SEC_IN_YEAR,
                "accepted_steps_after_0p1y": len(used_steps),
                "rejected_attempts": rejected,
                "last_timestep_seconds": proposed,
                "minimum_thickness_m": float(np.min(model.ice_thick)),
                "negative_volume_m3": negative_volume,
                "current_volume_drift_fraction": abs(float(model.volume_m3) - volume0) / volume0,
            })

    positive = model.ice_thick > 1e-6
    rows, cols = np.nonzero(positive)
    edge_distance = float(np.minimum.reduce([
        rows, positive.shape[0] - 1 - rows, cols, positive.shape[1] - 1 - cols
    ]).min() * dx)
    with gzip.open(NETWORK, "rb") as stream:
        network = pickle.load(stream)
    root_xy = np.asarray(network["branches"][int(network["main_branch_index"])]["coords_projected_m"], float)
    sampled = RegularGridInterpolator(
        (y[::-1], x), model.ice_thick[::-1], bounds_error=False, fill_value=np.nan
    )(np.column_stack([root_xy[:, 1], root_xy[:, 0]]))
    active = np.isfinite(sampled) & (sampled > 10.0)
    indices = np.flatnonzero(active)
    if len(indices):
        first, last = int(indices[0]), int(indices[-1])
        gaps = np.flatnonzero(~active[first:last + 1]) + first
        observed_length = float(control_lengths(root_xy)[active].sum())
    else:
        gaps, observed_length = np.array([], int), 0.0
    drift = abs(float(model.volume_m3) - volume0) / volume0
    length_error = abs(observed_length - REGISTERED_LENGTH_M) / REGISTERED_LENGTH_M
    current_negative_volume = float(-np.sum(model.ice_thick[model.ice_thick < 0]) * model.dxdy)
    checks = {
        "target_completed": failure is None and elapsed >= target_seconds,
        "numerical_zero_bound": float(np.min(model.ice_thick)) >= -NUMERICAL_ZERO_M,
        "negative_volume_budget": current_negative_volume <= NEGATIVE_VOLUME_BUDGET_M3,
        "volume_drift": drift <= 0.001,
        "per_step_mass_closure": max_residual <= closure_gate,
        "minimum_timestep": bool(used_steps and min(used_steps) >= MINIMUM_STEP_SECONDS),
        "ice_free_safety_boundary": edge_distance >= 5000.0,
        "observation_path_continuous": len(gaps) == 0,
        "registered_length": length_error <= 0.005,
    }
    passed = bool(all(checks.values()))
    if passed:
        np.savez_compressed(
            OUTPUT / "state_5y.npz", x_m=x, y_m=y,
            bed_elevation_m=bed.astype(np.float32),
            thickness_m=model.ice_thick.astype(np.float64), dx_m=np.float64(dx),
            elapsed_seconds=np.float64(elapsed),
        )
    summary = {
        "schema": "lia-191-planform2d-frozen-diffusivity-system14-zero-5y-v1",
        "status": "zero-mb-5y-pass" if passed else "zero-mb-5y-fail-mechanism-closed",
        "system_id": 14, "elapsed_year": elapsed / SEC_IN_YEAR,
        "accepted_steps_after_0p1y": len(used_steps), "rejected_attempts": rejected,
        "minimum_used_step_seconds": min(used_steps) if used_steps else None,
        "maximum_used_step_seconds": max(used_steps) if used_steps else None,
        "failure": failure, "last_rejection": last_rejection,
        "volume_drift_fraction": drift, "mass_closure_gate_m3": closure_gate,
        "maximum_absolute_step_volume_residual_m3": max_residual,
        "numerical_zero_tolerance_m": NUMERICAL_ZERO_M,
        "maximum_negative_volume_m3": max_negative_volume,
        "current_negative_volume_m3": current_negative_volume,
        "maximum_zeroed_cells_for_diffusivity_evaluation": max_zeroed_cells,
        "minimum_thickness_m": float(np.min(model.ice_thick)),
        "minimum_ice_to_grid_edge_m": edge_distance,
        "observed_length_m": observed_length,
        "registered_length_relative_error": length_error,
        "root_internal_gap_indices": gaps.astype(int).tolist(),
        "checks": checks, "state_clipping_used": False,
        "mass_correction_used": False,
        "next_authorized_stage": "steady-50y-design" if passed else None,
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
