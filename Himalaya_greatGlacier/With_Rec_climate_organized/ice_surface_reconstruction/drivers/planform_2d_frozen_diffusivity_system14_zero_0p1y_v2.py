"""Strict-nonnegative retry of system 14's 0.1 year semi-implicit gate."""

from __future__ import annotations

import gzip
import json
import pickle

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import planform_2d_frozen_diffusivity_semiimplicit_core_v1 as core
from .planform_2d_frozen_diffusivity_system14_zero_0p1y_v1 import (
    GLEN_A,
    INITIAL_STEP_SECONDS,
    MINIMUM_STEP_SECONDS,
    NETWORK,
    PROJECT,
    REGISTERED_LENGTH_M,
    STATE,
    STEP,
    TARGET_YEAR,
    control_lengths,
    write_json,
)


OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_planform2d_frozen_diffusivity_system14_zero_0p1y_v2"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    if json.loads(STEP.read_text(encoding="utf-8")).get("status") != "real-step-pass":
        raise RuntimeError("real-30s-step-gate-not-passed")
    scipy_cg = core.cg

    def cg_compat(operator, rhs, rtol=1e-5, atol=0.0, maxiter=None, M=None):
        return scipy_cg(operator, rhs, tol=rtol, atol=atol, maxiter=maxiter, M=M)

    core.cg = cg_compat
    state = np.load(STATE)
    bed = np.asarray(state["bed_elevation_m"], dtype=float)
    thickness0 = np.asarray(state["thickness_m"], dtype=float)
    x, y = np.asarray(state["x_m"], float), np.asarray(state["y_m"], float)
    dx = float(state["dx_m"])

    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.massbalance import ScalarMassBalance
    from oggm.core.sia2d import Upstream2D

    cfg.initialize(logging_level="WARNING")
    model = Upstream2D(
        bed, init_ice_thick=thickness0, dx=dx,
        mb_model=ScalarMassBalance(0.0), glen_a=GLEN_A, ice_thick_filter=None,
    )
    target_seconds = TARGET_YEAR * SEC_IN_YEAR
    volume0 = float(model.volume_m3)
    closure_gate = max(1e-6, volume0 * 1e-12)
    elapsed = 0.0
    accepted_steps = 0
    rejected_attempts = 0
    used_steps = []
    maximum_step_residual = 0.0
    failure = None
    last_rejection = None
    while elapsed < target_seconds:
        remaining = target_seconds - elapsed
        proposed = min(INITIAL_STEP_SECONDS, remaining)
        accepted = False
        floor = min(MINIMUM_STEP_SECONDS, remaining)
        while proposed >= floor:
            try:
                candidate, diagnostics = core.frozen_diffusivity_step(
                    model, proposed, rtol=1e-12, maxiter=4000
                )
                residual = abs(float(np.sum(candidate - model.ice_thick) * model.dxdy))
                candidate_min = float(np.min(candidate))
                valid = (
                    diagnostics["solver_info"] == 0
                    and diagnostics["matrix_finite"]
                    and diagnostics["matrix_positive_diagonal"]
                    and diagnostics["matrix_symmetric_max_abs"] == 0.0
                    and np.isfinite(candidate).all()
                    and candidate_min >= 0.0
                    and residual <= closure_gate
                )
                last_rejection = None if valid else {
                    "reason": "candidate-hard-gate",
                    "dt_seconds": proposed,
                    "minimum_thickness_m": candidate_min,
                    "volume_residual_m3": residual,
                    "solver_info": diagnostics["solver_info"],
                }
            except Exception as exc:
                valid = False
                last_rejection = {
                    "reason": f"{type(exc).__name__}:{exc}",
                    "dt_seconds": proposed,
                }
            if valid:
                model.ice_thick = candidate
                model.t += proposed
                elapsed += proposed
                accepted_steps += 1
                used_steps.append(proposed)
                maximum_step_residual = max(maximum_step_residual, residual)
                accepted = True
                break
            rejected_attempts += 1
            proposed /= 2.0
        if not accepted:
            failure = "no-strictly-nonnegative-conservative-step-at-or-above-30s"
            break
        write_json(OUTPUT / "progress.json", {
            "elapsed_year": elapsed / SEC_IN_YEAR,
            "accepted_steps": accepted_steps,
            "rejected_attempts": rejected_attempts,
            "last_timestep_seconds": proposed,
            "minimum_thickness_m": float(np.min(model.ice_thick)),
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
        first, last, gaps, observed_length = -1, -1, np.array([], int), 0.0
    final_volume = float(model.volume_m3)
    drift = abs(final_volume - volume0) / volume0
    length_error = abs(observed_length - REGISTERED_LENGTH_M) / REGISTERED_LENGTH_M
    checks = {
        "target_completed": failure is None and elapsed >= target_seconds,
        "strictly_nonnegative_without_clipping": bool(np.isfinite(model.ice_thick).all() and np.min(model.ice_thick) >= 0.0),
        "volume_drift": drift <= 0.001,
        "per_step_mass_closure": maximum_step_residual <= closure_gate,
        "minimum_timestep": bool(used_steps and min(used_steps) >= MINIMUM_STEP_SECONDS),
        "ice_free_safety_boundary": edge_distance >= 5000.0,
        "observation_path_continuous": len(gaps) == 0,
        "registered_length": length_error <= 0.005,
    }
    passed = bool(all(checks.values()))
    if passed:
        np.savez_compressed(
            OUTPUT / "state_0p1y.npz", x_m=x, y_m=y,
            bed_elevation_m=bed.astype(np.float32),
            thickness_m=model.ice_thick.astype(np.float32), dx_m=np.float64(dx),
            elapsed_seconds=np.float64(elapsed),
        )
    summary = {
        "schema": "lia-191-planform2d-frozen-diffusivity-system14-zero-0p1y-v2",
        "status": "zero-mb-0p1y-pass" if passed else "zero-mb-0p1y-fail-mechanism-closed",
        "supersedes": "v1-accepted-sub-tolerance-negative-thickness-then-negative-conductance",
        "system_id": 14,
        "elapsed_year": elapsed / SEC_IN_YEAR,
        "accepted_steps": accepted_steps,
        "rejected_attempts": rejected_attempts,
        "minimum_used_step_seconds": min(used_steps) if used_steps else None,
        "maximum_used_step_seconds": max(used_steps) if used_steps else None,
        "failure": failure,
        "last_rejection": last_rejection,
        "volume_drift_fraction": drift,
        "mass_closure_gate_m3": closure_gate,
        "maximum_absolute_step_volume_residual_m3": maximum_step_residual,
        "minimum_thickness_m": float(np.min(model.ice_thick)),
        "minimum_ice_to_grid_edge_m": edge_distance,
        "observed_length_m": observed_length,
        "registered_length_relative_error": length_error,
        "root_internal_gap_indices": gaps.astype(int).tolist(),
        "checks": checks,
        "clipping_used": False,
        "mass_correction_used": False,
        "next_authorized_stage": "zero-mb-5y" if passed else None,
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
