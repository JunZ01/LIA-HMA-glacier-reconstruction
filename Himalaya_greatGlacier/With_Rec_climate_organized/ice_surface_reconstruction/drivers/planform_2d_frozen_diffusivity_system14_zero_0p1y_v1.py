"""Rollback-safe 0.1 year zero-MB gate for system 14's semi-implicit 2-D state."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import planform_2d_frozen_diffusivity_semiimplicit_core_v1 as core


PROJECT = Path(__file__).resolve().parents[4]
STATE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_planform2d_source_native_padded_static_v1/system_14/source_native_padded_initial_state.npz"
STEP = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_planform2d_frozen_diffusivity_system14_step_v1/summary.json"
NETWORK = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3/14/component_0/lia_centerline_network.pkl.gz"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_planform2d_frozen_diffusivity_system14_zero_0p1y_v1"
GLEN_A = 6.642407336686561e-23
REGISTERED_LENGTH_M = 17608.26562508133
TARGET_YEAR = 0.1
INITIAL_STEP_SECONDS = 86400.0
MINIMUM_STEP_SECONDS = 30.0


def write_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.write_text(
        json.dumps(safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def control_lengths(coordinates: np.ndarray) -> np.ndarray:
    segments = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    result = np.empty(len(coordinates), dtype=float)
    result[0] = segments[0] / 2
    result[-1] = segments[-1] / 2
    result[1:-1] = (segments[:-1] + segments[1:]) / 2
    return result


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
    x = np.asarray(state["x_m"], dtype=float)
    y = np.asarray(state["y_m"], dtype=float)
    dx = float(state["dx_m"])

    from oggm import cfg
    from oggm.cfg import SEC_IN_YEAR
    from oggm.core.massbalance import ScalarMassBalance
    from oggm.core.sia2d import Upstream2D

    cfg.initialize(logging_level="WARNING")
    model = Upstream2D(
        bed,
        init_ice_thick=thickness0,
        dx=dx,
        mb_model=ScalarMassBalance(0.0),
        glen_a=GLEN_A,
        ice_thick_filter=None,
    )
    target_seconds = TARGET_YEAR * SEC_IN_YEAR
    volume0 = float(model.volume_m3)
    closure_gate = max(1e-6, volume0 * 1e-12)
    elapsed = 0.0
    accepted_steps = 0
    rejected_attempts = 0
    minimum_used = np.inf
    maximum_used = 0.0
    maximum_step_residual = 0.0
    minimum_thickness = float(np.min(thickness0))
    failure = None
    while elapsed < target_seconds:
        remaining = target_seconds - elapsed
        proposed = min(INITIAL_STEP_SECONDS, remaining)
        accepted = False
        while proposed >= min(MINIMUM_STEP_SECONDS, remaining):
            candidate, diagnostics = core.frozen_diffusivity_step(
                model, proposed, rtol=1e-12, maxiter=4000
            )
            residual = abs(float(np.sum(candidate - model.ice_thick) * model.dxdy))
            valid = (
                diagnostics["solver_info"] == 0
                and diagnostics["matrix_finite"]
                and diagnostics["matrix_positive_diagonal"]
                and diagnostics["matrix_symmetric_max_abs"] == 0.0
                and np.isfinite(candidate).all()
                and np.min(candidate) >= -1e-7
                and residual <= closure_gate
            )
            if valid:
                model.ice_thick = candidate
                model.t += proposed
                elapsed += proposed
                accepted_steps += 1
                minimum_used = min(minimum_used, proposed)
                maximum_used = max(maximum_used, proposed)
                maximum_step_residual = max(maximum_step_residual, residual)
                minimum_thickness = min(minimum_thickness, float(np.min(candidate)))
                accepted = True
                break
            rejected_attempts += 1
            proposed /= 2.0
        if not accepted:
            failure = "no-valid-semiimplicit-step-at-or-above-30s"
            break
        write_json(
            OUTPUT / "progress.json",
            {
                "elapsed_year": elapsed / SEC_IN_YEAR,
                "accepted_steps": accepted_steps,
                "rejected_attempts": rejected_attempts,
                "last_timestep_seconds": proposed,
                "current_volume_drift_fraction": abs(float(model.volume_m3) - volume0) / volume0,
            },
        )

    positive = model.ice_thick > 1e-6
    rows, cols = np.nonzero(positive)
    edge_distance = float(
        np.minimum.reduce(
            [rows, positive.shape[0] - 1 - rows, cols, positive.shape[1] - 1 - cols]
        ).min()
        * dx
    )
    with gzip.open(NETWORK, "rb") as stream:
        network = pickle.load(stream)
    root_xy = np.asarray(
        network["branches"][int(network["main_branch_index"])]["coords_projected_m"],
        dtype=float,
    )
    interpolator = RegularGridInterpolator(
        (y[::-1], x), model.ice_thick[::-1], bounds_error=False, fill_value=np.nan
    )
    sampled = interpolator(np.column_stack([root_xy[:, 1], root_xy[:, 0]]))
    active = np.isfinite(sampled) & (sampled > 10.0)
    indices = np.flatnonzero(active)
    if len(indices):
        first, last = int(indices[0]), int(indices[-1])
        gaps = np.flatnonzero(~active[first : last + 1]) + first
        observed_length = float(control_lengths(root_xy)[active].sum())
    else:
        first, last = -1, -1
        gaps = np.array([], dtype=int)
        observed_length = 0.0
    final_volume = float(model.volume_m3)
    drift = abs(final_volume - volume0) / volume0
    length_error = abs(observed_length - REGISTERED_LENGTH_M) / REGISTERED_LENGTH_M
    checks = {
        "target_completed": failure is None and elapsed >= target_seconds,
        "finite_nonnegative_without_clipping": bool(
            np.isfinite(model.ice_thick).all() and np.min(model.ice_thick) >= -1e-7
        ),
        "volume_drift": drift <= 0.001,
        "per_step_mass_closure": maximum_step_residual <= closure_gate,
        "minimum_timestep": minimum_used >= MINIMUM_STEP_SECONDS,
        "ice_free_safety_boundary": edge_distance >= 5000.0,
        "observation_path_continuous": len(gaps) == 0,
        "registered_length": length_error <= 0.005,
    }
    passed = bool(all(checks.values()))
    if passed:
        np.savez_compressed(
            OUTPUT / "state_0p1y.npz",
            x_m=x,
            y_m=y,
            bed_elevation_m=bed.astype(np.float32),
            thickness_m=model.ice_thick.astype(np.float32),
            dx_m=np.float64(dx),
            elapsed_seconds=np.float64(elapsed),
        )
    summary = {
        "schema": "lia-191-planform2d-frozen-diffusivity-system14-zero-0p1y-v1",
        "status": "zero-mb-0p1y-pass" if passed else "zero-mb-0p1y-fail-mechanism-closed",
        "system_id": 14,
        "target_year": TARGET_YEAR,
        "elapsed_year": elapsed / SEC_IN_YEAR,
        "initial_step_seconds": INITIAL_STEP_SECONDS,
        "minimum_allowed_step_seconds": MINIMUM_STEP_SECONDS,
        "accepted_steps": accepted_steps,
        "rejected_attempts": rejected_attempts,
        "minimum_used_step_seconds": None if not np.isfinite(minimum_used) else minimum_used,
        "maximum_used_step_seconds": maximum_used,
        "failure": failure,
        "initial_volume_m3": volume0,
        "final_volume_m3": final_volume,
        "volume_drift_fraction": drift,
        "mass_closure_gate_m3": closure_gate,
        "maximum_absolute_step_volume_residual_m3": maximum_step_residual,
        "minimum_thickness_m": minimum_thickness,
        "positive_cell_count": int(np.sum(positive)),
        "minimum_ice_to_grid_edge_m": edge_distance,
        "observed_length_m": observed_length,
        "registered_length_relative_error": length_error,
        "root_first_active_node": first,
        "root_last_active_node": last,
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
