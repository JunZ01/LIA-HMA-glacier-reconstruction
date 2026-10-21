"""One real 30 s semi-implicit step for system 14 after fixture promotion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import planform_2d_frozen_diffusivity_semiimplicit_core_v1 as core


PROJECT = Path(__file__).resolve().parents[4]
STATE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_planform2d_source_native_padded_static_v1/system_14/source_native_padded_initial_state.npz"
STATIC = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_planform2d_source_native_padded_static_v1/system_14/static_summary.json"
FIXTURE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_planform2d_frozen_diffusivity_fixtures_v2/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_planform2d_frozen_diffusivity_system14_step_v1"
GLEN_A = 6.642407336686561e-23
DT_SECONDS = 30.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    if fixture.get("status") != "fixture-pass-real-system-authorized":
        raise RuntimeError("fixture-gate-not-passed")
    if static.get("status") != "static-pass":
        raise RuntimeError("static-gate-not-passed")

    scipy_cg = core.cg

    def cg_compat(operator, rhs, rtol=1e-5, atol=0.0, maxiter=None, M=None):
        return scipy_cg(operator, rhs, tol=rtol, atol=atol, maxiter=maxiter, M=M)

    core.cg = cg_compat
    state = np.load(STATE)
    bed = np.asarray(state["bed_elevation_m"], dtype=float)
    thickness0 = np.asarray(state["thickness_m"], dtype=float)
    dx = float(state["dx_m"])

    from oggm import cfg
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
    thickness1, diagnostics = core.frozen_diffusivity_step(
        model, DT_SECONDS, rtol=1e-12, maxiter=4000
    )
    cell_area = dx * dx
    volume0 = float(np.sum(thickness0) * cell_area)
    volume1 = float(np.sum(thickness1) * cell_area)
    volume_residual = float(volume1 - volume0)
    closure_gate = max(1e-6, volume0 * 1e-12)
    negative = thickness1 < -1e-7
    positive = thickness1 > 1e-6
    rows, cols = np.nonzero(positive)
    edge_distance = float(
        np.minimum.reduce(
            [rows, positive.shape[0] - 1 - rows, cols, positive.shape[1] - 1 - cols]
        ).min()
        * dx
    )
    border_max = float(
        max(
            np.max(thickness1[0]),
            np.max(thickness1[-1]),
            np.max(thickness1[:, 0]),
            np.max(thickness1[:, -1]),
        )
    )
    face_symmetry = diagnostics["face_symmetry"]
    checks = {
        "fixture_pass": True,
        "static_pass": True,
        "solver_converged": diagnostics["solver_info"] == 0,
        "matrix_finite": diagnostics["matrix_finite"],
        "matrix_positive_diagonal": diagnostics["matrix_positive_diagonal"],
        "matrix_symmetric": diagnostics["matrix_symmetric_max_abs"] == 0.0,
        "face_symmetry": face_symmetry["east_max_relative_difference"] <= 1e-14
        and face_symmetry["south_max_relative_difference"] <= 1e-14,
        "mass_conservation": abs(volume_residual) <= closure_gate,
        "nonnegative_without_clipping": not np.any(negative),
        "finite": bool(np.isfinite(thickness1).all()),
        "boundary": border_max <= 10.0 and edge_distance >= 5000.0,
    }
    passed = bool(all(checks.values()))
    if passed:
        np.savez_compressed(
            OUTPUT / "state_after_30s.npz",
            bed_elevation_m=bed.astype(np.float32),
            thickness_m=thickness1.astype(np.float32),
            dx_m=np.float64(dx),
            elapsed_seconds=np.float64(DT_SECONDS),
        )
    summary = {
        "schema": "lia-191-planform2d-frozen-diffusivity-system14-step-v1",
        "status": "real-step-pass" if passed else "real-step-fail-mechanism-closed",
        "system_id": 14,
        "dt_seconds": DT_SECONDS,
        "source_state_sha256": sha256(STATE),
        "static_summary_sha256": sha256(STATIC),
        "fixture_summary_sha256": sha256(FIXTURE),
        "initial_volume_m3": volume0,
        "final_volume_m3": volume1,
        "volume_residual_m3": volume_residual,
        "mass_closure_gate_m3": closure_gate,
        "negative_cell_count_below_minus_1e7m": int(np.sum(negative)),
        "minimum_thickness_m": float(np.min(thickness1)),
        "positive_cell_count": int(np.sum(positive)),
        "minimum_ice_to_grid_edge_m": edge_distance,
        "border_maximum_thickness_m": border_max,
        "diagnostics": diagnostics,
        "checks": checks,
        "clipping_used": False,
        "mass_correction_used": False,
        "next_authorized_stage": "zero-mb-0p1y" if passed else None,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
