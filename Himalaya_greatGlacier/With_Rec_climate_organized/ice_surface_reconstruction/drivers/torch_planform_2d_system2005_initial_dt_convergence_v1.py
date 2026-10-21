"""Initial timestep-convergence screen for system 2005's PyTorch 2-D core."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

from .torch_planform_2d_frozen_diffusivity_core_v2 import frozen_diffusivity_step_chunked
from .torch_planform_2d_system2005_static_grid_smoke_v1 import (
    GLEN_N,
    GRAVITY_M_S2,
    ICE_DENSITY_KG_M3,
    SYSTEM_ID,
    _sha256,
)


TIMESTEPS_SECONDS = (0.01, 0.03, 0.1)
REFERENCE_DT_SECONDS = 0.01
TENDENCY_RELATIVE_L2_GATE = 0.05
SOLVER_RTOL = 1e-11
SOLVER_MAXITER = 4000
CHECK_INTERVAL = 8


def run(geology: Path, static_summary_path: Path, sensitivity_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    static_summary = json.loads(static_summary_path.read_text(encoding="utf-8"))
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    if static_summary.get("status") != "static-pass":
        raise RuntimeError("source-static-not-pass")
    if sensitivity.get("status") != "sensitivity-complete-0p1y-not-authorized":
        raise RuntimeError("previous-sensitivity-not-complete")
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
    h0 = torch.from_numpy(thickness).to(device="cuda:0", dtype=torch.float64)
    bed_tensor = torch.from_numpy(bed).to(device="cuda:0", dtype=torch.float64)
    cell_area = dx**2
    initial_volume = float(torch.sum(h0).item()) * cell_area
    closure_tolerance = max(1e-6, initial_volume * 1e-12)
    torch.cuda.synchronize()

    states: dict[float, torch.Tensor] = {}
    records: dict[str, dict[str, object]] = {}
    for dt in TIMESTEPS_SECONDS:
        started = time.perf_counter()
        state, diagnostics = frozen_diffusivity_step_chunked(
            h0,
            bed_tensor,
            dx=dx,
            dy=dx,
            glen_n=GLEN_N,
            gamma=gamma,
            dt=dt,
            rtol=SOLVER_RTOL,
            maxiter=SOLVER_MAXITER,
            check_interval=CHECK_INTERVAL,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        states[dt] = state
        delta = state - h0
        closure = float(torch.sum(delta).item()) * cell_area
        checks = {
            "solver_converged": bool(diagnostics["solver_converged"]),
            "state_finite": bool(diagnostics["state_finite"]),
            "negative_cell_count_zero": int(diagnostics["negative_cell_count"]) == 0,
            "mass_closure_pass": abs(closure) <= closure_tolerance,
        }
        records[f"{dt:g}"] = {
            "dt_seconds": dt,
            "gpu_seconds": elapsed,
            "maximum_absolute_change_m": float(torch.max(torch.abs(delta)).item()),
            "maximum_absolute_tendency_m_s": float(torch.max(torch.abs(delta / dt)).item()),
            "mass_closure_m3": closure,
            "mass_closure_tolerance_m3": closure_tolerance,
            "diagnostics": diagnostics,
            "checks": checks,
            "numerical_pass": all(bool(value) for value in checks.values()),
        }

    reference_tendency = (states[REFERENCE_DT_SECONDS] - h0) / REFERENCE_DT_SECONDS
    reference_l2 = max(float(torch.linalg.vector_norm(reference_tendency).item()), 1e-30)
    passing_candidates: list[float] = []
    for dt in TIMESTEPS_SECONDS:
        tendency = (states[dt] - h0) / dt
        difference = tendency - reference_tendency
        relative_l2 = float(torch.linalg.vector_norm(difference).item()) / reference_l2
        maximum_difference = float(torch.max(torch.abs(difference)).item())
        accuracy_pass = relative_l2 <= TENDENCY_RELATIVE_L2_GATE
        records[f"{dt:g}"]["tendency_l2_relative_to_0p01s"] = relative_l2
        records[f"{dt:g}"]["tendency_max_abs_difference_from_0p01s_m_s"] = maximum_difference
        records[f"{dt:g}"]["accuracy_gate_pass"] = accuracy_pass
        if bool(records[f"{dt:g}"]["numerical_pass"]) and accuracy_pass:
            passing_candidates.append(dt)
    selected_dt = max(passing_candidates) if passing_candidates else None
    source_hash_after = _sha256(geology)
    completed = selected_dt is not None and source_hash_after == source_hash_before
    summary: dict[str, object] = {
        "schema": "lia-191-torch-planform2d-system2005-initial-dt-convergence-v1",
        "status": "initial-dt-selected-adaptive-60s-eligible" if completed else "initial-dt-convergence-fail-stop",
        "system_id": SYSTEM_ID,
        "timesteps_seconds": list(TIMESTEPS_SECONDS),
        "reference_dt_seconds": REFERENCE_DT_SECONDS,
        "tendency_relative_l2_gate": TENDENCY_RELATIVE_L2_GATE,
        "selected_initial_dt_seconds": selected_dt,
        "records": records,
        "source_geology_sha256_before": source_hash_before,
        "source_geology_sha256_after": source_hash_after,
        "runtime_clipping": False,
        "runtime_limiter": False,
        "state_saved": False,
        "steps_chained": False,
        "zero_mb_60s_run": False,
        "zero_mb_0p1y_run": False,
        "steady_run": False,
        "historical_run": False,
        "next_authorized_stage": "adaptive-step-doubling-zero-mb-60s" if completed else None,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--sensitivity-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.geology, arguments.static_summary, arguments.sensitivity_summary, arguments.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
