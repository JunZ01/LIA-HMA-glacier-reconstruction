"""GPU-process outer correction using isolated implicit transport."""
from __future__ import annotations

import numpy as np


class GPUCouplingConvergenceError(RuntimeError):
    def __init__(self, history):
        super().__init__("large-step BP correction exceeded the frozen BP-call budget")
        self.history = history


def solve_gpu_coupled_step(
    thickness_old,
    velocity_old,
    bed,
    smb_requested,
    dt_years,
    transport_service,
    bp_solver,
    *,
    use_diffusion_predictor=True,
    damping=0.7,
    maximum_bp_calls=8,
    bp_tolerance=1e-6,
    thickness_equation_tolerance=1e-8,
):
    """Run the same joint gates as the CPU reference without importing SciPy."""
    h_old = np.asarray(thickness_old, dtype=np.float64)
    u_guess = np.asarray(velocity_old, dtype=np.float64)
    if use_diffusion_predictor:
        h = transport_service.predict(
            h_old, u_guess, np.asarray(bed), np.asarray(smb_requested), dt_years
        )
        predictor = "implicit-local-diffusion"
    else:
        h = h_old.copy()
        predictor = "disabled"
    history = []
    for outer in range(1, maximum_bp_calls + 1):
        u, info = bp_solver(h, u_guess)
        correction = transport_service.correct(
            h_old, h, u, np.asarray(smb_requested), dt_years
        )
        row = {
            "outer_iteration": outer,
            "bp_relative": float(info["relative_residual"]),
            "thickness_equation_relative": float(
                correction["thickness_equation_relative"]
            ),
            "fixed_point_relative": float(correction["fixed_point_relative"]),
            "mass_relative": float(correction["mass_relative"]),
            "active_iterations": int(correction["active_iterations"]),
            "bp_newton": int(info.get("iterations", 0)),
            "bp_cg": int(
                sum(x.get("cg_iterations", 0) for x in info.get("history", []))
            ),
        }
        history.append(row)
        if (
            row["bp_relative"] <= bp_tolerance
            and row["thickness_equation_relative"] <= thickness_equation_tolerance
            and row["fixed_point_relative"] <= thickness_equation_tolerance
            and row["mass_relative"] <= 1e-10
        ):
            return {
                "thickness": h,
                "velocity": u,
                "actual_smb": correction["actual_smb"],
                "boundary_export_volume": float(correction["boundary_export_volume"]),
                "actual_source_volume": float(correction["actual_source_volume"]),
                "mass_relative": row["mass_relative"],
                "thickness_equation_relative": row[
                    "thickness_equation_relative"
                ],
                "bp_relative": row["bp_relative"],
                "outer_iterations": outer,
                "bp_calls": outer,
                "predictor": predictor,
                "history": history,
            }
        h = (1.0 - damping) * h + damping * correction["thickness"]
        if np.any(h < 0):
            raise RuntimeError("convex outer correction violated non-negativity")
        u_guess = u
    raise GPUCouplingConvergenceError(history)
