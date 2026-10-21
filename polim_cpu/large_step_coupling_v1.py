"""Damped thickness--BP correction for the phase-1 large-step prototype."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Any

import numpy as np

from .large_step_transport_v1 import (
    ExistingMeshFluxAdapter,
    ThicknessStep,
    diffusion_conductance_from_flux,
    diffusion_predict,
    implicit_conservative_step,
)


Array = np.ndarray
BPSolver = Callable[[Array, Array], tuple[Array, Mapping[str, Any]]]


class CouplingConvergenceError(RuntimeError):
    def __init__(self, message: str, history: list[dict[str, float | int | str]]):
        super().__init__(message)
        self.history = history


@dataclass(frozen=True)
class CoupledStep:
    thickness: Array
    velocity: Array
    actual_smb: Array
    boundary_export_volume: float
    actual_source_volume: float
    mass_relative: float
    thickness_equation_relative: float
    bp_relative: float
    outer_iterations: int
    bp_calls: int
    predictor: str
    history: tuple[Mapping[str, Any], ...]


def _coupling_residual(
    candidate: Array,
    fixed: ThicknessStep,
    old_volume: float,
    source_scale: float,
    flux_scale: float,
) -> float:
    # ``fixed.actual_smb`` contains the complementarity multiplier from the
    # exact fixed-velocity solve.  Applying the same source to the outer
    # candidate exposes the local coupled equation error without cancellation.
    area_rhs = fixed.rhs_requested + fixed.unmet_ablation_volume
    residual = np.asarray(fixed.operator @ candidate - area_rhs)
    scale = max(old_volume + abs(source_scale) + abs(flux_scale), 1.0)
    return float(np.abs(residual).sum() / scale)


def solve_coupled_large_step(
    thickness_old: Array,
    velocity_old: Array,
    bed: Array,
    smb_requested: Array,
    dt_years: float,
    flux_adapter: ExistingMeshFluxAdapter,
    bp_solve: BPSolver,
    *,
    use_diffusion_predictor: bool = True,
    damping: float = 0.7,
    maximum_bp_calls: int = 8,
    bp_tolerance: float = 1e-6,
    thickness_equation_tolerance: float = 1e-8,
) -> CoupledStep:
    """Solve one large step and accept only the joint BP/mass fixed point.

    ``bp_solve(H, u_guess)`` must solve the existing, unmodified BP equations
    and return ``(u, info)`` with ``info['relative_residual']``.  Diffusion is
    used only to select the first ``H`` passed to that callback.
    """
    h_old = np.asarray(thickness_old, dtype=np.float64)
    u_old = np.asarray(velocity_old, dtype=np.float64)
    b = np.asarray(bed, dtype=np.float64)
    smb = np.asarray(smb_requested, dtype=np.float64)
    n = flux_adapter.topology.size
    if h_old.shape != (n,) or b.shape != (n,) or smb.shape != (n,):
        raise ValueError("thickness, bed and SMB must match the transport mesh")
    if not 0 < damping <= 1:
        raise ValueError("damping must lie in (0, 1]")
    if maximum_bp_calls < 1:
        raise ValueError("maximum_bp_calls must be positive")
    if np.any(h_old < 0) or not all(
        np.isfinite(x).all() for x in (h_old, u_old, b, smb)
    ):
        raise ValueError("coupled-step inputs must be finite and H must be non-negative")

    old_rates = flux_adapter.rates(u_old)
    if use_diffusion_predictor:
        conductance = diffusion_conductance_from_flux(h_old, b, old_rates)
        h = diffusion_predict(
            h_old, b, smb, dt_years, flux_adapter.topology, conductance
        )
        predictor = "implicit-local-diffusion"
    else:
        h = h_old.copy()
        predictor = "disabled"

    u_guess = u_old.copy()
    area = flux_adapter.topology.dual_area
    old_volume = float(area @ h_old)
    history: list[dict[str, float | int | str]] = []
    for outer in range(1, maximum_bp_calls + 1):
        u, info = bp_solve(h.copy(), u_guess.copy())
        u = np.asarray(u, dtype=np.float64)
        bp_relative = float(info["relative_residual"])
        if not np.isfinite(u).all() or not np.isfinite(bp_relative):
            raise RuntimeError("BP callback returned a non-finite result")
        rates = flux_adapter.rates(u)
        fixed = implicit_conservative_step(h_old, smb, dt_years, rates)
        flux_scale = float(dt_years) * rates.absolute_flux_rate(h)
        equation_relative = _coupling_residual(
            h,
            fixed,
            old_volume,
            fixed.actual_source_volume,
            flux_scale,
        )
        fixed_point_relative = float(
            np.sum(area * np.abs(fixed.thickness - h)) / max(old_volume, 1.0)
        )
        exported = float(dt_years) * rates.boundary_export_rate(h)
        source = float(dt_years) * (area @ fixed.actual_smb)
        mass_error = float(area @ h - old_volume - source + exported)
        mass_relative = abs(mass_error) / max(
            old_volume, abs(source), abs(exported), 1.0
        )
        row: dict[str, float | int | str] = {
            "outer_iteration": outer,
            "bp_relative": bp_relative,
            "thickness_equation_relative": equation_relative,
            "fixed_point_relative": fixed_point_relative,
            "transport_mass_relative": fixed.mass_relative,
            "candidate_mass_relative": mass_relative,
            "active_iterations": fixed.active_iterations,
            "bp_newton": int(info.get("iterations", 0)),
            "bp_cg": int(
                sum(x.get("cg_iterations", 0) for x in info.get("history", []))
            ),
        }
        history.append(row)
        if (
            bp_relative <= bp_tolerance
            and equation_relative <= thickness_equation_tolerance
            and fixed_point_relative <= thickness_equation_tolerance
            and mass_relative <= 1e-10
        ):
            if np.any(h < 0):
                raise RuntimeError("joint candidate failed non-negativity gate")
            return CoupledStep(
                thickness=h,
                velocity=u,
                actual_smb=fixed.actual_smb,
                boundary_export_volume=exported,
                actual_source_volume=source,
                mass_relative=mass_relative,
                thickness_equation_relative=equation_relative,
                bp_relative=bp_relative,
                outer_iterations=outer,
                bp_calls=outer,
                predictor=predictor,
                history=tuple(history),
            )
        h = (1.0 - damping) * h + damping * fixed.thickness
        if np.any(h < 0):
            raise RuntimeError("convex outer update violated non-negativity")
        u_guess = u
    raise CouplingConvergenceError(
        "large-step BP/thickness correction exceeded the frozen BP-call budget",
        history,
    )
